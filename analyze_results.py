"""Verify the returned Kaggle artifacts and produce reproducible report analyses."""
import argparse
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from kaggle_utils import fingerprint, write_json, assert_disjoint_sessions
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate, parse_output, _gt_boxes, _match, iou_xyxy

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ("e0", "e1", "e2", "e3", "e4_data_size_10")


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def semantic_content(name, data):
    if name.endswith(".jsonl"):
        text = data.decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if name.endswith(".json"):
        return json.loads(data.decode("utf-8"))
    return data


def import_archive(path):
    with zipfile.ZipFile(path) as archive:
        cfg = json.loads(archive.read("kaggle_run_config.json"))
        code_hash = hashlib.sha256(archive.read("kaggle_lab.py") + archive.read("kaggle_utils.py")).hexdigest()
        assert code_hash == cfg["code_sha256"], "Executed source fingerprint mismatch"
        for item in archive.infolist():
            name = item.filename
            if item.is_dir() or not name.startswith(("report/", "work/", "data/splits/")):
                continue
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f"Unsafe archive path: {name}")
            data = archive.read(item)
            if target.exists():
                if semantic_content(name, target.read_bytes()) != semantic_content(name, data):
                    raise ValueError(f"Existing artifact differs: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        provenance = ROOT / "report/provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        for name in ("kaggle_lab.py", "kaggle_utils.py", "kaggle_run_config.json"):
            (provenance / name).write_bytes(archive.read(name))
        write_json(provenance / "archive.json", {
            "file": Path(path).name, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "executed_code_sha256": code_hash, "source_fingerprint_verified": True})


def compare(actual, expected, prefix=""):
    if isinstance(actual, dict):
        for key in actual:
            compare(actual[key], expected[key], prefix + "." + key)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), prefix
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, prefix + f"[{i}]")
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, abs_tol=1e-12), (prefix, actual, expected)
    else:
        assert actual == expected, (prefix, actual, expected)


def prf(counts):
    tp, fp, fn = counts
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def analyze():
    analysis = ROOT / "report/analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    splits = {s: read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
    sessions = assert_disjoint_sessions(splits)
    cfg = load("report/provenance/kaggle_run_config.json")
    result = {"verified_scores": [], "sessions": sessions, "experiments": {}, "paired_bootstrap": {}}
    counts = {}
    for name in EXPERIMENTS:
        for split in ("val", "test"):
            suffix = "" if split == "test" else "_val"
            rows = read_jsonl(ROOT / f"work/{name}{suffix}.jsonl")
            ids = [r["image_id"] for r in rows]
            assert len(ids) == len(set(ids)) == len(splits[split]), (name, split)
            assert set(ids) == {r["image_id"] for r in splits[split]}, (name, split)
            pred = {r["image_id"]: r["raw"] for r in rows}
            evidence = load(f"report/{'evidence' if split == 'test' else 'validation'}/{name}.json")
            measured = evaluate(splits[split], pred)
            compare(measured, evidence, f"{name}/{split}")
            assert evidence["dataset_version"] == cfg["dataset_version"]
            result["verified_scores"].append(f"{name}/{split}")
        config = load(f"work/{name}/effective_config.json")
        assert load(f"work/{name}/completed.json")["fingerprint"] == fingerprint(config)
        run = load(f"work/{name}/run.json")
        if name != "e0":
            assert run["optimizer_steps"] == 300
            assert config["max_steps"] == 300 and config["batch_size"] * config["grad_accum"] == 8
            for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
                assert run[key] == evidence["meta"][key], (name, key)
        # Both parses and per-image counts follow the official scorer exactly.
        per_image = []
        diagnostics = Counter()
        for row in splits["test"]:
            parsed = parse_output(pred[row["image_id"]])
            if parsed is None:
                diagnostics["invalid_json_or_boxes"] += 1
            elif not parsed:
                diagnostics["empty_predictions"] += 1
            else:
                diagnostics["parsed_boxes"] += len(parsed)
                diagnostics["unknown_label_boxes"] += sum(o["label"] not in CLASSES for o in parsed)
                diagnostics["out_of_range_boxes"] += sum(any(v < 0 or v > 1000 for v in o["bbox"]) for o in parsed)
            per, _ = _match(_gt_boxes(row), parsed or [])
            per_image.append([sum(v[k] for v in per.values()) for k in range(3)])
        diagnostics["hit_token_limit_images"] = sum(r.get("hit_token_limit", False) for r in rows)
        counts[name] = np.array(per_image, dtype=np.int64)
        exp = {"output_diagnostics": dict(diagnostics), "inference": load(f"work/{name}/inference.json")}
        if name != "e0":
            logs = read_jsonl(ROOT / f"work/{name}/train_log.jsonl")
            valid = [r for r in logs if not r["update_skipped"]]
            assert len(valid) == 300 and all(math.isfinite(r["loss"]) for r in logs)
            exp["loss"] = {"mean_first_30_steps": float(np.mean([r["loss"] for r in valid[:30]])),
                           "mean_last_30_steps": float(np.mean([r["loss"] for r in valid[-30:]])),
                           "last_step": valid[-1]["loss"], "amp_skips": len(logs) - len(valid),
                           "attempts": len(logs), "approx_successful_image_presentations": 2400,
                           "approx_passes_per_training_image": 2400 / run["n_train_images"]}
        result["experiments"][name] = exp
    base = load("report/evidence/e3.json")
    ablation = load("report/evidence/e4_data_size_100.json")
    compare({k: v for k, v in base.items() if k not in ("name", "meta")}, ablation)
    result["verified_scores"].append("e4_data_size_100/test (reused E3)")
    # Paired image bootstrap uses the same resampled image indices for both models.
    # It conditions on this test set and does not model video-session dependence.
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(splits["test"]), size=(10000, len(splits["test"])))
    boot = {}
    for name, matrix in counts.items():
        totals = matrix[indices].sum(axis=1)
        boot[name] = 2 * totals[:, 0] / np.maximum(1, 2 * totals[:, 0] + totals[:, 1] + totals[:, 2])
    for newer, older in (("e1", "e0"), ("e2", "e1"), ("e3", "e2"), ("e3", "e1"), ("e3", "e4_data_size_10")):
        diff = boot[newer] - boot[older]
        result["paired_bootstrap"][f"{newer}-{older}"] = {
            "delta_f1": prf(counts[newer].sum(axis=0)) - prf(counts[older].sum(axis=0)),
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(), "n_boot": 10000, "seed": 42,
            "unit": "image (paired); session dependence not accounted for"}
    result["test_per_session"] = {}
    for session in sorted({r["session"] for r in splits["test"]}):
        positions = [i for i, r in enumerate(splits["test"]) if r["session"] == session]
        result["test_per_session"][session] = {
            "n_images": len(positions),
            "n_gt_boxes": sum(len(splits["test"][i]["objects"]) for i in positions),
            "micro_f1": {name: prf(matrix[positions].sum(axis=0)) for name, matrix in counts.items()}}
    result["dataset_box_area"] = {}
    for split, rows in splits.items():
        classes = {}
        for c in CLASSES:
            areas = [o["yolo"][2] * o["yolo"][3] for r in rows for o in r["objects"] if o["label"] == c]
            classes[c] = {"count": len(areas), "median_percent_frame": float(np.median(areas) * 100),
                          "median_pixels": float(np.median(areas) * 640 * 360)}
        result["dataset_box_area"][split] = classes
    subset_ids = set(load("work/e4_data_size_10/train_image_ids.json"))
    train_ids = {r["image_id"] for r in splits["train"]}
    assert len(subset_ids) == 142 and subset_ids <= train_ids
    subset = [r for r in splits["train"] if r["image_id"] in subset_ids]
    result["e4_subset"] = {"n_images": len(subset), "n_sessions": len({r["session"] for r in subset}),
                           "boxes": dict(Counter(o["label"] for r in subset for o in r["objects"]))}
    e3_pred = {r["image_id"]: parse_output(r["raw"]) or [] for r in read_jsonl(ROOT / "work/e3.jsonl")}
    outcomes = {c: {"matched_areas": [], "missed_areas": [], "miss_best_same_class_iou_below_01": 0,
                    "miss_best_same_class_iou_01_to_05": 0} for c in CLASSES}
    for row in splits["test"]:
        gt = _gt_boxes(row)
        pred = e3_pred[row["image_id"]]
        matched = set()
        for lab in CLASSES:
            g = [(i, o["bbox"]) for i, o in enumerate(gt) if o["label"] == lab]
            p = [o["bbox"] for o in pred if o["label"] == lab]
            pairs = sorted((iou_xyxy(a, b), i, j) for i, a in g for j, b in enumerate(p))[::-1]
            used_p = set()
            for v, i, j in pairs:
                if v < 0.5:
                    break
                if i not in matched and j not in used_p:
                    matched.add(i)
                    used_p.add(j)
            for i, a in g:
                obj = row["objects"][i]
                area = obj["yolo"][2] * obj["yolo"][3] * 100
                outcomes[lab]["matched_areas" if i in matched else "missed_areas"].append(area)
                if i not in matched:
                    best = max((iou_xyxy(a, b) for b in p), default=0)
                    if best < 0.1:
                        outcomes[lab]["miss_best_same_class_iou_below_01"] += 1
                    elif best < 0.5:
                        outcomes[lab]["miss_best_same_class_iou_01_to_05"] += 1
    for c, d in outcomes.items():
        for key in ("matched_areas", "missed_areas"):
            areas = d.pop(key)
            d[key.replace("areas", "count")] = len(areas)
            d[key.replace("areas", "median_percent_frame")] = float(np.median(areas)) if areas else None
    result["e3_gt_outcomes"] = outcomes
    result["all_verified"] = True
    # Preserve compact training evidence under report/ so GitHub need not contain work/.
    for name in EXPERIMENTS:
        dest = ROOT / "report/resources" / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ("run.json", "architecture.json", "effective_config.json", "inference.json",
                         "mask_check.json", "projector_reload.json", "trainable_parameters.json",
                         "checkpoint.json", "train_log.jsonl", "train_image_ids.json"):
            source = ROOT / "work" / name / filename
            if source.exists():
                shutil.copy2(source, dest / filename)
    write_json(analysis / "verified_analysis.json", result)
    plots(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def plots(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    names = [*EXPERIMENTS, "e4_data_size_100"]
    ev = [load(f"report/evidence/{n}.json") for n in names]
    labels = ["E0", "E1", "E2", "E3", "E4 10%", "E4 100%\n(reused E3)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    values = np.array([d["micro"]["f1"] for d in ev])
    lower = values - np.array([d["f1_ci95"][0] for d in ev])
    upper = np.array([d["f1_ci95"][1] for d in ev]) - values
    colors = ["#9298a1", "#448ac0", "#279e93", "#186951", "#d08c42", "#186951"]
    axes[0].bar(labels, values, color=colors, yerr=[lower, upper], capsize=4)
    axes[0].set(ylabel="Micro F1 at IoU >= 0.5", ylim=(0, 0.55), title="Test: official image-bootstrap 95% CI")
    for i, v in enumerate(values):
        axes[0].text(i, v + upper[i] + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    x = np.arange(4)
    for j, c in enumerate(CLASSES):
        axes[1].bar(x + (j - 1) * 0.23, [d["per_class"][c]["f1"] for d in ev[:4]],
                    width=0.23, label=c)
    axes[1].set(xticks=x, xticklabels=labels[:4], ylabel="Class F1", ylim=(0, 0.85),
                title="Manhole improves most; pothole remains hardest")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "report/analysis/f1_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for name in EXPERIMENTS[1:]:
        logs = [r for r in read_jsonl(ROOT / f"work/{name}/train_log.jsonl") if not r["update_skipped"]]
        losses = np.array([r["loss"] for r in logs])
        smooth = np.convolve(losses, np.ones(20) / 20, mode="valid")
        axes[0].plot(np.arange(20, len(losses) + 1), smooth, label=name.upper())
    axes[0].set(xlabel="Successful optimizer steps", ylabel="Training loss (20-step mean)",
                title="10% data fits training targets much more closely")
    axes[0].legend(frameon=False)
    for name, color in zip(EXPERIMENTS[1:4], colors[1:4]):
        d = load(f"report/evidence/{name}.json")
        axes[1].scatter(d["meta"]["train_minutes"], d["micro"]["f1"], s=140, color=color)
        axes[1].annotate(f"{name.upper()}\n{d['meta']['peak_vram_gb']:.2f} GiB",
                         (d["meta"]["train_minutes"], d["micro"]["f1"]), xytext=(6, 6),
                         textcoords="offset points")
    axes[1].set(xlabel="Training wall time (minutes)", ylabel="Test micro F1",
                title="Measured trade-off on one Kaggle T4", xlim=(48, 69), ylim=(0.23, 0.49))
    fig.tight_layout()
    fig.savefig(ROOT / "report/analysis/loss_and_resources.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="lab_day10_results.zip")
    args = ap.parse_args()
    import_archive(args.archive)
    analyze()
