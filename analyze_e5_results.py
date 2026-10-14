"""Import and independently verify measured E5 results, without loading a GPU model."""
import argparse
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_results import compare, semantic_content
from e5_reward import reward_details
from kaggle_utils import write_json, write_summary, yolo_to_box
from scripts.labcommon import read_jsonl
from scripts.score import evaluate, _gt_boxes, _match, parse_output

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def import_e5(path):
    provenance = ROOT / "report/provenance/e5"
    provenance.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(path) as archive:
        status = json.loads(archive.read("report/e5_status.json"))
        if status["status"] != "finished" or status["smoke_run"]:
            raise ValueError("Archive does not contain a finished full E5 run")
        for item in archive.infolist():
            name = item.filename
            if item.is_dir() or not name.startswith(("report/", "work/", "data/splits/")):
                continue
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError(f"Unsafe archive path: {name}")
            data = archive.read(item)
            if target.exists():
                current = target.read_bytes()
                if current != data and semantic_content(name, current) != semantic_content(name, data):
                    raise ValueError(f"Existing artifact differs: {name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for name in ("kaggle_e5.py", "e5_reward.py", "e5_trainer.py", "requirements-e5.txt", "e5_run_config.json"):
            data = archive.read(name)
            (provenance / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        # Analyze the reward implementation that actually ran, not a newer formula.
        if archive.read("e5_reward.py").decode().splitlines() != (ROOT / "e5_reward.py").read_text(encoding="utf-8").splitlines():
            raise ValueError("Executed reward differs from local code; inspect before recomputing rollouts")
    write_json(provenance / "archive.json", {"file": Path(path).name,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "executed_source_sha256": hashes,
        "reward_source_matches_local": True})


def verify_and_analyze(archive_path):
    cfg = load("report/provenance/e5/e5_run_config.json")
    run = load("work/e5/run.json")
    args = load("work/e5/grpo_args.json")
    meta = load("data/splits/meta.json")
    assert cfg["baseline"] == "e3" and cfg["dataset_version"] == meta["dataset_version"]
    assert cfg["max_steps"] == run["trainer_global_steps"] == 60
    assert args["num_generations"] == args["generation_batch_size"] == 4
    assert args["max_completion_length"] == 128 and args["beta"] == 0 and not args["use_vllm"]
    assert args["fp16"] and not args["bf16"]
    assert load("work/e5/precision_check.json")["verified"]
    assert load("work/e5/projector_frozen_check.json")["verified"]
    for split in ("val", "test"):
        rows = read_jsonl(ROOT / f"data/splits/{split}.jsonl")
        suffix = "" if split == "test" else "_val"
        for name in ("e3", "e5"):
            preds = read_jsonl(ROOT / f"work/{name}{suffix}.jsonl")
            ids = [r["image_id"] for r in preds]
            assert len(set(ids)) == len(ids) == len(rows)
            assert set(ids) == {r["image_id"] for r in rows}
            out = evaluate(rows, {r["image_id"]: r["raw"] for r in preds})
            ev = load(f"report/{'evidence' if split == 'test' else 'validation'}/{name}.json")
            compare(out, ev, f"{name}/{split}")
            assert ev["dataset_version"] == cfg["dataset_version"]
            if name == "e5":
                for key in ("trainable_params", "train_minutes", "peak_vram_gb"):
                    assert ev["meta"][key] == run[key]
    updates = read_jsonl(ROOT / "work/e5/optimizer_steps.jsonl")
    assert [r["trainer_step"] for r in updates] == list(range(1, 61))
    skipped = [r["trainer_step"] for r in updates if r["update_skipped"]]
    assert len(skipped) == run["amp_skipped_updates"] == 2
    assert run["successful_optimizer_updates"] == 60 - len(skipped) == 58
    train = {r["image_id"]: r for r in read_jsonl(ROOT / "data/splits/train.jsonl")}
    rollouts = read_jsonl(ROOT / "work/e5/rollouts.jsonl")
    assert len(rollouts) == 240
    for r in rollouts:
        row = train[r["image_id"]]
        valid = [o for o in row["objects"] if yolo_to_box(o["label"], o["yolo"]) is not None]
        recomputed = reward_details(r["raw"], _gt_boxes({**row, "objects": valid}))
        compare(recomputed, r, "rollout")
        assert r["hit_token_limit"] == (r["completion_tokens"] >= 128)
    groups = [rollouts[i:i + 4] for i in range(0, len(rollouts), 4)]
    for step, group in enumerate(groups):
        assert {r["trainer_step"] for r in group} == {step}
        assert len({r["image_id"] for r in group}) == 1
    logs = read_jsonl(ROOT / "work/e5/train_log.jsonl")
    steps = [r for r in logs if "loss" in r]
    assert len(steps) == 60 and all(math.isfinite(r["loss"]) for r in steps)
    for row, group in zip(steps, groups):
        assert math.isclose(row["reward"], float(np.mean([r["reward"] for r in group])), abs_tol=1e-7)
    import kaggle_e5
    original = load("report/e5_comparison.json")
    kaggle_e5.ROOT = ROOT
    kaggle_e5.config = lambda: cfg
    # Recompute score audit and paired CI without invoking the CLI's export step.
    kaggle_e5.compare_e5()
    measured = load("report/e5_comparison.json")
    compare(measured, original, "e5_comparison")
    # Preserve the byte-identical submitted comparison as the primary evidence.
    with zipfile.ZipFile(archive_path) as archive:
        (ROOT / "report/e5_comparison.json").write_bytes(archive.read("report/e5_comparison.json"))
        (ROOT / "report/e5_status.json").write_bytes(archive.read("report/e5_status.json"))
    result = {"all_verified": True, "score_checks": ["e3/val", "e3/test", "e5/val", "e5/test"],
              "recomputed_rewards": 240, "groups": 60, "unique_train_images": len({r["image_id"] for r in rollouts}),
              "trainer_global_steps": 60, "successful_updates": 58, "skipped_steps": skipped,
              "reward_first_10_groups": float(np.mean([r["reward"] for r in rollouts[:40]])),
              "reward_last_10_groups": float(np.mean([r["reward"] for r in rollouts[-40:]])),
              "mean_completion_tokens": float(np.mean([r["completion_tokens"] for r in rollouts])),
              "zero_variance_groups": sum(len({r["reward"] for r in g}) == 1 for g in groups),
              "invalid_and_at_limit": sum(not r["parse_valid"] and r["hit_token_limit"] for r in rollouts),
              "giant_rollouts": [r for r in rollouts if r["giant_boxes"]],
              "comparison": original, "test_per_session": {}}
    rows = read_jsonl(ROOT / "data/splits/test.jsonl")
    for session in sorted({r["session"] for r in rows}):
        selected = [r for r in rows if r["session"] == session]
        data = {"n_images": len(selected), "f1": {}}
        for name in ("e3", "e5"):
            pred = {r["image_id"]: r["raw"] for r in read_jsonl(ROOT / f"work/{name}.jsonl")}
            data["f1"][name] = evaluate(selected, pred, n_boot=100)["micro"]["f1"]
        result["test_per_session"][session] = data
    resources = ROOT / "report/resources/e5"
    resources.mkdir(parents=True, exist_ok=True)
    for file in (ROOT / "work/e5").iterdir():
        if file.is_file() and file.suffix in (".json", ".jsonl"):
            shutil.copy2(file, resources / file.name)
    write_json(ROOT / "report/analysis/e5_verified_analysis.json", result)
    write_summary(ROOT / "report/evidence", ROOT / "report/analysis/summary_with_e5.csv")
    plots(groups, result)
    print(json.dumps({k: v for k, v in result.items() if k != "comparison"}, ensure_ascii=False, indent=2))


def plots(groups, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    e3, e5 = [load(f"report/evidence/{n}.json") for n in ("e3", "e5")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for i, (name, ev) in enumerate(zip(("E3 SFT", "E5 GRPO"), (e3, e5))):
        value = ev["micro"]["f1"]
        axes[0].errorbar(i, value, yerr=[[value - ev["f1_ci95"][0]], [ev["f1_ci95"][1] - value]],
                         fmt="o", capsize=6, color=("#247a63", "#ae6a26")[i], markersize=9)
        axes[0].text(i, 0.487, f"F1={value:.4f}", ha="center")
    axes[0].set(xticks=[0, 1], xticklabels=["E3 SFT", "E5 GRPO"], xlim=(-0.6, 1.6), ylim=(0.37, 0.51),
                ylabel="Test micro F1", title="Official 95% CI overlaps")
    x = np.arange(3)
    for i, (name, ev) in enumerate(zip(("E3", "E5"), (e3, e5))):
        axes[1].bar(x + (i - 0.5) * 0.32, [ev["micro"][k] for k in ("precision", "recall", "f1")],
                    width=0.32, label=name)
    axes[1].set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylim=(0, 0.6),
                title="More recall, lower precision")
    axes[1].legend(frameon=False)
    means = [np.mean([r["reward"] for r in g]) for g in groups]
    axes[2].plot(np.arange(1, 61), means, color="#a4aab0", alpha=0.65, label="Group mean")
    axes[2].plot(np.arange(10, 61), np.convolve(means, np.ones(10) / 10, "valid"),
                 color="#3969a0", label="10-group mean")
    axes[2].set(xlabel="Trainer global step", ylabel="Sampled training IoU reward", ylim=(0, 1),
                title="Different train image per group")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "report/analysis/e5_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="lab_day10_e5_results.zip")
    args = parser.parse_args()
    import_e5(args.archive)
    verify_and_analyze(args.archive)
