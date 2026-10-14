"""CPU-only data and artifact helpers for the student Kaggle implementation."""
import csv
import hashlib
import json
import random
import re
from pathlib import Path

LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"
PROJECTORS = ("visual.merger", "visual.deepstack_merger_list")
PROMPT = (
    'Detect all road damage: pothole, crack, and manhole. Return ONLY a JSON list '
    'of objects with keys "label" and "bbox_2d". bbox_2d is [x1,y1,x2,y2], '
    'integer coordinates normalized to 0-1000 independently for image width and height. '
    'Use only these three labels. Return [] if there are no objects. No explanations.'
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def yolo_to_box(label, yolo):
    xc, yc, w, h = yolo
    coords = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
    box = [max(0, min(1000, round(v * 1000))) for v in coords]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects):
    boxes = [box for o in objects if (box := yolo_to_box(o["label"], o["yolo"])) is not None]
    boxes.sort(key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def subset_train(records, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    if fraction == 1:
        return list(records)
    indices = random.Random(seed).sample(range(len(records)), max(1, int(len(records) * fraction)))
    return [records[i] for i in sorted(indices)]


def assert_disjoint_sessions(splits):
    sessions = {s: {r["session"] for r in rows} for s, rows in splits.items()}
    names = list(sessions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sessions[a] & sessions[b]:
                raise ValueError(f"Session leakage between {a} and {b}")
    return {s: len(v) for s, v in sessions.items()}


def target_names(names, include_vit=False):
    regex = LLM_TARGET if not include_vit else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    return [n for n in names if re.fullmatch(regex, n)]


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def summary_rows(evidence_dir):
    rows = []
    for p in sorted(Path(evidence_dir).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "experiment": p.stem, "split": d["split"], "f1": d["micro"]["f1"],
            "ci95_low": d["f1_ci95"][0], "ci95_high": d["f1_ci95"][1],
            "fp_per_image": d["fp_per_image"], "parse_valid_rate": d["parse_valid_rate"],
            "missing_predictions": d["missing_predictions"],
            **{f"f1_{c}": d["per_class"][c]["f1"] for c in ("pothole", "crack", "manhole")},
            **{k: d.get("meta", {}).get(k, 0) for k in (
                "trainable_params", "peak_vram_gb", "train_minutes")},
        })
    return rows


def write_summary(evidence_dir, output):
    rows = summary_rows(evidence_dir)
    if rows:
        with Path(output).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
