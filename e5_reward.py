"""Pure Python IoU reward specified by GUIDE step 9; no GPU or TRL needed."""
import json
import math
from pathlib import Path

from scripts.score import parse_output, iou_xyxy


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(p.get("text", "") for p in content
                             if isinstance(p, dict) and p.get("type") == "text")
        return "".join(parts)
    return ""


def reward_details(completion, gt_boxes):
    """GT uses label/bbox in norm1000, same floating-point geometry as the scorer."""
    raw = completion_text(completion)
    pred = parse_output(raw)
    if pred is None:
        return {"reward": 0.0, "parse_valid": False, "n_pred": 0, "n_gt": len(gt_boxes),
                "empty": False, "matched_pairs": 0, "sum_iou": 0.0, "giant_boxes": 0}
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_gt, used_pred, total_iou = set(), set(), 0.0
    for value, i, j in pairs:
        if value < 0.5:
            break
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        total_iou += value
    reward = 1.0 if not gt_boxes and not pred else 0.2 + 0.8 * total_iou / max(len(gt_boxes), len(pred), 1)
    giant = sum((p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) >= 250000
                for p in pred)
    assert math.isfinite(reward) and 0 <= reward <= 1
    return {"reward": reward, "parse_valid": True, "n_pred": len(pred), "n_gt": len(gt_boxes),
            "empty": not pred, "matched_pairs": len(used_gt), "sum_iou": total_iou,
            "giant_boxes": giant}


def iou_reward(completion, gt_boxes):
    return reward_details(completion, gt_boxes)["reward"]


class RewardLogger:
    """TRL callable; stores rollouts locally and logs scalar hacking indicators."""
    __name__ = "iou_reward"

    def __init__(self, path, num_generations=4, max_completion_length=128):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length

    def __call__(self, completions, gt_boxes, image_id=None, completion_ids=None, **kwargs):
        if len(completions) != len(gt_boxes):
            raise ValueError("GT/completion batch lengths differ")
        ids = image_id if image_id is not None else [None] * len(completions)
        if len(ids) != len(completions):
            raise ValueError("image_id/completion batch lengths differ")
        details = [reward_details(c, g) for c, g in zip(completions, gt_boxes)]
        state = kwargs.get("trainer_state")
        step = getattr(state, "global_step", None)
        with self.path.open("a", encoding="utf-8") as log:
            for i, (c, d) in enumerate(zip(completions, details)):
                length = len(completion_ids[i]) if completion_ids is not None else None
                record = {"trainer_step": step, "image_id": ids[i], "raw": completion_text(c),
                          "completion_tokens": length,
                          "hit_token_limit": length is not None and length >= self.max_completion_length,
                          **d}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_metric = kwargs.get("log_metric")
        if log_metric and details:
            n = len(details)
            log_metric("reward_audit/parse_valid", sum(d["parse_valid"] for d in details) / n)
            log_metric("reward_audit/empty_on_nonempty_gt", sum(d["empty"] and d["n_gt"] > 0 for d in details) / n)
            log_metric("reward_audit/giant_box_image_rate", sum(d["giant_boxes"] > 0 for d in details) / n)
        return [d["reward"] for d in details]
