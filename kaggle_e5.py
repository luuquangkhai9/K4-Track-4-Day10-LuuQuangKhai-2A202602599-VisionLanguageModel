"""E5: continue SFT adapters with GRPO, verify checkpoint, score and audit reward."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

import kaggle_lab as lab
from kaggle_utils import PROJECTORS, PROMPT, assert_disjoint_sessions, write_json, yolo_to_box
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, evaluate, parse_output
from e5_reward import RewardLogger, reward_details

ROOT = Path(__file__).resolve().parent


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config():
    return load_json(ROOT / "e5_run_config.json")


def select_sft(root):
    """Use the original validation choice, never select E5 by test F1."""
    root = Path(root)
    vals = {n: load_json(root / f"report/validation/{n}.json") for n in ("e1", "e2", "e3")}
    best = max(vals, key=lambda n: vals[n]["micro"]["f1"])
    checkpoint = root / "work" / best
    run = load_json(checkpoint / "run.json")
    adapter = checkpoint / "adapter"
    if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"Missing SFT adapter: {adapter}. Add the OLD NOTEBOOK OUTPUT, not the results ZIP.")
    if best in ("e2", "e3") and not (checkpoint / "projector.pt").is_file():
        raise FileNotFoundError(f"Missing trained projector: {checkpoint / 'projector.pt'}")
    if run["optimizer_steps"] != 300 or run["status"] != "trained":
        raise ValueError("SFT source must be a completed 300-step run.")
    for split in ("test", "val"):
        suffix = "" if split == "test" else "_val"
        if not (root / "work" / f"{best}{suffix}.jsonl").is_file():
            raise FileNotFoundError(f"Missing cached SFT {split} predictions")
    return root, best, run


def find_sft(input_root, explicit=None):
    if explicit:
        return select_sft(Path(explicit))
    candidates = []
    for marker in Path(input_root).rglob("e4_selection.json"):
        if marker.parent.name == "report":
            root = marker.parent.parent
            best = load_json(marker)["selected_setup"]
            if (root / "work" / best / "adapter" / "adapter_config.json").exists():
                candidates.append(root)
    if len(candidates) != 1:
        raise FileNotFoundError("Need exactly one completed SFT notebook Output with weights. "
                                "Add the previous notebook Output or set SFT_SOURCE to its day10 directory. "
                                "lab_day10_results.zip has no model weights.")
    return select_sft(candidates[0])


def verify_optional_source_splits(source, splits):
    """Checkpoint exports may omit data/: compare original labels only if present."""
    checks = {}
    for split, rows in splits.items():
        path = Path(source) / f"data/splits/{split}.jsonl"
        if not path.is_file():
            checks[split] = "not_available_in_checkpoint_input"
            print(f"SFT {split} split not exported; using freshly prepared official split.", flush=True)
            continue
        if rows != read_jsonl(path):
            raise ValueError(f"Prepared {split} records/labels differ from the SFT source")
        checks[split] = "exact_records_match"
    return checks


def prepare():
    cfg = config()
    source, best, run = find_sft(cfg["input_root"], cfg.get("sft_source"))
    if run["config"]["prompt"] != PROMPT:
        raise ValueError("Current prompt differs from the saved SFT prompt; preserve it for fair evaluation")
    flag, dataset = lab.find_dataset(cfg["input_root"], cfg.get("data_source"))
    lab.command([sys.executable, "scripts/prepare_data.py", flag, dataset])
    splits = {s: read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
    assert_disjoint_sessions(splits)
    source_split_checks = verify_optional_source_splits(source, splits)
    meta = load_json(ROOT / "data/splits/meta.json")
    if meta["dataset_version"] != run["config"]["dataset_version"]:
        raise ValueError("Dataset version differs from SFT checkpoint")
    # Import cached SFT output, without re-running its test set to select/tune E5.
    for split in ("test", "val"):
        suffix = "" if split == "test" else "_val"
        pred_source = source / "work" / f"{best}{suffix}.jsonl"
        pred = read_jsonl(pred_source)
        ids = [r["image_id"] for r in pred]
        if len(set(ids)) != len(ids) or set(ids) != {r["image_id"] for r in splits[split]}:
            raise ValueError(f"Invalid SFT {split} image IDs")
        dest = ROOT / "work" / pred_source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_source, dest)
        evidence = source / f"report/{'evidence' if split == 'test' else 'validation'}/{best}.json"
        measured = evaluate(splits[split], {r["image_id"]: r["raw"] for r in pred})
        original = load_json(evidence)
        if original["dataset_version"] != meta["dataset_version"]:
            raise ValueError("SFT evidence dataset mismatch")
        if json.dumps(measured, sort_keys=True) != json.dumps(
                {k: original[k] for k in measured}, sort_keys=True):
            raise ValueError(f"SFT {split} evidence does not match raw predictions")
        out = ROOT / evidence.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evidence, out)
    train_bad = [{"image_id": r["image_id"], "object": o} for r in splits["train"]
                 for o in r["objects"] if yolo_to_box(o["label"], o["yolo"]) is None]
    cfg.update(sft_source=str(source), baseline=best, baseline_checkpoint=str(source / "work" / best),
               model_id=run["config"]["model_id"], model_revision=run["config"]["model_revision"],
               dataset_version=meta["dataset_version"], setup=best, seed=42,
               name="e5", max_new_tokens=run["config"]["max_new_tokens"], inference_batch_size=2,
               train_projector=False, train_policy="Continue SFT LoRA LLM/ViT; freeze restored SFT projector")
    write_json(ROOT / "e5_run_config.json", cfg)
    write_json(ROOT / "report/e5_warm_start.json", {
        "selected_by": "original full validation micro F1", "baseline": best,
        "test_used_for_selection": False, "baseline_run": run,
        "dataset_meta": meta, "degenerate_training_objects_skipped": train_bad,
        "source_split_checks": source_split_checks,
        "cached_sft_val_test_scores_reverified": True,
        "checkpoint_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "work" / best).rglob("*")
            if p.is_file() and p.suffix in (".pt", ".safetensors", ".bin")}})
    print(f"E5 starts from {best.upper()} at {cfg['baseline_checkpoint']}", flush=True)


def training_dataset(rows):
    from datasets import Dataset, Image as DatasetImage
    samples = []
    for row in rows:
        valid = [o for o in row["objects"] if yolo_to_box(o["label"], o["yolo"]) is not None]
        gt = _gt_boxes({**row, "objects": valid})
        samples.append({"prompt": [{"role": "user", "content": PROMPT}],
                        "image": str(ROOT / "data" / row["image_path"]),
                        "image_id": row["image_id"], "gt_boxes": gt})
    # Image column supplies PIL images; GRPO inserts them into conversational prompts.
    return Dataset.from_list(samples).cast_column("image", DatasetImage())


def load_policy(cfg, trainable):
    import torch
    from peft import PeftModel
    model, processor = lab.load_base(cfg)
    location = Path(cfg["baseline_checkpoint"]) if trainable else ROOT / "work/e5"
    model = PeftModel.from_pretrained(model, location / "adapter", is_trainable=trainable)
    params = dict(model.named_parameters())
    if cfg["setup"] in ("e2", "e3"):
        extras = torch.load(location / "projector.pt", map_location="cpu", weights_only=True)
        expected = {n for n in params if any(s in n for s in PROJECTORS)}
        if set(extras) != expected:
            raise ValueError("Projector checkpoint parameter names differ")
        for n, saved in extras.items():
            params[n].data = saved.to(params[n].device, dtype=torch.float32)
            params[n].requires_grad_(False)
        if not all(torch.equal(params[n].detach().cpu(), value) for n, value in extras.items()):
            raise ValueError("Projector reload not exact")
    if trainable:
        for n, p in model.named_parameters():
            if p.requires_grad:
                if "lora_" not in n:
                    raise ValueError(f"Unexpected GRPO trainable parameter: {n}")
                p.data = p.data.float()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model, processor


def make_grpo_args(cfg, use_cpu=False):
    from trl import GRPOConfig
    return GRPOConfig(
        output_dir=str(ROOT / "work/e5/trainer"), max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"], num_generations=4,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        generation_batch_size=4, max_completion_length=cfg["max_completion_length"],
        beta=0.0, use_vllm=False, use_transformers_continuous_batching=False,
        fp16=not use_cpu, bf16=False, use_cpu=use_cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", weight_decay=0.0, lr_scheduler_type="constant", warmup_steps=0,
        temperature=1.0, top_p=1.0, top_k=0, num_iterations=1, loss_type="grpo",
        scale_rewards="group", mask_truncated_completions=False,
        remove_unused_columns=False, logging_steps=1, logging_nan_inf_filter=False,
        report_to="none", save_strategy="no", eval_strategy="no", dataloader_num_workers=0,
        seed=cfg["seed"], data_seed=cfg["seed"], disable_tqdm=False)


def save_e5(model, processor, run_dir, step):
    import torch
    model.save_pretrained(run_dir / "adapter", safe_serialization=True)
    processor.save_pretrained(run_dir / "adapter")
    # The projector is frozen, but it is still required to reproduce the policy.
    extras = {n: p.detach().float().cpu() for n, p in model.named_parameters()
              if any(s in n for s in PROJECTORS)}
    if extras:
        tmp = run_dir / "projector.pt.tmp"
        torch.save(extras, tmp)
        tmp.replace(run_dir / "projector.pt")
    write_json(run_dir / "checkpoint.json", {"trainer_step": step, "projector_saved": bool(extras),
                                             "projector_trainable": False})


def train_e5():
    import torch
    from transformers import TrainerCallback
    from e5_trainer import T4GRPOTrainer
    if importlib.metadata.version("trl") != "1.9.2":
        raise RuntimeError("E5 requires trl==1.9.2")
    cfg = config()
    lab.seed_everything(cfg["seed"])
    run_dir = ROOT / "work/e5"
    run_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_policy(cfg, trainable=True)
    params = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    write_json(run_dir / "trainable_parameters.json", {"parameters": params,
                                                       "trainable_params": sum(v for _, v in params)})
    dataset = training_dataset(read_jsonl(ROOT / "data/splits/train.jsonl"))
    reward = RewardLogger(run_dir / "rollouts.jsonl", max_completion_length=cfg["max_completion_length"])

    class LocalLogs(TrainerCallback):
        def __init__(self):
            self.skipped = 0

        def on_train_begin(self, args, state, control, **kwargs):
            bad = [n for n, p in kwargs["model"].named_parameters()
                   if p.requires_grad and p.dtype != torch.float32]
            if bad:
                raise RuntimeError(f"Trainable weights must remain FP32 on T4: {bad[:3]}")

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "loss" in logs and not math.isfinite(logs["loss"]):
                raise RuntimeError("GRPO loss is NaN/inf")
            with (run_dir / "train_log.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"trainer_step": state.global_step, **logs}) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            skipped = trainer.accelerator.optimizer_step_was_skipped
            self.skipped += int(skipped)
            with (run_dir / "optimizer_steps.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"trainer_step": state.global_step, "update_skipped": skipped}) + "\n")
            if state.global_step % 20 == 0:
                save_e5(kwargs["model"], processor, run_dir, state.global_step)

    callbacks = LocalLogs()
    trainer = T4GRPOTrainer(model=model, args=make_grpo_args(cfg), reward_funcs=reward,
                            train_dataset=dataset, processing_class=processor, callbacks=[callbacks])
    write_json(run_dir / "precision_check.json", trainer.fp32_preservation)
    write_json(run_dir / "grpo_args.json", trainer.args.to_dict())
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    minutes = (time.perf_counter() - start) / 60
    save_e5(trainer.model, processor, run_dir, trainer.state.global_step)
    trainer.state.save_to_json(run_dir / "trainer_state.json")
    if cfg["setup"] in ("e2", "e3"):
        original = torch.load(Path(cfg["baseline_checkpoint"]) / "projector.pt", map_location="cpu", weights_only=True)
        final = torch.load(run_dir / "projector.pt", map_location="cpu", weights_only=True)
        same = set(original) == set(final) and all(torch.equal(original[n].float(), final[n]) for n in original)
        if not same:
            raise RuntimeError("Frozen SFT projector changed during GRPO")
        write_json(run_dir / "projector_frozen_check.json", {"verified": same, "tensors": len(final)})
    meta = {"trainable_params": sum(v for _, v in params),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "train_minutes": minutes}
    write_json(run_dir / "run.json", {**meta, "config": cfg, "trainer_global_steps": trainer.state.global_step,
                                      "successful_optimizer_updates": trainer.state.global_step - callbacks.skipped,
                                      "amp_skipped_updates": callbacks.skipped, "status": "trained",
                                      "vram_unit": "GiB (2**30 bytes)"})
    write_json(ROOT / "report/e5_environment.json", {n: importlib.metadata.version(n)
               for n in ("trl", "torch", "transformers", "peft", "bitsandbytes", "datasets")})


def evaluate_e5():
    cfg = config()
    run = load_json(ROOT / "work/e5/run.json")
    model, processor = load_policy(cfg, trainable=False)
    meta = {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
    inference = {}
    for split in ("val", "test"):
        rows = read_jsonl(ROOT / f"data/splits/{split}.jsonl")
        suffix = "" if split == "test" else "_val"
        inference[split] = lab.predict(model, processor, rows, ROOT / f"work/e5{suffix}.jsonl", cfg)
        lab.score("e5", split, meta)
        if split == "test":
            lab.make_examples("e5", rows)
    write_json(ROOT / "work/e5/inference.json", inference)
    compare_e5()


def audit_outputs(rows, records):
    raw = {r["image_id"]: r for r in records}
    summary = Counter()
    rewards = []
    for row in rows:
        record = raw[row["image_id"]]
        d = reward_details(record["raw"], _gt_boxes(row))
        rewards.append(d["reward"])
        summary["invalid_outputs"] += not d["parse_valid"]
        summary["empty_on_nonempty_gt"] += d["empty"] and d["n_gt"] > 0
        summary["giant_box_images"] += d["giant_boxes"] > 0
        summary["hit_token_limit_images"] += record.get("hit_token_limit", False)
        summary["predicted_boxes"] += d["n_pred"]
    return {"n_images": len(rows), **dict(summary), "mean_iou_reward": sum(rewards) / len(rewards)}


def compare_e5():
    import numpy as np
    cfg = config()
    baseline = cfg["baseline"]
    rows = read_jsonl(ROOT / "data/splits/test.jsonl")
    matrices, audits = {}, {}
    for name in (baseline, "e5"):
        records = read_jsonl(ROOT / f"work/{name}.jsonl")
        outputs = {r["image_id"]: r["raw"] for r in records}
        if len(outputs) != len(records) or set(outputs) != {r["image_id"] for r in rows}:
            raise ValueError(f"Missing/duplicate test output for {name}")
        evidence = load_json(ROOT / f"report/evidence/{name}.json")
        measured = evaluate(rows, outputs)
        if json.dumps(measured, sort_keys=True) != json.dumps({k: evidence[k] for k in measured}, sort_keys=True):
            raise ValueError(f"Cached {name} evidence mismatch")
        counts = []
        for row in rows:
            per, _ = _match(_gt_boxes(row), parse_output(outputs[row["image_id"]]) or [])
            counts.append([sum(v[k] for v in per.values()) for k in range(3)])
        matrices[name] = np.array(counts, dtype=np.int64)
        audits[name] = audit_outputs(rows, records)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(rows), size=(10000, len(rows)))
    f1 = {}
    for name, matrix in matrices.items():
        total = matrix[indices].sum(axis=1)
        f1[name] = 2 * total[:, 0] / np.maximum(1, 2 * total[:, 0] + total[:, 1] + total[:, 2])
    delta = f1["e5"] - f1[baseline]
    sft = load_json(ROOT / f"report/evidence/{baseline}.json")
    e5 = load_json(ROOT / "report/evidence/e5.json")
    rollouts = read_jsonl(ROOT / "work/e5/rollouts.jsonl")
    grouped = [rollouts[i:i + 4] for i in range(0, len(rollouts), 4)]
    if any(len(g) != 4 or len({r["image_id"] for r in g}) != 1 for g in grouped):
        raise ValueError("Rollout logs do not contain full groups of four completions for each prompt")
    rollout_audit = {"n_completions": len(rollouts), "n_groups": len(grouped),
                     "mean_reward": float(np.mean([r["reward"] for r in rollouts])),
                     "zero_variance_group_rate": float(np.mean([len({r["reward"] for r in g}) == 1 for g in grouped])),
                     "invalid_rate": float(np.mean([not r["parse_valid"] for r in rollouts])),
                     "empty_on_nonempty_gt_rate": float(np.mean([r["empty"] and r["n_gt"] > 0 for r in rollouts])),
                     "giant_box_image_rate": float(np.mean([r["giant_boxes"] > 0 for r in rollouts])),
                     "hit_token_limit_rate": float(np.mean([r["hit_token_limit"] for r in rollouts]))}
    comparison = {"baseline": baseline, "baseline_f1": sft["micro"]["f1"], "e5_f1": e5["micro"]["f1"],
                  "delta_f1": e5["micro"]["f1"] - sft["micro"]["f1"],
                  "baseline_ci95": sft["f1_ci95"], "e5_ci95": e5["f1_ci95"],
                  "paired_delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
                  "paired_bootstrap": {"n_boot": 10000, "seed": 42, "unit": "image, not session"},
                  "greedy_test_audit": audits, "training_rollout_audit": rollout_audit,
                  "limitations": ["One seed and only three test sessions.",
                                  "Training completion limit 128; shared SFT/E5 greedy evaluation limit 512.",
                                  "GRPO freezes the restored SFT projector and continues its LoRA adapters."]}
    write_json(ROOT / "report/e5_comparison.json", comparison)
    write_json(ROOT / "report/e5_status.json", {"status": "finished", "smoke_run": cfg["max_steps"] != 60,
                                               "config": cfg})
    print(json.dumps(comparison, indent=2), flush=True)


def export_results():
    dest = ROOT.parent / "lab_day10_e5_results.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("report", "data/splits", "work/e5"):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".png") \
                        and "adapter" not in path.relative_to(ROOT).parts:
                    archive.write(path, path.relative_to(ROOT))
        for pattern in ("work/*.jsonl",):
            for path in ROOT.glob(pattern):
                archive.write(path, path.relative_to(ROOT))
        for name in ("e5_run_config.json", "kaggle_e5.py", "e5_reward.py", "e5_trainer.py", "requirements-e5.txt"):
            path = ROOT / name
            if path.exists():
                archive.write(path, name)
    print(f"E5 analysis archive: {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("prepare", "train", "evaluate", "compare"), required=True)
    args = ap.parse_args()
    lab.ROOT = ROOT
    os.chdir(ROOT)
    try:
        {"prepare": prepare, "train": train_e5, "evaluate": evaluate_e5, "compare": compare_e5}[args.task]()
    except Exception as exc:
        write_json(ROOT / "report/e5_failure.json", {"task": args.task, "error": str(exc),
                                                     "traceback": traceback.format_exc()})
        raise
    finally:
        export_results()


if __name__ == "__main__":
    main()
