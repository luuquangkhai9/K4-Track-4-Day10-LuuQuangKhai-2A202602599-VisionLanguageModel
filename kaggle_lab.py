"""Student implementation: QLoRA E0-E4, run in separate GPU processes on Kaggle."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

from kaggle_utils import (LLM_TARGET, VIT_TARGET, PROJECTORS, PROMPT, assert_disjoint_sessions,
                          fingerprint, objects_to_answer, subset_train, target_names,
                          write_json, write_summary, yolo_to_box)
from scripts.labcommon import read_jsonl

ROOT = Path(__file__).resolve().parent


def command(args, log_path=None):
    args = [str(v) for v in args]
    if log_path is None:
        subprocess.run(args, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, args)


def image_path(row):
    return ROOT / "data" / row["image_path"]


def read_config():
    return json.loads((ROOT / "kaggle_run_config.json").read_text(encoding="utf-8"))


def user_message(row):
    return {"role": "user", "content": [
        {"type": "image", "image": str(image_path(row))},
        {"type": "text", "text": PROMPT},
    ]}


def make_quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    # Transformers 5.19 matches exclusions against full names from named_modules().
    # "visual" does NOT exclude children named "model.visual.blocks...".
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=["model.visual", "lm_head"])


def load_base(cfg):
    import torch
    import bitsandbytes as bnb
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Enable a Kaggle GPU accelerator.")
    quant = make_quantization_config()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], quantization_config=quant,
        dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    bad = [n for n, m in model.named_modules()
           if ("visual" in n or n == "lm_head") and isinstance(m, bnb.nn.Linear4bit)]
    if bad:
        raise RuntimeError(f"Visual/lm_head unexpectedly quantized: {bad[:5]}")
    quantized_names = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    if not quantized_names:
        raise RuntimeError("No NF4 layers: QLoRA was not activated.")
    print(f"NF4 layers: {len(quantized_names)}; visual and lm_head excluded from quantization", flush=True)
    return model, processor


def architecture(model):
    import bitsandbytes as bnb
    counts = {"llm_including_embeddings": 0, "vit_excluding_projector": 0,
              "projector": 0, "lm_head": 0, "other": 0}
    # Params4bit.numel() is packed; quant_state.shape holds the original dense shape.
    for n, p in model.named_parameters():
        if any(s in n for s in PROJECTORS):
            key = "projector"
        elif "visual" in n:
            key = "vit_excluding_projector"
        elif "language_model" in n:
            key = "llm_including_embeddings"
        elif "lm_head" in n:
            key = "lm_head"
        else:
            key = "other"
        if isinstance(p, bnb.nn.Params4bit) and p.quant_state is not None:
            import math
            size = math.prod(p.quant_state.shape)
        else:
            size = p.numel()
        counts[key] += size
    counts["total_dense_equivalent"] = sum(counts.values())
    return counts


def encode(row, processor):
    import torch
    msg = user_message(row)
    prefix = processor.apply_chat_template(
        [msg], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    answer = objects_to_answer(row["objects"])
    full = processor.apply_chat_template(
        [msg, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
    n = prefix["input_ids"].shape[1]
    if not torch.equal(prefix["input_ids"], full["input_ids"][:, :n]):
        raise RuntimeError("Chat template prefix mismatch: cannot safely mask assistant loss.")
    keep = {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
    encoded = {k: v for k, v in full.items() if k in keep}
    encoded["labels"] = full["input_ids"].clone()
    encoded["labels"][:, :n] = -100
    if not (encoded["labels"] != -100).any():
        raise RuntimeError("No supervised answer tokens.")
    encoded["pixel_values"] = encoded["pixel_values"].half()
    return encoded, answer, n


def collate(batch, pad_id):
    import torch
    from torch.nn.functional import pad
    length = max(x["input_ids"].shape[1] for x in batch)
    out = {}
    for key, fill in (("input_ids", pad_id), ("attention_mask", 0),
                      ("mm_token_type_ids", 0), ("labels", -100)):
        if not all(key in x for x in batch):
            if key == "mm_token_type_ids":
                continue
            raise ValueError(f"Missing {key}")
        out[key] = torch.cat([pad(x[key], (0, length - x[key].shape[1]), value=fill)
                              for x in batch], dim=0)
    for key in ("pixel_values", "image_grid_thw"):
        out[key] = torch.cat([x[key] for x in batch], dim=0)
    return out


def setup_model(model, cfg, run_dir):
    import torch
    from peft import LoraConfig, get_peft_model
    include_vit = cfg["setup"] == "e3"
    names = target_names([n for n, _ in model.named_modules()], include_vit)
    expected = 348 if include_vit else 252
    if len(names) != expected:
        raise RuntimeError(f"LoRA matched {len(names)} modules; expected {expected}. Check model tree.")
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    pattern = LLM_TARGET if not include_vit else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    model = get_peft_model(model, LoraConfig(
        r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=0.05,
        target_modules=pattern, bias="none", task_type="CAUSAL_LM"))
    if cfg["setup"] in ("e2", "e3"):
        for n, p in model.named_parameters():
            if any(s in n for s in PROJECTORS):
                p.requires_grad_(True)
    # GradScaler cannot unscale FP16 trainable gradients. Frozen visual remains FP16.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    write_json(run_dir / "trainable_parameters.json", {
        "matched_modules": names, "matched_count": len(names), "parameters": trainable,
        "trainable_params": sum(v for _, v in trainable)})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model, sum(v for _, v in trainable)


def save_weights(model, processor, run_dir, step):
    import torch
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    extras = {n: p.detach().float().cpu() for n, p in model.named_parameters()
              if p.requires_grad and any(s in n for s in PROJECTORS)}
    if extras:
        temp = run_dir / "projector.pt.tmp"
        torch.save(extras, temp)
        temp.replace(run_dir / "projector.pt")
    write_json(run_dir / "checkpoint.json", {"optimizer_step": step,
                                             "projector_saved": bool(extras)})


def reload_weights(model, run_dir, cfg):
    import torch
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, run_dir / "adapter", is_trainable=False)
    if cfg["setup"] in ("e2", "e3"):
        path = run_dir / "projector.pt"
        if not path.exists():
            raise RuntimeError("Trained projector missing: refusing to evaluate E2/E3.")
        extras = torch.load(path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        expected = {n for n in params if any(s in n for s in PROJECTORS)}
        if set(extras) != expected:
            raise RuntimeError("Projector checkpoint names do not match the loaded model.")
        with torch.no_grad():
            for n, saved in extras.items():
                # Keep the saved FP32 projector; autocast handles linear operations.
                params[n].data = saved.to(device=params[n].device, dtype=torch.float32)
        write_json(run_dir / "projector_reload.json", {
            "verified": all(torch.equal(params[n].detach().cpu(), t) for n, t in extras.items()),
            "tensors": len(extras)})
    return model


def train(model, processor, records, cfg, run_dir, trainable_params):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    records = subset_train(records, cfg["train_fraction"], cfg["seed"])
    write_json(run_dir / "train_image_ids.json", [r["image_id"] for r in records])
    encoded = []
    for i, row in enumerate(records):
        item, answer, n = encode(row, processor)
        if i == 0:
            supervised = item["labels"][item["labels"] != -100]
            decoded = processor.tokenizer.decode(supervised, skip_special_tokens=False)
            image_tokens = int((item["input_ids"] == model.config.image_token_id).sum())
            if (row["width"], row["height"]) == (640, 360) and image_tokens != 220:
                raise RuntimeError(f"Image has {image_tokens} vision tokens; expected 220.")
            if not decoded.startswith(answer) or "<|im_end|>" not in decoded:
                raise RuntimeError("Supervised tokens do not contain the complete answer and EOS.")
            write_json(run_dir / "mask_check.json", {
                "image_id": row["image_id"], "image_tokens": image_tokens,
                "masked_prefix_tokens": n, "supervised_tokens": int(supervised.numel()),
                "answer": answer, "decoded_supervised": decoded, "prefix_equal": True})
        encoded.append(item)
        if (i + 1) % 200 == 0:
            print(f"Encoded {i + 1}/{len(records)} training images", flush=True)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(encoded, batch_size=cfg["batch_size"], shuffle=True,
                        generator=generator, num_workers=0,
                        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
    iterator = iter(loader)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=max(1, round(cfg["max_steps"] * 0.05)),
        num_training_steps=cfg["max_steps"])
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    opt.zero_grad(set_to_none=True)
    step = attempts = skipped = 0
    with (run_dir / "train_log.jsonl").open("w", encoding="utf-8") as log:
        while step < cfg["max_steps"]:
            attempts += 1
            loss_sum = 0.0
            for _ in range(cfg["grad_accum"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    raw_loss = model(**batch).loss
                    loss = raw_loss / cfg["grad_accum"]
                if not torch.isfinite(raw_loss):
                    raise RuntimeError(f"Non-finite loss at attempt {attempts}: {raw_loss.item()}")
                loss_sum += raw_loss.detach().item() / cfg["grad_accum"]
                scaler.scale(loss).backward()
                del batch, loss, raw_loss
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            old_scale = scaler.get_scale()
            lr_used = opt.param_groups[0]["lr"]
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            did_step = scaler.get_scale() >= old_scale
            if did_step:
                scheduler.step()
                step += 1
                skipped = 0
            else:
                skipped += 1
            event = {"optimizer_step": step, "attempt": attempts, "loss": loss_sum,
                     "lr": lr_used, "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                     "amp_scale": scaler.get_scale(), "update_skipped": not did_step,
                     "elapsed_minutes": (time.perf_counter() - start) / 60}
            log.write(json.dumps(event) + "\n")
            log.flush()
            if skipped >= 10:
                raise RuntimeError("10 consecutive AMP overflows; inspect loss and trainable dtypes.")
            if did_step and (step == 1 or step % 10 == 0):
                print(f"{cfg['name']} step {step}/{cfg['max_steps']} loss={loss_sum:.4f} "
                      f"VRAM={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
            if did_step and (step % 100 == 0 or step == cfg["max_steps"]):
                save_weights(model, processor, run_dir, step)
    torch.cuda.synchronize()
    meta = {"trainable_params": trainable_params,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "train_minutes": (time.perf_counter() - start) / 60}
    write_json(run_dir / "run.json", {**meta, "config": cfg, "optimizer_steps": step,
                                      "attempts": attempts, "n_train_images": len(records),
                                      "status": "trained", "vram_unit": "GiB (2**30 bytes)"})
    return meta


def predict(model, processor, rows, output, cfg):
    import torch
    model.eval()
    model.config.use_cache = True
    batch_size = cfg["inference_batch_size"]
    index = 0
    start = time.perf_counter()
    tmp = output.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        while index < len(rows):
            batch_rows = rows[index:index + batch_size]
            try:
                # A list of conversations, one image per conversation.
                inputs = processor.apply_chat_template(
                    [[user_message(r)] for r in batch_rows], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    processor_kwargs={"padding": True})
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"],
                                         do_sample=False, use_cache=True,
                                         pad_token_id=processor.tokenizer.pad_token_id)
                raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                lengths = [int((tokens != processor.tokenizer.pad_token_id).sum())
                           for tokens in out[:, inputs["input_ids"].shape[1]:]]
                del inputs, out
            except torch.cuda.OutOfMemoryError:
                # OOM can leave partial tensors alive; release them before retrying.
                inputs = out = None
                gc.collect()
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"Inference OOM: retry with batch_size={batch_size}", flush=True)
                continue
            for row, text, length in zip(batch_rows, raw, lengths):
                f.write(json.dumps({"image_id": row["image_id"], "raw": text,
                                    "generated_tokens": length,
                                    "hit_token_limit": length >= cfg["max_new_tokens"]}) + "\n")
            f.flush()
            index += len(batch_rows)
            if index % 20 < batch_size or index == len(rows):
                print(f"{output.stem}: {index}/{len(rows)}", flush=True)
    tmp.replace(output)
    return {"minutes": (time.perf_counter() - start) / 60, "batch_size_final": batch_size}


def score(name, split, meta):
    suffix = "" if split == "test" else "_val"
    dest = ROOT / "report" / ("evidence" if split == "test" else "validation") / f"{name}.json"
    args = [sys.executable, "scripts/score.py", "--pred", f"work/{name}{suffix}.jsonl",
            "--split", split, "--name", name.upper(), "--out", dest]
    for k, v in meta.items():
        args += ["--meta", f"{k}={v}"]
    command(args)
    return json.loads(dest.read_text(encoding="utf-8"))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def experiment(cfg):
    import torch
    name = cfg["name"]
    seed_everything(cfg["seed"])
    run_dir = ROOT / "work" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "effective_config.json", cfg)
    splits = {s: read_jsonl(ROOT / "data/splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    model, processor = load_base(cfg)
    write_json(run_dir / "architecture.json", architecture(model))
    (run_dir / "module_names.txt").write_text("\n".join(n for n, _ in model.named_modules()), encoding="utf-8")
    if name != "e0":
        model, n = setup_model(model, cfg, run_dir)
        meta = train(model, processor, splits["train"], cfg, run_dir, n)
        # Reload into a fresh base model: evaluation verifies serialization as well.
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model, processor = load_base(cfg)
        model = reload_weights(model, run_dir, cfg)
    else:
        meta = {"trainable_params": 0, "peak_vram_gb": 0, "train_minutes": 0}
        torch.cuda.reset_peak_memory_stats()
    inference = {}
    for split in ("val", "test"):
        suffix = "" if split == "test" else "_val"
        inference[split] = predict(model, processor, splits[split],
                                  ROOT / "work" / f"{name}{suffix}.jsonl", cfg)
        if name == "e0":
            meta["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        score(name, split, meta)
    write_json(run_dir / "inference.json", inference)
    if name == "e0":
        write_json(run_dir / "run.json", {**meta, "config": cfg, "status": "zero_shot",
                                           "inference_minutes": sum(d["minutes"] for d in inference.values()),
                                           "vram_unit": "GiB (2**30 bytes)"})
    make_examples(name, splits["test"])
    write_json(run_dir / "completed.json", {"fingerprint": fingerprint(cfg), "config": cfg})


def draw_boxes(row, pred, output, caption):
    from PIL import Image, ImageDraw
    with Image.open(image_path(row)) as image:
        image = image.convert("RGB")
    w, h = image.size
    canvas = Image.new("RGB", (w, h + 65), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for o in row["objects"]:
        converted = yolo_to_box(o["label"], o["yolo"])
        if converted is None:
            continue
        bb = converted["bbox_2d"]
        box = [bb[0] * w / 1000, bb[1] * h / 1000, bb[2] * w / 1000, bb[3] * h / 1000]
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], max(0, box[1] - 12)), "GT:" + o["label"], fill="lime")
    for o in pred or []:
        bb = o["bbox"]
        box = [max(0, min(w, bb[0] * w / 1000)), max(0, min(h, bb[1] * h / 1000)),
               max(0, min(w, bb[2] * w / 1000)), max(0, min(h, bb[3] * h / 1000))]
        if box[2] >= box[0] and box[3] >= box[1]:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], max(0, box[1])), "P:" + o["label"], fill="red")
    draw.text((5, h + 4), caption[:100], fill="black")
    draw.text((5, h + 22), "GT=green | prediction=red | coordinates: 0-1000", fill="black")
    draw.text((5, h + 40), row["image_id"], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_examples(name, rows):
    from scripts.score import parse_output, _gt_boxes, _match
    predictions = {r["image_id"]: r for r in read_jsonl(ROOT / "work" / f"{name}.jsonl")}
    details = []
    for row in rows:
        raw = predictions[row["image_id"]]["raw"]
        pred = parse_output(raw)
        per, _ = _match(_gt_boxes(row), pred or [])
        tp, fp, fn = [sum(v[k] for v in per.values()) for k in range(3)]
        details.append({"image_id": row["image_id"], "tp": tp, "fp": fp, "fn": fn,
                        "parse_valid": pred is not None, "raw": raw,
                        "hit_token_limit": predictions[row["image_id"]]["hit_token_limit"]})
    write_json(ROOT / "report" / "errors" / f"{name}.json", details)
    good = sorted(details, key=lambda d: (d["fp"] + d["fn"], -d["tp"], d["image_id"]))[:3]
    good_ids = {d["image_id"] for d in good}
    bad = sorted((d for d in details if d["image_id"] not in good_ids),
                 key=lambda d: (-(d["fp"] + d["fn"]), d["image_id"]))[:3]
    by_id = {r["image_id"]: r for r in rows}
    captions = []
    for i, d in enumerate(good + bad):
        caption = f"{name.upper()} TP={d['tp']} FP={d['fp']} FN={d['fn']} valid={d['parse_valid']}"
        path = ROOT / "report" / "figures" / name / f"{i + 1}_{d['image_id']}.png"
        draw_boxes(by_id[d["image_id"]], parse_output(d["raw"]), path, caption)
        captions.append({**d, "figure": str(path.relative_to(ROOT)), "caption": caption})
    write_json(ROOT / "report" / "figures" / name / "captions.json", captions)


def find_dataset(input_root, explicit):
    if explicit:
        path = Path(explicit)
        if path.is_file() and path.suffix.lower() == ".zip":
            return "--zip", path
        if (path / "images").is_dir() and (path / "labels-YOLO").is_dir():
            return "--source", path
        raise FileNotFoundError(f"Invalid DATA_SOURCE: {path}")
    roots = sorted({p.parent for p in Path(input_root).rglob("labels-YOLO")
                    if p.is_dir() and (p.parent / "images").is_dir()
                    and "work" not in p.relative_to(input_root).parts})
    if len(roots) == 1:
        return "--source", roots[0]
    if len(roots) > 1:
        raise RuntimeError("Multiple datasets found; set DATA_SOURCE explicitly.")
    zips = []
    for p in Path(input_root).rglob("*.zip"):
        try:
            with zipfile.ZipFile(p) as z:
                if any("labels-YOLO/" in n for n in z.namelist()):
                    zips.append(p)
        except zipfile.BadZipFile:
            continue
    if len(zips) == 1:
        return "--zip", zips[0]
    raise FileNotFoundError("Add the road-damage dataset in Kaggle Input, or set DATA_SOURCE.")


def bundle_results():
    rows = write_summary(ROOT / "report/evidence", ROOT / "report/summary.csv")
    manifest = {"evidence": rows, "report_written": False,
                "notes": ["Measured results only; REPORT.md must be completed after analysis.",
                          "Image-bootstrap CI is not a paired significance test or session-bootstrap CI."]}
    if (ROOT / "report/failure.json").exists():
        manifest["failure"] = json.loads((ROOT / "report/failure.json").read_text(encoding="utf-8"))
    write_json(ROOT / "report/results_manifest.json", manifest)
    paths = [*sorted((ROOT / "report").rglob("*")),
             *sorted((ROOT / "data/splits").glob("*.json*")),
             *sorted((ROOT / "work").rglob("*.json")),
             *sorted((ROOT / "work").rglob("*.jsonl")),
             *sorted((ROOT / "work").rglob("*.txt"))]
    # Exclude model weights, optimizer states, dataset images and HF model cache.
    excluded = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "added_tokens.json", "vocab.json", "merges.txt", "chat_template.json"}
    dest = ROOT.parent / "lab_day10_results.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.is_file() and "adapter" not in p.relative_to(ROOT).parts and p.name not in excluded:
                z.write(p, p.relative_to(ROOT))
        for p in [ROOT / "kaggle_run_config.json", ROOT / "kaggle_lab.py", ROOT / "kaggle_utils.py"]:
            if p.exists():
                z.write(p, p.name)
    print(f"Results for analysis: {dest}", flush=True)


def config_for(base, name, setup=None, fraction=1.0):
    import yaml
    setup = setup or name
    if setup == "e0":
        original = yaml.safe_load((ROOT / "configs/e0.yaml").read_text(encoding="utf-8"))
        original.update(lr=0, seed=42, max_steps=0)
    else:
        original = yaml.safe_load((ROOT / f"configs/{setup}.yaml").read_text(encoding="utf-8"))
    return {"name": name, "setup": setup, "model_id": original["model_id"],
            "model_revision": base["model_revision"], "seed": original["seed"],
            "rank": original.get("lora", {}).get("r", 16), "lr": float(original["lr"]),
            "max_steps": base["max_steps"] if setup != "e0" else 0,
            "batch_size": base["batch_size"], "grad_accum": base["grad_accum"],
            "max_new_tokens": base["max_new_tokens"],
            "inference_batch_size": base["inference_batch_size"],
            "train_fraction": fraction, "prompt": PROMPT,
            "dataset_version": base["dataset_version"], "code_sha256": base["code_sha256"]}


def restore_completed(base, cfg):
    source = base.get("resume_source")
    if not source:
        return False
    source = Path(source)
    name = cfg["name"]
    marker = source / "work" / name / "completed.json"
    if not marker.exists():
        return False
    done = json.loads(marker.read_text(encoding="utf-8"))
    if done.get("fingerprint") != fingerprint(cfg):
        raise RuntimeError(f"Resume config mismatch for {name}; clear RESUME_SOURCE or use identical config.")
    required = [source / "report/evidence" / f"{name}.json",
                source / "report/validation" / f"{name}.json",
                source / "work" / f"{name}.jsonl", source / "work" / f"{name}_val.jsonl"]
    if not all(p.exists() for p in required):
        raise RuntimeError(f"Incomplete completed-run artifacts for {name}")
    shutil.copytree(source / "work" / name, ROOT / "work" / name, dirs_exist_ok=True)
    for p in required:
        dest = ROOT / p.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for part in (f"report/figures/{name}",):
        if (source / part).exists():
            shutil.copytree(source / part, ROOT / part, dirs_exist_ok=True)
    if (source / f"report/errors/{name}.json").exists():
        dest = ROOT / f"report/errors/{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / f"report/errors/{name}.json", dest)
    print(f"Reused completed {name} from previous output", flush=True)
    return True


def run_all():
    from huggingface_hub import model_info
    base = read_config()
    (ROOT / "work").mkdir(exist_ok=True)
    (ROOT / "report").mkdir(exist_ok=True)
    flag, src = find_dataset(base["input_root"], base.get("data_source"))
    command([sys.executable, "scripts/prepare_data.py", flag, src])
    meta = json.loads((ROOT / "data/splits/meta.json").read_text(encoding="utf-8"))
    base["dataset_version"] = meta["dataset_version"]
    splits = {s: read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
    sessions = assert_disjoint_sessions(splits)
    degenerate = [{"split": s, "image_id": r["image_id"], "object": o}
                  for s, rows in splits.items() for r in rows for o in r["objects"]
                  if yolo_to_box(o["label"], o["yolo"]) is None]
    if any(d["split"] != "train" for d in degenerate):
        raise RuntimeError("Degenerate val/test boxes: inspect dataset before scoring.")
    write_json(ROOT / "report/data_checks.json", {
        "sessions": sessions, "meta": meta, "degenerate_training_objects": degenerate,
        "policy": "Skip degenerate objects in training answers only; original splits/GT unchanged."})
    # Pin one immutable model revision for all independent experiments.
    base["model_revision"] = base.get("model_revision") or model_info("Qwen/Qwen3-VL-4B-Instruct").sha
    base["code_sha256"] = hashlib.sha256(
        (ROOT / "kaggle_lab.py").read_bytes() + (ROOT / "kaggle_utils.py").read_bytes()).hexdigest()
    write_json(ROOT / "kaggle_run_config.json", base)
    env = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "torchvision", "transformers", "peft", "bitsandbytes", "accelerate"):
        env["packages"][name] = importlib.metadata.version(name)
    env["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    write_json(ROOT / "report/environment.json", env)
    for row in read_jsonl(ROOT / "data/splits/train.jsonl")[:6]:
        draw_boxes(row, [], ROOT / "report/figures/data" / f"{row['image_id']}.png", "Training label check")
    for name in ("e0", "e1", "e2", "e3"):
        cfg = config_for(base, name)
        if not restore_completed(base, cfg):
            write_json(ROOT / "work" / f"{name}_config.json", cfg)
            command([sys.executable, "-u", "kaggle_lab.py", "--task", "experiment", "--name", name],
                    ROOT / f"report/logs/{name}.txt")
        bundle_results()
    # Validation chooses E4's baseline. Test predictions never select the setup.
    vals = {n: json.loads((ROOT / f"report/validation/{n}.json").read_text(encoding="utf-8"))
            for n in ("e1", "e2", "e3")}
    best = max(vals, key=lambda n: vals[n]["micro"]["f1"])
    selection = {"selected_setup": best, "criterion": "full validation micro F1; ties prefer E1 then E2",
                 "validation_f1": {n: d["micro"]["f1"] for n, d in vals.items()},
                 "variable": "data_size", "values": [0.1, 1.0],
                 "baseline_100_percent_reuses": best, "test_used_for_selection": False}
    write_json(ROOT / "report/e4_selection.json", selection)
    baseline = json.loads((ROOT / f"report/evidence/{best}.json").read_text(encoding="utf-8"))
    baseline["name"] = "E4_DATA_SIZE_100"
    baseline["meta"].update(ablation_variable="data_size", train_fraction=1.0, baseline_setup=best,
                            reused_from=best)
    write_json(ROOT / "report/evidence/e4_data_size_100.json", baseline)
    cfg = config_for(base, "e4_data_size_10", setup=best, fraction=0.1)
    write_json(ROOT / "work/e4_data_size_10_config.json", cfg)
    if not restore_completed(base, cfg):
        command([sys.executable, "-u", "kaggle_lab.py", "--task", "experiment", "--name", cfg["name"]],
                ROOT / "report/logs/e4_data_size_10.txt")
    evidence_path = ROOT / "report/evidence/e4_data_size_10.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["meta"].update(ablation_variable="data_size", train_fraction=0.1, baseline_setup=best)
    write_json(evidence_path, evidence)
    # Diagnose coordinate confusion from the SAME predictions, without tuning models.
    command([sys.executable, "scripts/score.py", "--pred", "work/e1.jsonl", "--coord", "pixel",
             "--name", "E1_WRONG_COORD_DIAGNOSTIC", "--out", "report/diagnostics/e1_wrong_coord.json"])
    e0 = json.loads((ROOT / "report/evidence/e0.json").read_text(encoding="utf-8"))
    e1 = json.loads((ROOT / "report/evidence/e1.json").read_text(encoding="utf-8"))
    quality = {"e1_parse_valid_at_least_095": e1["parse_valid_rate"] >= 0.95,
               "e1_f1_gain_at_least_010": e1["micro"]["f1"] - e0["micro"]["f1"] >= 0.10,
               "e1_f1_gain": e1["micro"]["f1"] - e0["micro"]["f1"],
               "all_test_predictions_present": all(d["missing_predictions"] == 0 for d in
                   [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "report/evidence").glob("*.json")]),
               "status": "finished", "report_and_git_checks_pending": True,
               "smoke_run": base["max_steps"] != 300}
    write_json(ROOT / "report/quality_checks.json", quality)
    # This checker also checks the unfilled report and git history: record, do not fake PASS.
    result = subprocess.run([sys.executable, "scripts/check_submission.py"], cwd=ROOT,
                            capture_output=True, text=True)
    (ROOT / "report/submission_check.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    failure = ROOT / "report/failure.json"
    if failure.exists():
        failure.unlink()
    print(json.dumps(quality, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", "experiment"), default="all")
    parser.add_argument("--name")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.task == "experiment":
        cfg = json.loads((ROOT / "work" / f"{args.name}_config.json").read_text(encoding="utf-8"))
        experiment(cfg)
        return
    try:
        run_all()
    except Exception as exc:
        write_json(ROOT / "report/failure.json", {"error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        bundle_results()


if __name__ == "__main__":
    main()
