"""Build a portable Kaggle notebook embedding the authorized lab source files."""
import base64
import ast
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def build():
    paths = [ROOT / name for name in ("README.md", "GUIDE.md", "RUBRIC.md",
                                      "requirements-kaggle.txt", "kaggle_lab.py", "kaggle_utils.py",
                                      "KAGGLE.md", "pytest.ini")]
    for directory, suffix in (("scripts", ".py"), ("tests", ".py"),
                              ("kaggle_checks", ".py"), ("configs", ".yaml")):
        paths += sorted((ROOT / directory).glob(f"*{suffix}"))
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        # The static original template prevents recursive notebook payloads when
        # both generated notebooks are rebuilt. The complete code is visible in cells.
        template = zipfile.ZipInfo("lab_day10.ipynb", date_time=(2026, 1, 1, 0, 0, 0))
        template.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(template, (ROOT / "lab_day10_template.ipynb").read_bytes())
        # A new runtime starts with the report template, never with results from
        # a previous run or the user's completed report in the shared workspace.
        report_template = zipfile.ZipInfo("REPORT.md", date_time=(2026, 1, 1, 0, 0, 0))
        report_template.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(report_template, (ROOT / "REPORT_template.md").read_bytes())
    encoded = base64.b64encode(content.getvalue()).decode("ascii")
    cells = [markdown("""# Lab Day 10 trên Kaggle: QLoRA Qwen3-VL-4B

Chạy E0, E1, E2, E3 và E4 (10% / 100% dữ liệu). Notebook nhúng toàn bộ mã cần thiết.
1. Chọn GPU **T4** (nếu T4 x2, mã dùng GPU 0), bật **Internet**.
2. Add Input: dataset `lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.
3. Chọn **Save Version → Save & Run All**. Không phụ thuộc biến của phiên interactive.
4. Tải `/kaggle/working/lab_day10_results.zip` trong Output để gửi phân tích.

E5/GRPO là bonus, chưa có trong notebook này. Số liệu không được tạo giả; báo cáo được viết sau khi chạy.
Nếu yêu cầu lớp học bắt buộc Colab, xác nhận với giảng viên việc dùng Kaggle.
"""), markdown("## 0. Cấu hình lượt chạy"), code('''# Mặc định chạy đầy đủ 300 bước cho mỗi lượt train.
# Có thể đặt MAX_STEPS=2 để kiểm tra pipeline; lượt này không dùng làm bài nộp.
DATA_SOURCE = None  # tự tìm trong /kaggle/input; hoặc đường dẫn root có images/ và labels-YOLO/
RESUME_SOURCE = None  # root day10 của Output cũ đã Add Input; chỉ tái dùng lượt đã hoàn thành
MODEL_REVISION = None  # tự ghim SHA model; khi resume, dùng SHA từ kaggle_run_config.json cũ
MAX_STEPS = 300
BATCH_SIZE = 1
GRAD_ACCUM = 8
INFERENCE_BATCH_SIZE = 2
MAX_NEW_TOKENS = 512  # dùng cùng giới hạn cho mọi setup, giảm nguy cơ cắt JSON nhiều box
'''), markdown("## 1. Khôi phục mã nguồn vào working"), code(f'''import base64, io, json, os, subprocess, sys, zipfile
from pathlib import Path

BASE = Path("/kaggle/working")
BASE.mkdir(parents=True, exist_ok=True)
PROJECT = BASE / "day10"
PROJECT.mkdir(exist_ok=True)
PAYLOAD = "{encoded}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(PROJECT)
os.environ["HF_HOME"] = "/tmp/day10_hf_cache"  # model gốc không nằm trong Output
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
config = {{"input_root": "/kaggle/input", "data_source": DATA_SOURCE,
          "resume_source": RESUME_SOURCE, "model_revision": MODEL_REVISION,
          "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
          "inference_batch_size": INFERENCE_BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS}}
if BATCH_SIZE * GRAD_ACCUM != 8:
    raise ValueError("Giữ effective batch size = 8 cho các thí nghiệm.")
if not all(isinstance(v, int) and v > 0 for v in
           [MAX_STEPS, BATCH_SIZE, GRAD_ACCUM, INFERENCE_BATCH_SIZE, MAX_NEW_TOKENS]):
    raise ValueError("Các cấu hình bước/batch/token phải là số nguyên dương.")
(PROJECT / "kaggle_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("Project:", PROJECT)
'''), markdown("## 2. Cài thư viện, kiểm tra GPU và công cụ đề"), code('''# Cài trước khi kernel import torch/transformers. Giữ bản CUDA torch/vision của Kaggle.
subprocess.run(["nvidia-smi", "-L"], check=True)
probe = subprocess.run([sys.executable, "-c",
    "import torch, torchvision; assert torch.cuda.is_available(), 'GPU required'; "
    "print('torch',torch.__version__,'torchvision',torchvision.__version__); "
    "assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,4), 'torch>=2.4 required'"],
    check=True)
import importlib.metadata as package_metadata
constraints = PROJECT / "kaggle_torch_constraints.txt"
versions = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        versions.append(name + "==" + package_metadata.version(name))
    except package_metadata.PackageNotFoundError:
        pass
constraints.write_text(chr(10).join(versions) + chr(10), encoding="utf-8")
print("Keeping Kaggle CUDA packages:", versions)
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "-r", str(PROJECT / "requirements-kaggle.txt"), "-c", str(constraints)], check=True)
# This pipeline uses bitsandbytes NF4. A preinstalled TorchAO can break PEFT's
# optional backend detection (PEFT 0.21.2 rejects torchao < 0.16.0).
try:
    torchao_version = package_metadata.version("torchao")
except package_metadata.PackageNotFoundError:
    torchao_version = None
if torchao_version is not None:
    print("Removing unused optional TorchAO backend:", torchao_version)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "kaggle_checks"],
               cwd=PROJECT, check=True)
'''), markdown("## 3. Run All: dữ liệu → E0–E3 → chọn bằng val → E4 → xuất kết quả"), code('''# Các setup chạy ở subprocess riêng, giải phóng GPU giữa các thí nghiệm.
# subprocess.check_call truyền log trực tiếp vào Output của Save Version.
try:
    subprocess.run([sys.executable, "-u", "kaggle_lab.py", "--task", "all"],
                   cwd=PROJECT, check=True)
except subprocess.CalledProcessError:
    print("Pipeline dừng vì lỗi. Xem day10/report/failure.json và log bên trên.")
    print("ZIP chứa kết quả đã hoàn thành vẫn được tạo nếu runner đã khởi động.")
    raise
'''), markdown("## 4. Xem kết quả và tải file"), code('''import csv
from IPython.display import FileLink, display
summary = PROJECT / "report/summary.csv"
if summary.exists():
    import pandas as pd
    display(pd.read_csv(summary))
quality = PROJECT / "report/quality_checks.json"
if quality.exists():
    print(quality.read_text(encoding="utf-8"))
display(FileLink("lab_day10_results.zip"))
print("Checkpoint: /kaggle/working/day10/work/<experiment>/adapter/")
print("Projector E2/E3: /kaggle/working/day10/work/<experiment>/projector.pt")
print("Gửi ZIP kết quả để phân tích và viết REPORT.md. Không commit trọng số/dữ liệu.")
''')]
    # Keep the bootstrapping and environment checks; show the implementation itself
    # in executable notebook cells rather than launching --task all.
    cells = staged_cells(cells)
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                             "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}}}
    for i, cell in enumerate(cells):
        cell["id"] = f"day10-{i:02d}"
    output = ROOT / "lab_day10_kaggle.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(output)


def function_source(filename, names):
    source = (ROOT / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return "\n\n".join(ast.get_source_segment(source, nodes[name]) for name in names) + "\n"


def staged_cells(old_cells):
    loader = function_source("kaggle_lab.py", ["user_message", "make_quantization_config", "load_base",
                                               "architecture", "seed_everything"])
    generation = function_source("kaggle_lab.py", ["predict", "score", "make_examples"])
    cells = [markdown("""# Lab Day 10 trên Kaggle: QLoRA Qwen3-VL-4B

Notebook được trình bày theo mục 0–8 của `lab_day10.ipynb`.
Mã chuyển nhãn, mask loss, LoRA, train và inference hiển thị trực tiếp trong các cell.
E1/E2/E3 có cell huấn luyện và cell đánh giá riêng; Save Version → Save & Run All chạy từ trên xuống.

Chọn GPU T4, bật Internet, Add Input dataset gốc
`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`.
Nếu T4 x2, mã dùng GPU 0. Kết quả: `/kaggle/working/lab_day10_results.zip`.
E5/GRPO là bonus, chưa triển khai. Báo cáo sẽ được viết từ số liệu thật sau khi chạy.
"""), markdown("## 0. Môi trường và cấu hình Kaggle"), old_cells[2],
        markdown("### Khôi phục công cụ đề và mã nguồn dùng cho kiểm thử\nCell bootstrap tạo scripts/tests/configs và các file Python đối chiếu. Phần thực nghiệm được thực thi bằng các cell bên dưới."),
        old_cells[4], old_cells[6], code('''import gc, hashlib, importlib.metadata, json, os, platform
import random, re, shutil, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import torch
from IPython.display import display, FileLink

ROOT = PROJECT
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.labcommon import read_jsonl
base = json.loads((ROOT / "kaggle_run_config.json").read_text(encoding="utf-8"))
'''), code(function_source("kaggle_utils.py", ["write_json", "assert_disjoint_sessions"])
                      + "\n" + function_source("kaggle_lab.py", ["command", "image_path", "find_dataset"])),
        markdown("## 1. Dữ liệu: tải và chia split"), code('''(ROOT / "work").mkdir(exist_ok=True)
(ROOT / "report").mkdir(exist_ok=True)
flag, source = find_dataset(base["input_root"], base.get("data_source"))
command([sys.executable, "scripts/prepare_data.py", flag, source])
splits = {s: read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
dataset_meta = json.loads((ROOT / "data/splits/meta.json").read_text(encoding="utf-8"))
base["dataset_version"] = dataset_meta["dataset_version"]
session_counts = assert_disjoint_sessions(splits)
for split, rows in splits.items():
    print(split, "images=", len(rows), "sessions=", session_counts[split],
          "boxes=", dataset_meta["counts"][split]["boxes"])
print("dataset_version:", base["dataset_version"])
'''), code('''# Vẽ 6 ảnh với nhãn YOLO gốc, trước khi chuyển sang JSON.
from PIL import Image, ImageDraw
for row in splits["train"][:6]:
    with Image.open(image_path(row)) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for obj in row["objects"]:
        xc, yc, w, h = obj["yolo"]
        box = [(xc-w/2)*width, (yc-h/2)*height, (xc+w/2)*width, (yc+h/2)*height]
        draw.rectangle(box, outline="lime", width=2)
        draw.text((box[0], box[1]), obj["label"], fill="lime")
    display(image)
'''), markdown("## 2. Nhãn: YOLO → JSON thang 0–1000"),
        code((ROOT / "kaggle_utils.py").read_text(encoding="utf-8")),
        code(function_source("kaggle_lab.py", ["draw_boxes"])), code('''# Bỏ nhãn train suy biến khỏi đáp án; giữ nguyên split và GT của đề.
degenerate = [{"split": s, "image_id": row["image_id"], "object": obj}
              for s, rows in splits.items() for row in rows for obj in row["objects"]
              if yolo_to_box(obj["label"], obj["yolo"]) is None]
if any(item["split"] != "train" for item in degenerate):
    raise RuntimeError("Có nhãn val/test suy biến; kiểm tra dataset trước khi chấm.")
write_json(ROOT / "report/data_checks.json", {
    "sessions": session_counts, "meta": dataset_meta,
    "degenerate_training_objects": degenerate,
    "policy": "Skip degenerate objects in training answers only; original splits/GT unchanged."})
print("Prompt:", PROMPT)
print("Example answer:", objects_to_answer(splits["train"][0]["objects"]))
print("Degenerate training objects:", degenerate)
for row in splits["train"][:6]:
    path = ROOT / "report/figures/data" / f"{row['image_id']}.png"
    draw_boxes(row, [], path, "Training label check (normalized round-trip)")
display(Image.open(ROOT / "report/figures/data" / f"{splits['train'][0]['image_id']}.png"))
'''), markdown("## 3. E0: zero-shot"), code(loader), code(generation),
        code(function_source("kaggle_lab.py", ["config_for", "restore_completed", "bundle_results"])),
        code('''# Cùng một revision model cho mọi thí nghiệm độc lập.
from huggingface_hub import model_info
base["model_revision"] = base.get("model_revision") or model_info("Qwen/Qwen3-VL-4B-Instruct").sha
base["code_sha256"] = hashlib.sha256(
    (ROOT / "kaggle_lab.py").read_bytes() + (ROOT / "kaggle_utils.py").read_bytes()).hexdigest()
base["execution_style"] = "notebook_cells"
write_json(ROOT / "kaggle_run_config.json", base)
environment = {"python": sys.version, "platform": platform.platform(), "packages": {}}
for name in ("torch", "torchvision", "transformers", "peft", "bitsandbytes", "accelerate"):
    environment["packages"][name] = importlib.metadata.version(name)
environment["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
write_json(ROOT / "report/environment.json", environment)
print("Model revision:", base["model_revision"])
'''), code(EVALUATE_SETUP), code('''cfg_e0 = config_for(base, "e0")
evaluate_setup(cfg_e0)
print("E0 test:", json.loads((ROOT / "report/evidence/e0.json").read_text())["micro"])
'''), markdown("## 4. Dữ liệu huấn luyện: chat template và mask loss"),
        code(function_source("kaggle_lab.py", ["encode", "collate"])), code('''from transformers import AutoProcessor
processor_check = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", revision=base["model_revision"])
example, answer, prefix_length = encode(splits["train"][0], processor_check)
supervised = example["labels"][example["labels"] != -100]
decoded = processor_check.tokenizer.decode(supervised, skip_special_tokens=False)
image_id = processor_check.tokenizer.convert_tokens_to_ids("<|image_pad|>")
vision_tokens = int((example["input_ids"] == image_id).sum())
assert vision_tokens == 220
assert decoded.startswith(answer) and "<|im_end|>" in decoded
write_json(ROOT / "report/mask_example.json", {
    "vision_tokens": vision_tokens, "masked_prefix_tokens": prefix_length,
    "supervised_tokens": int(supervised.numel()), "decoded_supervised": decoded})
print("Vision tokens:", vision_tokens, "Masked prefix:", prefix_length)
print("Supervised answer:", decoded)
print("Collated shapes:", {k: tuple(v.shape) for k, v in
      collate([example, example], processor_check.tokenizer.pad_token_id).items()})
del example, supervised, processor_check
'''), markdown("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
        code(function_source("kaggle_lab.py", ["setup_model"])), code('''# In cấu hình và số module mục tiêu trước khi train.
modules = (ROOT / "work/e0/module_names.txt").read_text(encoding="utf-8").splitlines()
for name in ("e1", "e2", "e3"):
    cfg = config_for(base, name)
    matched = target_names(modules, include_vit=(name == "e3"))
    assert len(matched) == (348 if name == "e3" else 252)
    print(name.upper(), "LoRA modules:", len(matched), "rank:", cfg["rank"],
          "train_projector:", name in ("e2", "e3"))
print("Architecture:", json.loads((ROOT / "work/e0/architecture.json").read_text()))
'''), markdown("## 6. Vòng huấn luyện"),
        code(function_source("kaggle_lab.py", ["save_weights", "train"])), code(TRAIN_SETUP)]
    for name in ("e1", "e2", "e3"):
        cells += [markdown(f"### Huấn luyện {name.upper()} độc lập từ model gốc"),
                  code(f'cfg_{name} = config_for(base, "{name}")\ntrain_setup(cfg_{name})\n')]
    cells += [markdown("## 7. Suy luận và chấm E1–E3"),
              code(function_source("kaggle_lab.py", ["reload_weights"]))]
    for name in ("e1", "e2", "e3"):
        cells += [markdown(f"### Nạp checkpoint và đánh giá {name.upper()}"),
                  code(f'evaluate_setup(cfg_{name})\n')]
    cells += [code('''write_summary(ROOT / "report/evidence", ROOT / "report/summary.csv")
import pandas as pd
display(pd.read_csv(ROOT / "report/summary.csv"))
'''), markdown("## 8. E4: ablation data_size; E5 là bonus tùy chọn"), code(E4_SELECTION),
              code('''cfg_e4 = config_for(base, "e4_data_size_10", setup=best, fraction=0.1)
train_setup(cfg_e4)
'''), code('''evaluate_setup(cfg_e4)
evidence_path = ROOT / "report/evidence/e4_data_size_10.json"
evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
evidence["meta"].update(ablation_variable="data_size", train_fraction=0.1, baseline_setup=best)
write_json(evidence_path, evidence)
bundle_results()
'''), markdown("### Kiểm tra bằng chứng và xuất kết quả để viết báo cáo"), code(FINISH_RESULTS),
              code("".join(old_cells[10]["source"]).replace(
                  'display(FileLink("lab_day10_results.zip"))',
                  'display(FileLink(str(ROOT.parent / "lab_day10_results.zip")))'))]
    return cells


EVALUATE_SETUP = '''def evaluate_setup(cfg):
    """Đánh giá một setup; luôn giải phóng GPU và đóng gói kết quả khi kết thúc cell."""
    name = cfg["name"]
    run_dir = ROOT / "work" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = processor = None
    try:
        if restore_completed(base, cfg):
            make_examples(name, splits["test"])
            return
        seed_everything(cfg["seed"])
        write_json(run_dir / "effective_config.json", cfg)
        if name != "e0":
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if fingerprint(run["config"]) != fingerprint(cfg) or run["optimizer_steps"] != cfg["max_steps"]:
                raise RuntimeError("Checkpoint/config không khớp lượt train đầy đủ.")
            meta = {k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
        else:
            meta = {"trainable_params": 0, "peak_vram_gb": 0, "train_minutes": 0}
        model, processor = load_base(cfg)
        write_json(run_dir / "architecture.json", architecture(model))
        (run_dir / "module_names.txt").write_text(
            "\\n".join(n for n, _ in model.named_modules()), encoding="utf-8")
        if name != "e0":
            model = reload_weights(model, run_dir, cfg)
        else:
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
                "inference_minutes": sum(v["minutes"] for v in inference.values()),
                "vram_unit": "GiB (2**30 bytes)"})
        make_examples(name, splits["test"])
        write_json(run_dir / "completed.json", {"fingerprint": fingerprint(cfg), "config": cfg})
    except Exception as exc:
        write_json(ROOT / "report/failure.json", {
            "phase": "evaluate", "experiment": name, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        bundle_results()
'''

TRAIN_SETUP = '''def train_setup(cfg):
    """Huấn luyện một setup độc lập. Chạy lại cell có thể dùng checkpoint train hoàn tất cùng config."""
    name = cfg["name"]
    run_dir = ROOT / "work" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = processor = None
    try:
        if restore_completed(base, cfg):
            return
        existing = run_dir / "run.json"
        if existing.exists() and (run_dir / "adapter/adapter_config.json").exists():
            run = json.loads(existing.read_text(encoding="utf-8"))
            if (run.get("status") == "trained" and run.get("optimizer_steps") == cfg["max_steps"]
                and fingerprint(run["config"]) == fingerprint(cfg)
                and (cfg["setup"] == "e1" or (run_dir / "projector.pt").exists())):
                print("Dùng lại checkpoint train đã hoàn thành:", name)
                return
        # Xóa marker lượt cũ trước khi train lại để không tái dùng nhầm bằng chứng.
        if (run_dir / "completed.json").exists():
            (run_dir / "completed.json").unlink()
        seed_everything(cfg["seed"])
        write_json(run_dir / "effective_config.json", cfg)
        model, processor = load_base(cfg)
        write_json(run_dir / "architecture.json", architecture(model))
        model, trainable_params = setup_model(model, cfg, run_dir)
        print(name, "trainable_params:", trainable_params)
        train(model, processor, splits["train"], cfg, run_dir, trainable_params)
    except Exception as exc:
        write_json(ROOT / "report/failure.json", {
            "phase": "train", "experiment": name, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        bundle_results()
'''

E4_SELECTION = '''# Chọn baseline chỉ bằng full validation F1; không dùng test để chọn.
vals = {n: json.loads((ROOT / f"report/validation/{n}.json").read_text(encoding="utf-8"))
        for n in ("e1", "e2", "e3")}
best = max(vals, key=lambda n: vals[n]["micro"]["f1"])
write_json(ROOT / "report/e4_selection.json", {
    "selected_setup": best, "criterion": "full validation micro F1; ties prefer E1 then E2",
    "validation_f1": {n: d["micro"]["f1"] for n, d in vals.items()},
    "variable": "data_size", "values": [0.1, 1.0], "baseline_100_percent_reuses": best,
    "test_used_for_selection": False})
baseline = json.loads((ROOT / f"report/evidence/{best}.json").read_text(encoding="utf-8"))
baseline["name"] = "E4_DATA_SIZE_100"
baseline["meta"].update(ablation_variable="data_size", train_fraction=1.0,
                        baseline_setup=best, reused_from=best)
write_json(ROOT / "report/evidence/e4_data_size_100.json", baseline)
print("E4 baseline:", best, "10% vs 100%, mọi cấu hình train khác giữ nguyên")
'''

FINISH_RESULTS = '''# Chấm cùng output E1 theo thang pixel sai để giải thích câu 3 của báo cáo.
command([sys.executable, "scripts/score.py", "--pred", "work/e1.jsonl", "--coord", "pixel",
         "--name", "E1_WRONG_COORD_DIAGNOSTIC", "--out", "report/diagnostics/e1_wrong_coord.json"])
e0 = json.loads((ROOT / "report/evidence/e0.json").read_text(encoding="utf-8"))
e1 = json.loads((ROOT / "report/evidence/e1.json").read_text(encoding="utf-8"))
quality = {
    "e1_parse_valid_at_least_095": e1["parse_valid_rate"] >= 0.95,
    "e1_f1_gain_at_least_010": e1["micro"]["f1"] - e0["micro"]["f1"] >= 0.10,
    "e1_f1_gain": e1["micro"]["f1"] - e0["micro"]["f1"],
    "all_test_predictions_present": all(json.loads(p.read_text())["missing_predictions"] == 0
                                        for p in (ROOT / "report/evidence").glob("*.json")),
    "status": "finished", "report_and_git_checks_pending": True,
    "smoke_run": base["max_steps"] != 300}
write_json(ROOT / "report/quality_checks.json", quality)
result = subprocess.run([sys.executable, "scripts/check_submission.py"], cwd=ROOT,
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
(ROOT / "report/submission_check.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
if (ROOT / "report/failure.json").exists():
    (ROOT / "report/failure.json").unlink()
bundle_results()
print(json.dumps(quality, indent=2))
'''


if __name__ == "__main__":
    build()
