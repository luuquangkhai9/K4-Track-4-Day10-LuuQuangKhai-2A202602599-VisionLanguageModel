"""Generate the completed Colab lab using the same visible, tested algorithm cells."""
import json
from pathlib import Path
from build_kaggle_notebook import build as build_kaggle, code, markdown

ROOT = Path(__file__).resolve().parent


CONFIG = '''# Colab: Runtime → Change runtime type → T4 GPU.
# ZIP: upload archive.zip vào /content, rồi đặt DATA_ZIP="/content/archive.zip".
# Hoặc để DATA_ZIP=None, RAW_DATA_DIR=None và dùng Colab Secrets cho Kaggle CLI.
DATA_ZIP = None
RAW_DATA_DIR = None  # thư mục gốc có images/ và labels-YOLO/, nếu đã giải nén
MOUNT_DRIVE = True  # đặt False nếu chỉ lưu trong session hiện tại
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/lab_day10_backup"
RESUME_SOURCE = None  # có thể trỏ vào DRIVE_BACKUP_DIR của lượt hoàn thành trước
MODEL_REVISION = None  # khi resume, dùng SHA trong kaggle_run_config.json cũ
MAX_STEPS = 300
BATCH_SIZE = 1
GRAD_ACCUM = 8
INFERENCE_BATCH_SIZE = 2
MAX_NEW_TOKENS = 512
DOWNLOAD_RESULTS = True
'''


DRIVE = '''if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP_ROOT = Path(DRIVE_BACKUP_DIR)
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
else:
    BACKUP_ROOT = None
print("Checkpoint backup:", BACKUP_ROOT)


def persist_outputs():
    """Backup logs, predictions, checkpoint and results; exclude raw dataset/cache."""
    if BACKUP_ROOT is None:
        return
    work = ROOT / "work"
    if work.exists():
        (BACKUP_ROOT / "work").mkdir(parents=True, exist_ok=True)
        for item in work.iterdir():
            if item.name in ("raw", "dataset.zip"):
                continue
            destination = BACKUP_ROOT / "work" / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
    if (ROOT / "report").exists():
        shutil.copytree(ROOT / "report", BACKUP_ROOT / "report", dirs_exist_ok=True)
    shutil.copy2(ROOT / "kaggle_run_config.json", BACKUP_ROOT / "kaggle_run_config.json")
    archive = ROOT.parent / "lab_day10_results.zip"
    if archive.exists():
        shutil.copy2(archive, BACKUP_ROOT / archive.name)
    print("Saved to Drive:", BACKUP_ROOT)
'''


DATA = '''(ROOT / "work").mkdir(exist_ok=True)
(ROOT / "report").mkdir(exist_ok=True)
if DATA_ZIP is not None and RAW_DATA_DIR is not None:
    raise ValueError("Chọn đúng một nguồn: DATA_ZIP hoặc RAW_DATA_DIR.")
if RAW_DATA_DIR is not None:
    command([sys.executable, "scripts/prepare_data.py", "--source", RAW_DATA_DIR])
elif DATA_ZIP is not None:
    if not Path(DATA_ZIP).is_file():
        raise FileNotFoundError("Upload ZIP trước khi chạy, hoặc để DATA_ZIP=None để dùng Kaggle Secrets.")
    command([sys.executable, "scripts/prepare_data.py", "--zip", DATA_ZIP])
else:
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    command([sys.executable, "scripts/prepare_data.py", "--kaggle"])
splits = {s: read_jsonl(ROOT / f"data/splits/{s}.jsonl") for s in ("train", "val", "test")}
dataset_meta = json.loads((ROOT / "data/splits/meta.json").read_text(encoding="utf-8"))
base["dataset_version"] = dataset_meta["dataset_version"]
session_counts = assert_disjoint_sessions(splits)
for split, rows in splits.items():
    print(split, "images=", len(rows), "sessions=", session_counts[split],
          "boxes=", dataset_meta["counts"][split]["boxes"])
print("dataset_version:", base["dataset_version"])
'''


EXPORT = '''write_summary(ROOT / "report/evidence", ROOT / "report/summary.csv")
import pandas as pd
display(pd.read_csv(ROOT / "report/summary.csv"))
print((ROOT / "report/quality_checks.json").read_text(encoding="utf-8"))
persist_outputs()
archive = ROOT.parent / "lab_day10_results.zip"
print("Results:", archive)
print("Checkpoint:", ROOT / "work")
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(archive))
print("Gửi ZIP để phân tích và hoàn thành REPORT.md bằng số liệu thật.")
'''


def build():
    build_kaggle()
    notebook = json.loads((ROOT / "lab_day10_kaggle.ipynb").read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if source.startswith("# Lab Day 10 trên Kaggle:"):
                source = """# Lab Day 10: Finetune Qwen3-VL-4B bằng QLoRA trên Colab

Các cell code đã được điền theo các mục 0–8 của mẫu lab.
Chọn Colab T4, rồi chạy từ trên xuống. Có thể dùng ZIP hoặc Kaggle Secrets.
Mã bootstrap cung cấp công cụ đề; thuật toán huấn luyện và suy luận hiển thị trực tiếp trong cell.
Sao lưu adapter/projector mỗi 100 bước lên Drive nếu MOUNT_DRIVE=True.
Kết quả: `/content/lab_day10_results.zip`; số liệu được tạo bởi score.py gốc.
E0–E4 là phần bắt buộc đầy đủ; E5/GRPO vẫn là bonus chưa triển khai.
"""
            elif source.startswith("## 0."):
                source = "## 0. Môi trường (Colab T4) và cấu hình"
            cells.append(markdown(source))
            continue
        if source.startswith("# Mặc định chạy đầy đủ 300 bước"):
            source = CONFIG
        elif "PAYLOAD =" in source:
            source = source.replace('Path("/kaggle/working")', 'Path("/content")')
            source = source.replace('"/tmp/day10_hf_cache"', '"/content/day10_hf_cache"')
            source = source.replace('"input_root": "/kaggle/input", "data_source": DATA_SOURCE,',
                                    '"input_root": "/content", "data_source": RAW_DATA_DIR,')
        elif "kaggle_torch_constraints.txt" in source:
            source = source.replace("Kaggle", "Colab").replace("kaggle_torch_constraints.txt", "colab_torch_constraints.txt")
            source = source.replace('str(PROJECT / "requirements-kaggle.txt"),',
                                    'str(PROJECT / "requirements-kaggle.txt"), "kaggle",')
        elif source.startswith('(ROOT / "work").mkdir(exist_ok=True)'):
            source = DATA
        elif source.startswith("# Cùng một revision model"):
            source = source.replace('"notebook_cells"', '"colab_notebook_cells"')
        elif source.startswith("def save_weights("):
            # Saving weights uses local storage first; copy each checkpoint to Drive.
            target = '\n\ndef train('
            before, after = source.split(target, 1)
            before += '\n    if BACKUP_ROOT is not None:\n        shutil.copytree(run_dir, BACKUP_ROOT / "work" / run_dir.name, dirs_exist_ok=True)\n'
            source = before + target + after
        elif source.startswith("def evaluate_setup(") or source.startswith("def train_setup("):
            source += "\n"
            # finally backs up both completed runs and partial logs after a failure.
            source = source.replace('        bundle_results()\n',
                                    '        bundle_results()\n        persist_outputs()\n')
        elif source.startswith("# Chấm cùng output E1"):
            source += "\npersist_outputs()\n"
        elif source.startswith("import csv"):
            source = EXPORT
        cells.append(code(source))
        if source.startswith("import gc, hashlib"):
            cells.append(markdown("### Mount Google Drive để lưu checkpoint và kết quả"))
            cells.append(code(DRIVE))
    notebook["cells"] = cells
    for i, cell in enumerate(cells):
        cell["id"] = f"colab-day10-{i:02d}"
    notebook["metadata"]["colab"] = {"name": "lab_day10.ipynb", "provenance": []}
    (ROOT / "lab_day10.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(ROOT / "lab_day10.ipynb")


if __name__ == "__main__":
    build()
