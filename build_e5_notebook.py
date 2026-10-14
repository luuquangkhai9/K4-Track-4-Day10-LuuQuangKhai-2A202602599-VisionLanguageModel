"""Create the self-contained E5 Kaggle notebook without changing E0-E4 notebooks."""
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def build():
    files = [ROOT / n for n in ("kaggle_e5.py", "e5_trainer.py", "e5_reward.py", "kaggle_lab.py",
                                "kaggle_utils.py", "requirements-e5.txt", "requirements-kaggle.txt",
                                "E5.md", "README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "pytest.ini",
                                "lab_day10.ipynb")]
    for folder, suffix in (("scripts", ".py"), ("tests", ".py"), ("e5_checks", ".py"), ("configs", ".yaml")):
        files += sorted((ROOT / folder).glob(f"*{suffix}"))
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = base64.b64encode(blob.getvalue()).decode("ascii")
    cells = [md("""# E5 bonus — GRPO reward IoU từ checkpoint SFT

Chọn GPU T4, bật Internet. Nếu T4 x2, mã chỉ dùng GPU 0.
**Add Input hai nguồn:** dataset road-damage gốc và **Output của notebook E0–E4 đã chạy**.
Output cần chứa `day10/work/e3/adapter/` và `day10/work/e3/projector.pt`.
ZIP `lab_day10_results.zip` không có trọng số, không dùng thay Output checkpoint.

Notebook chỉ chạy E5 (60 bước), không huấn luyện lại E0–E4. Baseline chọn theo validation
đã lưu, thường là E3. Reward được kiểm thử trước khi nạp GPU model.
Save Version → Save & Run All; tải `lab_day10_e5_results.zip` ở Output để phân tích bonus.
"""), md("## 0. Cấu hình"), code('''SFT_SOURCE = None  # tự tìm Output cũ; hoặc "/kaggle/input/.../day10"
DATA_SOURCE = None  # tự tìm dataset gốc; hoặc root có images/ và labels-YOLO/
MAX_STEPS = 60      # 2 để smoke test; bản 2 bước không dùng làm bằng chứng hoàn thành
MAX_COMPLETION_LENGTH = 128
LEARNING_RATE = 5e-6
'''), md("## 1. Khôi phục mã và cài TRL 1.9.2"), code(f'''import base64, io, json, os, subprocess, sys, zipfile
from pathlib import Path
PROJECT = Path("/kaggle/working/day10_e5")
PROJECT.mkdir(parents=True, exist_ok=True)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
os.chdir("/kaggle/working")
PAYLOAD = "{payload}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(PROJECT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # before any torch import, one physical T4
os.environ["HF_HOME"] = "/tmp/day10_hf_cache"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
config = {{"input_root": "/kaggle/input", "sft_source": SFT_SOURCE, "data_source": DATA_SOURCE,
          "max_steps": MAX_STEPS, "max_completion_length": MAX_COMPLETION_LENGTH,
          "learning_rate": LEARNING_RATE}}
if not isinstance(MAX_STEPS, int) or MAX_STEPS <= 0 or MAX_COMPLETION_LENGTH <= 0 or LEARNING_RATE <= 0:
    raise ValueError("Steps, completion length and learning rate must be positive.")
(PROJECT / "e5_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
'''), code('''import importlib.metadata as metadata
subprocess.run(["nvidia-smi", "-L"], check=True)
subprocess.run([sys.executable, "-c",
    "import torch,torchvision; assert torch.cuda.is_available(); "
    "print('Visible GPUs:',torch.cuda.device_count()); assert torch.cuda.device_count()==1; "
    "assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,4)"], check=True)
constraints = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        constraints.append(name + "==" + metadata.version(name))
    except metadata.PackageNotFoundError:
        pass
lock = PROJECT / "torch_constraints.txt"
lock.write_text(chr(10).join(constraints) + chr(10), encoding="utf-8")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(PROJECT / "requirements-e5.txt"),
               "-c", str(lock)], check=True)
try:
    metadata.version("torchao")
except metadata.PackageNotFoundError:
    pass
else:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
# REPORT.md is completed; exclude only the teacher test requiring an unfilled template.
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "e5_checks",
               "-k", "not report_template_has_student_markers_and_ten_questions"], cwd=PROJECT, check=True)
'''), md("## 2. Reward IoU — mã và kiểm thử"), code((ROOT / "e5_reward.py").read_text(encoding="utf-8")),
        code('''# Công thức đề: JSON hỏng=0; GT và pred rỗng=1;
# còn lại 0.2 + 0.8 * sum(matched IoU) / max(n_gt,n_pred,1).
gt = [{"label":"pothole", "bbox":[100,100,200,200]}]
perfect = '[{"label":"pothole","bbox_2d":[100,100,200,200]}]'
print("Perfect:", iou_reward(perfect, gt))
print("Malformed:", iou_reward("not JSON", gt))
print("Empty on positive GT:", iou_reward("[]", gt))
print("Huge box:", iou_reward('[{"label":"pothole","bbox_2d":[0,0,1000,1000]}]', gt))
'''), md("## 3. Kiểm tra checkpoint và split; không chọn bằng test"), code('''if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import kaggle_e5
def run_task(task):
    log_path = PROJECT / "report/logs" / f"e5_{task}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "kaggle_e5.py", "--task", task],
                cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, process.args)
    finally:
        kaggle_e5.export_results()
run_task("prepare")
print((PROJECT / "report/e5_warm_start.json").read_text(encoding="utf-8"))
'''), md("## 4. Huấn luyện GRPO 60 bước"),
        md("""`num_generations=4`, generation batch=4, train microbatch=1 và accumulation=4.
Tiếp tục LoRA đã học; projector SFT được nạp nguyên trạng và đóng băng.
`T4GRPOTrainer` giữ chính xác fp32 adapter khi TRL khởi tạo và autocast fp16 cho VLM.
Log reward/output trong `work/e5/rollouts.jsonl`; checkpoint mỗi 20 bước.
"""), code((ROOT / "e5_trainer.py").read_text(encoding="utf-8")), code('''run_task("train")
print((PROJECT / "work/e5/run.json").read_text(encoding="utf-8"))
'''), md("## 5. Nạp lại E5, suy luận val/test, so sánh SFT và reward hacking"), code('''# Greedy evaluation uses the SAME 512-token limit as the saved SFT predictions.
run_task("evaluate")
print((PROJECT / "report/e5_comparison.json").read_text(encoding="utf-8"))
'''), md("## 6. Tải bằng chứng bonus"), code('''from IPython.display import FileLink, display
display(FileLink("lab_day10_e5_results.zip"))
print("Tải ZIP trong Output của phiên Save Version và gửi lại để cập nhật báo cáo.")
print("Adapter + projector E5 nằm riêng ở day10_e5/work/e5/. Không commit trọng số.")
''')]
    for i, c in enumerate(cells):
        c["id"] = f"e5-{i:02d}"
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}}}
    output = ROOT / "lab_day10_kaggle_e5.ipynb"
    output.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    build()
