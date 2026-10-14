"""Execute generated cells in order with CPU fixtures; no real metrics are published."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("IPython")
pytest.importorskip("pandas")
from transformers import AutoProcessor
from huggingface_hub import model_info
from kaggle_checks.test_model_contract import tiny_model
from scripts.score import main as score_main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("notebook_name", ["lab_day10_kaggle.ipynb", "lab_day10.ipynb"])
def test_generated_cells_run_e0_through_e4_in_order(tmp_path, monkeypatch, notebook_name):
    torch.set_num_threads(1)
    from PIL import Image
    import huggingface_hub
    working = tmp_path / "working"
    project = working / "day10"
    input_root = tmp_path / "input"
    (input_root / "road/images").mkdir(parents=True)
    (input_root / "road/labels-YOLO").mkdir()
    monkeypatch.chdir(tmp_path)
    train_calls, prediction_calls, reload_calls = [], [], []

    class Processor:
        def __init__(self):
            self.answer = "[]"
            self.tokenizer = SimpleNamespace(pad_token_id=0, padding_side="left",
                decode=lambda ids, **kwargs: self.answer + "<|im_end|>\n",
                convert_tokens_to_ids=lambda token: 99)

        def apply_chat_template(self, messages, **kwargs):
            ids = [1] + [99] * 220 + [2]
            if len(messages) == 2:
                self.answer = messages[1]["content"][0]["text"]
                ids += [3, 4, 5]
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[0] + [1] * 220 + [0] * (len(ids) - 221)]),
                    "pixel_values": torch.zeros(880, 1536),
                    "image_grid_thw": torch.tensor([[1, 22, 40]])}

    processor = Processor()
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **kw: SimpleNamespace(sha="fixture-sha"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 6 * 2**30)
    original_run = subprocess.run
    drive_calls, download_calls = [], []
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=lambda path: drive_calls.append(path))
    colab.files = SimpleNamespace(download=lambda path: download_calls.append(path))
    try:
        __import__("google")
    except ImportError:
        google = ModuleType("google")
        google.__path__ = []
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def run(args, **kwargs):
        if args[0] == "nvidia-smi":
            return SimpleNamespace(stdout="CPU test fixture", stderr="", returncode=0)
        return original_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    notebook = json.loads((ROOT / notebook_name).read_text(encoding="utf-8"))
    context = {}

    def prepare_fixture():
        raw = project / "work/raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "dataset-sentinel.txt").write_text("raw dataset must not be copied to Drive")
        counts = {}
        for split in ("train", "val", "test"):
            rows = []
            for i in range(6):
                row = {"image_id": f"{split}_{i}", "image_path": f"images/{split}_{i}.jpg",
                       "width": 640, "height": 360, "session": split,
                       "objects": [{"label": lab, "yolo": [0.3 + j * 0.2, 0.5, 0.1, 0.2]}
                                   for j, lab in enumerate(("pothole", "crack", "manhole"))]}
                path = project / "data" / row["image_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (640, 360), "gray").save(path)
                rows.append(row)
            split_path = project / f"data/splits/{split}.jsonl"
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            counts[split] = {"images": 6, "boxes": {c: 6 for c in ("pothole", "crack", "manhole")}}
        (project / "data/splits/meta.json").write_text(
            json.dumps({"dataset_version": "cpu-fixture-only", "counts": counts}), encoding="utf-8")

    def command(args, **kwargs):
        if str(args[1]) == "scripts/prepare_data.py":
            prepare_fixture()
        elif str(args[1]) == "scripts/score.py":
            score_main([str(a) for a in args[2:]])
        else:
            raise AssertionError(f"Unexpected command: {args}")

    def fake_load(cfg):
        return tiny_model(), processor

    def fake_train(model, proc, rows, cfg, run_dir, n):
        train_calls.append(cfg["name"])
        (run_dir / "adapter").mkdir(exist_ok=True)
        (run_dir / "adapter/adapter_config.json").write_text("{}")
        if cfg["setup"] != "e1":
            (run_dir / "projector.pt").write_bytes(b"fixture-only")
        meta = {"trainable_params": n, "peak_vram_gb": 6, "train_minutes": 1}
        context["write_json"](run_dir / "run.json", {**meta, "config": cfg,
            "optimizer_steps": cfg["max_steps"], "status": "trained"})
        return meta

    def fake_predict(model, proc, rows, path, cfg):
        prediction_calls.append((cfg["name"], path.stem))
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                raw = "[]" if cfg["name"] == "e0" else context["objects_to_answer"](row["objects"])
                f.write(json.dumps({"image_id": row["image_id"], "raw": raw,
                                    "hit_token_limit": False}) + "\n")
        return {"minutes": 1, "batch_size_final": 2}

    def fake_reload(model, run_dir, cfg):
        reload_calls.append(cfg["name"])
        return model

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "_torch_constraints.txt" in source:
            continue  # Installation is tested separately; never alter the test environment.
        if "PAYLOAD =" in source:
            source = source.replace('Path("/kaggle/working")', f"Path({str(working)!r})")
            source = source.replace('Path("/content")', f"Path({str(working)!r})")
        exec(compile(source, cell["id"], "exec"), context)
        if "MAX_STEPS = 300" in source:
            context["MAX_STEPS"] = 2
            if notebook_name == "lab_day10.ipynb":
                zip_path = input_root / "fixture.zip"
                with zipfile.ZipFile(zip_path, "w"):
                    pass
                context["DATA_ZIP"] = str(zip_path)
                context["DRIVE_BACKUP_DIR"] = str(tmp_path / "drive-backup")
        if "PAYLOAD =" in source:
            config_path = project / "kaggle_run_config.json"
            config = json.loads(config_path.read_text())
            config["input_root"] = str(input_root)
            config_path.write_text(json.dumps(config))
        context.update(command=command, load_base=fake_load, predict=fake_predict,
                       train=fake_train, reload_weights=fake_reload, display=lambda *a, **kw: None)

    assert train_calls == ["e1", "e2", "e3", "e4_data_size_10"]
    assert [name for name, path in prediction_calls[::2]] == ["e0", "e1", "e2", "e3", "e4_data_size_10"]
    assert reload_calls == ["e1", "e2", "e3", "e4_data_size_10"]
    assert context["best"] == "e1"  # validation ties select the cheaper first setup
    evidence = list((project / "report/evidence").glob("*.json"))
    assert len(evidence) == 6
    quality = json.loads((project / "report/quality_checks.json").read_text())
    assert quality["status"] == "finished" and quality["smoke_run"]
    assert quality["all_test_predictions_present"]
    with zipfile.ZipFile(working / "lab_day10_results.zip") as z:
        assert "report/evidence/e4_data_size_100.json" in z.namelist()
        assert "report/evidence/e4_data_size_10.json" in z.namelist()
    # Rerunning a training cell must reuse its full matching checkpoint.
    context["train_setup"](context["cfg_e1"])
    assert train_calls == ["e1", "e2", "e3", "e4_data_size_10"]
    if notebook_name == "lab_day10.ipynb":
        assert drive_calls == ["/content/drive"]
        assert download_calls == [str(working / "lab_day10_results.zip")]
        assert (tmp_path / "drive-backup/lab_day10_results.zip").exists()
        assert (tmp_path / "drive-backup/work/e2/projector.pt").exists()
        assert not (tmp_path / "drive-backup/work/raw").exists()
