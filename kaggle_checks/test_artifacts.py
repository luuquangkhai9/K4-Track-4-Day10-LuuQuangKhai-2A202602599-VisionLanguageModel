import json
import zipfile
from pathlib import Path
from PIL import Image
import kaggle_lab
from kaggle_utils import objects_to_answer, write_json
from scripts.score import evaluate


def test_scoring_figures_and_analysis_zip_exclude_weights(tmp_path, monkeypatch):
    root = tmp_path / "day10"
    root.mkdir()
    monkeypatch.setattr(kaggle_lab, "ROOT", root)
    rows = []
    for i in range(6):
        row = {"image_id": f"img{i}", "image_path": f"images/img{i}.jpg", "width": 640,
               "height": 360, "session": "test-session", "objects": [
                   {"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
        path = root / "data" / row["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), "gray").save(path)
        rows.append(row)
    predictions = {r["image_id"]: objects_to_answer(r["objects"]) for r in rows}
    result = evaluate(rows, predictions, n_boot=20)
    result.update(name="E1", split="test", meta={"trainable_params": 10,
                                              "peak_vram_gb": 1, "train_minutes": 1})
    write_json(root / "report/evidence/e1.json", result)
    (root / "work/e1/adapter").mkdir(parents=True)
    (root / "work/e1/adapter/adapter_model.safetensors").write_bytes(b"weights")
    (root / "work/e1/projector.pt").write_bytes(b"weights")
    write_json(root / "work/e1/run.json", {"train_minutes": 1})
    with (root / "work/e1.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"image_id": r["image_id"], "raw": predictions[r["image_id"]],
                                "hit_token_limit": False}) + "\n")
    kaggle_lab.make_examples("e1", rows)
    kaggle_lab.bundle_results()
    with zipfile.ZipFile(tmp_path / "lab_day10_results.zip") as archive:
        names = archive.namelist()
        assert "report/evidence/e1.json" in names
        assert "work/e1.jsonl" in names and "work/e1/run.json" in names
        assert sum(n.endswith(".png") for n in names) == 6
        assert not any(n.endswith((".pt", ".safetensors", ".jpg")) for n in names)
        assert json.loads(archive.read("report/evidence/e1.json"))["micro"]["f1"] == 1
