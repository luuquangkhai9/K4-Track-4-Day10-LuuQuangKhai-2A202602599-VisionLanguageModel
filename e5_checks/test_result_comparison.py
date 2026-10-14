import json
import zipfile
import kaggle_e5
from e5_reward import reward_details
from kaggle_utils import write_json
from scripts.score import evaluate


def test_verified_sft_comparison_and_weight_free_export(tmp_path, monkeypatch):
    root = tmp_path / "day10_e5"
    root.mkdir()
    monkeypatch.setattr(kaggle_e5, "ROOT", root)
    write_json(root / "e5_run_config.json", {"baseline": "e3", "max_steps": 60})
    rows = [{"image_id": f"img{i}", "width": 640, "height": 360, "session": "session",
             "objects": [{"label": "pothole", "yolo": [0.15, 0.15, 0.1, 0.1]}]} for i in range(6)]
    raw = '[{"label":"pothole","bbox_2d":[100,100,200,200]}]'
    splits = root / "data/splits"
    splits.mkdir(parents=True)
    (splits / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    work = root / "work/e5"
    work.mkdir(parents=True)
    (work / "projector.pt").write_bytes(b"fixture weights")
    (work / "adapter").mkdir()
    (work / "adapter/adapter_model.safetensors").write_bytes(b"fixture weights")
    for name in ("e3", "e5"):
        pred = {r["image_id"]: raw if name == "e5" else "[]" for r in rows}
        (root / f"work/{name}.jsonl").write_text("\n".join(json.dumps({
            "image_id": r["image_id"], "raw": pred[r["image_id"]], "hit_token_limit": False}) for r in rows))
        ev = evaluate(rows, pred)
        ev.update(split="test", dataset_version="fixture")
        write_json(root / f"report/evidence/{name}.json", ev)
    rollout = {"trainer_step": 0, "image_id": "img0", "raw": raw, "hit_token_limit": False,
               **reward_details(raw, [{"label": "pothole", "bbox": [100, 100, 200, 200]}])}
    (work / "rollouts.jsonl").write_text("\n".join(json.dumps(rollout) for _ in range(4)))
    kaggle_e5.compare_e5()
    result = json.loads((root / "report/e5_comparison.json").read_text())
    assert result["delta_f1"] == 1
    assert result["training_rollout_audit"]["zero_variance_group_rate"] == 1
    assert result["greedy_test_audit"]["e3"]["empty_on_nonempty_gt"] == 6
    kaggle_e5.export_results()
    with zipfile.ZipFile(tmp_path / "lab_day10_e5_results.zip") as archive:
        names = archive.namelist()
        assert "report/evidence/e5.json" in names and "work/e3.jsonl" in names
        assert not any(n.endswith((".pt", ".safetensors")) for n in names)
