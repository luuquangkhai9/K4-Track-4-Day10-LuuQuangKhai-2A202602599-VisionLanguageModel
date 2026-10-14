import ast
import json
from pathlib import Path
import pytest
from kaggle_e5 import find_sft, select_sft, verify_optional_source_splits


def make_source(root, projector=True):
    for name, f1 in (("e1", 0.2), ("e2", 0.3), ("e3", 0.4)):
        path = root / f"report/validation/{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"micro": {"f1": f1}}))
    marker = root / "report/e4_selection.json"
    marker.write_text(json.dumps({"selected_setup": "e3"}))
    adapter = root / "work/e3/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture only")
    (root / "work/e3/run.json").write_text(json.dumps({"optimizer_steps": 300, "status": "trained"}))
    if projector:
        (root / "work/e3/projector.pt").write_bytes(b"fixture only")
    for name in ("e3.jsonl", "e3_val.jsonl"):
        (root / "work" / name).write_text("")


def test_find_completed_best_sft_by_validation(tmp_path):
    root = tmp_path / "input/old-output/day10"
    make_source(root)
    found, best, _ = find_sft(tmp_path / "input")
    assert found == root and best == "e3"


def test_missing_projector_is_not_silently_replaced_by_base(tmp_path):
    make_source(tmp_path, projector=False)
    with pytest.raises(FileNotFoundError, match="projector"):
        select_sft(tmp_path)


def test_results_without_weights_are_not_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="weights"):
        find_sft(tmp_path)


def test_checkpoint_without_data_uses_fresh_splits(tmp_path):
    splits = {s: [{"image_id": s}] for s in ("train", "val", "test")}
    assert verify_optional_source_splits(tmp_path, splits) == {
        s: "not_available_in_checkpoint_input" for s in splits}


def test_available_source_split_is_compared_and_mismatch_rejected(tmp_path):
    directory = tmp_path / "data/splits"
    directory.mkdir(parents=True)
    rows = [{"image_id": "a", "objects": []}]
    (directory / "train.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert verify_optional_source_splits(tmp_path, {"train": rows}) == {"train": "exact_records_match"}
    with pytest.raises(ValueError, match="differ"):
        verify_optional_source_splits(tmp_path, {"train": [{"image_id": "changed"}]})


@pytest.mark.parametrize("corrupt_evidence", [False, True])
def test_prepare_without_data_still_rechecks_sft_evidence(tmp_path, monkeypatch, corrupt_evidence):
    import kaggle_e5
    from kaggle_utils import PROMPT, write_json
    from scripts.score import evaluate
    source = tmp_path / "input/day10"
    make_source(source)
    root = tmp_path / "day10_e5"
    root.mkdir()
    monkeypatch.setattr(kaggle_e5, "ROOT", root)
    cfg = {"input_root": str(tmp_path / "input"), "sft_source": str(source),
           "max_steps": 60, "max_completion_length": 128, "learning_rate": 5e-6}
    write_json(root / "e5_run_config.json", cfg)
    version = "fixture-version"
    write_json(source / "work/e3/run.json", {"optimizer_steps": 300, "status": "trained", "config": {
        "prompt": PROMPT, "dataset_version": version, "model_id": "fixture-model",
        "model_revision": "fixture-revision", "max_new_tokens": 512}})
    splits = {s: [{"image_id": s, "image_path": f"images/{s}.jpg", "session": s,
                  "width": 640, "height": 360, "objects": [
                      {"label": "pothole", "yolo": [0.15, 0.15, 0.1, 0.1]}]}]
              for s in ("train", "val", "test")}
    raw = '[{"label":"pothole","bbox_2d":[100,100,200,200]}]'
    for split in ("val", "test"):
        suffix = "_val" if split == "val" else ""
        (source / f"work/e3{suffix}.jsonl").write_text(json.dumps({"image_id": split, "raw": raw}) + "\n")
        evidence = evaluate(splits[split], {split: raw})
        evidence.update(dataset_version=version, split=split)
        if corrupt_evidence and split == "test":
            evidence["micro"]["f1"] = 0.5
        write_json(source / f"report/{'validation' if split == 'val' else 'evidence'}/e3.json", evidence)

    def prepare_official_splits(args):
        directory = root / "data/splits"
        directory.mkdir(parents=True)
        for split, rows in splits.items():
            (directory / f"{split}.jsonl").write_text(json.dumps(rows[0]) + "\n")
        write_json(directory / "meta.json", {"dataset_version": version})

    monkeypatch.setattr(kaggle_e5.lab, "find_dataset", lambda *a: ("--source", tmp_path / "dataset"))
    monkeypatch.setattr(kaggle_e5.lab, "command", prepare_official_splits)
    assert not (source / "data").exists()
    if corrupt_evidence:
        with pytest.raises(ValueError, match="does not match raw predictions"):
            kaggle_e5.prepare()
    else:
        kaggle_e5.prepare()
        warm_start = json.loads((root / "report/e5_warm_start.json").read_text())
        assert warm_start["cached_sft_val_test_scores_reverified"]
        assert set(warm_start["source_split_checks"].values()) == {"not_available_in_checkpoint_input"}
        assert json.loads((root / "e5_run_config.json").read_text())["baseline"] == "e3"


def test_notebook_syntax_and_single_gpu_before_torch():
    root = Path(__file__).resolve().parents[1]
    path = root / "lab_day10_kaggle_e5.ipynb"
    if not path.exists():
        pytest.skip("E5 notebook has not been built")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(src)
    all_code = "\n".join(code)
    assert all_code.index('os.environ["CUDA_VISIBLE_DEVICES"] = "0"') < all_code.index("import torch")
    assert 'run_task("prepare")' in all_code and 'run_task("train")' in all_code
    assert 'run_task("evaluate")' in all_code
