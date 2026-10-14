import ast
import json
from pathlib import Path
import pytest
from kaggle_e5 import find_sft, select_sft


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
