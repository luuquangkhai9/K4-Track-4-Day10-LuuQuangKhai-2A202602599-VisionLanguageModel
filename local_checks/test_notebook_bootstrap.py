"""Local regression checks for the generated Kaggle notebook setup cell."""
import ast
import importlib.metadata
import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_constraints_keep_cuda_versions_and_optional_audio(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / "lab_day10_kaggle.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "kaggle_torch_constraints.txt" in "".join(c["source"]))
    start = source.index("import importlib.metadata as package_metadata")
    end = source.index('subprocess.run([sys.executable, "-m", "pip"', start)
    versions = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128"}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    exec(compile(source[start:end], "generated-kaggle-constraints", "exec"), {"PROJECT": tmp_path})
    assert (tmp_path / "kaggle_torch_constraints.txt").read_text(encoding="utf-8") == (
        "torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n")


def test_all_notebook_cells_parse():
    notebook = json.loads((ROOT / "lab_day10_kaggle.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_shows_lab_sections_and_real_implementation():
    notebook = json.loads((ROOT / "lab_day10_kaggle.ipynb").read_text(encoding="utf-8"))
    headings = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"]
    for step in range(9):
        assert any(h.startswith(f"## {step}.") for h in headings)
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert not any('"--task", "all"' in src for src in sources)
    definitions = {}
    for src in sources:
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = node
    for name in ("yolo_to_box", "objects_to_answer", "load_base", "predict", "encode", "collate",
                 "setup_model", "train", "save_weights", "reload_weights", "train_setup", "evaluate_setup"):
        assert name in definitions
    # Confirm the displayed core functions use the same tested implementation.
    core = ast.parse((ROOT / "kaggle_lab.py").read_text(encoding="utf-8"))
    for node in core.body:
        if isinstance(node, ast.FunctionDef) and node.name in definitions:
            assert ast.dump(node) == ast.dump(definitions[node.name])
    for name in ("e1", "e2", "e3"):
        assert any(f"train_setup(cfg_{name})" in src for src in sources)
        assert any(f"evaluate_setup(cfg_{name})" in src for src in sources)


def test_optional_torchao_is_removed_before_peft_tests(monkeypatch):
    notebook = json.loads((ROOT / "lab_day10_kaggle.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "kaggle_torch_constraints.txt" in "".join(c["source"]))
    start = source.index('try:\n    torchao_version')
    end = source.index('subprocess.run([sys.executable, "-m", "pytest"', start)
    calls = []
    subprocess = SimpleNamespace(run=lambda args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.15.0")
    context = {"package_metadata": importlib.metadata, "subprocess": subprocess, "sys": sys}
    exec(compile(source[start:end], "generated-torchao-cleanup", "exec"), context)
    assert calls == [([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], {"check": True})]

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    calls.clear()
    exec(compile(source[start:end], "generated-torchao-cleanup", "exec"), context)
    assert calls == []
