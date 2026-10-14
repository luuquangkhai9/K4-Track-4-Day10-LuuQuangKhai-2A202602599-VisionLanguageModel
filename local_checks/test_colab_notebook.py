import ast
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_has_complete_visible_cells_and_drive_checkpoint_backup():
    nb = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    definitions = {}
    for src in sources:
        assert "YOUR CODE HERE" not in src
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef):
                definitions[node.name] = node
    for name in ("encode", "collate", "setup_model", "train", "predict", "reload_weights", "persist_outputs"):
        assert name in definitions
    save_source = next(s for s in sources if s.startswith("def save_weights("))
    assert 'shutil.copytree(run_dir, BACKUP_ROOT / "work" / run_dir.name' in save_source
    assert any("drive.mount" in s for s in sources)
    assert any('"--zip", DATA_ZIP' in s and 'userdata.get("KAGGLE_KEY")' in s for s in sources)
    assert not any('"--task", "all"' in s for s in sources)


def test_embedded_template_prevents_recursive_notebook_payloads():
    for name in ("lab_day10.ipynb", "lab_day10_kaggle.ipynb"):
        nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
        src = next("".join(c["source"]) for c in nb["cells"] if "PAYLOAD =" in "".join(c["source"]))
        tree = ast.parse(src)
        payload = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "PAYLOAD" for t in n.targets))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            assert archive.read("lab_day10.ipynb") == (ROOT / "lab_day10_template.ipynb").read_bytes()
            assert b"PAYLOAD" not in archive.read("lab_day10.ipynb")
            assert archive.read("REPORT.md") == (ROOT / "REPORT_template.md").read_bytes()
            assert "<điền" in archive.read("REPORT.md").decode("utf-8")
