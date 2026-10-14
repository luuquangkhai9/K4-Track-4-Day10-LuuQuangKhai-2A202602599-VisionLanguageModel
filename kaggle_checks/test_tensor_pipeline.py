"""Checks actual processor tensor handling without downloading model weights."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from kaggle_lab import collate, encode


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        ids = [1, 99, 99, 2]
        if len(messages) == 2:
            ids += [3, 4, 5]
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 1, 0] + [0] * (len(ids) - 4)]),
                "pixel_values": torch.randn(8, 12), "image_grid_thw": torch.tensor([[1, 2, 4]])}


def test_answer_mask_and_variable_length_multimodal_collation():
    row = {"image_path": "images/unused.jpg", "objects": []}
    item, answer, n = encode(row, Processor())
    assert answer == "[]" and n == 4
    assert item["labels"].tolist() == [[-100] * 4 + [3, 4, 5]]
    shorter = {k: v.clone() for k, v in item.items()}
    for k in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        shorter[k] = shorter[k][:, :-1]
    shorter["pixel_values"] = shorter["pixel_values"][:4]
    batch = collate([item, shorter], pad_id=0)
    assert batch["input_ids"].shape == (2, 7)
    assert batch["labels"][1, -1] == -100
    assert batch["attention_mask"][1, -1] == 0
    assert batch["mm_token_type_ids"][1, -1] == 0
    assert batch["pixel_values"].shape == (12, 12)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_chat_prefix_mismatch_stops_training():
    class Broken(Processor):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if len(messages) == 2:
                result["input_ids"][0, 0] = 42
            return result
    with pytest.raises(RuntimeError, match="prefix mismatch"):
        encode({"image_path": "unused.jpg", "objects": []}, Broken())
