import json
import pytest
from kaggle_utils import (assert_disjoint_sessions, objects_to_answer, subset_train,
                          target_names, yolo_to_box)


def test_yolo_round_trip_and_empty_answer():
    box = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])["bbox_2d"]
    assert box == [400, 300, 600, 700]
    assert [box[0] / 1000 * 640, box[1] / 1000 * 360] == [256, 108]
    assert objects_to_answer([]) == "[]"
    clipped = yolo_to_box("crack", [0.02, 0.98, 0.1, 0.1])["bbox_2d"]
    assert clipped == [0, 930, 70, 1000]


def test_target_patterns_exclude_projectors_and_embedding():
    names = ["model.language_model.layers.0.self_attn.q_proj",
             "model.language_model.layers.35.mlp.down_proj",
             "model.visual.blocks.0.attn.qkv", "model.visual.blocks.23.mlp.linear_fc2",
             "model.visual.merger.mlp.0", "model.language_model.embed_tokens"]
    assert target_names(names) == names[:2]
    assert target_names(names, True) == names[:4]


def test_session_leakage_is_rejected():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_sessions({"train": [{"session": "a"}], "test": [{"session": "a"}]})


def test_ablation_is_reproducible_subset_and_keeps_full_baseline():
    records = [{"image_id": str(i)} for i in range(100)]
    small = subset_train(records, 0.1, 42)
    assert len(small) == 10 and small == subset_train(records, 0.1, 42)
    assert subset_train(records, 1, 42) == records
    assert all(r in records for r in small)


def test_answers_have_spatial_order():
    ans = json.loads(objects_to_answer([
        {"label": "crack", "yolo": [0.5, 0.8, 0.2, 0.2]},
        {"label": "pothole", "yolo": [0.5, 0.2, 0.2, 0.2]}]))
    assert [o["label"] for o in ans] == ["pothole", "crack"]


def test_real_degenerate_training_annotation_is_not_a_broken_json_box():
    invalid = {"label": "crack", "yolo": [0.639062, 0.786111, 0.003125, 0.0]}
    assert yolo_to_box(invalid["label"], invalid["yolo"]) is None
    assert objects_to_answer([invalid]) == "[]"
