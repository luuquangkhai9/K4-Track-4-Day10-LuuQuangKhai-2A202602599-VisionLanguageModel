import json
from types import SimpleNamespace
import pytest
from e5_reward import RewardLogger, completion_text, iou_reward, reward_details

GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]


def output(label="pothole", box=(100, 100, 200, 200)):
    return json.dumps([{"label": label, "bbox_2d": list(box)}])


def test_perfect_and_conversational_completion():
    assert iou_reward(output(), GT) == 1.0
    completion = [{"role": "assistant", "content": [{"type": "text", "text": output()}]}]
    assert completion_text(completion) == output()
    assert iou_reward(completion, GT) == 1.0


@pytest.mark.parametrize("bad", ["not JSON", '{"label":"pothole"}',
                                   '[{"label":"pothole","bbox_2d":[1,2,1,3]}]',
                                   '[{"label":"pothole","bbox_2d":[1,2,3,NaN]}]'])
def test_malformed_json_or_invalid_box_gets_zero(bad):
    assert iou_reward(bad, GT) == 0


def test_wrong_label_and_unknown_label_receive_format_reward_only():
    assert iou_reward(output("crack"), GT) == pytest.approx(0.2)
    assert iou_reward(output("unknown"), GT) == pytest.approx(0.2)


def test_extra_box_penalty_and_one_to_one_duplicate_matching():
    two = json.dumps(json.loads(output()) * 2)
    assert iou_reward(two, GT) == pytest.approx(0.6)
    details = reward_details(two, GT)
    assert details["matched_pairs"] == 1
    assert iou_reward(output(), GT * 2) == pytest.approx(0.6)


def test_empty_cases_do_not_reward_omitting_nonempty_gt():
    assert iou_reward("[]", []) == 1
    assert iou_reward("[]", GT) == pytest.approx(0.2)
    assert iou_reward(output(), []) == pytest.approx(0.2)


def test_huge_box_is_not_matched_to_a_tiny_gt():
    raw = output(box=(0, 0, 1000, 1000))
    assert iou_reward(raw, GT) == pytest.approx(0.2)
    assert reward_details(raw, GT)["giant_boxes"] == 1


def test_iou_threshold_and_fractional_reward():
    # IoU 0.5: half-width prediction, exactly at the accepted threshold.
    assert iou_reward(output(box=(100, 100, 150, 200)), GT) == pytest.approx(0.6)
    assert iou_reward(output(box=(100, 100, 149, 200)), GT) == pytest.approx(0.2)


def test_reward_is_order_invariant_and_matching_is_same_class():
    gt = GT + [{"label": "manhole", "bbox": [300, 300, 400, 400]}]
    pred = json.loads(output()) + [{"label": "manhole", "bbox_2d": [300, 300, 400, 400]}]
    assert iou_reward(json.dumps(pred[::-1]), gt) == 1


def test_logger_records_rollouts_and_trl_extra_kwargs(tmp_path):
    logger = RewardLogger(tmp_path / "rollouts.jsonl")
    metrics = {}
    rewards = logger(completions=[output(), "[]"], gt_boxes=[GT, GT], image_id=["a", "a"],
                     completion_ids=[[1, 2], [3]], trainer_state=SimpleNamespace(global_step=7),
                     log_metric=lambda k, v: metrics.update({k: v}), extra_unused=True)
    assert rewards == [1, 0.2]
    rows = [json.loads(s) for s in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert rows[0]["trainer_step"] == 7
    assert metrics["reward_audit/empty_on_nonempty_gt"] == 0.5


def test_logger_rejects_misaligned_ground_truth(tmp_path):
    with pytest.raises(ValueError, match="lengths differ"):
        RewardLogger(tmp_path / "log.jsonl")([output()], [])
