"""Real TRL/VLM one-step CPU smoke test; no pretrained downloads or GPU required."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("datasets")
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                          Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor)
from e5_reward import RewardLogger
from e5_trainer import T4GRPOTrainer
from kaggle_e5 import make_grpo_args
from kaggle_utils import LLM_TARGET, VIT_TARGET


def test_grpo_group_configuration_is_compatible_with_pinned_trl():
    args = make_grpo_args({"max_steps": 60, "learning_rate": 5e-6,
                           "max_completion_length": 128, "seed": 42}, use_cpu=True)
    assert args.num_generations == 4
    assert args.generation_batch_size == 4
    assert args.steps_per_generation == 4
    assert not args.bf16 and not args.use_vllm


def test_real_multimodal_grpo_one_step_and_exact_fp32_warm_start(tmp_path):
    torch.set_num_threads(1)
    vocab = {f"token{i}": i for i in range(512)}
    for i, special in enumerate(["<pad>", "<unk>", "<|im_start|>", "<|im_end|>",
                                 "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "[]", "<|video_pad|>"]):
        del vocab[f"token{i}"]
        vocab[special] = i
    inner = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    inner.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=inner, pad_token="<pad>", unk_token="<unk>",
                                 bos_token="<|im_start|>", eos_token="<|im_end|>",
                                 additional_special_tokens=["<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"],
                                 padding_side="left")
    template = ("{% for m in messages %}<|im_start|>{{m['role']}}\n"
                "{% for c in m['content'] %}{% if c['type']=='image' %}"
                "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{c['text']}}{% endif %}"
                "{% endfor %}<|im_end|>\n{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    tok.chat_template = template
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(size={"shortest_edge": 4096, "longest_edge": 4096},
                                              patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(), tokenizer=tok, chat_template=template)
    cfg = Qwen3VLConfig(
        image_token_id=4, vision_start_token_id=5, vision_end_token_id=6,
        text_config={"vocab_size": 512, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                     "head_dim": 8, "max_position_embeddings": 512,
                     "bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 0},
        vision_config={"depth": 3, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 16, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "deepstack_visual_indexes": [0, 1, 2]})
    model = get_peft_model(Qwen3VLForConditionalGeneration(cfg), LoraConfig(
        r=2, lora_alpha=4, target_modules=f"(?:{LLM_TARGET}|{VIT_TARGET})", task_type="CAUSAL_LM"))
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.00012345)
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    # Exercise TRL's quantized-model BF16 cast even though this tiny CPU model is dense.
    model.is_loaded_in_4bit = True
    args = make_grpo_args({"max_steps": 1, "learning_rate": 5e-6,
                           "max_completion_length": 4, "seed": 42}, use_cpu=True)
    args.output_dir = str(tmp_path / "trainer")
    args.generation_kwargs = {"suppress_tokens": [2, 4, 5, 6, 8]}
    image = Image.new("RGB", (64, 64), "gray")
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": "detect damage"}], "image": image,
         "image_id": f"img{i}", "gt_boxes": [{"label": "pothole", "bbox": [100, 100, 200, 200]}]}
        for i in range(2)])
    reward = RewardLogger(tmp_path / "rollouts.jsonl", max_completion_length=4)

    def test_reward_with_signal(completions, gt_boxes, **kwargs):
        # The random tiny model cannot emit meaningful boxes. Keep the real IoU
        # audit, but inject known advantages ONLY in this test to verify backward.
        reward(completions=completions, gt_boxes=gt_boxes, **kwargs)
        return [i / 3 for i in range(len(completions))]

    trainer = T4GRPOTrainer(model=model, args=args, reward_funcs=test_reward_with_signal,
                            train_dataset=dataset, processing_class=processor)
    assert trainer.fp32_preservation["verified"]
    assert all(torch.equal(dict(model.named_parameters())[n], value) for n, value in initial.items())
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    model.is_loaded_in_4bit = False
    trainer.train()
    assert trainer.state.global_step == 1
    import json
    rows = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len({r["image_id"] for r in rows}) == 1
    assert all(0 <= r["reward"] <= 1 for r in rows)
    assert any(not torch.equal(dict(model.named_parameters())[n], value) for n, value in initial.items())


def test_dataset_keeps_pil_image_and_exact_gt_but_drops_zero_height(tmp_path, monkeypatch):
    import kaggle_e5
    monkeypatch.setattr(kaggle_e5, "ROOT", tmp_path)
    (tmp_path / "data/images").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "data/images/a.jpg")
    row = {"image_id": "a", "image_path": "images/a.jpg", "objects": [
        {"label": "pothole", "yolo": [0.15, 0.15, 0.1, 0.1]},
        {"label": "crack", "yolo": [0.5, 0.5, 0.1, 0.0]}]}
    dataset = kaggle_e5.training_dataset([row])
    sample = dataset[0]
    assert sample["image"].size == (64, 64)
    assert len(sample["gt_boxes"]) == 1 and sample["gt_boxes"][0]["label"] == "pothole"
    assert sample["gt_boxes"][0]["bbox"] == pytest.approx([100, 100, 200, 200])
