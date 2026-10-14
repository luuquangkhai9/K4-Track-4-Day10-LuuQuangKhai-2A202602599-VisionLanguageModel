"""Small real Qwen3-VL forward/backward and PEFT serialization tests on CPU."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from kaggle_lab import make_quantization_config, reload_weights, save_weights, setup_model


def tiny_model():
    config = Qwen3VLConfig(
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        text_config={"vocab_size": 512, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 36, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8,
                     "max_position_embeddings": 1024},
        vision_config={"depth": 24, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32, "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "deepstack_visual_indexes": [5, 11, 17]})
    return Qwen3VLForConditionalGeneration(config)


class SaveProcessor:
    def save_pretrained(self, path):
        pass


def test_actual_bnb_conversion_keeps_visual_and_head_fp16():
    bnb = pytest.importorskip("bitsandbytes")
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear
    from transformers.quantizers.quantizers_utils import should_convert_module

    # Reproduce the old exclusion's failure with the exact library matcher.
    visual_qkv = "model.visual.blocks.0.attn.qkv"
    assert should_convert_module(visual_qkv, ["visual", "lm_head"])
    quant = make_quantization_config()
    assert not should_convert_module(visual_qkv, quant.llm_int8_skip_modules)
    model = replace_with_bnb_linear(tiny_model().half(),
                                   modules_to_not_convert=quant.llm_int8_skip_modules,
                                   quantization_config=quant)
    quantized = [name for name, module in model.named_modules()
                 if isinstance(module, bnb.nn.Linear4bit)]
    assert len(quantized) == 252
    assert all(name.startswith("model.language_model.layers.") for name in quantized)
    assert isinstance(model.model.visual.blocks[0].attn.qkv, torch.nn.Linear)
    assert not isinstance(model.model.visual.blocks[0].attn.qkv, bnb.nn.Linear4bit)
    assert all(p.dtype == torch.float16 for p in model.model.visual.parameters())
    assert model.lm_head.weight.dtype == torch.float16


@pytest.mark.parametrize("setup", ["e1", "e2", "e3"])
def test_real_model_gradients_and_saved_projector(tmp_path, setup):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = {"setup": setup, "rank": 4}
    model, n = setup_model(tiny_model(), cfg, tmp_path)
    assert n > 0
    ids = torch.tensor([[10, 501, 500, 500, 500, 500, 502, 11, 12, 13]])
    labels = ids.clone()
    labels[:, :8] = -100
    model.train()
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                   mm_token_type_ids=(ids == 500).long(),
                   pixel_values=torch.randn(16, 1536), image_grid_thw=torch.tensor([[1, 4, 4]]))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    params = dict(model.named_parameters())
    assert any(p.grad is not None for name, p in params.items() if "lora_B" in name)
    projector = {name: p for name, p in params.items() if "visual.merger" in name
                 or "visual.deepstack_merger_list" in name}
    assert all(p.requires_grad == (setup != "e1") for p in projector.values())
    if setup != "e1":
        assert any(p.grad is not None for p in projector.values())
        with torch.no_grad():
            for p in projector.values():
                p.add_(0.05)
    vision_lora = [p for name, p in params.items() if "visual.blocks" in name and "lora_B" in name]
    assert bool(vision_lora) == (setup == "e3")
    if vision_lora:
        assert any(p.grad is not None for p in vision_lora)
    save_weights(model, SaveProcessor(), tmp_path, 1)
    reloaded = reload_weights(tiny_model(), tmp_path, cfg)
    restored = dict(reloaded.named_parameters())
    if setup != "e1":
        assert all(torch.equal(restored[name], value) for name, value in projector.items())
