"""Compatibility for TRL 1.9.2's QLoRA/VLM path on a single fp16 T4."""
from contextlib import nullcontext
import torch
from trl import GRPOTrainer


class T4GRPOTrainer(GRPOTrainer):
    def __init__(self, model, *args, **kwargs):
        # TRL 1.9.2 unconditionally converts quantized-model trainables to BF16.
        # A dtype conversion back alone would lose SFT precision; restore exact values.
        original = {n: p.detach().float().cpu().clone()
                    for n, p in model.named_parameters() if p.requires_grad}
        super().__init__(model, *args, **kwargs)
        restored = dict(self.model.named_parameters())
        for n, value in original.items():
            if n not in restored or not restored[n].requires_grad:
                raise RuntimeError(f"Trainer changed trainable parameter identity: {n}")
            restored[n].data = value.to(device=restored[n].device, dtype=torch.float32)
        verified = all(torch.equal(restored[n].detach().cpu(), value) for n, value in original.items())
        if not verified:
            raise RuntimeError("SFT initialization was not preserved exactly.")
        self.fp32_preservation = {"verified": verified, "trainable_tensors": len(original),
                                  "trainable_params": sum(v.numel() for v in original.values()),
                                  "precision": "fp32 trainables; fp16 CUDA autocast"}

    def _precision_context(self):
        if self.accelerator.device.type == "cuda" and self.args.fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _generate_single_turn(self, *args, **kwargs):
        # Generation is outside Trainer.compute_loss_context_manager. The frozen
        # FP32 projector still needs autocast with the FP16 vision backbone.
        with self._precision_context():
            return super()._generate_single_turn(*args, **kwargs)

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        # Covers both the detached old-policy logprobs and the training forward.
        with self._precision_context():
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)
