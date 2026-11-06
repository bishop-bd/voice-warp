"""Opt-in numerical check of the qualified ROCm convolution policy.

Uses a trusted local standard RVC checkpoint and generated features. All model
noise is zeroed only inside this verification process so backend differences can
be compared directly. This is not a listening or live latency qualification.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import onnxruntime as ort
    from export_models import zero_torch_noise
    from voice_warp_backend import configure_torch_policy
    from voice_warp_rvc import TorchRvcInferencer, model_metadata
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    slot, checkpoint = model_metadata(args.model, ort)
    model = TorchRvcInferencer(checkpoint, slot, 'cuda:0')
    cases = []
    with torch.no_grad(), zero_torch_noise(torch):
        for frames in (32, 64, 113, 192):
            # These are generated content features, not captured audio.
            inputs = [torch.zeros(1, frames, slot.embChannels, device='cuda:0'),
                      torch.tensor([frames], device='cuda:0'),
                      torch.full((1, frames), 120, dtype=torch.long, device='cuda:0'),
                      torch.full((1, frames), 180.0, device='cuda:0'),
                      torch.zeros(1, dtype=torch.long, device='cuda:0'), None]
            inputs[0][:, :, ::3] = 0.1
            inputs[0][:, :, 1::3] = -0.1
            torch.backends.cudnn.enabled = True
            expected = model.infer(*inputs).cpu()
            policy = configure_torch_policy(torch, {'backend': 'rocm', 'index': 0})
            if policy['convolution'] != 'ATen GPU':
                raise ValueError('This runtime/device does not match the qualified convolution policy')
            actual = model.infer(*inputs).cpu()
            delta = (actual - expected).abs()
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)
            cases.append({'frames': frames, 'max_absolute_error': delta.max().item(),
                          'mean_absolute_error': delta.mean().item(), 'samples': actual.numel()})
    print(json.dumps({'passed': True, 'torch': torch.__version__, 'policy': policy,
                      'atol': 1e-3, 'rtol': 1e-2, 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
