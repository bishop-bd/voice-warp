"""Opt-in numerical validation against the original Fairseq HuBERT implementation.

Create a reference with --write-reference in the legacy runtime, then compare it
in the modern runtime without installing Fairseq there. Fixtures contain only
synthetic waveforms/features and are not a listening or speech-quality test.
"""
import argparse
from pathlib import Path
import json


def waveforms(torch):
    torch.manual_seed(42)
    for length in (8192, 16000, 32160):
        t = torch.arange(length, dtype=torch.float32) / 16000
        yield f"tone_{length}", (0.15 * torch.sin(2 * torch.pi * 180 * t) + 0.05 * torch.sin(2 * torch.pi * 370 * t)).unsqueeze(0)
        yield f"noise_{length}", 0.05 * torch.randn(1, length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--runtime-policy", action="store_true",
                        help="Apply the worker's qualified device policy before comparing")
    args = parser.parse_args()
    import numpy as np
    import torch
    from voice_warp_hubert import HUBERT_SHA256, load_hubert
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    policy = None
    if args.runtime_policy:
        from voice_warp_backend import configure_torch_policy
        policy = configure_torch_policy(torch, {"backend": "rocm" if torch.version.hip else "cpu",
                                               "index": torch.device(args.device).index or 0})
    if args.write_reference:
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)])
        legacy = models[0].float().eval()
        values = {"metadata": np.array(json.dumps({"hubert_sha256": HUBERT_SHA256,
                                                    "reference": "fairseq0.12.2", "torch": torch.__version__}))}
        with torch.no_grad():
            for name, waveform in waveforms(torch):
                values[name] = waveform.numpy()
                for layer, project in ((9, True), (12, False)):
                    output = legacy.extract_features(source=waveform, padding_mask=torch.zeros_like(waveform, dtype=torch.bool), output_layer=layer)[0]
                    if project:
                        output = legacy.final_proj(output)
                    values[f"{name}_layer{layer}"] = output.numpy()
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.reference, **values)
        print(f"Wrote synthetic reference: {args.reference}")
        return
    model = load_hubert(args.checkpoint, args.device)
    reference = np.load(args.reference, allow_pickle=False)
    if json.loads(str(reference["metadata"]))["hubert_sha256"] != HUBERT_SHA256:
        raise ValueError("Reference belongs to another shared asset")
    results = []
    with torch.no_grad():
        for name in reference.files:
            if name == "metadata" or "_layer" in name:
                continue
            waveform = torch.from_numpy(reference[name]).to(args.device)
            for layer, project in ((9, True), (12, False)):
                output = model(waveform, layer, project).cpu()
                expected = torch.from_numpy(reference[f"{name}_layer{layer}"])
                difference = (output - expected).abs()
                result = {"input": name, "layer": layer, "max_abs": difference.max().item(),
                          "mean_abs": difference.mean().item()}
                results.append(result)
                torch.testing.assert_close(output, expected, atol=5e-5, rtol=3e-4)
    print(json.dumps({"torch": torch.__version__, "device": args.device, "passed": True,
                      "policy": policy, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
