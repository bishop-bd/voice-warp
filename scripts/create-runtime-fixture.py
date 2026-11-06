#!/usr/bin/env python3
"""Create deterministic random RVC weights for packaging tests, not a real voice."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--upstream-server', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream_server.resolve()))
    import torch
    from voice_changer.RVC.inferencer.rvc_models.infer_pack.models import SynthesizerTrnMs256NSFsid
    torch.manual_seed(60842)
    torch.set_num_threads(2)
    configuration = [1025, 32, 16, 16, 32, 2, 2, 3, 0.0, '1', [3], [[1, 3, 5]],
                     [10, 8, 2, 2], 64, [20, 16, 4, 4], 1, 16, 32000]
    model = SynthesizerTrnMs256NSFsid(*configuration, is_half=False).eval()
    weights = {name: tensor.cpu() for name, tensor in model.state_dict().items() if not name.startswith('enc_q.')}
    torch.save({'config': configuration, 'weight': weights, 'f0': 1, 'version': 'v1',
                'info': 'Deterministic random packaging fixture; no trained voice or speech-quality claim'}, args.destination)


if __name__ == '__main__':
    main()
