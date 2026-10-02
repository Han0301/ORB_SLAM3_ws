#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import onnx
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Export a fixed-batch D-FINE detector to ONNX for the ROS '
            'TensorRT node.'
        )
    )
    parser.add_argument('--dfine-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.dfine_root.resolve()))

    from src.core import YAMLConfig

    config = YAMLConfig(str(args.config.resolve()), resume=str(args.checkpoint.resolve()))
    if 'HGNetv2' in config.yaml_cfg:
        config.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    config.model.load_state_dict(state)

    class ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = config.model.deploy()
            self.postprocessor = config.postprocessor.deploy()

        def forward(
            self, images: torch.Tensor, orig_target_sizes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = ExportModel().eval()
    images = torch.rand(1, 3, 640, 640, dtype=torch.float32)
    orig_target_sizes = torch.tensor([[640, 640]], dtype=torch.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (images, orig_target_sizes),
            str(args.output),
            input_names=['images', 'orig_target_sizes'],
            output_names=['labels', 'boxes', 'scores'],
            opset_version=16,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    print(f'ONNX export checked: {args.output}')


if __name__ == '__main__':
    main()
