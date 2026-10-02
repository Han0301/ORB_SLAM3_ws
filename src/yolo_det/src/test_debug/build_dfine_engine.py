#!/usr/bin/env python3

import argparse
from pathlib import Path

import modelopt.onnx.autocast as autocast
import onnx
import tensorrt as trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Build a fixed-batch FP16 TensorRT engine from a D-FINE ONNX '
            'model.'
        )
    )
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--fp16-onnx', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--workspace-gib', type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fp16_onnx.parent.mkdir(parents=True, exist_ok=True)
    converted = autocast.convert_to_mixed_precision(
        onnx_path=str(args.onnx.resolve()),
        low_precision_type='fp16',
        keep_io_types=True,
        providers=['cpu'],
    )
    onnx.save(converted, str(args.fp16_onnx))

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TensorRT 10+ only supports explicit-batch networks and removed the old flag.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(args.fp16_onnx.resolve())):
        errors = '\n'.join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(
            f'TensorRT could not parse {args.fp16_onnx}:\n{errors}'
        )

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gib * 1024 * 1024 * 1024)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(f'FP16 TensorRT engine written: {args.engine}')


if __name__ == '__main__':
    main()
