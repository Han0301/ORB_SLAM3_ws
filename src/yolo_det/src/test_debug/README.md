# D-FINE-S 模型转换

生产节点不依赖 D-FINE 的 Python 源码；这些脚本只负责在开发机上生成本机 TensorRT engine。

当前验证组合：D-FINE commit `956d1709314c2c6a4df6f34de232054578a7449f`、PyTorch 2.5.1、TensorRT 11.3、RTX 4060 Laptop GPU。

官方 `dfine_s_obj2coco.pth` 大小为 41,847,790 字节，SHA-256：

```text
75bedc333cdf665f9cddd2c6a2a9baa34e0e5e9a38db7bb8e6110edc3b4ba8eb
```

导出固定批次 ONNX：

```bash
/home/h/ORBSLAM_ws/.venv-dfine-export/bin/python \
  /home/h/ORBSLAM_ws/src/yolo_det/src/test_debug/export_dfine_s.py \
  --dfine-root /home/h/ORBSLAM_ws/local/D-FINE \
  --config /home/h/ORBSLAM_ws/local/D-FINE/configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml \
  --checkpoint /home/h/ORBSLAM_ws/src/yolo_det/models/dfine_s_obj2coco.pth \
  --output /home/h/ORBSLAM_ws/src/yolo_det/models/dfine_s_obj2coco.onnx
```

TensorRT 11 已移除旧的 FP16 builder flag，因此构建脚本先用 NVIDIA ModelOpt 生成混合 FP16 ONNX，再生成强类型 engine：

```bash
/home/h/ORBSLAM_ws/.venv-yolo-export/bin/python \
  /home/h/ORBSLAM_ws/src/yolo_det/src/test_debug/build_dfine_engine.py \
  --onnx /home/h/ORBSLAM_ws/src/yolo_det/models/dfine_s_obj2coco.onnx \
  --fp16-onnx /home/h/ORBSLAM_ws/src/yolo_det/models/dfine_s_obj2coco.fp16.onnx \
  --engine /home/h/ORBSLAM_ws/src/yolo_det/models/dfine_s_obj2coco.engine \
  --workspace-gib 2
```

`.pth`、`.onnx`、`.engine` 均由 `models/.gitignore` 排除，不应提交到仓库。

节点联调时可用独立测试发布器重复发送一张图像，不需要修改生产代码：

```bash
python3 /home/h/ORBSLAM_ws/src/yolo_det/src/test_debug/publish_test_image.py \
  /home/h/yolo_ws/yolo_det/assets/test_frame.jpg
```
