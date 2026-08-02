# stiching

一个 C++ 图像拼接实验项目，将 SuperPoint 特征提取和 LightGlue 特征匹配接入 OpenCV Stitcher，并通过 ONNX Runtime 执行神经网络推理。

> 仓库名称保留为 `stiching`；生成的可执行文件名为 `mapstitch`。

## 功能

- 使用 SuperPoint 提取图像关键点与描述子
- 使用 LightGlue 完成特征匹配
- 支持 OpenCV `PANORAMA` 与 `SCANS` 拼接模式
- 输出拼接图像
- 输出匹配点坐标与匹配可视化
- 可将输入图像切分为重叠区域以增加实验性匹配机会

## 依赖

- 支持 C++17 的编译器
- CMake 3.10+
- OpenCV
- ONNX Runtime
- CGAL
- GMP
- FFTW3
- SuperPoint 与 LightGlue ONNX 模型

当前 `CMakeLists.txt` 默认从以下位置查找 ONNX Runtime：

```text
/opt/onnxruntime
```

若安装在其他位置，请修改 `ONNXRUNTIME_ROOT`。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

生成的程序通常位于：

```text
build/mapstitch
```

## 使用

```bash
./build/mapstitch \
  --sp ./models/superpoint.onnx \
  --lg ./models/lightglue.onnx \
  --mthresh 0.2 \
  --mode scans \
  --output result.jpg \
  --match matches.txt \
  image1.jpg image2.jpg
```

主要参数：

```text
--sp <path>             SuperPoint ONNX 模型
--lg <path>             LightGlue ONNX 模型
--mthresh <float>       匹配阈值
--mode panorama|scans   OpenCV Stitcher 模式
--output <path>         拼接结果文件
--match <path>          匹配点坐标输出文件
--d3                    将每张图像切分为多个重叠区域
```

查看内置帮助：

```bash
./build/mapstitch --help
```

## 输出

- 拼接后的图像写入 `--output` 指定路径。
- 匹配点写入 `--match` 指定文件。
- 代码还会生成和显示匹配可视化，因此在无图形界面的服务器上运行时可能需要移除 `imshow` / `waitKey` 或使用虚拟显示。

## 当前限制

- ONNX Runtime 路径在 CMake 中硬编码。
- 模型输入输出格式必须与仓库中的 SuperPoint/LightGlue 封装匹配。
- 部分默认输出路径和参数缺少完整校验。
- 代码仍包含实验性模块与调试显示，不是无头批处理工具。
- 拼接效果依赖图像重叠、视角变化、模型权重和匹配阈值。
