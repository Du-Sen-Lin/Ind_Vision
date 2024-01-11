# Windows部署

# 前置

1、安装visualstudio2022 【参考从c.pdf】

2、安装配置 opencv 4.7.0 【opencv-4.7.0-windows.exe】

https://opencv.org/releases/

可参考：https://blog.csdn.net/weixin_43729127/article/details/132635245

解压位置：D:\package\vs\opencv

配置环境变量：

```
D:\package\vs\opencv\opencv\build\x64\vc16\bin
D:\package\vs\opencv\opencv\build\x64\vc16\lib
```

3、vs2022创建opencv_demo项目测试

Cmake: https://learn.microsoft.com/zh-cn/cpp/build/cmake-projects-in-visual-studio?view=msvc-170&viewFallbackFrom=vs-2019

参考：https://zhuanlan.zhihu.com/p/659412062

严重性	代码	说明	项目	文件	行	禁止显示状态
错误	LNK1107	文件无效或损坏: 无法在 0x340 处读取	D:\Wood\Code\C\vswork\CPPMakeProject\out\build\x64-debug\CPPMakeProject	D:\Wood\Code\C\vswork\CPPMakeProject\lib\dymcdemo.dll	1	

参考：

```
https://blog.csdn.net/zpc20000929/article/details/126731162?
https://github.com/joenali/cmake_win_lib_dll
```



4、其他环境

```python
#本地基础环境安装：
# windows3060ti方案：nvidia驱动（是安装ok的）cuda cudnn anaconda pytorch pycharm
# nvidia-smi查看显卡驱动 30系显卡支持11以上，与10.x版本兼容有问题：11.6  表示显卡驱动最高支持11.6，因此下载cuda版本不能高于这个
# https://pytorch.org/get-started/previous-versions/ 提前查看准备安装的pytorh版本：v1.12.0版本，windows下要求是11.6/11.3，可以向下兼容
# cuda: https://developer.nvidia.com/cuda-toolkit-archive 选择安装 11.6
# 安装路径默认：C:\Users\19478\AppData\Local\Temp\CUDA 选择自定义【C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6】，将驱动组件取消，其他保持勾选
# nvcc -V验证 【不需要手动设置环境变量，环境变量都是自动设置的，可去查看位置。】

# cudnn: https://developer.nvidia.com/rdp/cudnn-archive 选择安装 v8.7.0 for cuda 11.x
# nvidia 账号：1947885050@qq.com 密码：****
# 将cuda文件夹内的文件复制到安装CUDA所在的对应目录下

# anaconda python3.9版本 路径：C:\Users\19478\anaconda3
# 配置环境变量 测试Spyder
conda activate base

# 训练自定义数据集
# 数据标注准备
pip install labelimg
```

4-1、CUDA11.6 

4-2、CUDNN 8.7.0

4-3、opencv 4.7.0

4-4、TensorRT-8.5.3.1

https://developer.nvidia.com/

选择 TensorRT 8.5 GA Update 2 for Windows 10 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 ZIP Package

【TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip】

解压到【D:\package\vs\TensorRT-8.5.3.1】

添加环境变量：【D:\package\vs\TensorRT-8.5.3.1\lib】



5、vs2022属性表创建配置

- 新建一个C++孔项目，项目设置为Debug、X64模式【DeployDemo】
- [属性窗口] -> [右击Debug|x64] -> [添加新项目属性表] 【OpenCV4.7.0_DebugX64.props】【OpenCV4.7.0_ReleaseX64.props】
- 编辑属性表

```python
# opencv ， 创建OpenCV属性表 参考 https://blog.csdn.net/m0_72734364/article/details/128865904?spm=1001.2014.3001.5501
# 【拷贝include路径】[通用属性] -> [VC++目录] -> [包含目录] -> [编辑] 将两个OpenCV两个头文件目录拷贝进去 
D:\package\vs\opencv\opencv\build\include
D:\package\vs\opencv\opencv\build\include\opencv2
# 【拷贝lib路径，外加设置dll到系统环境变量】[通用属性] -> [VC++目录] -> [库目录] -> [编辑] 
D:\package\vs\opencv\opencv\build\x64\vc16\lib
# 【拷贝lib文件名称】[通用属性] -> [链接器] -> [输入] -> [附加依赖项] -> 将文件名"opencv_world470d.lib"拷贝进去

# 创建TensorRT属性表 【TensorRT_X64.props】， debug与release下配置一致, 配置好之后主要release配置直接添加现有属性表即可。
# include路径
path/to/TensorRT-8.5.3.1/include
path/to/TensorRT-8.5.3.1/samples/common
path/to/TensorRT-8.5.3.1/samples/common/windows
# lib路径
path/to/TensorRT-8.5.3.1/lib
# lib文件名称（for release& debug）
nvinfer.lib
nvinfer_plugin.lib
nvonnxparser.lib
nvparsers.lib
# 最后，修改tensorrt属性表：[通用属性] -> [C/C++] -> [预处理器] -> [预处理器定义] -> 添加指令：_CRT_SECURE_NO_WARNINGS -> [确认]


# 创建CUDA属性表，直接添加现有属性表即可。
# CUDA属性表直接使用官方的，路径为 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\extras\visual_studio_integration\MSBuildExtensions\CUDA 11.6.props

```

- cuda 和tensorrt的属性表同时兼容release x64 和debug x64，你再新建TensorRT-Alpha中yolov8 yolov7 yolov6 等项目后，只需要把上述提前做好的属性表引入到工程就行了，当然项目还需要进行简单设置(设置NVCC，避免tensorrt的坑)，在后文提到。属性表做到了一次新建，到处使用。

- yolov8模型部署测试

```python
# Classes
names:
  0: abnormal_ob
  1: plastic
  2: insect
  3: black_dot
  4: stomium
# /root/project/bp_algo/common/YOLO/ultralytics_yolov8/runs/detect/train10/weights/best.pt 
# '/root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val/Image_20230310171144605.bmp'

# 模型转换,导出 onnx 文件 best_ds.onnx
yolo export model=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/runs/detect/train10/weights/best.pt format=onnx dynamic=True simplify=True
# 模型转换,导出 onnx 文件 best.onnx
yolo export model=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/runs/detect/train10/weights/best.pt format=onnx

# 使用 TensorRT 编译 onnx 文件
D:/package/vs/TensorRT-8.5.3.1/bin/trtexec.exe --onnx=./best.onnx --saveEngine=./best.trt --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
# error:
"""
[W] [TRT] onnx2trt_utils.cpp:377: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[01/11/2024-14:50:34] [I] Finish parsing network model
[01/11/2024-14:50:34] [E] Static model does not take explicit shapes since the shape of inference tensors will be determined by the model itself
[01/11/2024-14:50:34] [E] Network And Config setup failed
[01/11/2024-14:50:34] [E] Building engine failed
[01/11/2024-14:50:34] [E] Failed to create engine from model or file.
[01/11/2024-14:50:34] [E] Engine set up failed
"""
# 警告原因以及解决方案：
原因：NNX模型的参数类型是INT64, 这个可以从netron中看到。 TensorRT本身不支持INT64. 
https://github.com/FeiYull/TensorRT-Alpha/issues/69 
解决方案1：使用优化后的./best_ds.onnx， 忽略警告，看是否能够正常编译。
# 错误原因
# 静态模型不采用显式形状，因为推理张量的形状将由模型本身决定，使用dynamic导出的模型

D:/package/vs/TensorRT-8.5.3.1/bin/trtexec.exe --onnx=./best_ds.onnx --saveEngine=./best_ds.trt --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
"""
[01/11/2024-15:31:31] [W] [TRT] onnx2trt_utils.cpp:377: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[01/11/2024-15:31:32] [I] Finish parsing network model
[01/11/2024-15:31:33] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +966, GPU +346, now: CPU 14092, GPU 1665 (MiB)
[01/11/2024-15:31:33] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +153, GPU +56, now: CPU 14245, GPU 1721 (MiB)
[01/11/2024-15:31:33] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
Could not locate zlibwapi.dll. Please make sure it is in your library path!
"""
# 错误原因分析： 没有安装zlip.
# 安装并配置zlib： https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows
# 解决参考：https://blog.csdn.net/AugustMe/article/details/127791707
# 下载：http://www.winimage.com/zLibDll/ 【选择AMD64/Intel EM64T，下载zlib123dllx64.zip】
dll_x64文件夹下的zlibwapi.dll复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin
【我没有移动lib文件，有的技术博文说也要移，如果移动的话，复制到lib文件放到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib】
===成功转trt，存在很多警告warning。

```

```python
# 参考 https://blog.csdn.net/weixin_42166222/article/details/130669596
git clone https://github.com/FeiYull/TensorRT-Alpha.git
# 将 TensorRT-Alpha/yolov8 中选中的文件拷贝到项目的 源文件 中

# 将TensorRT-Alpha/utils 中选中的文件拷贝到项目的 头文件 中

# 最后将TensorRT-8.4.3.1/samples/common 下的 logger.cpp、sampleOptions.cpp 文件拷贝到项目的 资源文件 中

# 接下来设置 生成依赖项，选择 CUDA 11.6（若没有，见下文 遇到的问题 中有解决方案）

# 然后设置 NVCC 编译 .cu及其对应头文件
# 选择.cu文件右键 属性->项类型 更改为 CUDA C/C++ ，然后点击应用、确定即可

# 最后，右键 DeployDemo -> 属性 -> 配置属性 -> 高级 -> 字符集，设置为 未设置

# 点击 生成 -> 生成解决方案，直到成功
# error: 解决方案：在资源文件中，将sampleOptions.cpp移除掉【https://blog.csdn.net/m0_72734364/article/details/128865904?spm=1001.2014.3001.5501】，ok！
"""
严重性	代码	说明	项目	文件	行	禁止显示状态
错误	LNK2019	无法解析的外部符号 "class std::vector<class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >,class std::allocator<class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> > > > __cdecl sample::splitToStringVec(class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> > const &,char)" (?splitToStringVec@sample@@YA?AV?$vector@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@V?$allocator@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@2@@std@@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@D@Z)，函数 "struct std::pair<class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> >,double> __cdecl sample::`anonymous namespace'::splitNameAndValue<double>(class std::basic_string<char,struct std::char_traits<char>,class std::allocator<char> > const &)" (??$splitNameAndValue@N@?A0xb137924a@sample@@YA?AU?$pair@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@N@std@@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@@Z) 中引用了该符号	DeployDemo	D:\Wood\Code\C\vswork\DeployDemo\sampleOptions.obj	1	
"""

# 编译运行:修改为 类别数和类别名,类别数 在 app_yolov8.cpp 中修改, 类别名 在 utils.h 中修改
{0: 'abnormal_ob', 1: 'plastic', 2: 'insect', 3: 'black_dot', 4: 'stomium'}

# 使用如下命令进行图像的推理
--model=D:/Wood/Code/C/vswork/yolov8_models/best_ds.trt --size=640 --batch_size=1  --img=D:/Wood/Code/C/vswork/yolov8_models/Image_20230310171144605.bmp  --savePath=D:/Wood/Code/C/vswork/yolov8_models/results/result # --show
# 右键项目 -> 属性 -> 属性配置 -> 调试 -> 命令参数，将上述命令添加进去

# 点击 本地Windows调试器 ok，结果保存在D:/Wood/Code/C/vswork/yolov8_models/results/result_0.jpg
```





# 一、YOLOv5

可参考：

```python
# 通过 TensorRT 网络定义 API 实现流行的深度学习网络。
https://github.com/wang-xinyu/tensorrtx/tree/master

# 把 TensorRT C++ api推理 YOLOv5的代码，打包成动态链接库(ubuntu系统so文件)，并通过 Python 调用。
https://github.com/emptysoal/YOLOv5-TensorRT-lib-Python

# 基于 TensorRT 的 C++ 高性能推理库。
https://github.com/l-sf/Linfer

# C++推理的tensorRT引擎文件的案例。 系统：Ubuntu 编译：VScode + Cmake
https://github.com/ZhengChuan-1/YoloV5-TensorRT-inference

# 本仓库提供深度学习CV领域模型加速部署案例，仓库实现的cuda c支持多batch图像预处理、推理、decode、NMS。大部分模型转换流程为：torch->onnx->tensorrt。
https://github.com/FeiYull/tensorrt-alpha

# 【YOLO】Windows 下 YOLOv8 使用 TensorRT 进行模型加速部署
https://blog.csdn.net/weixin_42166222/article/details/130669596

# 适用于anomalib导出的onnx格式的模型，测试了patchcore,fastflow,efficient_ad模型
https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp
```



