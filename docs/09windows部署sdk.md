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
- [属性窗口] -> [右击Debug|x64] -> [添加新项目属性表] 【OpenCV4.7.0_DebugX64.props】
- 编辑属性表

```python
# opencv ， 创建OpenCV属性表 参考 https://blog.csdn.net/m0_72734364/article/details/128865904?spm=1001.2014.3001.5501
# 【拷贝include路径】[通用属性] -> [VC++目录] -> [包含目录] -> [编辑] 将两个OpenCV两个头文件目录拷贝进去 
D:\package\vs\opencv\opencv\build\include
D:\package\vs\opencv\opencv\build\include\opencv2
# 【拷贝lib路径，外加设置dll到系统环境变量】[通用属性] -> [VC++目录] -> [库目录] -> [编辑] 
D:\package\vs\opencv\opencv\build\x64\vc16\lib
# 【拷贝lib文件名称】[通用属性] -> [链接器] -> [输入] -> [附加依赖项] -> 将文件名"opencv_world470d.lib"拷贝进去

# 创建TensorRT属性表


# 创建CUDA属性表


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



