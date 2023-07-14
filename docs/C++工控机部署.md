# C++ 工控机环境部署（fastdeploy版本部署环境）

## **一、算法环境（fastdeploy版本部署环境）**

### 1、**nvidia驱动安装：[熟悉官网不同显卡对应的不同版本的nvidia驱动；CUDA版本； CUDNN版本]**

```Shell
# 根据显卡，内核，选择对应 驱动下载：NVIDIA-Linux-x86_64-525.89.02.run

# 查看nouveau是否禁用
lsmod | grep nouveau
# 禁用nouveau
sudo vim /etc/modprobe.d/blacklist.conf
在文件末尾输入
blacklist nouveau
options nouveau modeset=0
保存并退出终端
sudo update-initramfs -u
重启电脑，在终端输入以下命令，若无输出，则表示禁用成功
lsmod | grep nouveau

apt install gcc
apt-get install make

./NVIDIA-Linux-x86_64-525.89.02.run
```

### 2、**cuda 安装：**

```Shell
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo sh cuda_11.6.0_510.39.01_linux.run
# continue->accept->取消driver安装后install
```

### 3、**cudnn安装：**

```Shell
https://developer.nvidia.com/rdp/cudnn-download 官网注册账号下载对应版本（cuda11.*）
解压安装：
cp lib/* /usr/local/cuda-11.6/lib64/
cp include/* /usr/local/cuda-11.6/include/
```

### 4、**python 环境安装：**

```Shell
apt update

apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz
tar -xf Python-3.10.*.tgz
cd Python-3.10.*/

./configure --enable-optimizations --prefix=/usr/local/python

make -j8
make install
```

### 5、**PyTorch安装:**

```Shell
apt install python3-pip
# CUDA 11.6 python_base:3.8
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 6、**yolov5 测试显卡调用:**

```Shell
git clone https://github.com/ultralytics/yolov5

# -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt

python3 detect.py --weights yolov5s.pt --source data/images/bus.jpg --device 0
```

### 7、**Cmake 安装：**

```Shell
# https://cmake.org/download/
wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3.tar.gz

tar -zxvf cmake-3.25.3.tar.gz

apt install g++
apt install build-essential
apt install libssl-dev

cd cmake-3.25.3
./bootstrap

make -j8
make install
```

### 8、**opencv安装配置：**

```Shell
apt-get install build-essential 

# libgtk2.0-dev 失败
apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

wget -O opencv-4.6.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip

cd opencv-4.6.0

mkdir -p build && cd build
cmake ..
make -j8
make install
```

### 9、**spdlog 和 nlohmann:**

```Shell
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake ..
make
make install

git clone https://github.com/nlohmann/json.git
mkdir build && cd build
cmake ..
make
make install
```

### 10、**fastdeploy:**

```Shell
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-1.0.4.tgz
tar xvf fastdeploy-linux-x64-gpu-1.0.4.tgz

source {dir}/fastdeploy-linux-x64-gpu-1.0.4/fastdeploy_init.sh


# 解决 在~/.bashrc中添加环境变量 再source 
export LD_LIBRARY_PATH={dir}/fastdeploy-linux-x64-gpu-1.0.4/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/onnxruntime/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/paddle_inference/paddle/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/openvino/runtime/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/openvino/runtime/3rdparty/omp/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/tensorrt/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/opencv/lib64:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/fast_tokenizer/lib:{dir}/fastdeploy-linux-x64-gpu-1.0.4/third_libs/install/paddle2onnx/lib
```