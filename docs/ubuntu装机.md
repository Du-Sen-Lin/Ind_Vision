# 一、ubuntu20.04系统安装

## 1、下载ubuntu系统

参考：http://www.taodudu.cc/news/show-2883749.html

```
# https://www.ubuntu.com/download/desktop
ubuntu-18.04.6-desktop-amd64.iso
ubuntu-20.04.6-desktop-amd64.iso
```

## 2、制作U盘

```
# https://rufus.akeo.ie/
```

（1）打开 Rufus；（2）插好你的 U盘；

![image-20230508150810874](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230508150810874.png)

## 3、系统安装

按esc或delete进入bios, 修改ubuntu启动盘作为启动项。

一路安装：可选择双系统/只安装ubuntu

拔掉U盘，将BIOS启动顺序调回，重启。



## 4、基础环境安装

3060Ti;  

内核版本(arch命令)：x86_64

### 4-1、驱动安装

![image-20230526174841161](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230526174841161.png)

| 版本:     | 525.116.04            |
| --------- | --------------------- |
| 发布日期: | 2023.5.9              |
| 操作系统: | Linux 64-bit [x86_64] |
| 语言:     | Chinese (Simplified)  |
| 文件大小: | 394.19 MB             |

```python
下载 NVIDIA-Linux-x86_64-525.116.04.run
# 禁用nouveau ， lsmod | grep nouveau查看是否禁用
sudo gedit /etc/modprobe.d/blacklist.conf 末尾添加
blacklist nouveau
options nouveau modeset=0

# 执行命令 sudo update-initramfs -u
# 重启电脑：reboot

# 安装gcc, make: sudo apt install gcc/make
sudo chmod 777 NVIDIA-Linux-x86_64-525.116.04.run
sudo ./NVIDIA-Linux-x86_64-525.116.04.run
```



### 4-2、cuda安装; cudnn安装（未完成，因docker开发，暂时不安装）

```python
cuda安装 cudnn安装
下载12.0.1 https://developer.nvidia.cn/cuda-toolkit-archive
wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_525.85.12_linux.run
sudo ./cuda_12.0.1_525.85.12_linux.run # 按空格去掉安装显卡驱动的选项

Please make sure that
 -   PATH includes /usr/local/cuda-12.0/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.0/lib64, or, add /usr/local/cuda-12.0/lib64 to /etc/ld.so.conf and run ldconfig as root

# https://developer.nvidia.cn/rdp/cudnn-download 未登录成功网络问题
```

### 4-3、docker安装 ，nvidia-docker安装

```python
3、docker安装 nvidia-docker安装
# 参考 https://docs.docker.com/engine/install/ubuntu/#prerequisites 
# https://blog.csdn.net/qq_31635851/article/details/127645999

# nvidia-docker 安装参考 https://blog.csdn.net/weixin_44633882/article/details/115362059
```



### 4-4、常用工具安装：

```python
安装百度网盘，todesk
# 百度网盘 sudo dpkg -i **.deb
# todesk https://www.todesk.com/linux.html

Ubuntu20.04安装ssh并开启远程访问登录
https://blog.csdn.net/weixin_43085712/article/details/128562116

shell工具：final shell
http://www.hostbuf.com/t/1059.html


安装vscode
https://code.visualstudio.com/
https://blog.csdn.net/weixin_48661404/article/details/127308771

su 失败解决方法
https://blog.csdn.net/weixin_41480156/article/details/113528690

anaconda安装：
主机安装anoconda: 参考 https://zhuanlan.zhihu.com/p/600780684 在root权限下
```

