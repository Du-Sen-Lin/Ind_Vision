# YOLO系列

## 一、前置

1、基础知识：

IoU:交并⽐； FPS: 每秒帧率； NMS: ⾮极⼤值抑制；

2、YOLO v3:

Mul-scales: 多尺度融合；

图⽚缩放：按⽐例缩放，其他区域mask；

3、YOLO v4:

输⼊端：Mosaic数据增强（丰富训练数据集；减少GPU数量）；cmBN; SAT⾃对抗训练；

Backbone: CSPDarkNet(参考cspnet, 增强CNN学习能⼒；降低计算瓶颈；降低内存成本)；Mish 激活函数；Dropblock（Dropout 随机失活，⼤量⽤在全连接层中；dropblock, ⽤在卷积神经⽹络中，受到cutout数据增强思想启发，多个局部区域，整体删减丢弃）;

Neck: SPP（四种最⼤池化⽅式，1x1; 5x5; 9x9; 13x13）; FPN(上采样，分辨率变⼩，感受野增⼤，⾃顶向下，传达强语义特征) + PAN（下采样，⾃底向上，传达强定义特征）

FPN: 先进行下采样，然后再进行上采样来构建多尺度的特征金字塔。

```markdown
FPN层自顶向下传达强语义特征，而特征金字塔则自底向上传达强定位特征。
FPN 高维度向低维度传递语义信息（大目标更明确）,自上而下，上采样。
PAN 低维度向高维度再传递一次语义信息（小目标也更明确），自下而上，下采样。
深层的feature map携带有更强的语义特征，较弱的定位信息。而浅层的feature map携带有较强的位置信息，和较弱的语义特征。
FPN就是把深层的语义特征传到浅层，从而增强多个尺度上的语义表达。
而PAN则相反把浅层的定位信息传导到深层，增强多个尺度上的定位能力。
```

![image-20230626143924417](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230626143924417.png)





![image-20230627162117438](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230627162117438.png)

输出层：CIoU_loss（⽤于坐标的回归损失函数）; DIoU_nms(缓解遮挡问题);

### 1-1、Yolov3:

```
作者: Joseph Redmon
代码（darket版本）: https://github.com/pjreddie/darknet
pytorch版本：https://github.com/ultralytics/yolov3
```

Backbone + FPN + YoloHead

![image-20230322115144156](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230322115144156.png)

特点：全卷积网络，无池化层和全连接层；包含Darknet-53网络结构，anchor锚框，FPN等结构；

降采样：不使用pooling; 使用conv进行降采样，即每个Res模块前面的CBL都起到下采样的作用；

最后一个255：3  x (80 + 4  + 1)  == > 每个网格单元预测3个box x  (80类别 + 4 坐标框信息 + 1置信度信息) 

**Concat：**张量拼接，会扩充两个张量的维度，例如26x26x256和26x26x512两个张量拼接，结果是26x26x768。Concat和cfg文件中的route功能一样。

**add：**张量相加，张量直接相加，不会扩充维度，例如104x104x128和104x104x128相加，结果还是104x104x128。add和cfg文件中的shortcut功能一样。

损失函数：

- 一个是xywh部分带来的误差，也就是bbox带来的loss

坐标xy采用的二分类交叉熵损失; 宽高wh则采用的差方和

- 一个是置信度带来的误差，也就是obj带来的loss

置信度损失也采用的二分类交叉熵损失

- 最后一个是类别带来的误差，也就是class带来的loss

![image-20230505142837680](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230505142837680.png)

二分类交叉熵损失, 分类损失则只计算了有目标的损失，不计算无目标的框的损失。

**理解：**

参考：https://zhuanlan.zhihu.com/p/143747206

 (1) DarkNet-53: 残差思想, 缓解梯度消失的问题，使得模型更容易收敛； 深、浅层特征融合
+多层特征图（multi-scales）；⽆池化层（使⽤卷积步⻓为2）；
（2）多尺度预测：使⽤聚类算法得到9种不同⼤⼩宽⾼的先验框，3个特征图，每个特征图预测3个先
验框；COCO数据集：⼀个先验框：80个类别预测值 + 4个位置预测值 + 1个置信度预测值；3个预测框
3x(80+5) =255=每⼀个特征图的预测通道数；
（3）使⽤Logistic函数（sigmoid）代替Softmax函数处理类别的预测得分，Softmax只能预测⼀个
类别，Logistic分类器相互独⽴，可以实现多类别的预测, Softmax可以被多个独⽴的Logistic分类器取
代，并且准 确率不会下降；

### 1-2、Yolov4：

理解：https://zhuanlan.zhihu.com/p/143747206

![image-20230324165502819](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230324165502819.png)

![image-20230920160523116](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230920160523116.png)



**算法理解：**

```markdown
参考 https://zhuanlan.zhihu.com/p/143747206
输⼊端：Mosaic数据增强（丰富训练数据集；减少GPU数量）；cmBN; SAT⾃对抗训练；

Backbone: CSPDarkNet(参考cspnet, 增强CNN学习能⼒；降低计算瓶颈；降低内存成本)；Mish
激活函数；Dropblock（Dropout 随机失活，⼤量⽤在全连接层中；dropblock, ⽤在卷积神经⽹络
中，受到cutout数据增强思想启发，多个局部区域，整体删减丢弃）;

Neck: SPP（四种最⼤池化⽅式，1x1; 5x5; 9x9; 13x13）; FPN(上采样，分辨率变⼩，感受野增
⼤，⾃顶向下，传达强语义特征) + PAN（下采样，⾃底向上，传达强定义特征），理解可参考 http
s://blog.csdn.net/junqing_wu/article/details/105598849

输出层：CIoU_loss（⽤于坐标的回归损失函数）; DIoU_nms(缓解遮挡问题);
```

**mosaic数据增强**：采用了4张图片，随机缩放、随机裁剪、随机排布的方式进行拼接



## 二、yolov5

### 2-1、模型结构：

```markdown
官方：https://github.com/ultralytics/yolov5
训练汉化：https://github.com/wudashuo/yolov5

作者：Ultralytics团队 ⽆论⽂。相关讨论：https://cloud.tencent.com/developer/article/1661447
```

backbone + PANFPN + Yolohead

```
https://zhuanlan.zhihu.com/p/172121380
```

![image-20230324180221335](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230324180221335.png)

![image-20230706170930620](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230706170930620.png)

Yolov5**大分辨率图像小目标检测**：

对大分辨率图片先进行分割，变成一张张小图，再进行检测。为了避免两张小图之间，一些目标正好被分割截断，所以两个小图之间设置**overlap重叠区域**，比如分割的小图是**960\*960**像素大小，则overlap可以设置为**960\*20%=192**像素。每个小图检测完成后，再将所有的框放到大图上，对大图整体做一次**nms操作**，将重叠区域的很多重复框去除。

**损失函数：**

分类损失：BCE loss, Binary CrossEntropy 损失函数，多标签分类任务的损失函数BCE，BCE主要适用于二分类的任务，而且多标签分类任务可以简单地理解为多个二元分类任务叠加；

置信度损失：BCE loss

定位损失：CIOU loss

```markdown
理解：
•输⼊端：Mosaic数据增强；⾃适应锚框计算；⾃适应图⽚缩放；
•Backbone: Focus结构；CSP结构； 
•Neck: FPN + PAN结构，同yolov4; 
•输出端：Bounding box的损失函数是CIoU_loss, 同yolov4; ⾮极⼤值抑制：DIoU_nms，同yolov4;

与yolov4的不同：
模型规模：
	YOLOv5相对于YOLOv4来说更轻量级。YOLOv5是一个单一版本的网络，而YOLOv4则有许多变体（如YOLOv4-tiny、YOLOv4-large等），包括了多种规模和速度的选择。
模型结构：
	YOLOv5采用了一种不同的模型结构，它取消了YOLOv4中的CSPNet结构（Cross-Stage Partial Networks），并使用了PANet（Path Aggregation Network）来进行特征融合。此外，YOLOv5引入了SE模块（Squeeze-and-Excitation）来增强特征的表达能力。
Backbone网络：
	YOLOv5使用了一种名为CSPDarknet53的新型骨干网络，相比YOLOv4中的 CSPDarknet53，它做了一些改动和优化，使得网络更适合目标检测任务。
推理速度：
	由于YOLOv5相对于YOLOv4来说更轻量级，因此在相同的硬件条件下，YOLOv5可能会具有更快的推理速度。
```



```python
# 环境安装和验证
git clone https://github.com/ultralytics/yolov5
pip install -r requirements.txt
python3 detect.py --weights yolov5s.pt --source data/images/bus.jpg --device 0

# 训练
python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight pretrained/yolov5s.pt --epochs 100 --batch-size 16 --device 0,1
python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight runs/train/exp/weights/last.pt --epochs 10 --batch-size 16 --device 0,1

# 测试
python detect.py --weights /root/project/wdcv/common/WDCV/wdcv/re_algo/detection/yolo/yolov5/runs/train/exp/weights/best.pt --source /root/dataset/public/dataset_yolo/images/val/Image_20230310171432581.bmp --device 0
# Image_20230310171144605.bmp
python detect.py --weights /root/project/cv_algo/commom/WDCV/wdcv/re_algo/detection/yolo/yolov5/runs/train/exp4/weights/best.pt --source /root/dataset/public/dataset_yolo/images/val/Image_20230310171144605.bmp --device 0

# 转onnx
python export.py --data data/hayao.yaml --weights runs/train/exp/weights/best.pt --include onnx --device 0 --opset 12
```



### 2-2、C++ 推理

Yolov5 C++ 推理fastdeploy部署：参考hayao_algo

Yolov5 C++ batch推理fastdeploy部署：参考lvqi

Yolov5 C++ batch推理onnxruntime部署：参考hayao_demo

### 2-3、代码分析（Yolov5 v7.0）

```markdown
快速训练（分类/检测/分割）：https://zhuanlan.zhihu.com/p/594299913
参数解释：https://blog.csdn.net/weixin_43694096/article/details/124378167
代码分析：https://blog.csdn.net/weixin_51322383/article/details/130353834
```

#### 2-3-1、目标检测任务：

```shell
# 测试
python detect.py --weights yolov5s.pt --source data/images/bus.jpg --device 0
```

```shell
# 训练
python train.py --data data/hayao.yaml --cfg models/hayao_yolov5s.yaml --weight pretrained/yolov5s.pt --epochs 100 --batch-size 16 --device 2,3
```

#### 2-3-2、分类任务

#### 2-3-3、分割任务

## 三、Yolox

细节可以参考：《Wood_LE_2022.pdf》

```markdown
作者: 旷视BaseDetection团队
代码: https://github.com/Megvii-BaseDetection/YOLOX
blog: https://zhuanlan.zhihu.com/p/397993315
标准⽹络结构：Yolox-s、Yolox-m、Yolox-l、Yolox-x、Yolox-Darknet53；
轻量级⽹络结构：Yolox-Nano、Yolox-Tiny；
```

Backbone + FPN + DecoupledHead （neck部分也可以采用FPN+PAN的结）

### 改进思路： 

#### (1)基准模型：Yolov3_spp 

选择Yolov3_spp结构，并添加⼀些常⽤的改进⽅式，作为Yolov3 baseline基准模型；

![image-20230920173012848](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230920173012848.png)

**训练过程改进**：a、添加了EMA权值更新、Cosine学习率机制等训练技巧；b、使⽤IOU损失函数训练 reg分⽀，BCE损失函数训练cls与obj分⽀；c、添加了RandomHorizontalFlip、ColorJitter以及多尺 度数据增⼴，移除了RandomResizedCrop；

#### (2)Yolox-Darknet53:

 对Yolov3 baseline基准模型，添加各种trick，⽐如Decoupled Head、SimOTA等，得到YoloxDarknet53版本；

![image-20230920173415254](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230920173415254.png)

- 输⼊端：Mosaic数据增强（随机缩放、随机裁剪、随机排布的⽅式进⾏拼接）；Mixup数据增强 （两张图⽚融合）；训练的最后15个epoch，这两个数据增强会被关闭掉；由于采取了更强的数据 增强⽅式，作者在研究中发现，ImageNet预训练将毫⽆意义，因此，所有的模型，均是从头开始 训练的； 

- Backbone: Yolov3 baseline 

- Neck: FPN结构融合（FPN⾃顶向下，将⾼层的特征信息，通过上采样的⽅式进⾏传递融合，得到 进⾏预测的特征图）

-  Prediction层： 

  a、Decoupled Head: 

  ​		 三个Decoupled Head, 称“解耦头”；细节：三个分⽀，类别分⽀-N个类别 的⼆分类判断，Sigmoid激活函数；前景背景判断分⽀；⽬标框坐标信息分⽀；三个分⽀ concat融合，再进⾏总体的concat, 得到8400*85的预测信息，8400指预测框的数量。将检测 头解耦，会增加运算的复杂度⸺可使⽤ 1个1x1 的卷积先进⾏降维。

  b、Anchor-free：

  ​		Yolov3、Yolov4、Yolov5中，通常都是采⽤Anchor Based的⽅式，来提取⽬标 框，进⽽和标注的groundtruth进⾏⽐对，判断两者的差距。Anchor Free⽅式：参数量少了 2/3；Anchor框信息，巧妙的将Backbone下采样的⼤⼩信息引⼊进来。400个框，对应锚框⼤ ⼩2^5=32, 1600个预测框对应锚框⼤⼩16x16，6400个预测框对应锚框⼤⼩8x8； 锚框相当于 桥梁，将8400给锚框和图⽚上⽬标框进⾏关联，挑选出正样本锚框，正样本锚框所对应的位 置，就可以将正样本预测框。关联⽅式--标签分配。

  c、 标签分配：如何挑选正样本锚框？两个关键点：初步筛选，SimOTA;

  初步筛选：根据中⼼点/根据⽬标框（如从8400提出来1000个）；

  SimOTA(精细化筛选): 初筛正样本信息提取：拿到第⼀步初步筛选提出的框的位置信息+前景背 景信息+类别信息； Loss函数计算：对初步筛选的框与gt计算loss函数；cost成本计算：根据上 ⼀步计算的位置损失和类别损失计算cost成本函数；SimOTA: 例如假设三个⽬标框，为每个⽬ 标框挑选10个iou最⼤的候选框，根据iou和计算每个⽬标框分配⼏个候选框（如分别3，4， 3），再去1000个候选框中选择对应数量的cost最⼩的框；如果候选框和多个⽬标检测框关联， 选择较⼩cost的；

- Loss计算：

​		经过第三部分的标签匹配，⽬标框和正样本预测框对应起来了。然后计算两者的Loss, 位置损失函数：iou_loss/giou_loss; ⽬标损失函数：BCE_loss (⼆分类交叉熵损失函数); 分类损 失：BCE_loss. 注意：（1） 前⾯精细化筛选中，使⽤了reg_loss和cls_loss，筛选出和⽬标框所 对应的预测框。因此这⾥的iou_loss和cls_loss，只针对⽬标框和筛选出的正样本预测框进⾏计 算。⽽obj_loss，则还是针对8400个预测框； （2）在Decoupled Head中，cls_output和 obj_output使⽤了sigmoid函数进⾏归⼀化，但是在训练时，并没有使⽤sigmoid函数，原因是训 练时⽤的nn.BCEWithLogitsLoss函数，已经包含了sigmoid操作。

#### (3) 版本:

Yolox-s、Yolox-m、Yolox-l、Yolox-x系列 对Yolov5的四个版本，采⽤这些有效的trick，逐⼀进⾏改进，得到Yolox-s、Yolox-m、Yolox-l、 Yolox-x四个版本；

####  (4)轻量级⽹络：

Yolox-Nano, Yolox-Tiny;



## 四、yolov6

```markdown
作者：美团
代码：GitHub - meituan/YOLOv6: YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications
blog：https://blog.csdn.net/Pariya_Official/article/details/125794741
blog: https://tech.meituan.com/
```

### 4-1、网络结构

Backbone:参考RepVGG网络设计了RepBlock来替代CSPDarknet53模块

```markdown
EfficientRep Backbone：
	在 Backbone 设计方面，我们基于以上 Rep 算子设计了一个高效的Backbone。相比于 YOLOv5 采用的 CSP-Backbone，该 Backbone 能够高效利用硬件（如 GPU）算力的同时，还具有较强的表征能力。
	下图 4 为 EfficientRep Backbone 具体设计结构图，我们将 Backbone 中 stride=2 的普通 Conv 层替换成了 stride=2 的 RepConv层。同时，将原始的 CSP-Block 都重新设计为 RepBlock，其中 RepBlock 的第一个 RepConv 会做 channel 维度的变换和对齐。另外，我们还将原始的 SPPF 优化设计为更加高效的 SimSPPF。
```

Neck： FPN-PAN； 激活函数都为ReLU，从而提升网络训练的速度，且使用transpose反卷积进行上采样，并将neck中的CSP模块也使用RepBlock进行替代(Rep-PAN)；

```
Rep-PAN：
	在 Neck 设计方面，为了让其在硬件上推理更加高效，以达到更好的精度与速度的平衡，我们基于硬件感知神经网络设计思想，为 YOLOv6 设计了一个更有效的特征融合网络结构。
	Rep-PAN 基于 PAN[6] 拓扑方式，用 RepBlock 替换了 YOLOv5 中使用的 CSP-Block，同时对整体 Neck 中的算子进行了调整，目的是在硬件上达到高效推理的同时，保持较好的多尺度特征融合能力（Rep-PAN 结构图如下图 5 所示）。
```

Head: 沿用了YOLOX的解耦头设计，不过并未对各个检测头进行降维的操作，而是选择减少网络的深度来减少各个部分的内存占用。此外，在anchor free的锚框分配策略中也沿用了SimOTA等方法来提升训练速度。参考了SloU边界框回归损失函数来监督网络的学习，通过引入了所需回归之间的向量角度，重新定义了距离损失，有效降低了回归的自由度，加快网络收敛，进一步提升了回归精度。

```
对解耦头进行了精简设计，同时综合考虑到相关算子表征能力和硬件上计算开销这两者的平衡，采用 Hybrid Channels 策略重新设计了一个更高效的解耦头结构，在维持精度的同时降低了延时，缓解了解耦头中 3x3 卷积带来的额外延时开销
```



### 4-2、其他训练策略：

吸收借鉴了学术界和业界其他检测框架的先进研究进展：Anchor-free 无锚范式 、SimOTA 标签分配策略以及 SIoU 边界框回归损失。

#### 1、**Anchor-free 无锚范式**

传统的目标检测算法（如Faster R-CNN）使用预定义的锚框来进行目标检测，而Anchor-free方法则直接在特征图上预测目标的位置和类别，省略了锚框的设计和调整过程。

Anchor-free: yolox, yolov6。      YOLOX 使用了一种称为“Detection Head”的结构来预测目标的位置和类别。这个结构包括了一系列的卷积层和激活函数，最终输出预测的目标位置、类别以及置信度。YOLOX 是一种基于 Anchor-free 无锚范式的目标检测算法，它没有预定义的锚框，直接在特征图上预测目标的位置和类别。**目标位置的预测**：YOLOX 使用了特定的结构来直接预测目标的中心点坐标、尺寸等信息，从而实现目标检测。

Anchor-based：yolov5。      YOLOv5 是一种基于锚框的目标检测算法，它在网络中预定义了一组锚框，用于在特征图上进行目标检测。 **锚框的选择**：在 YOLOv5 中，一般会通过聚类等方式来自动选择一组适应于数据集的锚框，这样可以使模型更好地适应具体的目标。 **模型结构**：YOLOv5 包括一个骨干网络（backbone）和若干个检测头（detection head），检测头负责预测目标的位置、类别和置信度。

#### 2、**SimOTA 标签分配策略**

将真实目标与预测框进行匹配，从而确定哪些预测框是正样本，哪些是负样本。

具体的分配策略如下：

1. 对于每一个真实目标，计算其与所有预测框的相似度。
2. 根据相似度，将真实目标与相似度最高的预测框进行匹配，将该预测框标记为正样本。
3. 对于剩余的预测框，如果其与任何真实目标的相似度高于一个阈值（通常是一个较低的值），则将其标记为正样本。
4. 对于未被匹配的真实目标，如果其与任何预测框的相似度高于一个阈值（通常是一个较高的值），则将其视为漏检目标。
5. 对于剩余的预测框，将其标记为负样本。

需要注意的是，相似度的计算可以采用多种方式，常用的有 IoU（Intersection over Union）和 GIoU（Generalized Intersection over Union）等。

#### 3、**SIoU 边界框回归损失**

SIoU 损失函数的计算公式如下：

SIoU Loss=1−SIoU

其中，SIoU 的计算公式如下：

SIoU=Area of Intersection / Area of Smallest Enclosing Box

具体步骤如下：

1. 计算预测框和真实框的交集区域的面积（Intersection）。
2. 计算包围这两个框的最小矩形框的面积（Smallest Enclosing Box）。
3. 计算 SIoU。
4. SIoU Loss 就是 1 减去 SIoU。

相比于传统的 IoU 损失，SIoU 考虑了最小包围框，因此对于有重叠但不完全重合的情况，SIoU 的损失会更准确。

SIoU 考虑了最小包围框，因此对于有重叠但不完全重合的情况，SIoU 的损失会更准确。



## 五、yolov7

```
官方代码：https://github.com/WongKinYiu/yolov7
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
```



## 六、yolov8

```
https://github.com/ultralytics/ultralytics
```

C++部署：https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/yolov8

