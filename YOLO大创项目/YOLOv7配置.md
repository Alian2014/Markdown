# YOLOv7配置


## 一、 环境配置

### 1. 安装Anaconda

- Anaconda包括Conda、Python以及一大堆安装好的工具包，比如：numpy、pandas等
- conda是一个开源的包、环境管理器，可以用于在同一个机器上安装不同版本的软件包及其依赖，并能够在不同的环境之间切换

### 2. 安装Cuda及Cudnn

- 在NVIDIA控制面板查询电脑显卡支持的最高版本的cuda
- 用conda建立虚拟环境后使用命令安装对应版本的pytorch
- 注意一些cuda版本没有适配的pytorch
--b

### 3. 安装labelimg

- 在命令行窗口中依次输入下列代码，安装labelimg依赖的第三方库

```shell
pip install PyQt5

pip install pyqt5-tools

pip install lxml

pip install labelimg
```

### 4. 下载YOLOv7并测试

- 从Github克隆YOLOv7文件夹，在该进入虚拟环境并运行代码安装yolov7所需的依赖

- 运行detect.py，测试环境是否正常

---

## 二、 准备数据集

### 1. 新建文件夹VOC2007并整理格式

```
VOC2007/                #数据集  
├── Annotations/        #存放labelimg标注图片生成的xml标签文件
│   ├── images.xml      #对应的xml标签文件
│   └── ...             
├── dataSet_path/       #存放所有训练用图片的地址的txt文件
│   ├── test.txt        
│   ├── train.txt       
│   ├── val.txt         
│   ├── train.cache     #运行yolo产生的缓存文件
│   ├── val.cache       
│   └──...              
├── ImageSets/           
│   └── Main/           #存放数据集划分情况
│       ├── test.txt    #测试集列表
│       ├── train.txt   #训练集列表
│       ├── trainval.txt#测试预测集
│       └── val.txt     #预测集列表
├── JPEGImages/         #存放用于训练的图片
│   ├── images.jpg      
│   └── ...
├── labels/             #存yolo类型文件，与训练用图片一一对应
│   ├── images.txt      #yolo类型文件
│   └── ...
├── get_all_name.py     #获取所有的标签类型，输出到all_classes.txt
├── xml_to_yolo.py      #将xml文件转化为yolo文件，以及生成存放所有训练用图片的地址的txt文件
└── all_classes.txt     #所有的标签类型
```

### 2. 配置自己的data文件

- 在data文件夹中新建一个myvoc.yaml，用于配置训练集路径

---

## 三、 调整YOLOv7默认设置

### 1. 基础设置

- --weights：yolov7_training.py

- --cfg：yolov7.yaml

- --data：myvoc.yaml

- --device：0

- --name：VOC2007

### 2. 防止过拟合

- --epochs：100

- --label-smoothing：0.1

- --linear-lr：开启

- hyp.scratch.p5.yaml：lr0: 0.001  lrf: 0.1

### 3. 防止爆显存

- --batch-size：2

### 4. 报错：UnicodeDecodeError: ‘gbk’ codec can’t decode type 0xaf in position 525: illegal multibyte sequence

- 原因是python和win10系统，打开文件时默认的编码方式冲突导致：python默认的是gbk，而win10默认的是utf-8，所以只要改一下python打开文件时，默认的编码就行，代码就在下面复制就行了。

```python 
    #with open(opt.data) as f:
    with open(opt.data, 'r', encoding='utf-8') as f:
```

### 5. 报错：AssertionError: train: No labels in E:\Learning\yolov7\VOC2007\dataSet_path\train.cache. Can not train without labels. 

- 原因是datasets.py中制定的字符串转换以通过图片路径寻找标签路径的方法与数据集文件夹名称的差异，如下修改即可：

```python
    # sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    sa, sb = os.sep + 'JPEGImages' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
```
