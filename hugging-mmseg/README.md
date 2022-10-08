# mmseg-frame

#### mmsegmentation做图像分割的整体框架
##### 1、数据集准备：
（1）数据标注；
（2）转换为语义分割特征图（掩模图像）；
（3）数据集划分；
（4）datasets注册

##### 2、模型训练：
（1）模型配置（config文件配置），包括：
* model：选用的网络
* datasets：数据集的配置，包括，数据集位置、数据预处理、数据增强等
* default runtime：运行配置
* schedules：优化算法、迭代次数等

（2）开始训练：
* 导入配置文件；
* 构造数据集；
* 构造分割模型；
* 创建工作目录；
* 执行训练。

##### 3、模型推理（预测）
* 导入模型配置文件；
* 导入模型训练结果（权重）；
* 读取测试集；
* 执行推理。

##### 4、预训练模型使用：主要配置config文件中的pretrained和load_from两个选项
* 直接使用预训练模型：pretrained=''（权重文件链接）
* 从零训练模型：pretrained=None， load_from=None
* 对预训练模型再次训练：pretrained=None， load_from=''（权重文件本地路径）
