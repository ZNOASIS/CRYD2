# CRYD2
## 一、训练策略
由于骨骼关键点的序列数据天然具有时空图的结构，适用于作为图神经网络的输入，目前主流的基于骨骼关键点的动作识别算法大多基于图神经网络。相较于基于CNN与RNN的模型，图神经网络能够在较少的参数量下取得较高的识别精度。为了利用GCN模型和Transformer模型的互补能力，本次比赛中，我们的模型使用目前较为优秀的骨骼关键点识别模型CTR-GCN，SkateFormer,ske_former作为backbone，并使用带Label Smoothing的交叉熵作为损失函数，提升模型对困难样本的区分能力。在数据处理阶段，我们使用上采样策略缓解类别不均衡问题，利用分段随机采样的方法处理序列长度不均衡的问题，并引入了骨骼向量、关节角度、骨骼角度、以及运动速度等多种特征特征训练模型，提升模型的准确率与鲁棒性。模型的框图如图1.1所示。

![](https://github.com/ZNOASIS/CRYD2/blob/main/1.1.png)

<p align="center">图1.1 模型框图</p>

## 二、损失函数与数据处理
### 1、	标签平滑的损失函数
我们没有使用传统的交叉熵函数而是改进为标签平滑的损失函数（Label Smoothing Loss），它适用于分类任务，标签平滑的损失函数可以有效提高模型的泛化能力，减轻过拟合。
损失函数的表达式:

$$
L=(1-smoothing) \times(- \log( P_{target} ) )+smoothing \times(-{{1}\over {C}} \sum\limits_{i=1}^C \log( p_{i} )  )
$$

- smoothing 是标签平滑参数 ，通常设量为一个小的正数，用于平滑目标标签，避免将模型的预测强制逼近一个精确的类别。
- $P_{target} $是模型预测的目标类别的概率
- C 是类别数， $p_{i} $是预测的类别i的概率
### 2、余弦退火
在训练策略的选择上，我们使用带Momentum的SGD算法作为模型的优化器，并使用带Warm-Up的余弦退火(Consine Annealing)策略作为学习率衰减的方法。实验中我们发现逐epoch的学习率衰减性能优于逐iteration的学习率衰减策略，因此我们最终选用了逐epoch的余弦退火作为我们的学习率衰减策略。
所有的实验中，我们取epoch = 90，max_lr =0.1，start_lr = 0.01，momentum =0.9。同时，为了进一步缓解模型的过拟合，我们对网络参数加上了幅度为4e-4的L2 weight decay项。

<div align=center>
<img src="https://github.com/ZNOASIS/CRYD2/blob/main/2.2.png"  style="width: 60%; height: auto;"> 
</div>

<p align="center">图2.1 余弦退火</p>

## 三、	数据处理
### 1、上采样
由于个别类别数据量少，为了保证少样本类别识别效果，我们对数据进行了上采样操作，将每一个类别都补充到相同的个数。

<div align=center>
<img src="https://github.com/ZNOASIS/CRYD2/blob/main/3.1.png"  style="width: 60%; height: auto;"> 
</div>

<p align="center">图3.1 上采样</p>

### 2、分段随机采样
对于序列识别任务，通常需要选取一个合适的序列长度作为模型的输入，并将所有输入样本的长度对齐至该选定的序列长度，所选择的序列长度以及样本长度的对齐方法往往能够显著影响模型性能。
如图所示，通过对训练集的观察我们发现，训练集中的关键点序列大部分都是300帧长，为了更好充分利用数据，我们使用了分段随机采样选取256帧进行处理。对于长度小于256的样本，我们将其添加全0帧补全。对与长度大于256帧的样本，我们使用分段均匀采样方法，将其长度压缩至256帧。
具体的做法为：将样本按照有效帧的长度划分为256个区间，在训练过程中，每次分别从每一段中随机采样一帧组成一个长度为256帧的样本作为模型的输入。

<div align=center>
<img src="https://github.com/ZNOASIS/CRYD2/blob/main/3.2.png"  style="width: 60%; height: auto;"> 
</div>

<p align="center">图3.2 分段随机采样</p>

### 3、数据增强
我们在数据投喂阶段依概率随机进行通道交换，剪切，旋转，缩放，空间翻转，时间翻转，高斯噪声，高斯滤波等一种或多种方式，提高了模型泛化能力。
## 四、高阶特征提取
原始训练数据仅包含关键点的坐标与置信度，这属于一阶信息。尽管图神经网络具有抽取节点间高阶特征的能力，但是人为的引入与动作识别有关的高阶特征依然能够为网络提供更加丰富的信息，提升网络的性能。本次比赛中，我们选取了5种特征，详细描述见表1。其中角度特征的抽取如图4.1所示。

表1
| 特征|描述|
|-------|---------------------|
|Joint|	原始关节点坐标|
|Bone	|由源关节点指向汇关节点的二维向量，代表骨骼信息|
|Angles	|骨骼的夹角信息，共有六种夹角，详见代码说明部分|
|Joint Motion	|同一关节点坐标在相邻帧间的差值|
|Bone Motion	|同一骨骼向量在相邻帧间的差值|


![](https://github.com/ZNOASIS/CRYD2/blob/main/4.png)


<p align="center">图4.1 人体骨架图</p>

为了保证输入数据维度一致性，我们分别在六种角度上，对每一个骨架节点都构造并求得了一个角度，得到特征形状为（6，300，17，2）的特征，方便后续特征融合。角度的求法我们选用向量的内积除以向量模长得到。

表2
|特征组合（缩写）	|特征维度|
|--------|----------|
|Joint（J）|	3|
|Bone（B）|	3|
|Joint+Angle（JA）|	9|
|Bone+Angle（BA）	|9|
|Joint+Motion（JM）|	3|
|Bone Motion（BM）	|3|

以上各个模态我们都没有预先生成，均在训练过程中根据对应config，在feeder中动态处理出各个模态，获得带有角度模态的代码参见feeder相同目录下的tools.py文件中get_JA函数。
## 五、全流程复现
使用根目录下的UP.py文件对数据和标签进行上采样处理，请注意修改config和命令行为自己的路径，然后分别用三个模型对各个模态进行训练。

```
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_j.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir/j --device 2 3
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_ja.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir/ja --device 0 1 
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_b.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir/b --device 0 1
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_ba.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir/ba --device 2 3
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_jm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir/jm --device 6
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_bm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir/bm --device 4

python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/default.yaml --work-dir autodl-tmp/CTRGCN/work_dir/j --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/ja.yaml --work-dir autodl-tmp/CTRGCN/work_dir/ja --device 1
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/ba.yaml --work-dir autodl-tmp/CTRGCN/work_dir/ba --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/bone.yaml --work-dir autodl-tmp/CTRGCN/work_dir/b --device 1
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/joint_motion.yaml --work-dir autodl-tmp/CTRGCN/work_dir/jm --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/bone_motion.yaml --work-dir autodl-tmp/CTRGCN/work_dir/bm --device 1


python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_J.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/j --device 0
python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JA.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/ja --device 1
python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_B.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/b --device 2
python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BA.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/ba --device 0
python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JM.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/jm --device 1
python CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BM.yaml --work-dir CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/bm --device 2
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2 --device 0
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2a.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2a --device 1
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2M.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2m --device 0 1 2
```
然后用如下代码进行测试，得到置信度文件
```
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/default.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/j --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/j/runs-76-23332.pt --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/ja.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/ja --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/ja/runs-83-25481.pt --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/bone.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/b --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/b/runs-86-26402.pt --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/ba.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/ba --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/ba/runs-87-26709.pt --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/joint_motion.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/jm --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/jm/runs-97-29779.pt --device 0
python CTR-GCN-main/main.py --config CTR-GCN-main/config/nturgbd-cross-subject/bone_motion.yaml --work-dir autodl-tmp/CTRGCN/work_dir_test/bm --phase test --save-score True --weights autodl-tmp/CTRGCN/work_dir/bm/runs-92-28244.pt --device 0

python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_j.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir_test/j --phase test --save-score True --weights autodl-tmp/SkateFormer-main/work_dir/j/runs-440-270600.pt --device 0
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_ja.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir_test/ja --phase test --save-score True --weights autodl-tmp/SkateFormer-main/work_dir/ja/runs-482-296430.pt --device 1
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_ba.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir_test/ba --phase test --save-score True --weights autodl-tmp/SkateFormer-main/work_dir/ba/runs-435-267525.pt --device 2
python autodl-tmp/SkateFormer-main/main.py --config autodl-tmp/SkateFormer-main/config/train/ntu_cs/SkateFormer_b.yaml --work-dir autodl-tmp/SkateFormer-main/work_dir_test/b --phase test --save-score True --weights autodl-tmp/SkateFormer-main/work_dir/b/runs-486-298890.pt --device 3
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_jm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test/jm --phase test --save-score True --weights CRYD-main/SkateFormer-main/work_dir/jm/runs-487-149509.pt --device 4
python CRYD-main/SkateFormer-main/main.py --config CRYD-main/SkateFormer-main/config/train/ntu_cs/SkateFormer_bm.yaml --work-dir CRYD-main/SkateFormer-main/work_dir_test/bm --phase test --save-score True --weights CRYD-main/SkateFormer-main/work_dir/bm/runs-460-141220.pt --device 4


python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_J.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/j --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/j/runs-97-14841.pt --device 0
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JA.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ja --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/ja/runs-92-14076.pt --device 1
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_B.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/b --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/b/runs-97-14841.pt --device 2
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BA.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ba --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/ba/runs-100-15300.pt --device 0
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_BM.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/bm --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/bm/runs-97-14841.pt --device 1
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_JM.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/jm --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/jm/runs-97-14841.pt --device 0
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2 --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2/runs-100-15300.pt --device 0
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2a.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2a --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2a/runs-89-13617.pt --device 1
python data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/main.py --config data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/config/mixformer_V2_k2M.yaml --work-dir data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2M --phase test --save-score True --weights data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir/k2m/runs-94-14382.pt --device 2
```
用根目录下的search_best.py对验证集进行最优融合参数搜索，然后用ensemble.py测试集进行合成.注意，我们认为效果不好的模型能够给权重融合带了反例，因此我们搜索范围下界是负数。
