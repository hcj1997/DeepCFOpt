# DeepCFOpt
环境：
tf2.0 
python 3.6 
keras 2.2.4

DeepCF原作者为Zhi-Hong Deng，论文发表在2019AAAI
论文题目:DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System

改进地方：
  1、原作者的网络结构深度不够，并且一层的神经元长度太多，这导致网络准确率低训练复杂度高。
  CFNet.py中将原作者的网络深度加深，并且每一层的节点数减少
  2、在原作者的网络中，网络分支结合的地方用的是Concatenate()
  将部分Concatenate更改为Multiply效果更佳，推荐时点积运算依然有着不错的效果
总结：
  1、改变网络的深度和宽度
  2、改变网络分支连接效果
  3、网络准确率更高，训练时间更短、参数量变为原来的1/4
  （以上均为网络整体训练的效果，不预训练网络分支）
