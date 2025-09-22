# 基于Transformer的IMDB情感分类器

这是一个完全手搓实现的基于Transformer架构的NLP神经网络，用于IMDB电影评论的情感分类任务。

## 功能特点

### 🧠 核心架构
- **多头注意力机制**: 实现完整的自注意力机制，支持多头并行计算
- **位置编码**: 使用正弦和余弦函数的位置编码，帮助模型理解序列位置信息
- **前馈网络**: 每个编码器层包含前馈神经网络
- **残差连接**: 防止梯度消失，提高训练稳定性
- **层归一化**: 加速训练收敛

### 📊 数据处理
- **文本预处理**: 清理HTML标签、特殊字符，统一大小写
- **词汇表构建**: 自动构建词汇表，支持未知词处理
- **序列填充**: 统一序列长度，支持批处理
- **数据分割**: 自动划分训练集、验证集、测试集

### 🎯 模型特性
- **可配置参数**: 支持调整模型维度、层数、注意力头数等
- **注意力可视化**: 可视化注意力权重，理解模型决策过程
- **训练监控**: 实时显示训练损失、准确率变化
- **模型保存**: 自动保存训练好的模型

## 文件结构

```
nlp_base/
├── transformer_sentiment_classifier.py  # 主程序文件
├── run_transformer_sentiment.py         # 运行脚本
└── README_transformer.md                # 说明文档
```

## 使用方法

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib seaborn scikit-learn
```

### 2. 运行程序

```bash
# 方法1: 直接运行主程序
python transformer_sentiment_classifier.py

# 方法2: 使用运行脚本
python run_transformer_sentiment.py
```

### 3. 程序输出

程序运行后会生成以下文件：
- `transformer_sentiment_model.pth`: 训练好的模型
- `transformer_training_history.png`: 训练历史图表
- `transformer_confusion_matrix.png`: 混淆矩阵
- `transformer_attention_visualization.png`: 注意力权重可视化

## 模型架构详解

### 1. 位置编码 (PositionalEncoding)
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 2. 多头注意力 (MultiHeadAttention)
- 将输入分割为多个头并行计算
- 使用缩放点积注意力机制
- 支持注意力掩码

### 3. 前馈网络 (FeedForward)
- 两层全连接网络
- 使用ReLU激活函数
- 支持dropout正则化

### 4. 编码器层 (TransformerEncoderLayer)
- 自注意力 + 残差连接 + 层归一化
- 前馈网络 + 残差连接 + 层归一化

## 参数配置

```python
# 模型参数
D_MODEL = 128        # 模型维度
N_HEADS = 8          # 注意力头数
D_FF = 512           # 前馈网络维度
N_LAYERS = 4         # 编码器层数
MAX_LEN = 128        # 最大序列长度

# 训练参数
BATCH_SIZE = 32      # 批大小
NUM_EPOCHS = 10      # 训练轮数
LEARNING_RATE = 0.001 # 学习率
```

## 性能优化建议

### 1. 硬件要求
- **CPU**: 支持多核处理器
- **内存**: 建议8GB以上
- **GPU**: 可选，支持CUDA加速

### 2. 参数调优
- **模型维度**: 增加d_model提高表达能力，但会增加计算量
- **层数**: 增加层数提高模型复杂度，但要注意过拟合
- **注意力头数**: 通常设置为8的倍数，与d_model匹配
- **学习率**: 使用学习率调度器自动调整

### 3. 数据增强
- 增加训练数据量
- 使用数据增强技术
- 调整词汇表大小

## 可视化功能

### 1. 训练历史
- 损失函数变化曲线
- 准确率变化曲线
- 学习率调整过程

### 2. 注意力权重
- 可视化模型关注的词汇
- 理解模型决策过程
- 分析注意力模式

### 3. 混淆矩阵
- 分类结果统计
- 错误分析
- 性能评估

## 扩展功能

### 1. 支持真实IMDB数据
```python
# 下载真实IMDB数据集
# 替换模拟数据生成函数
```

### 2. 多分类支持
```python
# 修改num_classes参数
# 支持更多情感类别
```

### 3. 模型集成
```python
# 训练多个模型
# 使用投票或平均方法
```

## 常见问题

### Q: 训练速度慢怎么办？
A: 
- 使用GPU加速
- 减少模型参数
- 使用更小的批大小
- 启用混合精度训练

### Q: 准确率不高怎么办？
A:
- 增加训练数据
- 调整模型架构
- 使用预训练词向量
- 增加训练轮数

### Q: 内存不足怎么办？
A:
- 减少批大小
- 减少序列长度
- 使用梯度累积
- 启用内存优化

## 技术细节

### 注意力机制公式
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

### 位置编码公式
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
```

### 前馈网络
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

## 参考文献

1. Vaswani, A., et al. "Attention is all you need." NIPS 2017.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." ACL 2019.
3. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI 2019.

---

**注意**: 这是一个教学示例，使用了模拟数据。在实际应用中，请使用真实的IMDB数据集进行训练。
