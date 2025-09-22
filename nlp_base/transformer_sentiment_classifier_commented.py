"""
基于Transformer的IMDB情感分类器 - 详细注释版
手搓实现Transformer架构，用于电影评论情感分析
每一行都有详细的中文注释，便于学习和理解
"""

# 导入PyTorch深度学习框架相关模块
import torch  # PyTorch核心库，提供张量操作和自动微分
import torch.nn as nn  # 神经网络模块，包含各种层和激活函数
import torch.nn.functional as F  # 函数式接口，提供各种激活函数和损失函数
import torch.optim as optim  # 优化器模块，包含各种优化算法
from torch.utils.data import DataLoader, TensorDataset  # 数据加载器，用于批处理数据

# 导入数值计算和数据处理库
import numpy as np  # NumPy库，用于数值计算和数组操作
import matplotlib.pyplot as plt  # Matplotlib绘图库，用于可视化
from sklearn.model_selection import train_test_split  # 数据集划分工具
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # 评估指标
import seaborn as sns  # Seaborn统计绘图库，用于美化图表

# 导入文本处理和系统相关模块
import re  # 正则表达式库，用于文本清理
import os  # 操作系统接口，用于文件操作
from collections import Counter  # 计数器，用于统计词频
import pickle  # 序列化库，用于保存和加载对象
import time  # 时间库，用于计时

# 设置随机种子，确保结果可重现
torch.manual_seed(42)  # 设置PyTorch随机种子
np.random.seed(42)  # 设置NumPy随机种子

class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为序列中的每个位置添加位置信息，因为Transformer没有循环结构
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        Args:
            d_model: 模型维度，即词嵌入的维度
            max_len: 最大序列长度，用于预计算位置编码
        """
        super(PositionalEncoding, self).__init__()  # 调用父类初始化方法
        
        # 创建位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # 初始化为零矩阵
        
        # 创建位置索引，从0到max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 形状为(max_len, 1)
        
        # 计算分母项，用于正弦和余弦函数的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))  # 使用对数缩放
        
        # 为偶数位置计算正弦编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 每隔一个位置使用正弦
        # 为奇数位置计算余弦编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 每隔一个位置使用余弦
        
        # 调整维度顺序，添加batch维度
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状变为(1, max_len, d_model)
        
        # 将位置编码注册为缓冲区，不参与梯度计算但会随模型移动
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播，将位置编码添加到输入
        Args:
            x: 输入张量，形状为(seq_len, batch_size, d_model)
        Returns:
            添加位置编码后的张量
        """
        return x + self.pe[:x.size(0), :]  # 将位置编码加到输入上

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    Transformer的核心组件，允许模型同时关注序列中的不同位置
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        初始化多头注意力
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout概率
        """
        super(MultiHeadAttention, self).__init__()  # 调用父类初始化
        assert d_model % n_heads == 0  # 确保模型维度能被头数整除
        
        # 保存参数
        self.d_model = d_model  # 模型维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # Query变换矩阵
        self.W_k = nn.Linear(d_model, d_model)  # Key变换矩阵
        self.W_v = nn.Linear(d_model, d_model)  # Value变换矩阵
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换矩阵
        
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算
        Args:
            Q: Query矩阵
            K: Key矩阵
            V: Value矩阵
            mask: 注意力掩码
        Returns:
            注意力输出和权重
        """
        # 计算注意力分数：Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 如果有掩码，将掩码位置设为负无穷
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        # 应用dropout防止过拟合
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权和：attention_weights * V
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        多头注意力的前向传播
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 注意力掩码
        Returns:
            注意力输出和权重
        """
        batch_size = query.size(0)  # 获取批次大小
        
        # 线性变换并分割为多头
        # 将输入通过线性层，然后重塑为多头格式
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用缩放点积注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头：转置回原始维度并拼接
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """
    前馈网络
    每个Transformer层都包含一个前馈网络
    由两个线性层和ReLU激活函数组成
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(FeedForward, self).__init__()  # 调用父类初始化
        
        # 定义两个线性层
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层：d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层：d_ff -> d_model
        self.dropout = nn.Dropout(dropout)  # dropout层
        
    def forward(self, x):
        """
        前馈网络前向传播
        Args:
            x: 输入张量
        Returns:
            前馈网络输出
        """
        # 第一层线性变换 + ReLU激活 + dropout
        # 第二层线性变换
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含自注意力机制、前馈网络、残差连接和层归一化
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        初始化编码器层
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super(TransformerEncoderLayer, self).__init__()  # 调用父类初始化
        
        # 定义子模块
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)  # 自注意力
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一个层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二个层归一化
        self.dropout = nn.Dropout(dropout)  # dropout层
        
    def forward(self, x, mask=None):
        """
        编码器层前向传播
        Args:
            x: 输入张量
            mask: 注意力掩码
        Returns:
            输出张量和注意力权重
        """
        # 自注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attention(x, x, x, mask)  # 自注意力
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 层归一化
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)  # 前馈网络
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接 + 层归一化
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    由多个编码器层堆叠而成
    """
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        """
        初始化编码器
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: 编码器层数
            dropout: dropout概率
        """
        super(TransformerEncoder, self).__init__()  # 调用父类初始化
        
        # 创建编码器层列表
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)  # 创建n_layers个编码器层
        ])
        
    def forward(self, x, mask=None):
        """
        编码器前向传播
        Args:
            x: 输入张量
            mask: 注意力掩码
        Returns:
            输出张量和所有层的注意力权重
        """
        attention_weights = []  # 存储每层的注意力权重
        for layer in self.layers:  # 遍历每个编码器层
            x, attn_weights = layer(x, mask)  # 通过编码器层
            attention_weights.append(attn_weights)  # 保存注意力权重
        return x, attention_weights

class TransformerSentimentClassifier(nn.Module):
    """
    基于Transformer的情感分类器
    完整的模型架构，包含词嵌入、位置编码、Transformer编码器和分类器
    """
    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512, 
                 n_layers=6, max_len=512, num_classes=2, dropout=0.1):
        """
        初始化分类器
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: 编码器层数
            max_len: 最大序列长度
            num_classes: 分类类别数
            dropout: dropout概率
        """
        super(TransformerSentimentClassifier, self).__init__()  # 调用父类初始化
        
        # 保存模型维度
        self.d_model = d_model
        
        # 词嵌入层：将词汇索引映射为密集向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        # 分类器：将编码器输出映射为类别概率
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        模型前向传播
        Args:
            x: 输入序列，形状为(batch_size, seq_len)
            mask: 注意力掩码
        Returns:
            分类输出和注意力权重
        """
        # 词嵌入：将词汇索引转换为密集向量
        x = self.embedding(x) * np.sqrt(self.d_model)  # 乘以sqrt(d_model)进行缩放
        
        # 位置编码：添加位置信息
        x = self.pos_encoding(x)
        x = self.dropout(x)  # 应用dropout
        
        # Transformer编码器：通过多层编码器
        x, attention_weights = self.transformer_encoder(x, mask)
        
        # 全局平均池化：将序列维度平均为单个向量
        x = x.mean(dim=1)
        
        # 分类：通过线性层得到类别概率
        output = self.classifier(x)
        
        return output, attention_weights

class IMDBDataProcessor:
    """
    IMDB数据处理器
    负责文本清理、词汇表构建、序列转换等数据预处理工作
    """
    def __init__(self, max_vocab_size=10000, max_len=512):
        """
        初始化数据处理器
        Args:
            max_vocab_size: 最大词汇表大小
            max_len: 最大序列长度
        """
        self.max_vocab_size = max_vocab_size  # 最大词汇表大小
        self.max_len = max_len  # 最大序列长度
        self.word_to_idx = {}  # 词汇到索引的映射
        self.idx_to_word = {}  # 索引到词汇的映射
        self.vocab_size = 0  # 词汇表大小
        
    def clean_text(self, text):
        """
        清理文本数据
        Args:
            text: 原始文本
        Returns:
            清理后的文本
        """
        # 转换为小写，统一大小写
        text = text.lower()
        # 移除HTML标签，清理网页格式
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符，只保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 移除多余空格，规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts):
        """
        构建词汇表
        Args:
            texts: 文本列表
        """
        word_counts = Counter()  # 创建词频计数器
        for text in texts:  # 遍历所有文本
            words = text.split()  # 按空格分割单词
            word_counts.update(words)  # 更新词频统计
        
        # 选择最常见的词，限制词汇表大小
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # 添加特殊标记
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}  # 填充和未知词标记
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}  # 反向映射
        
        # 构建词汇表映射
        for i, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = i + 2  # 词汇到索引
            self.idx_to_word[i + 2] = word  # 索引到词汇
        
        self.vocab_size = len(self.word_to_idx)  # 更新词汇表大小
        print(f"词汇表大小: {self.vocab_size}")  # 打印词汇表大小
    
    def text_to_sequence(self, text):
        """
        将文本转换为数字序列
        Args:
            text: 输入文本
        Returns:
            数字序列
        """
        words = text.split()  # 按空格分割单词
        sequence = []  # 初始化序列
        for word in words:  # 遍历每个单词
            if word in self.word_to_idx:  # 如果词汇在词汇表中
                sequence.append(self.word_to_idx[word])  # 添加对应索引
            else:  # 如果词汇不在词汇表中
                sequence.append(self.word_to_idx['<UNK>'])  # 添加未知词标记
        return sequence
    
    def pad_sequence(self, sequence):
        """
        填充序列到固定长度
        Args:
            sequence: 输入序列
        Returns:
            填充后的序列
        """
        if len(sequence) > self.max_len:  # 如果序列超过最大长度
            return sequence[:self.max_len]  # 截断到最大长度
        else:  # 如果序列不足最大长度
            return sequence + [0] * (self.max_len - len(sequence))  # 用0填充
    
    def process_data(self, texts, labels):
        """
        处理数据
        Args:
            texts: 文本列表
            labels: 标签列表
        Returns:
            处理后的序列和标签
        """
        # 清理文本
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # 构建词汇表
        self.build_vocab(cleaned_texts)
        
        # 转换为序列
        sequences = []  # 初始化序列列表
        for text in cleaned_texts:  # 遍历清理后的文本
            seq = self.text_to_sequence(text)  # 转换为序列
            seq = self.pad_sequence(seq)  # 填充序列
            sequences.append(seq)  # 添加到序列列表
        
        return np.array(sequences), np.array(labels)  # 转换为numpy数组

def load_imdb_data():
    """
    加载IMDB数据集（模拟数据，实际使用时需要下载真实数据）
    Returns:
        文本列表和标签列表
    """
    print("正在生成模拟IMDB数据...")  # 打印提示信息
    
    # 生成模拟数据
    np.random.seed(42)  # 设置随机种子
    n_samples = 1000  # 样本数量
    
    # 正面评论模板
    positive_templates = [
        "This movie is absolutely fantastic and amazing!",
        "I loved every minute of this brilliant film.",
        "Outstanding performance and incredible story.",
        "This is one of the best movies I have ever seen.",
        "Excellent direction and wonderful acting.",
        "A masterpiece that everyone should watch.",
        "Incredible cinematography and perfect script.",
        "This film exceeded all my expectations.",
        "Brilliant storytelling and amazing characters.",
        "A must-watch for all movie lovers."
    ]
    
    # 负面评论模板
    negative_templates = [
        "This movie is terrible and boring.",
        "Waste of time and money, completely awful.",
        "Poor acting and terrible storyline.",
        "One of the worst movies I have ever seen.",
        "Disappointing and overrated film.",
        "Bad direction and terrible script.",
        "Awful cinematography and poor editing.",
        "This film was a complete disaster.",
        "Terrible storytelling and bad characters.",
        "Avoid this movie at all costs."
    ]
    
    texts = []  # 初始化文本列表
    labels = []  # 初始化标签列表
    
    for i in range(n_samples):  # 生成指定数量的样本
        if i < n_samples // 2:  # 前半部分为正面样本
            # 生成正面评论
            template = np.random.choice(positive_templates)  # 随机选择正面模板
            # 添加一些变化
            variations = ["really", "very", "so", "extremely", "incredibly"]  # 变化词
            variation = np.random.choice(variations)  # 随机选择变化词
            text = template.replace("absolutely", variation).replace("incredible", variation)  # 替换词汇
            texts.append(text)  # 添加到文本列表
            labels.append(1)  # 添加正面标签
        else:  # 后半部分为负面样本
            # 生成负面评论
            template = np.random.choice(negative_templates)  # 随机选择负面模板
            # 添加一些变化
            variations = ["really", "very", "so", "extremely", "incredibly"]  # 变化词
            variation = np.random.choice(variations)  # 随机选择变化词
            text = template.replace("terrible", variation + " terrible").replace("awful", variation + " awful")  # 替换词汇
            texts.append(text)  # 添加到文本列表
            labels.append(0)  # 添加负面标签
    
    return texts, labels  # 返回文本和标签

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    训练模型
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
    Returns:
        训练历史数据
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.to(device)  # 将模型移动到设备
    
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # 学习率调度器
    
    # 初始化训练历史记录
    train_losses = []  # 训练损失列表
    val_losses = []  # 验证损失列表
    train_accuracies = []  # 训练准确率列表
    val_accuracies = []  # 验证准确率列表
    
    print(f"开始训练，使用设备: {device}")  # 打印设备信息
    
    for epoch in range(num_epochs):  # 遍历每个训练轮
        # 训练阶段
        model.train()  # 设置模型为训练模式
        train_loss = 0  # 初始化训练损失
        train_correct = 0  # 初始化正确预测数
        train_total = 0  # 初始化总样本数
        
        for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练批次
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            
            optimizer.zero_grad()  # 清零梯度
            output, _ = model(data)  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += loss.item()  # 累加损失
            _, predicted = torch.max(output.data, 1)  # 获取预测结果
            train_total += target.size(0)  # 累加样本数
            train_correct += (predicted == target).sum().item()  # 累加正确数
        
        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0  # 初始化验证损失
        val_correct = 0  # 初始化正确预测数
        val_total = 0  # 初始化总样本数
        
        with torch.no_grad():  # 禁用梯度计算
            for data, target in val_loader:  # 遍历验证批次
                data, target = data.to(device), target.to(device)  # 移动数据到设备
                output, _ = model(data)  # 前向传播
                loss = criterion(output, target)  # 计算损失
                
                val_loss += loss.item()  # 累加损失
                _, predicted = torch.max(output.data, 1)  # 获取预测结果
                val_total += target.size(0)  # 累加样本数
                val_correct += (predicted == target).sum().item()  # 累加正确数
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)  # 平均训练损失
        avg_val_loss = val_loss / len(val_loader)  # 平均验证损失
        train_acc = 100. * train_correct / train_total  # 训练准确率
        val_acc = 100. * val_correct / val_total  # 验证准确率
        
        # 记录训练历史
        train_losses.append(avg_train_loss)  # 记录训练损失
        val_losses.append(avg_val_loss)  # 记录验证损失
        train_accuracies.append(train_acc)  # 记录训练准确率
        val_accuracies.append(val_acc)  # 记录验证准确率
        
        scheduler.step(avg_val_loss)  # 更新学习率
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies  # 返回训练历史

def evaluate_model(model, test_loader):
    """
    评估模型
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
    Returns:
        评估结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.eval()  # 设置模型为评估模式
    
    all_predictions = []  # 所有预测结果
    all_targets = []  # 所有真实标签
    
    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:  # 遍历测试批次
            data, target = data.to(device), target.to(device)  # 移动数据到设备
            output, _ = model(data)  # 前向传播
            _, predicted = torch.max(output, 1)  # 获取预测结果
            
            all_predictions.extend(predicted.cpu().numpy())  # 收集预测结果
            all_targets.extend(target.cpu().numpy())  # 收集真实标签
    
    accuracy = accuracy_score(all_targets, all_predictions)  # 计算准确率
    report = classification_report(all_targets, all_predictions, 
                                 target_names=['负面', '正面'])  # 生成分类报告
    
    return accuracy, report, all_predictions, all_targets  # 返回评估结果

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练历史
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # 创建子图
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')  # 绘制训练损失
    ax1.plot(val_losses, label='验证损失', color='red')  # 绘制验证损失
    ax1.set_title('训练和验证损失')  # 设置标题
    ax1.set_xlabel('Epoch')  # 设置x轴标签
    ax1.set_ylabel('损失')  # 设置y轴标签
    ax1.legend()  # 显示图例
    ax1.grid(True)  # 显示网格
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue')  # 绘制训练准确率
    ax2.plot(val_accuracies, label='验证准确率', color='red')  # 绘制验证准确率
    ax2.set_title('训练和验证准确率')  # 设置标题
    ax2.set_xlabel('Epoch')  # 设置x轴标签
    ax2.set_ylabel('准确率 (%)')  # 设置y轴标签
    ax2.legend()  # 显示图例
    ax2.grid(True)  # 显示网格
    
    plt.tight_layout()  # 调整布局
    plt.savefig('nlp_base/transformer_training_history.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片

def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    """
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵
    plt.figure(figsize=(8, 6))  # 创建图形
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'], 
                yticklabels=['负面', '正面'])  # 绘制热力图
    plt.title('混淆矩阵')  # 设置标题
    plt.xlabel('预测标签')  # 设置x轴标签
    plt.ylabel('真实标签')  # 设置y轴标签
    plt.savefig('nlp_base/transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片

def visualize_attention(model, text, processor, device):
    """
    可视化注意力权重
    Args:
        model: 训练好的模型
        text: 输入文本
        processor: 数据处理器
        device: 计算设备
    """
    model.eval()  # 设置模型为评估模式
    
    # 处理文本
    cleaned_text = processor.clean_text(text)  # 清理文本
    sequence = processor.text_to_sequence(cleaned_text)  # 转换为序列
    sequence = processor.pad_sequence(sequence)  # 填充序列
    
    # 转换为张量
    input_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)  # 转换为张量并添加批次维度
    
    with torch.no_grad():  # 禁用梯度计算
        output, attention_weights = model(input_tensor)  # 前向传播
        prediction = torch.softmax(output, dim=1)  # 计算概率分布
    
    # 获取第一个注意力头的权重
    first_layer_attention = attention_weights[0][0, 0, :, :].cpu().numpy()  # 提取注意力权重
    
    # 可视化
    words = cleaned_text.split()[:len(first_layer_attention)]  # 获取词汇列表
    plt.figure(figsize=(12, 8))  # 创建图形
    sns.heatmap(first_layer_attention[:len(words), :len(words)], 
                xticklabels=words, yticklabels=words, 
                cmap='Blues', annot=True, fmt='.2f')  # 绘制注意力热力图
    plt.title(f'注意力权重可视化\n预测: {"正面" if prediction[0, 1] > 0.5 else "负面"} (置信度: {max(prediction[0]):.3f})')  # 设置标题
    plt.xlabel('Key')  # 设置x轴标签
    plt.ylabel('Query')  # 设置y轴标签
    plt.xticks(rotation=45)  # 旋转x轴标签
    plt.yticks(rotation=0)  # 旋转y轴标签
    plt.tight_layout()  # 调整布局
    plt.savefig('nlp_base/transformer_attention_visualization.png', dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()  # 显示图片

def main():
    """
    主函数
    执行完整的训练和评估流程
    """
    print("=" * 60)  # 打印分隔线
    print("基于Transformer的IMDB情感分类器")  # 打印标题
    print("=" * 60)  # 打印分隔线
    
    # 设置参数
    BATCH_SIZE = 32  # 批次大小
    MAX_LEN = 128  # 最大序列长度
    D_MODEL = 128  # 模型维度
    N_HEADS = 8  # 注意力头数
    D_FF = 512  # 前馈网络隐藏层维度
    N_LAYERS = 4  # 编码器层数
    NUM_EPOCHS = 10  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    
    # 加载数据
    print("1. 加载IMDB数据集...")  # 打印步骤信息
    texts, labels = load_imdb_data()  # 加载数据
    print(f"数据集大小: {len(texts)}")  # 打印数据集大小
    print(f"正面样本: {sum(labels)}")  # 打印正面样本数
    print(f"负面样本: {len(labels) - sum(labels)}")  # 打印负面样本数
    
    # 数据预处理
    print("\n2. 数据预处理...")  # 打印步骤信息
    processor = IMDBDataProcessor(max_vocab_size=5000, max_len=MAX_LEN)  # 创建数据处理器
    sequences, labels = processor.process_data(texts, labels)  # 处理数据
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels)  # 第一次划分
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 第二次划分
    
    print(f"训练集大小: {len(X_train)}")  # 打印训练集大小
    print(f"验证集大小: {len(X_val)}")  # 打印验证集大小
    print(f"测试集大小: {len(X_test)}")  # 打印测试集大小
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))  # 训练数据集
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))  # 验证数据集
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))  # 测试数据集
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # 训练数据加载器
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 验证数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 测试数据加载器
    
    # 创建模型
    print("\n3. 创建Transformer模型...")  # 打印步骤信息
    model = TransformerSentimentClassifier(  # 创建模型
        vocab_size=processor.vocab_size,  # 词汇表大小
        d_model=D_MODEL,  # 模型维度
        n_heads=N_HEADS,  # 注意力头数
        d_ff=D_FF,  # 前馈网络隐藏层维度
        n_layers=N_LAYERS,  # 编码器层数
        max_len=MAX_LEN,  # 最大序列长度
        num_classes=2  # 分类类别数
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数数
    print(f"模型参数总数: {total_params:,}")  # 打印总参数数
    print(f"可训练参数: {trainable_params:,}")  # 打印可训练参数数
    
    # 训练模型
    print("\n4. 开始训练...")  # 打印步骤信息
    start_time = time.time()  # 记录开始时间
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)  # 训练模型
    training_time = time.time() - start_time  # 计算训练时间
    print(f"训练完成，用时: {training_time:.2f}秒")  # 打印训练时间
    
    # 评估模型
    print("\n5. 评估模型...")  # 打印步骤信息
    test_accuracy, test_report, predictions, targets = evaluate_model(model, test_loader)  # 评估模型
    print(f"测试准确率: {test_accuracy:.4f}")  # 打印测试准确率
    print("\n分类报告:")  # 打印分类报告标题
    print(test_report)  # 打印分类报告
    
    # 可视化结果
    print("\n6. 生成可视化结果...")  # 打印步骤信息
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)  # 绘制训练历史
    plot_confusion_matrix(targets, predictions)  # 绘制混淆矩阵
    
    # 注意力可视化
    print("\n7. 注意力权重可视化...")  # 打印步骤信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
    model.to(device)  # 移动模型到设备
    
    # 测试一些样本
    test_samples = [  # 测试样本列表
        "This movie is absolutely fantastic and amazing!",
        "This movie is terrible and boring.",
        "I loved every minute of this brilliant film.",
        "Waste of time and money, completely awful."
    ]
    
    for sample in test_samples:  # 遍历测试样本
        print(f"\n测试样本: '{sample}'")  # 打印测试样本
        visualize_attention(model, sample, processor, device)  # 可视化注意力
    
    # 保存模型
    print("\n8. 保存模型...")  # 打印步骤信息
    torch.save({  # 保存模型
        'model_state_dict': model.state_dict(),  # 模型状态字典
        'processor': processor,  # 数据处理器
        'model_config': {  # 模型配置
            'vocab_size': processor.vocab_size,  # 词汇表大小
            'd_model': D_MODEL,  # 模型维度
            'n_heads': N_HEADS,  # 注意力头数
            'd_ff': D_FF,  # 前馈网络隐藏层维度
            'n_layers': N_LAYERS,  # 编码器层数
            'max_len': MAX_LEN,  # 最大序列长度
            'num_classes': 2  # 分类类别数
        }
    }, 'nlp_base/transformer_sentiment_model.pth')  # 保存路径
    
    print("\n" + "=" * 60)  # 打印分隔线
    print("训练完成！模型已保存为 'transformer_sentiment_model.pth'")  # 打印完成信息
    print("可视化结果已保存到当前目录")  # 打印保存信息
    print("=" * 60)  # 打印分隔线

if __name__ == "__main__":  # 如果作为主程序运行
    main()  # 调用主函数
