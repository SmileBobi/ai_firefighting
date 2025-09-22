"""
基于Transformer的IMDB情感分类器
手搓实现Transformer架构，用于电影评论情感分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import re
import os
from collections import Counter
import pickle
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分割为多头
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        return x, attention_weights

class TransformerSentimentClassifier(nn.Module):
    """基于Transformer的情感分类器"""
    def __init__(self, vocab_size, d_model=128, n_heads=8, d_ff=512, 
                 n_layers=6, max_len=512, num_classes=2, dropout=0.1):
        super(TransformerSentimentClassifier, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 词嵌入
        x = self.embedding(x) * np.sqrt(self.d_model)
        
        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码器
        x, attention_weights = self.transformer_encoder(x, mask)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output, attention_weights

class IMDBDataProcessor:
    """IMDB数据处理器"""
    def __init__(self, max_vocab_size=10000, max_len=512):
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
    def clean_text(self, text):
        """清理文本"""
        # 转换为小写
        text = text.lower()
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocab(self, texts):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # 添加特殊标记
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"词汇表大小: {self.vocab_size}")
    
    def text_to_sequence(self, text):
        """将文本转换为序列"""
        words = text.split()
        sequence = []
        for word in words:
            if word in self.word_to_idx:
                sequence.append(self.word_to_idx[word])
            else:
                sequence.append(self.word_to_idx['<UNK>'])
        return sequence
    
    def pad_sequence(self, sequence):
        """填充序列到固定长度"""
        if len(sequence) > self.max_len:
            return sequence[:self.max_len]
        else:
            return sequence + [0] * (self.max_len - len(sequence))
    
    def process_data(self, texts, labels):
        """处理数据"""
        # 清理文本
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # 构建词汇表
        self.build_vocab(cleaned_texts)
        
        # 转换为序列
        sequences = []
        for text in cleaned_texts:
            seq = self.text_to_sequence(text)
            seq = self.pad_sequence(seq)
            sequences.append(seq)
        
        return np.array(sequences), np.array(labels)

def load_imdb_data():
    """加载IMDB数据集（模拟数据，实际使用时需要下载真实数据）"""
    print("正在生成模拟IMDB数据...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    
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
    
    texts = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples // 2:
            # 生成正面评论
            template = np.random.choice(positive_templates)
            # 添加一些变化
            variations = ["really", "very", "so", "extremely", "incredibly"]
            variation = np.random.choice(variations)
            text = template.replace("absolutely", variation).replace("incredible", variation)
            texts.append(text)
            labels.append(1)  # 正面
        else:
            # 生成负面评论
            template = np.random.choice(negative_templates)
            # 添加一些变化
            variations = ["really", "very", "so", "extremely", "incredibly"]
            variation = np.random.choice(variations)
            text = template.replace("terrible", variation + " terrible").replace("awful", variation + " awful")
            texts.append(text)
            labels.append(0)  # 负面
    
    return texts, labels

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"开始训练，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, 
                                 target_names=['负面', '正面'])
    
    return accuracy, report, all_predictions, all_targets

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率', color='blue')
    ax2.plot(val_accuracies, label='验证准确率', color='red')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('nlp_base/transformer_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'], 
                yticklabels=['负面', '正面'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('nlp_base/transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_attention(model, text, processor, device):
    """可视化注意力权重"""
    model.eval()
    
    # 处理文本
    cleaned_text = processor.clean_text(text)
    sequence = processor.text_to_sequence(cleaned_text)
    sequence = processor.pad_sequence(sequence)
    
    # 转换为张量
    input_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output, attention_weights = model(input_tensor)
        prediction = torch.softmax(output, dim=1)
    
    # 获取第一个注意力头的权重
    first_layer_attention = attention_weights[0][0, 0, :, :].cpu().numpy()
    
    # 可视化
    words = cleaned_text.split()[:len(first_layer_attention)]
    plt.figure(figsize=(12, 8))
    sns.heatmap(first_layer_attention[:len(words), :len(words)], 
                xticklabels=words, yticklabels=words, 
                cmap='Blues', annot=True, fmt='.2f')
    plt.title(f'注意力权重可视化\n预测: {"正面" if prediction[0, 1] > 0.5 else "负面"} (置信度: {max(prediction[0]):.3f})')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('nlp_base/transformer_attention_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("基于Transformer的IMDB情感分类器")
    print("=" * 60)
    
    # 设置参数
    BATCH_SIZE = 32
    MAX_LEN = 128
    D_MODEL = 128
    N_HEADS = 8
    D_FF = 512
    N_LAYERS = 4
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # 加载数据
    print("1. 加载IMDB数据集...")
    texts, labels = load_imdb_data()
    print(f"数据集大小: {len(texts)}")
    print(f"正面样本: {sum(labels)}")
    print(f"负面样本: {len(labels) - sum(labels)}")
    
    # 数据预处理
    print("\n2. 数据预处理...")
    processor = IMDBDataProcessor(max_vocab_size=5000, max_len=MAX_LEN)
    sequences, labels = processor.process_data(texts, labels)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print("\n3. 创建Transformer模型...")
    model = TransformerSentimentClassifier(
        vocab_size=processor.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        num_classes=2
    )
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 训练模型
    print("\n4. 开始训练...")
    start_time = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)
    training_time = time.time() - start_time
    print(f"训练完成，用时: {training_time:.2f}秒")
    
    # 评估模型
    print("\n5. 评估模型...")
    test_accuracy, test_report, predictions, targets = evaluate_model(model, test_loader)
    print(f"测试准确率: {test_accuracy:.4f}")
    print("\n分类报告:")
    print(test_report)
    
    # 可视化结果
    print("\n6. 生成可视化结果...")
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(targets, predictions)
    
    # 注意力可视化
    print("\n7. 注意力权重可视化...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 测试一些样本
    test_samples = [
        "This movie is absolutely fantastic and amazing!",
        "This movie is terrible and boring.",
        "I loved every minute of this brilliant film.",
        "Waste of time and money, completely awful."
    ]
    
    for sample in test_samples:
        print(f"\n测试样本: '{sample}'")
        visualize_attention(model, sample, processor, device)
    
    # 保存模型
    print("\n8. 保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'model_config': {
            'vocab_size': processor.vocab_size,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'd_ff': D_FF,
            'n_layers': N_LAYERS,
            'max_len': MAX_LEN,
            'num_classes': 2
        }
    }, 'nlp_base/transformer_sentiment_model.pth')
    
    print("\n" + "=" * 60)
    print("训练完成！模型已保存为 'transformer_sentiment_model.pth'")
    print("可视化结果已保存到当前目录")
    print("=" * 60)

if __name__ == "__main__":
    main()
