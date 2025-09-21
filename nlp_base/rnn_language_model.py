#!/usr/bin/env python
"""
RNN语言建模
基于WikiText数据集的快速实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import random


class SimpleRNNDataset(Dataset):
    """RNN数据集"""
    
    def __init__(self, text_data, vocab, seq_length=20):
        self.text_data = text_data
        self.vocab = vocab
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.text_data) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.text_data[idx:idx + self.seq_length]
        target = self.text_data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SimpleLSTMModel(nn.Module):
    """LSTM语言模型"""
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.3):
        super(SimpleLSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM层
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)
    
    def forward(self, x, hidden):
        """前向传播"""
        embedded = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out.contiguous().view(-1, self.hidden_size))
        return output, hidden


class SimpleLanguageModelTrainer:
    """语言模型训练器"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 初始化隐藏状态
            hidden = self.model.init_hidden(data.size(0), self.device)
            
            self.optimizer.zero_grad()
            output, hidden = self.model(data, hidden)
            
            # 计算损失
            loss = self.criterion(output, target.view(-1))
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'批次 [{batch_idx}/{len(train_loader)}], 损失: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        perplexity = math.exp(avg_loss)
        
        self.train_losses.append(avg_loss)
        self.train_perplexities.append(perplexity)
        
        return avg_loss, perplexity
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                hidden = self.model.init_hidden(data.size(0), self.device)
                output, hidden = self.model(data, hidden)
                loss = self.criterion(output, target.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        perplexity = math.exp(avg_loss)
        
        self.val_losses.append(avg_loss)
        self.val_perplexities.append(perplexity)
        
        return avg_loss, perplexity
    
    def train(self, train_loader, val_loader, epochs=5):
        """训练模型"""
        print(f"开始训练，共{epochs}个epoch...")
        print(f"设备: {self.device}")
        print("-" * 40)
        
        best_val_perplexity = float('inf')
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_perplexity = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_perplexity = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练 - 损失: {train_loss:.4f}, 困惑度: {train_perplexity:.2f}')
            print(f'  验证 - 损失: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}')
            print(f'  时间: {epoch_time:.2f}秒')
            
            # 保存最佳模型
            if val_perplexity < best_val_perplexity:
                best_val_perplexity = val_perplexity
                torch.save(self.model.state_dict(), 'best_simple_rnn_model.pth')
                print(f'  ✓ 保存最佳模型 (困惑度: {val_perplexity:.2f})')
            
            print("-" * 40)
        
        print(f"训练完成！最佳验证困惑度: {best_val_perplexity:.2f}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 困惑度曲线
        ax2.plot(self.train_perplexities, label='训练困惑度', color='blue')
        ax2.plot(self.val_perplexities, label='验证困惑度', color='red')
        ax2.set_title('训练和验证困惑度')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('困惑度')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('simple_rnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class SimpleTextGenerator:
    """文本生成器"""
    
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.idx_to_token = {idx: token for token, idx in vocab.get_stoi().items()}
    
    def generate_text(self, seed_text, max_length=50, temperature=1.0):
        """生成文本"""
        self.model.eval()
        
        # 处理种子文本
        tokens = seed_text.split()
        indices = [self.vocab[token] for token in tokens if token in self.vocab]
        
        if not indices:
            indices = [self.vocab['<unk>']]
        
        generated_indices = indices.copy()
        
        with torch.no_grad():
            hidden = self.model.init_hidden(1, self.device)
            
            # 处理种子文本
            for idx in indices[:-1]:
                input_tensor = torch.tensor([[idx]], device=self.device)
                _, hidden = self.model(input_tensor, hidden)
            
            # 生成新文本
            for _ in range(max_length):
                input_tensor = torch.tensor([[indices[-1]]], device=self.device)
                output, hidden = self.model(input_tensor, hidden)
                
                # 应用温度采样
                logits = output / temperature
                probabilities = F.softmax(logits, dim=-1)
                next_token_idx = torch.multinomial(probabilities, 1).item()
                
                generated_indices.append(next_token_idx)
                indices = [next_token_idx]
        
        # 转换为文本
        generated_text = ' '.join([self.idx_to_token[idx] for idx in generated_indices])
        return generated_text


def load_simple_data(seq_length=20, batch_size=32, vocab_size_limit=10000):
    """加载数据集"""
    print("加载WikiText2数据集...")
    
    # 获取训练数据
    train_iter = torchtext.datasets.WikiText2(split='train')
    val_iter = torchtext.datasets.WikiText2(split='valid')
    
    # 分词器
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    # 构建词汇表
    print("构建词汇表...")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), 
                                     specials=['<unk>', '<pad>'],
                                     max_tokens=vocab_size_limit)
    vocab.set_default_index(vocab['<unk>'])
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 处理数据
    print("处理数据...")
    train_tokens = []
    for text in train_iter:
        tokens = tokenizer(text)
        train_tokens.extend([vocab[token] for token in tokens])
    
    val_tokens = []
    for text in val_iter:
        tokens = tokenizer(text)
        val_tokens.extend([vocab[token] for token in tokens])
    
    # 创建数据集
    train_dataset = SimpleRNNDataset(train_tokens, vocab, seq_length)
    val_dataset = SimpleRNNDataset(val_tokens, vocab, seq_length)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    return train_loader, val_loader, vocab


def main():
    """主函数"""
    print("=" * 50)
    print("RNN语言建模")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    dropout = 0.3
    seq_length = 20
    batch_size = 32
    vocab_size_limit = 10000
    
    # 加载数据
    train_loader, val_loader, vocab = load_simple_data(
        seq_length=seq_length, 
        batch_size=batch_size,
        vocab_size_limit=vocab_size_limit
    )
    
    # 创建模型
    print(f"\n创建LSTM语言模型...")
    model = SimpleLSTMModel(
        vocab_size=len(vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = SimpleLanguageModelTrainer(model, device, learning_rate=0.001)
    
    # 训练模型
    print("\n开始训练...")
    trainer.train(train_loader, val_loader, epochs=5)
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    trainer.plot_training_history()
    
    # 文本生成演示
    print("\n文本生成演示...")
    generator = SimpleTextGenerator(model, vocab, device)
    
    seed_texts = [
        "The history of",
        "In the beginning",
        "The most important",
        "According to"
    ]
    
    for seed in seed_texts:
        print(f"\n种子文本: '{seed}'")
        generated = generator.generate_text(seed, max_length=20, temperature=0.8)
        print(f"生成文本: {generated}")
    
    print("\n程序完成！")
    print("生成的文件:")
    print("- best_simple_rnn_model.pth (最佳模型)")
    print("- simple_rnn_training_history.png (训练历史)")


if __name__ == "__main__":
    main()
