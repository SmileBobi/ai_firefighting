"""
基于PyTorch和HuggingFace Transformers的文本生成程序
使用WikiText-2数据集实现文本生成任务
支持GPT-2模型训练、微调和文本生成
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import os
import time
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入HuggingFace库
try:
    from transformers import (
        GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        get_linear_schedule_with_warmup
    )
    from datasets import load_dataset
    HF_AVAILABLE = True
    print("✅ HuggingFace Transformers库可用")
except ImportError:
    HF_AVAILABLE = False
    print("❌ HuggingFace Transformers库不可用，将使用模拟数据")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class WikiTextDataset(Dataset):
    """
    WikiText-2数据集处理类
    处理文本数据，支持训练和验证
    """
    def __init__(self, texts, tokenizer, max_length=512):
        """
        初始化数据集
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            编码后的文本张量
        """
        text = self.texts[idx]
        
        # 使用分词器编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # 对于语言模型，标签就是输入
        }

class TextGenerator:
    """
    文本生成器类
    基于GPT-2模型实现文本生成功能
    """
    def __init__(self, model_name='gpt2', device=None):
        """
        初始化文本生成器
        Args:
            model_name: 模型名称
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        if HF_AVAILABLE:
            # 加载预训练模型和分词器
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            print(f"✅ 加载模型: {model_name}")
        else:
            # 创建模拟模型
            self.tokenizer = self._create_mock_tokenizer()
            self.model = self._create_mock_model()
            print("⚠️ 使用模拟模型（HuggingFace库不可用）")
    
    def _create_mock_tokenizer(self):
        """创建模拟分词器"""
        class MockTokenizer:
            def __init__(self):
                self.vocab = {
                    '<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3,
                    'the': 4, 'a': 5, 'an': 6, 'and': 7, 'or': 8, 'but': 9,
                    'is': 10, 'are': 11, 'was': 12, 'were': 13,
                    'i': 14, 'you': 15, 'he': 16, 'she': 17, 'it': 18,
                    'we': 19, 'they': 20, 'this': 21, 'that': 22,
                    'in': 23, 'on': 24, 'at': 25, 'to': 26, 'for': 27,
                    'of': 28, 'with': 29, 'by': 30, 'from': 31
                }
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
                self.pad_token = '<pad>'
                self.eos_token = '</s>'
                self.bos_token = '<s>'
            
            def encode(self, text, **kwargs):
                """编码文本"""
                words = text.lower().split()
                ids = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
                return ids
            
            def decode(self, ids, **kwargs):
                """解码文本"""
                words = [self.reverse_vocab.get(id, '<unk>') for id in ids]
                return ' '.join(words)
            
            def __call__(self, text, **kwargs):
                """分词器调用"""
                return {'input_ids': torch.tensor(self.encode(text))}
        
        return MockTokenizer()
    
    def _create_mock_model(self):
        """创建模拟模型"""
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {'vocab_size': 32})()
            
            def generate(self, input_ids, max_length=50, **kwargs):
                """模拟文本生成"""
                batch_size, seq_len = input_ids.shape
                # 简单的随机生成
                generated = input_ids.clone()
                for i in range(seq_len, max_length):
                    next_token = torch.randint(1, 32, (batch_size, 1))
                    generated = torch.cat([generated, next_token], dim=1)
                return generated
            
            def to(self, device):
                return self
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        return MockModel()
    
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_p=0.9, num_return_sequences=1):
        """
        生成文本
        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: 核采样参数
            num_return_sequences: 生成序列数量
        Returns:
            生成的文本列表
        """
        self.model.eval()
        
        # 编码输入提示
        if HF_AVAILABLE:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        else:
            input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        with torch.no_grad():
            if HF_AVAILABLE:
                # 使用HuggingFace模型生成
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            else:
                # 使用模拟模型生成
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length
                )
        
        # 解码生成的文本
        generated_texts = []
        for output in outputs:
            if HF_AVAILABLE:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
            else:
                text = self.tokenizer.decode(output.tolist())
            generated_texts.append(text)
        
        return generated_texts

class WikiTextProcessor:
    """
    WikiText-2数据处理器
    处理原始文本数据，准备训练数据
    """
    def __init__(self):
        """初始化数据处理器"""
        self.texts = []
        self.processed_texts = []
    
    def load_wikitext_data(self, split='train'):
        """
        加载WikiText-2数据
        Args:
            split: 数据集分割（train/validation/test）
        Returns:
            文本列表
        """
        if HF_AVAILABLE:
            try:
                # 使用HuggingFace datasets库加载
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
                texts = [item['text'] for item in dataset if item['text'].strip()]
                print(f"✅ 加载WikiText-2 {split}数据: {len(texts)}条")
                return texts
            except Exception as e:
                print(f"❌ 加载WikiText-2失败: {e}")
                return self._create_mock_data()
        else:
            return self._create_mock_data()
    
    def _create_mock_data(self):
        """创建模拟WikiText数据"""
        print("⚠️ 创建模拟WikiText数据")
        
        mock_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with human language understanding.",
            "Deep learning models can process large amounts of text data.",
            "Transformers have revolutionized the field of NLP.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Language models can generate coherent and fluent text.",
            "Text generation is an important task in natural language processing.",
            "Neural networks can learn complex patterns from data.",
            "Pre-trained models have improved performance on many NLP tasks.",
            "The development of large language models has advanced AI capabilities.",
            "Text understanding and generation are key challenges in AI.",
            "Modern NLP systems use transformer architectures extensively.",
            "Language models can be fine-tuned for specific tasks.",
            "The quality of generated text depends on training data and model size."
        ]
        
        # 扩展数据
        extended_texts = []
        for _ in range(100):  # 生成更多样本
            for text in mock_texts:
                extended_texts.append(text)
        
        return extended_texts
    
    def preprocess_text(self, texts, min_length=10, max_length=512):
        """
        预处理文本数据
        Args:
            texts: 原始文本列表
            min_length: 最小文本长度
            max_length: 最大文本长度
        Returns:
            处理后的文本列表
        """
        processed = []
        
        for text in texts:
            # 清理文本
            text = text.strip()
            
            # 过滤空文本和过短文本
            if len(text) < min_length:
                continue
            
            # 截断过长的文本
            if len(text) > max_length:
                text = text[:max_length]
            
            # 移除特殊字符
            text = re.sub(r'[^\w\s.,!?;:]', '', text)
            
            processed.append(text)
        
        print(f"✅ 预处理完成: {len(processed)}条有效文本")
        return processed

class TextGenerationTrainer:
    """
    文本生成训练器
    负责模型训练和评估
    """
    def __init__(self, model, tokenizer, device):
        """
        初始化训练器
        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def train_model(self, train_dataset, val_dataset, num_epochs=3, batch_size=4, learning_rate=5e-5):
        """
        训练模型
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        print("🚀 开始训练模型...")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 设置优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 设置损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                if HF_AVAILABLE:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                else:
                    # 模拟训练
                    loss = torch.tensor(0.5, requires_grad=True)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if HF_AVAILABLE:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                    else:
                        loss = torch.tensor(0.3)
                    
                    val_loss += loss.item()
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"  训练损失: {avg_train_loss:.4f}")
            print(f"  验证损失: {avg_val_loss:.4f}")
        
        return train_losses, val_losses

class TextGenerationEvaluator:
    """
    文本生成评估器
    计算各种评估指标
    """
    def __init__(self, tokenizer):
        """
        初始化评估器
        Args:
            tokenizer: 分词器
        """
        self.tokenizer = tokenizer
    
    def calculate_perplexity(self, model, test_dataset):
        """
        计算困惑度
        Args:
            model: 语言模型
            test_dataset: 测试数据集
        Returns:
            困惑度值
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for item in test_dataset:
                input_ids = item['input_ids'].unsqueeze(0)
                labels = item['labels'].unsqueeze(0)
                
                if HF_AVAILABLE:
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                else:
                    loss = torch.tensor(0.4)  # 模拟损失
                
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_generation_quality(self, generated_texts, reference_texts):
        """
        评估生成质量
        Args:
            generated_texts: 生成的文本
            reference_texts: 参考文本
        Returns:
            评估指标字典
        """
        # 计算基本统计
        avg_length = np.mean([len(text.split()) for text in generated_texts])
        unique_words = len(set(word for text in generated_texts for word in text.split()))
        
        # 计算重复率
        repetition_rate = self._calculate_repetition_rate(generated_texts)
        
        # 计算流畅度（简单的n-gram重叠）
        fluency_score = self._calculate_fluency(generated_texts)
        
        return {
            'average_length': avg_length,
            'unique_words': unique_words,
            'repetition_rate': repetition_rate,
            'fluency_score': fluency_score
        }
    
    def _calculate_repetition_rate(self, texts):
        """计算重复率"""
        total_repetitions = 0
        total_words = 0
        
        for text in texts:
            words = text.split()
            total_words += len(words)
            
            # 计算重复的n-gram
            for n in [2, 3, 4]:
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                unique_ngrams = set(ngrams)
                repetitions = len(ngrams) - len(unique_ngrams)
                total_repetitions += repetitions
        
        return total_repetitions / max(total_words, 1)
    
    def _calculate_fluency(self, texts):
        """计算流畅度"""
        # 简单的流畅度计算：基于词汇多样性
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        unique_words = set(all_words)
        total_words = len(all_words)
        
        return len(unique_words) / max(total_words, 1)

def visualize_training_history(train_losses, val_losses):
    """
    可视化训练历史
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue', marker='o')
    plt.plot(val_losses, label='验证损失', color='red', marker='s')
    plt.title('训练历史')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 损失对比
    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_losses) + 1)
    x = np.arange(len(epochs))
    width = 0.35
    
    plt.bar(x - width/2, train_losses, width, label='训练损失', alpha=0.8)
    plt.bar(x + width/2, val_losses, width, label='验证损失', alpha=0.8)
    plt.title('损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.xticks(x, epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nlp_base/text_generation_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_generation_results(generated_texts, metrics):
    """
    可视化生成结果
    Args:
        generated_texts: 生成的文本
        metrics: 评估指标
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 文本长度分布
    lengths = [len(text.split()) for text in generated_texts]
    axes[0, 0].hist(lengths, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('生成文本长度分布')
    axes[0, 0].set_xlabel('文本长度（词数）')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 评估指标雷达图
    categories = ['平均长度', '独特词汇', '流畅度', '重复率']
    values = [
        metrics['average_length'] / 50,  # 归一化
        metrics['unique_words'] / 1000,  # 归一化
        metrics['fluency_score'],
        1 - metrics['repetition_rate']  # 反转重复率
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    axes[0, 1].plot(angles, values, 'o-', linewidth=2, color='red')
    axes[0, 1].fill(angles, values, alpha=0.25, color='red')
    axes[0, 1].set_xticks(angles[:-1])
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].set_title('生成质量评估')
    axes[0, 1].grid(True)
    
    # 词汇频率
    all_words = []
    for text in generated_texts:
        all_words.extend(text.split())
    
    word_counts = Counter(all_words)
    top_words = dict(word_counts.most_common(10))
    
    axes[1, 0].bar(range(len(top_words)), list(top_words.values()))
    axes[1, 0].set_title('高频词汇')
    axes[1, 0].set_xlabel('词汇')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].set_xticks(range(len(top_words)))
    axes[1, 0].set_xticklabels(list(top_words.keys()), rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 生成示例
    axes[1, 1].text(0.1, 0.9, '生成文本示例:', fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    sample_text = generated_texts[0][:200] + '...' if len(generated_texts[0]) > 200 else generated_texts[0]
    axes[1, 1].text(0.1, 0.7, sample_text, fontsize=10, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', wrap=True)
    axes[1, 1].set_title('生成示例')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('nlp_base/text_generation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数
    执行完整的文本生成流程
    """
    print("=" * 80)
    print("🤖 基于PyTorch和HuggingFace Transformers的文本生成程序")
    print("📚 使用WikiText-2数据集实现文本生成任务")
    print("=" * 80)
    
    # 设置参数
    BATCH_SIZE = 4
    MAX_LENGTH = 256
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 1. 加载和处理数据
    print("\n📊 1. 加载和处理WikiText-2数据...")
    processor = WikiTextProcessor()
    
    # 加载数据
    train_texts = processor.load_wikitext_data('train')
    val_texts = processor.load_wikitext_data('validation')
    
    # 预处理数据
    train_texts = processor.preprocess_text(train_texts, min_length=20, max_length=MAX_LENGTH)
    val_texts = processor.preprocess_text(val_texts, min_length=20, max_length=MAX_LENGTH)
    
    print(f"✅ 训练数据: {len(train_texts)}条")
    print(f"✅ 验证数据: {len(val_texts)}条")
    
    # 2. 初始化模型和分词器
    print("\n🧠 2. 初始化模型和分词器...")
    generator = TextGenerator(device=device)
    
    # 创建数据集
    train_dataset = WikiTextDataset(train_texts, generator.tokenizer, MAX_LENGTH)
    val_dataset = WikiTextDataset(val_texts, generator.tokenizer, MAX_LENGTH)
    
    print(f"✅ 训练数据集: {len(train_dataset)}个样本")
    print(f"✅ 验证数据集: {len(val_dataset)}个样本")
    
    # 3. 训练模型
    print("\n🚀 3. 开始训练模型...")
    trainer = TextGenerationTrainer(generator.model, generator.tokenizer, device)
    
    start_time = time.time()
    train_losses, val_losses = trainer.train_model(
        train_dataset, val_dataset, 
        num_epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE
    )
    training_time = time.time() - start_time
    
    print(f"✅ 训练完成，用时: {training_time:.2f}秒")
    
    # 4. 评估模型
    print("\n📈 4. 评估模型性能...")
    evaluator = TextGenerationEvaluator(generator.tokenizer)
    
    # 计算困惑度
    perplexity = evaluator.calculate_perplexity(generator.model, val_dataset)
    print(f"📊 困惑度: {perplexity:.2f}")
    
    # 5. 生成文本
    print("\n✨ 5. 生成文本示例...")
    
    # 测试提示
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Natural language processing",
        "Deep learning models",
        "Transformers have revolutionized"
    ]
    
    generated_texts = []
    for prompt in test_prompts:
        print(f"\n🎯 提示: '{prompt}'")
        generated = generator.generate_text(
            prompt, 
            max_length=100, 
            temperature=0.8, 
            num_return_sequences=1
        )
        print(f"📝 生成: '{generated[0]}'")
        generated_texts.extend(generated)
    
    # 6. 评估生成质量
    print("\n📊 6. 评估生成质量...")
    metrics = evaluator.evaluate_generation_quality(generated_texts, val_texts[:len(generated_texts)])
    
    print("📈 生成质量指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 7. 可视化结果
    print("\n📊 7. 生成可视化结果...")
    visualize_training_history(train_losses, val_losses)
    visualize_generation_results(generated_texts, metrics)
    
    # 8. 保存模型和结果
    print("\n💾 8. 保存模型和结果...")
    
    # 保存模型
    if HF_AVAILABLE:
        generator.model.save_pretrained('nlp_base/text_generation_model')
        generator.tokenizer.save_pretrained('nlp_base/text_generation_model')
        print("✅ 模型已保存")
    
    # 保存结果
    results = {
        'training_time': training_time,
        'perplexity': perplexity,
        'metrics': metrics,
        'generated_texts': generated_texts[:5],  # 保存前5个生成样本
        'model_config': {
            'model_name': generator.model_name,
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open('nlp_base/text_generation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ 结果已保存到 text_generation_results.json")
    
    print("\n" + "=" * 80)
    print("🎉 文本生成程序执行完成！")
    print("📁 生成的文件:")
    print("  - text_generation_training_history.png (训练历史)")
    print("  - text_generation_results.png (生成结果)")
    print("  - text_generation_results.json (详细结果)")
    if HF_AVAILABLE:
        print("  - text_generation_model/ (保存的模型)")
    print("=" * 80)

if __name__ == "__main__":
    main()
