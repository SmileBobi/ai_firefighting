"""
DeepSpeed训练加速示例
展示如何使用DeepSpeed进行模型训练加速
"""

import os
import time
import json
import torch

# 设置环境变量消除 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import deepspeed
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup
)
import numpy as np
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataset(Dataset):
    """简单的文本分类数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DeepSpeedTrainer:
    """DeepSpeed训练器"""
    
    def __init__(self, 
                 model_name: str = "bert-base-chinese",
                 deepspeed_config: str = "./nlp_base2/deepspeed_config.json",
                 output_dir: str = "./nlp_base2/deepspeed_output"):
        """
        初始化DeepSpeed训练器
        
        Args:
            model_name: 预训练模型名称
            deepspeed_config: DeepSpeed配置文件路径
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.deepspeed_config = deepspeed_config
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        
    def create_deepspeed_config(self):
        """创建DeepSpeed配置文件"""
        config = {
            "train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5,
                    "warmup_num_steps": 100
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            },
            "wall_clock_breakdown": False
        }
        
        with open(self.deepspeed_config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DeepSpeed配置文件已创建: {self.deepspeed_config}")
        return config
    
    def setup_model_and_tokenizer(self):
        """设置模型和分词器"""
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"分词器加载成功: {self.model_name}")
            
            # 加载模型
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # 添加分类头
            num_labels = 2  # 二分类
            self.model.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
            
            logger.info(f"模型加载成功: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"模型和分词器设置失败: {e}")
            return False
    
    def create_sample_data(self, num_samples: int = 1000) -> tuple:
        """创建示例数据"""
        # 生成示例文本和标签
        texts = []
        labels = []
        
        # 正样本（包含积极词汇）
        positive_words = ["好", "棒", "优秀", "完美", "喜欢", "满意", "推荐"]
        for i in range(num_samples // 2):
            text = f"这个产品很{positive_words[i % len(positive_words)]}，我非常满意。"
            texts.append(text)
            labels.append(1)
        
        # 负样本（包含消极词汇）
        negative_words = ["差", "坏", "糟糕", "失望", "讨厌", "不推荐", "问题"]
        for i in range(num_samples // 2):
            text = f"这个产品很{negative_words[i % len(negative_words)]}，我很失望。"
            texts.append(text)
            labels.append(0)
        
        return texts, labels
    
    def train_with_deepspeed(self, 
                           texts: List[str], 
                           labels: List[int],
                           num_epochs: int = 3,
                           batch_size: int = 8,
                           learning_rate: float = 2e-5):
        """使用DeepSpeed进行训练"""
        
        # 创建数据集
        dataset = SimpleDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 创建优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 创建学习率调度器
        total_steps = len(dataloader) * num_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # 初始化DeepSpeed
        model_engine, optimizer, returned_dataloader, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self.deepspeed_config
        )
        
        # 更新优化器和学习率调度器引用
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # 关键修复：如果返回的dataloader为None，使用原始的dataloader
        if returned_dataloader is None:
            logger.info("DeepSpeed返回的dataloader为None，使用原始dataloader")
            working_dataloader = dataloader
        else:
            working_dataloader = returned_dataloader
        
        # 训练循环
        logger.info("开始DeepSpeed训练...")
        start_time = time.time()
        
        model_engine.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(working_dataloader):
                # 需要手动将数据移动到正确的设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
                logits = model_engine.classifier(outputs.last_hidden_state[:, 0])
                
                # 计算损失
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                
                # 反向传播
                model_engine.backward(loss)
                model_engine.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            total_loss += avg_epoch_loss
            logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"DeepSpeed训练完成!")
        logger.info(f"总训练时间: {training_time:.2f} 秒")
        logger.info(f"平均每轮时间: {training_time/num_epochs:.2f} 秒")
        
        # 保存模型
        model_engine.save_checkpoint(self.output_dir)
        logger.info(f"模型已保存到: {self.output_dir}")
        
        return {
            "training_time": training_time,
            "avg_epoch_time": training_time / num_epochs,
            "total_loss": total_loss / num_epochs
        }
    
    def train_without_deepspeed(self, 
                               texts: List[str], 
                               labels: List[int],
                               num_epochs: int = 3,
                               batch_size: int = 8,
                               learning_rate: float = 2e-5):
        """不使用DeepSpeed的标准训练（用于对比）"""
        
        # 创建数据集
        dataset = SimpleDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # 创建优化器和损失函数
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        # 训练循环
        logger.info("开始标准训练...")
        start_time = time.time()
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = self.model.classifier(outputs.last_hidden_state[:, 0])
                
                # 计算损失
                loss = loss_fn(logits, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            total_loss += avg_epoch_loss
            logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"标准训练完成!")
        logger.info(f"总训练时间: {training_time:.2f} 秒")
        logger.info(f"平均每轮时间: {training_time/num_epochs:.2f} 秒")
        
        return {
            "training_time": training_time,
            "avg_epoch_time": training_time / num_epochs,
            "total_loss": total_loss / num_epochs
        }

def performance_comparison():
    """性能对比测试"""
    print("=== DeepSpeed vs 标准训练性能对比 ===\n")
    
    # 初始化训练器
    trainer = DeepSpeedTrainer()
    
    # 创建DeepSpeed配置
    trainer.create_deepspeed_config()
    
    # 设置模型和分词器
    if not trainer.setup_model_and_tokenizer():
        print("模型设置失败，退出测试")
        return
    
    # 创建示例数据
    print("创建示例数据...")
    texts, labels = trainer.create_sample_data(num_samples=500)
    print(f"数据量: {len(texts)} 个样本")
    
    # 测试参数
    num_epochs = 2
    batch_size = 4
    
    print(f"\n测试参数:")
    print(f"  训练轮数: {num_epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  样本数量: {len(texts)}")
    
    # 1. 标准训练
    print("\n" + "="*50)
    print("1. 标准训练测试")
    print("="*50)
    
    standard_results = trainer.train_without_deepspeed(
        texts, labels, num_epochs, batch_size
    )
    
    # 2. DeepSpeed训练
    print("\n" + "="*50)
    print("2. DeepSpeed训练测试")
    print("="*50)
    
    deepspeed_results = trainer.train_with_deepspeed(
        texts, labels, num_epochs, batch_size
    )
    
    # 3. 性能对比
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    
    speedup = standard_results["training_time"] / deepspeed_results["training_time"]
    
    print(f"标准训练时间: {standard_results['training_time']:.2f} 秒")
    print(f"DeepSpeed训练时间: {deepspeed_results['training_time']:.2f} 秒")
    print(f"加速比: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"✅ DeepSpeed加速了 {speedup:.2f} 倍!")
    else:
        print(f"⚠️ DeepSpeed在此配置下未显示加速效果")
    
    print(f"\n标准训练平均损失: {standard_results['total_loss']:.4f}")
    print(f"DeepSpeed训练平均损失: {deepspeed_results['total_loss']:.4f}")

def main():
    """主函数"""
    print("🚀 DeepSpeed训练加速示例")
    print("="*60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练（DeepSpeed效果有限）")
    
    print("\n开始性能对比测试...")
    performance_comparison()

if __name__ == "__main__":
    main()
