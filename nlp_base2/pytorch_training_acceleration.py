"""
PyTorch训练加速示例
使用PyTorch原生优化技术实现训练加速，避免DeepSpeed的复杂依赖
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
import numpy as np
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedDataset(Dataset):
    """优化的数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预编码所有文本以提高效率
        self.encoded_data = []
        self._preprocess_data()
    
    def _preprocess_data(self):
        """预处理数据，预编码文本"""
        logger.info("开始预处理数据...")
        for i, text in enumerate(self.texts):
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.encoded_data.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[i], dtype=torch.long)
            })
        logger.info(f"数据预处理完成，共 {len(self.encoded_data)} 个样本")
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return self.encoded_data[idx]

class TrainingAccelerator:
    """训练加速器"""
    
    def __init__(self, 
                 model_name: str = "bert-base-chinese",
                 output_dir: str = "./nlp_base2/accelerated_output",
                 use_mixed_precision: bool = True,
                 use_data_parallel: bool = True):
        """
        初始化训练加速器
        
        Args:
            model_name: 预训练模型名称
            output_dir: 输出目录
            use_mixed_precision: 是否使用混合精度训练
            use_data_parallel: 是否使用数据并行
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_mixed_precision = use_mixed_precision
        self.use_data_parallel = use_data_parallel
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.device = None
        
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
            
            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # 数据并行
            if self.use_data_parallel and torch.cuda.device_count() > 1:
                self.model = DataParallel(self.model)
                logger.info(f"使用数据并行，GPU数量: {torch.cuda.device_count()}")
            
            # 混合精度训练
            if self.use_mixed_precision and self.device.type == "cuda":
                self.scaler = GradScaler()
                logger.info("启用混合精度训练")
            
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
        positive_words = ["好", "棒", "优秀", "完美", "喜欢", "满意", "推荐", "赞", "棒极了", "太棒了"]
        for i in range(num_samples // 2):
            word = positive_words[i % len(positive_words)]
            text = f"这个产品很{word}，我非常满意，强烈推荐给大家。"
            texts.append(text)
            labels.append(1)
        
        # 负样本（包含消极词汇）
        negative_words = ["差", "坏", "糟糕", "失望", "讨厌", "不推荐", "问题", "垃圾", "太差了", "很烂"]
        for i in range(num_samples // 2):
            word = negative_words[i % len(negative_words)]
            text = f"这个产品很{word}，我很失望，不推荐购买。"
            texts.append(text)
            labels.append(0)
        
        return texts, labels
    
    def train_with_optimizations(self, 
                                texts: List[str], 
                                labels: List[int],
                                num_epochs: int = 3,
                                batch_size: int = 8,
                                learning_rate: float = 2e-5):
        """使用各种优化技术进行训练"""
        
        # 创建数据集
        dataset = OptimizedDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # 多进程数据加载
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        # 创建优化器（使用AdamW优化器）
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 创建学习率调度器
        total_steps = len(dataloader) * num_epochs
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # 损失函数
        loss_fn = nn.CrossEntropyLoss()
        
        # 训练循环
        logger.info("开始优化训练...")
        start_time = time.time()
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # 优化器零梯度
                self.optimizer.zero_grad()
                
                # 前向传播（使用混合精度）
                if self.use_mixed_precision and self.device.type == "cuda":
                    with autocast():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(self.model, 'module'):  # DataParallel
                            logits = self.model.module.classifier(outputs.last_hidden_state[:, 0])
                        else:
                            logits = self.model.classifier(outputs.last_hidden_state[:, 0])
                        loss = loss_fn(logits, labels)
                    
                    # 反向传播（混合精度）
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准训练
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if hasattr(self.model, 'module'):  # DataParallel
                        logits = self.model.module.classifier(outputs.last_hidden_state[:, 0])
                    else:
                        logits = self.model.classifier(outputs.last_hidden_state[:, 0])
                    loss = loss_fn(logits, labels)
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                
                # 更新学习率
                self.lr_scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches
            total_loss += avg_epoch_loss
            logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"优化训练完成!")
        logger.info(f"总训练时间: {training_time:.2f} 秒")
        logger.info(f"平均每轮时间: {training_time/num_epochs:.2f} 秒")
        
        # 保存模型
        self._save_model()
        
        return {
            "training_time": training_time,
            "avg_epoch_time": training_time / num_epochs,
            "total_loss": total_loss / num_epochs
        }
    
    def train_without_optimizations(self, 
                                   texts: List[str], 
                                   labels: List[int],
                                   num_epochs: int = 3,
                                   batch_size: int = 8,
                                   learning_rate: float = 2e-5):
        """不使用优化的标准训练（用于对比）"""
        
        # 创建数据集
        dataset = OptimizedDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(self.model, 'module'):  # DataParallel
                    logits = self.model.module.classifier(outputs.last_hidden_state[:, 0])
                else:
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
    
    def _save_model(self):
        """保存模型"""
        model_path = os.path.join(self.output_dir, "model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict()
        }, model_path)
        logger.info(f"模型已保存到: {model_path}")
    
    def get_system_info(self) -> dict:
        """获取系统信息"""
        info = {
            "CUDA可用": torch.cuda.is_available(),
            "设备": str(self.device),
            "GPU数量": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "混合精度": self.use_mixed_precision,
            "数据并行": self.use_data_parallel
        }
        
        if torch.cuda.is_available():
            info["GPU名称"] = torch.cuda.get_device_name(0)
            info["显存"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return info

def performance_comparison():
    """性能对比测试"""
    print("=== PyTorch训练优化 vs 标准训练性能对比 ===\n")
    
    # 初始化训练器
    accelerator = TrainingAccelerator(
        use_mixed_precision=True,
        use_data_parallel=True
    )
    
    # 设置模型和分词器
    if not accelerator.setup_model_and_tokenizer():
        print("模型设置失败，退出测试")
        return
    
    # 显示系统信息
    print("系统信息:")
    system_info = accelerator.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # 创建示例数据
    print("\n创建示例数据...")
    texts, labels = accelerator.create_sample_data(num_samples=500)
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
    
    standard_results = accelerator.train_without_optimizations(
        texts, labels, num_epochs, batch_size
    )
    
    # 2. 优化训练
    print("\n" + "="*50)
    print("2. 优化训练测试")
    print("="*50)
    
    optimized_results = accelerator.train_with_optimizations(
        texts, labels, num_epochs, batch_size
    )
    
    # 3. 性能对比
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    
    speedup = standard_results["training_time"] / optimized_results["training_time"]
    
    print(f"标准训练时间: {standard_results['training_time']:.2f} 秒")
    print(f"优化训练时间: {optimized_results['training_time']:.2f} 秒")
    print(f"加速比: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"✅ 优化技术加速了 {speedup:.2f} 倍!")
    else:
        print(f"⚠️ 在此配置下优化效果不明显")
    
    print(f"\n标准训练平均损失: {standard_results['total_loss']:.4f}")
    print(f"优化训练平均损失: {optimized_results['total_loss']:.4f}")
    
    # 优化技术说明
    print(f"\n使用的优化技术:")
    print(f"  - 混合精度训练 (FP16): {'是' if accelerator.use_mixed_precision else '否'}")
    print(f"  - 数据并行: {'是' if accelerator.use_data_parallel else '否'}")
    print(f"  - 预编码数据: 是")
    print(f"  - 多进程数据加载: 是")
    print(f"  - 内存固定: {'是' if accelerator.device.type == 'cuda' else '否'}")

def main():
    """主函数"""
    print("🚀 PyTorch训练加速示例")
    print("="*60)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA不可用，将使用CPU训练")
    
    print("\n开始性能对比测试...")
    performance_comparison()

if __name__ == "__main__":
    main()
