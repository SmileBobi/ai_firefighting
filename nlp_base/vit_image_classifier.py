"""
基于Vision Transformer (ViT) 的图像分类程序
使用HuggingFace Transformers库实现图像分类任务
支持多种数据集和模型微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import time
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入HuggingFace库
try:
    from transformers import (
        ViTForImageClassification, ViTImageProcessor, ViTConfig,
        TrainingArguments, Trainer, AutoImageProcessor, AutoModelForImageClassification
    )
    from datasets import load_dataset
    import PIL.Image
    HF_AVAILABLE = True
    print("✅ HuggingFace Transformers库可用")
except ImportError:
    HF_AVAILABLE = False
    print("❌ HuggingFace Transformers库不可用，将使用模拟数据")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class ImageDataset(Dataset):
    """
    图像数据集处理类
    处理图像数据，支持训练和验证
    """
    def __init__(self, images, labels, processor, transform=None):
        """
        初始化数据集
        Args:
            images: 图像列表
            labels: 标签列表
            processor: 图像处理器
            transform: 数据增强变换
        """
        self.images = images
        self.labels = labels
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            处理后的图像和标签
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # 如果是PIL图像，直接使用
        if hasattr(image, 'convert'):
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            # 如果是numpy数组，转换为PIL图像
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = PIL.Image.fromarray(image)
            else:
                # 创建模拟图像
                image = PIL.Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        
        # 使用处理器处理图像
        if HF_AVAILABLE:
            inputs = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # 模拟处理
            return {
                'pixel_values': torch.randn(3, 224, 224),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class ViTImageClassifier:
    """
    Vision Transformer图像分类器
    基于ViT模型实现图像分类功能
    """
    def __init__(self, model_name='google/vit-base-patch16-224', num_labels=10, device=None):
        """
        初始化图像分类器
        Args:
            model_name: 模型名称
            num_labels: 分类类别数
            device: 计算设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        
        if HF_AVAILABLE:
            # 加载预训练模型和处理器
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            self.model.to(self.device)
            print(f"✅ 加载ViT模型: {model_name}")
        else:
            # 创建模拟模型
            self.processor = self._create_mock_processor()
            self.model = self._create_mock_model()
            print("⚠️ 使用模拟模型（HuggingFace库不可用）")
    
    def _create_mock_processor(self):
        """创建模拟图像处理器"""
        class MockProcessor:
            def __call__(self, images, return_tensors="pt", **kwargs):
                return {
                    'pixel_values': torch.randn(1, 3, 224, 224)
                }
        
        return MockProcessor()
    
    def _create_mock_model(self):
        """创建模拟ViT模型"""
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {'num_labels': self.num_labels})()
            
            def forward(self, pixel_values, labels=None):
                """模拟前向传播"""
                batch_size = pixel_values.size(0)
                logits = torch.randn(batch_size, self.num_labels)
                
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    return type('Output', (), {'loss': loss, 'logits': logits})()
                else:
                    return type('Output', (), {'logits': logits})()
            
            def to(self, device):
                return self
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        return MockModel()
    
    def predict(self, images):
        """
        预测图像类别
        Args:
            images: 输入图像列表
        Returns:
            预测结果
        """
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for image in images:
                if HF_AVAILABLE:
                    # 处理单个图像
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                else:
                    # 模拟预测
                    logits = torch.randn(1, self.num_labels)
                
                # 计算概率
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                
                predictions.append(pred.cpu().numpy()[0])
                probabilities.append(probs.cpu().numpy()[0])
        
        return predictions, probabilities

class ImageDataProcessor:
    """
    图像数据处理器
    处理图像数据，准备训练数据
    """
    def __init__(self):
        """初始化数据处理器"""
        self.images = []
        self.labels = []
        self.class_names = []
    
    def load_cifar10_data(self):
        """
        加载CIFAR-10数据集
        Returns:
            图像和标签列表
        """
        if HF_AVAILABLE:
            try:
                # 使用HuggingFace datasets库加载CIFAR-10
                dataset = load_dataset('cifar10', split='train')
                images = [PIL.Image.fromarray(np.array(item['img'])) for item in dataset]
                labels = [item['label'] for item in dataset]
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck']
                print(f"✅ 加载CIFAR-10数据: {len(images)}张图像")
                return images, labels, class_names
            except Exception as e:
                print(f"❌ 加载CIFAR-10失败: {e}")
                return self._create_mock_data()
        else:
            return self._create_mock_data()
    
    def _create_mock_data(self):
        """创建模拟图像数据"""
        print("⚠️ 创建模拟图像数据")
        
        # 创建模拟图像
        images = []
        labels = []
        class_names = ['cat', 'dog', 'bird', 'car', 'tree', 'house', 'flower', 'mountain', 'ocean', 'sky']
        
        for i in range(200):  # 创建200张模拟图像
            # 创建随机颜色的图像
            image = PIL.Image.new('RGB', (224, 224), 
                                 color=(np.random.randint(0, 256), 
                                       np.random.randint(0, 256), 
                                       np.random.randint(0, 256)))
            images.append(image)
            labels.append(i % len(class_names))  # 循环分配标签
        
        return images, labels, class_names
    
    def preprocess_images(self, images, labels, test_size=0.2):
        """
        预处理图像数据
        Args:
            images: 图像列表
            labels: 标签列表
            test_size: 测试集比例
        Returns:
            处理后的数据集
        """
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"✅ 训练集: {len(X_train)}张图像")
        print(f"✅ 测试集: {len(X_test)}张图像")
        
        return X_train, X_test, y_train, y_test

class ViTTrainer:
    """
    ViT模型训练器
    负责模型训练和评估
    """
    def __init__(self, model, processor, device):
        """
        初始化训练器
        Args:
            model: ViT模型
            processor: 图像处理器
            device: 计算设备
        """
        self.model = model
        self.processor = processor
        self.device = device
        
    def train_model(self, train_dataset, val_dataset, num_epochs=3, batch_size=8, learning_rate=5e-5):
        """
        训练模型
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
        """
        print("🚀 开始训练ViT模型...")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 设置优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 设置损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 移动数据到设备
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                if HF_AVAILABLE:
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    # 模拟训练
                    loss = torch.tensor(0.5, requires_grad=True)
                    logits = torch.randn(labels.size(0), self.model.config.num_labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch['pixel_values'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    if HF_AVAILABLE:
                        outputs = self.model(pixel_values=pixel_values, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits
                    else:
                        loss = torch.tensor(0.3)
                        logits = torch.randn(labels.size(0), self.model.config.num_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        return train_losses, val_losses, train_accuracies, val_accuracies

class ImageClassificationEvaluator:
    """
    图像分类评估器
    计算各种评估指标
    """
    def __init__(self, class_names):
        """
        初始化评估器
        Args:
            class_names: 类别名称列表
        """
        self.class_names = class_names
    
    def evaluate_model(self, model, test_loader, device):
        """
        评估模型性能
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            device: 计算设备
        Returns:
            评估结果
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                if HF_AVAILABLE:
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                else:
                    logits = torch.randn(labels.size(0), model.config.num_labels)
                
                probabilities = torch.softmax(logits, dim=-1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                    target_names=self.class_names)
        
        return accuracy, report, all_predictions, all_labels, all_probabilities
    
    def calculate_class_metrics(self, y_true, y_pred, y_prob):
        """
        计算各类别指标
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
        Returns:
            类别指标字典
        """
        metrics = {}
        
        for i, class_name in enumerate(self.class_names):
            # 计算该类别的精确率、召回率、F1分数
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(y_true == i)
            }
        
        return metrics

def visualize_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    可视化训练历史
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accuracies: 训练准确率列表
        val_accuracies: 验证准确率列表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', marker='s')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='训练准确率', marker='o')
    ax2.plot(epochs, val_accuracies, 'r-', label='验证准确率', marker='s')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 损失对比
    x = np.arange(len(epochs))
    width = 0.35
    ax3.bar(x - width/2, train_losses, width, label='训练损失', alpha=0.8)
    ax3.bar(x + width/2, val_losses, width, label='验证损失', alpha=0.8)
    ax3.set_title('损失对比')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('损失')
    ax3.set_xticks(x)
    ax3.set_xticklabels(epochs)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 准确率对比
    ax4.bar(x - width/2, train_accuracies, width, label='训练准确率', alpha=0.8)
    ax4.bar(x + width/2, val_accuracies, width, label='验证准确率', alpha=0.8)
    ax4.set_title('准确率对比')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(epochs)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nlp_base/vit_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_classification_results(y_true, y_pred, class_names, y_prob=None):
    """
    可视化分类结果
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        y_prob: 预测概率
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('混淆矩阵')
    ax1.set_xlabel('预测标签')
    ax1.set_ylabel('真实标签')
    
    # 类别分布
    true_counts = [np.sum(y_true == i) for i in range(len(class_names))]
    pred_counts = [np.sum(y_pred == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    ax2.bar(x - width/2, true_counts, width, label='真实分布', alpha=0.8)
    ax2.bar(x + width/2, pred_counts, width, label='预测分布', alpha=0.8)
    ax2.set_title('类别分布对比')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('样本数量')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 准确率分析
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            accuracy = np.sum((y_pred[class_mask] == i)) / np.sum(class_mask)
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0)
    
    ax3.bar(class_names, class_accuracies, alpha=0.7, color='skyblue')
    ax3.set_title('各类别准确率')
    ax3.set_xlabel('类别')
    ax3.set_ylabel('准确率')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 预测置信度分布
    if y_prob is not None:
        max_probs = np.max(y_prob, axis=1)
        ax4.hist(max_probs, bins=20, alpha=0.7, color='lightgreen')
        ax4.set_title('预测置信度分布')
        ax4.set_xlabel('最大预测概率')
        ax4.set_ylabel('频次')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '置信度数据不可用', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('预测置信度分布')
    
    plt.tight_layout()
    plt.savefig('nlp_base/vit_classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sample_predictions(images, predictions, class_names, num_samples=8):
    """
    可视化样本预测结果
    Args:
        images: 图像列表
        predictions: 预测结果
        class_names: 类别名称
        num_samples: 显示样本数量
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # 显示图像
        if hasattr(images[i], 'convert'):
            ax.imshow(images[i])
        else:
            # 如果是numpy数组
            ax.imshow(images[i])
        
        # 设置标题
        pred_class = class_names[predictions[i]]
        ax.set_title(f'预测: {pred_class}', fontsize=12)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('nlp_base/vit_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数
    执行完整的图像分类流程
    """
    print("=" * 80)
    print("🖼️ 基于Vision Transformer (ViT) 的图像分类程序")
    print("📚 使用HuggingFace Transformers库实现图像分类任务")
    print("=" * 80)
    
    # 设置参数
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    NUM_LABELS = 10
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 1. 加载和处理数据
    print("\n📊 1. 加载和处理图像数据...")
    processor = ImageDataProcessor()
    
    # 加载数据
    images, labels, class_names = processor.load_cifar10_data()
    
    # 预处理数据
    X_train, X_test, y_train, y_test = processor.preprocess_images(images, labels)
    
    print(f"✅ 训练数据: {len(X_train)}张图像")
    print(f"✅ 测试数据: {len(X_test)}张图像")
    print(f"✅ 类别: {class_names}")
    
    # 2. 初始化模型和处理器
    print("\n🧠 2. 初始化ViT模型和处理器...")
    classifier = ViTImageClassifier(
        model_name='google/vit-base-patch16-224',
        num_labels=NUM_LABELS,
        device=device
    )
    
    # 创建数据集
    train_dataset = ImageDataset(X_train, y_train, classifier.processor)
    test_dataset = ImageDataset(X_test, y_test, classifier.processor)
    
    print(f"✅ 训练数据集: {len(train_dataset)}个样本")
    print(f"✅ 测试数据集: {len(test_dataset)}个样本")
    
    # 3. 训练模型
    print("\n🚀 3. 开始训练模型...")
    trainer = ViTTrainer(classifier.model, classifier.processor, device)
    
    start_time = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = trainer.train_model(
        train_dataset, test_dataset, 
        num_epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE
    )
    training_time = time.time() - start_time
    
    print(f"✅ 训练完成，用时: {training_time:.2f}秒")
    
    # 4. 评估模型
    print("\n📈 4. 评估模型性能...")
    evaluator = ImageClassificationEvaluator(class_names)
    
    # 创建测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 评估模型
    accuracy, report, predictions, true_labels, probabilities = evaluator.evaluate_model(
        classifier.model, test_loader, device
    )
    
    print(f"📊 测试准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(report)
    
    # 5. 生成预测结果
    print("\n✨ 5. 生成预测结果...")
    
    # 选择一些测试样本进行预测
    sample_images = X_test[:8]
    sample_predictions, sample_probabilities = classifier.predict(sample_images)
    
    print("📝 样本预测结果:")
    for i, (img, pred) in enumerate(zip(sample_images, sample_predictions)):
        pred_class = class_names[pred]
        confidence = np.max(sample_probabilities[i])
        print(f"  样本 {i+1}: {pred_class} (置信度: {confidence:.3f})")
    
    # 6. 计算详细指标
    print("\n📊 6. 计算详细指标...")
    class_metrics = evaluator.calculate_class_metrics(true_labels, predictions, probabilities)
    
    print("📈 各类别指标:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}:")
        print(f"    精确率: {metrics['precision']:.4f}")
        print(f"    召回率: {metrics['recall']:.4f}")
        print(f"    F1分数: {metrics['f1_score']:.4f}")
        print(f"    支持数: {metrics['support']}")
    
    # 7. 可视化结果
    print("\n📊 7. 生成可视化结果...")
    visualize_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    visualize_classification_results(true_labels, predictions, class_names, probabilities)
    visualize_sample_predictions(sample_images, sample_predictions, class_names)
    
    # 8. 保存模型和结果
    print("\n💾 8. 保存模型和结果...")
    
    # 保存模型
    if HF_AVAILABLE:
        classifier.model.save_pretrained('nlp_base/vit_image_classifier_model')
        classifier.processor.save_pretrained('nlp_base/vit_image_classifier_model')
        print("✅ 模型已保存")
    
    # 保存结果
    results = {
        'training_time': training_time,
        'test_accuracy': accuracy,
        'class_metrics': class_metrics,
        'sample_predictions': [
            {
                'image_index': i,
                'predicted_class': class_names[pred],
                'confidence': float(np.max(prob))
            }
            for i, (pred, prob) in enumerate(zip(sample_predictions, sample_probabilities))
        ],
        'model_config': {
            'model_name': classifier.model_name,
            'num_labels': NUM_LABELS,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open('nlp_base/vit_classification_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ 结果已保存到 vit_classification_results.json")
    
    print("\n" + "=" * 80)
    print("🎉 图像分类程序执行完成！")
    print("📁 生成的文件:")
    print("  - vit_training_history.png (训练历史)")
    print("  - vit_classification_results.png (分类结果)")
    print("  - vit_sample_predictions.png (样本预测)")
    print("  - vit_classification_results.json (详细结果)")
    if HF_AVAILABLE:
        print("  - vit_image_classifier_model/ (保存的模型)")
    print("=" * 80)

if __name__ == "__main__":
    main()
