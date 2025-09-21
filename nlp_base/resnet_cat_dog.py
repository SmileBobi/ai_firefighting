#!/usr/bin/env python
"""
ResNet猫狗识别
使用预训练模型进行迁移学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image


class ResNetCatDog:
    """ResNet猫狗识别类"""
    
    def __init__(self, device, num_classes=2):
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def create_model(self, use_pretrained=True):
        """创建ResNet模型"""
        if use_pretrained:
            # 使用预训练的ResNet-18
            self.model = torchvision.models.resnet18(pretrained=True)
            # 修改最后一层以适应二分类
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        else:
            # 从头开始训练
            self.model = torchvision.models.resnet18(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        
        self.model = self.model.to(self.device)
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model
    
    def get_data_loaders(self, data_dir='./data/cats_and_dogs', batch_size=32):
        """获取数据加载器"""
        
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            print(f"数据目录 {data_dir} 不存在！")
            print("请先运行 download_cat_dog_data.py 准备数据集")
            return None, None, None
        
        # 训练集
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transform_train
        )
        
        # 分割训练集和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # 验证集使用不同的变换
        val_dataset.dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transform_val
        )
        
        # 测试集
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'test'),
            transform=transform_val
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'训练批次 [{batch_idx}/{len(train_loader)}], '
                      f'损失: {loss.item():.4f}, '
                      f'准确率: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=0.001):
        """训练模型"""
        print(f"开始训练ResNet模型，共{epochs}个epoch...")
        print(f"设备: {self.device}")
        print("-" * 50)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 学习率调度
            scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
            print(f'  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%')
            print(f'  时间: {epoch_time:.2f}秒')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_resnet_cat_dog.pth')
                print(f'  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)')
            
            print("-" * 50)
        
        print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('resnet_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader):
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算准确率
        accuracy = 100. * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
        
        print(f"测试准确率: {accuracy:.2f}%")
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(all_targets, all_preds, target_names=['猫', '狗']))
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['猫', '狗'], yticklabels=['猫', '狗'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.savefig('resnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def visualize_samples(self, data_loader, num_samples=8):
        """可视化样本数据"""
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        class_names = ['猫', '狗']
        
        for i in range(min(num_samples, len(images))):
            # 反归一化图像
            img = images[i].clone()
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f'标签: {class_names[labels[i].item()]}')
            axes[i].axis('off')
        
        plt.suptitle('猫狗数据集样本')
        plt.tight_layout()
        plt.savefig('cat_dog_samples.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("ResNet 猫狗识别")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ResNetCatDog(device)
    model.create_model(use_pretrained=True)
    
    # 获取数据
    print("\n加载猫狗数据集...")
    train_loader, val_loader, test_loader = model.get_data_loaders(batch_size=16)
    
    if train_loader is None:
        print("请先准备猫狗数据集！")
        print("运行: python download_cat_dog_data.py")
        return
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 可视化样本
    print("\n可视化样本数据...")
    model.visualize_samples(train_loader)
    
    # 训练模型
    print("\n开始训练...")
    model.train(train_loader, val_loader, epochs=10)
    
    # 绘制训练历史
    print("\n绘制训练历史...")
    model.plot_training_history()
    
    # 评估模型
    print("\n评估模型...")
    model.evaluate_model(test_loader)
    
    print("\n程序完成！")
    print("生成的文件:")
    print("- best_resnet_cat_dog.pth (最佳模型)")
    print("- resnet_training_history.png (训练历史)")
    print("- resnet_confusion_matrix.png (混淆矩阵)")
    print("- cat_dog_samples.png (样本数据)")


if __name__ == "__main__":
    main()
