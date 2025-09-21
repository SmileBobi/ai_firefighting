#!/usr/bin/env python
"""
简化版AlexNet实现MINST手写数字识别
针对MINST数据集优化的轻量级版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


class SimpleAlexNet(nn.Module):
    """
    简化版AlexNet，专门针对MINST数据集优化
    输入: 1x28x28 (灰度图像)
    输出: 10个类别的概率
    """
    
    def __init__(self, num_classes=10):
        super(SimpleAlexNet, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积层: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x7x7
            
            # 第二个卷积层: 32x7x7 -> 64x3x3
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x1x1
            
            # 第三个卷积层: 64x1x1 -> 128x1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 自适应全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # 自适应池化到1x1
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, device, epochs=10):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"开始训练，共{epochs}个epoch...")
    print("-" * 50)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练 - 损失: {avg_train_loss:.4f}, 准确率: {train_acc:.2f}%')
        print(f'  验证 - 损失: {avg_val_loss:.4f}, 准确率: {val_acc:.2f}%')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print("-" * 50)
    
    return train_losses, train_accs, val_losses, val_accs


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"测试准确率: {accuracy:.2f}%")
    return accuracy


def plot_results(train_losses, train_accs, val_losses, val_accs):
    """绘制训练结果"""
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
    ax2.plot(train_accs, label='训练准确率', color='blue')
    ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_alexnet_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, test_loader, device, num_samples=8):
    """可视化预测结果"""
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
        pred_label = predictions[i].item()
        true_label = labels[i].item()
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'预测: {pred_label}, 真实: {true_label}', color=color)
        axes[i].axis('off')
    
    plt.suptitle('AlexNet预测结果')
    plt.tight_layout()
    plt.savefig('alexnet_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("简化版AlexNet MINST手写数字识别")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    print("\n加载MINST数据集...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("\n创建简化版AlexNet模型...")
    model = SimpleAlexNet(num_classes=10).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n开始训练...")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=10
    )
    
    # 绘制结果
    print("\n绘制训练结果...")
    plot_results(train_losses, train_accs, val_losses, val_accs)
    
    # 评估模型
    print("\n评估模型...")
    test_accuracy = evaluate_model(model, test_loader, device)
    
    # 可视化预测
    print("\n可视化预测结果...")
    visualize_predictions(model, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'simple_alexnet_mnist.pth')
    print("\n模型已保存为: simple_alexnet_mnist.pth")
    
    print("\n程序完成！")
    print("生成的文件:")
    print("- simple_alexnet_mnist.pth (训练好的模型)")
    print("- simple_alexnet_results.png (训练结果)")
    print("- alexnet_predictions.png (预测结果)")


if __name__ == "__main__":
    main()
