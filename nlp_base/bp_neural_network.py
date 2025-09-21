#!/usr/bin/env python
"""
BP神经网络实现MNIST手写数字识别
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time


class BPNetwork:
    """BP神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化神经网络
        
        Args:
            input_size: 输入层神经元数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出层神经元数量
            learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        # 输入层到隐藏层
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        # 隐藏层到输出层
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # 存储训练历史
        self.losses = []
        self.accuracies = []
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        # 防止溢出
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax激活函数"""
        # 防止溢出
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        
        Args:
            X: 输入数据 (batch_size, input_size)
        
        Returns:
            z1: 隐藏层线性组合
            a1: 隐藏层激活值
            z2: 输出层线性组合
            a2: 输出层激活值
        """
        # 输入层到隐藏层
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        
        # 隐藏层到输出层
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        
        return z1, a1, z2, a2
    
    def backward(self, X, y, z1, a1, z2, a2):
        """
        反向传播
        
        Args:
            X: 输入数据
            y: 真实标签 (one-hot编码)
            z1, a1: 隐藏层的线性组合和激活值
            z2, a2: 输出层的线性组合和激活值
        """
        m = X.shape[0]  # 样本数量
        
        # 输出层误差
        dz2 = a2 - y
        
        # 隐藏层误差
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(a1)
        
        # 计算梯度
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """更新权重和偏置"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def compute_loss(self, y_pred, y_true):
        """计算交叉熵损失"""
        # 防止log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # 计算交叉熵损失
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def compute_accuracy(self, y_pred, y_true):
        """计算准确率"""
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        训练神经网络
        
        Args:
            X_train: 训练数据
            y_train: 训练标签 (one-hot编码)
            X_val: 验证数据
            y_val: 验证标签 (one-hot编码)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程
        """
        print(f"开始训练BP神经网络...")
        print(f"网络结构: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        print(f"训练样本数: {X_train.shape[0]}")
        print(f"验证样本数: {X_val.shape[0]}")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        print(f"学习率: {self.learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0
            train_accuracy = 0
            num_batches = 0
            
            # 随机打乱训练数据
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批次训练
            for i in range(0, X_train.shape[0], batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                z1, a1, z2, a2 = self.forward(batch_X)
                
                # 计算损失和准确率
                batch_loss = self.compute_loss(a2, batch_y)
                batch_accuracy = self.compute_accuracy(a2, batch_y)
                
                train_loss += batch_loss
                train_accuracy += batch_accuracy
                num_batches += 1
                
                # 反向传播
                dW1, db1, dW2, db2 = self.backward(batch_X, batch_y, z1, a1, z2, a2)
                
                # 更新权重
                self.update_weights(dW1, db1, dW2, db2)
            
            # 计算平均训练损失和准确率
            avg_train_loss = train_loss / num_batches
            avg_train_accuracy = train_accuracy / num_batches
            
            # 验证阶段
            _, _, _, val_output = self.forward(X_val)
            val_loss = self.compute_loss(val_output, y_val)
            val_accuracy = self.compute_accuracy(val_output, y_val)
            
            # 记录训练历史
            self.losses.append(avg_train_loss)
            self.accuracies.append(val_accuracy)
            
            # 显示训练进度
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成！用时: {training_time:.2f}秒")
        print(f"最终训练准确率: {avg_train_accuracy:.4f}")
        print(f"最终验证准确率: {val_accuracy:.4f}")
    
    def predict(self, X):
        """预测"""
        _, _, _, output = self.forward(X)
        return output
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions = self.predict(X_test)
        test_loss = self.compute_loss(predictions, y_test)
        test_accuracy = self.compute_accuracy(predictions, y_test)
        
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失曲线
        ax1.plot(self.losses, label='Training Loss', color='blue')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(self.accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def load_mnist_data():
    """加载MNIST数据集"""
    print("正在加载MNIST数据集...")
    
    # 使用sklearn加载MNIST数据
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    except TypeError:
        # 兼容旧版本sklearn
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    print(f"数据集大小: {X.shape}")
    print(f"标签范围: {np.min(y)} - {np.max(y)}")
    
    # 数据预处理
    # 归一化到[0,1]
    X = X / 255.0
    
    # 将标签转换为one-hot编码
    y_onehot = np.zeros((len(y), 10))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y, y_onehot


def visualize_samples(X, y, num_samples=10):
    """可视化样本"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Label: {y[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("BP神经网络MNIST手写数字识别")
    print("=" * 60)
    
    # 1. 加载数据
    X, y, y_onehot = load_mnist_data()
    
    # 2. 数据分割
    print("\n正在分割数据集...")
    X_train, X_temp, y_train, y_temp, y_train_onehot, y_temp_onehot = train_test_split(
        X, y, y_onehot, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test, y_val_onehot, y_test_onehot = train_test_split(
        X_temp, y_temp, y_temp_onehot, test_size=0.5, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 3. 可视化样本
    print("\n显示样本图像...")
    visualize_samples(X_train, y_train)
    
    # 4. 创建神经网络
    print("\n创建BP神经网络...")
    # 网络结构: 784 -> 128 -> 10
    network = BPNetwork(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.1
    )
    
    # 5. 训练网络
    print("\n开始训练...")
    network.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=50,
        batch_size=64,
        verbose=True
    )
    
    # 6. 评估模型
    print("\n评估模型...")
    test_loss, test_accuracy = network.evaluate(X_test, y_test_onehot)
    
    # 7. 绘制训练历史
    print("\n绘制训练历史...")
    network.plot_training_history()
    
    # 8. 预测示例
    print("\n预测示例...")
    sample_indices = np.random.choice(X_test.shape[0], 10, replace=False)
    sample_X = X_test[sample_indices]
    sample_y = y_test[sample_indices]
    
    predictions = network.predict(sample_X)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print("预测结果:")
    for i in range(10):
        confidence = np.max(predictions[i])
        print(f"样本 {i+1}: 真实标签={sample_y[i]}, 预测标签={predicted_labels[i]}, 置信度={confidence:.4f}")
    
    # 9. 可视化预测结果
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        axes[i].imshow(sample_X[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'真实: {sample_y[i]}, 预测: {predicted_labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    main()
