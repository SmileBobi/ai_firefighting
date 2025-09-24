# 基于Vision Transformer (ViT) 的图像分类程序

这是一个完整的图像分类程序，使用PyTorch和HuggingFace Transformers库，基于Vision Transformer (ViT) 模型实现图像分类任务。

## 🚀 功能特点

### 🧠 核心功能
- **ViT模型**: 使用预训练的Vision Transformer模型进行图像分类
- **CIFAR-10数据集**: 自动加载和处理CIFAR-10数据集
- **模型微调**: 支持在自定义数据上微调模型
- **高精度分类**: 多种分类策略和参数控制
- **质量评估**: 全面的分类质量评估指标

### 📊 数据处理
- **自动数据加载**: 从HuggingFace Hub加载CIFAR-10数据集
- **图像预处理**: 标准化和增强图像数据
- **数据集划分**: 自动划分训练、验证和测试集
- **批处理**: 高效的数据批处理机制

### 🎯 模型训练
- **预训练模型**: 基于ViT的预训练模型
- **微调支持**: 在特定数据上进行模型微调
- **训练监控**: 实时监控训练损失和验证损失
- **学习率调度**: 自动学习率调整

### 📈 评估指标
- **准确率**: 计算模型分类准确率
- **混淆矩阵**: 分析分类错误模式
- **各类别指标**: 精确率、召回率、F1分数
- **置信度分析**: 预测置信度分布

## 📁 文件结构

```
nlp_base/
├── vit_image_classifier.py              # 主程序文件
├── run_vit_classifier.py               # 运行脚本
├── demo_vit_classifier.py              # 演示脚本
├── README_vit_classifier.md            # 说明文档
└── results/                             # 输出结果目录
    ├── vit_training_history.png
    ├── vit_classification_results.png
    ├── vit_sample_predictions.png
    ├── vit_classification_results.json
    └── vit_image_classifier_model/       # 保存的模型
```

## 🛠️ 安装依赖

### 必需依赖
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install matplotlib seaborn scikit-learn
pip install numpy pandas pillow
```

### 可选依赖
```bash
pip install accelerate  # 加速训练
pip install wandb       # 实验跟踪
pip install tensorboard  # 可视化
pip install timm         # 更多视觉模型
```

## 🚀 使用方法

### 1. 基本运行
```bash
# 直接运行主程序
python vit_image_classifier.py

# 或使用运行脚本
python run_vit_classifier.py
```

### 2. 自定义参数
```python
# 在代码中修改参数
BATCH_SIZE = 16         # 批次大小
NUM_EPOCHS = 5          # 训练轮数
LEARNING_RATE = 3e-5    # 学习率
NUM_LABELS = 10         # 分类类别数
```

### 3. 自定义模型
```python
# 使用不同的ViT模型
classifier = ViTImageClassifier(
    model_name='google/vit-large-patch16-224',
    num_labels=10,
    device=device
)
```

## 📊 程序输出

### 1. 训练过程
```
🚀 开始训练ViT模型...
📚 Epoch 1/3
  Batch 0/25, Loss: 2.3456
  Batch 10/25, Loss: 2.1234
  ...
  训练损失: 2.1234, 训练准确率: 45.67%
  验证损失: 1.9876, 验证准确率: 52.34%
```

### 2. 分类结果
```
📊 测试准确率: 0.8567
📝 样本预测结果:
  样本 1: cat (置信度: 0.923)
  样本 2: dog (置信度: 0.876)
  样本 3: bird (置信度: 0.945)
```

### 3. 评估指标
```
📈 各类别指标:
  cat:
    精确率: 0.8567
    召回率: 0.8234
    F1分数: 0.8398
    支持数: 100
```

## 🎨 可视化功能

### 1. 训练历史
- 训练和验证损失曲线
- 训练和验证准确率曲线
- 损失和准确率对比柱状图

### 2. 分类结果
- 混淆矩阵热力图
- 类别分布对比
- 各类别准确率分析
- 预测置信度分布

### 3. 样本预测
- 样本图像展示
- 预测结果标注
- 置信度显示

## ⚙️ 配置选项

### 模型配置
```python
# 模型参数
MODEL_NAME = 'google/vit-base-patch16-224'  # 模型名称
NUM_LABELS = 10                             # 分类类别数
BATCH_SIZE = 8                              # 批次大小
```

### 训练配置
```python
# 训练参数
NUM_EPOCHS = 3                              # 训练轮数
LEARNING_RATE = 5e-5                        # 学习率
WARMUP_STEPS = 100                          # 预热步数
```

### 数据配置
```python
# 数据参数
TEST_SIZE = 0.2                            # 测试集比例
IMAGE_SIZE = 224                           # 图像大小
```

## 📈 性能优化

### 1. 硬件优化
- **GPU加速**: 自动检测并使用CUDA
- **混合精度**: 支持FP16训练
- **梯度累积**: 处理大批次数据

### 2. 内存优化
- **梯度检查点**: 减少内存使用
- **动态填充**: 按批次动态调整图像大小
- **数据并行**: 支持多GPU训练

### 3. 训练优化
- **学习率调度**: 自动调整学习率
- **早停机制**: 防止过拟合
- **模型检查点**: 定期保存模型

## 🔧 高级功能

### 1. 自定义数据集
```python
# 加载自定义数据集
def load_custom_dataset(image_dir, label_file):
    images = []
    labels = []
    # 自定义数据加载逻辑
    return images, labels
```

### 2. 数据增强
```python
# 应用数据增强
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 3. 模型集成
```python
# 训练多个模型
models = [
    ViTImageClassifier('google/vit-base-patch16-224'),
    ViTImageClassifier('google/vit-large-patch16-224'),
    ViTImageClassifier('microsoft/beit-base-patch16-224')
]
```

## 📊 评估指标详解

### 1. 准确率 (Accuracy)
- **定义**: 正确预测的样本数 / 总样本数
- **计算**: (TP + TN) / (TP + TN + FP + FN)
- **意义**: 整体分类性能，越高越好

### 2. 精确率 (Precision)
- **定义**: 正确预测为正类的样本数 / 预测为正类的样本数
- **计算**: TP / (TP + FP)
- **意义**: 预测准确性，越高越好

### 3. 召回率 (Recall)
- **定义**: 正确预测为正类的样本数 / 实际为正类的样本数
- **计算**: TP / (TP + FN)
- **意义**: 覆盖完整性，越高越好

### 4. F1分数 (F1-Score)
- **定义**: 精确率和召回率的调和平均
- **计算**: 2 * (Precision * Recall) / (Precision + Recall)
- **意义**: 综合性能指标，越高越好

## 🤖 支持的ViT模型

### 1. 基础模型
- `google/vit-base-patch16-224`
- `google/vit-base-patch32-224`
- `google/vit-base-patch16-384`

### 2. 大型模型
- `google/vit-large-patch16-224`
- `google/vit-large-patch32-224`
- `google/vit-large-patch16-384`

### 3. 其他变体
- `microsoft/beit-base-patch16-224`
- `microsoft/swin-base-patch4-window7-224`
- `facebook/deit-base-patch16-224`

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 
- 减少批次大小
- 使用更小的模型
- 启用混合精度训练
- 使用梯度累积

### Q: 训练速度慢怎么办？
A:
- 使用GPU加速
- 启用混合精度
- 使用预训练模型
- 减少图像分辨率

### Q: 分类准确率低怎么办？
A:
- 增加训练数据
- 调整学习率
- 使用数据增强
- 尝试更大的模型

### Q: 模型过拟合怎么办？
A:
- 增加正则化
- 使用早停
- 减少模型复杂度
- 增加训练数据

## 📚 学习资源

### 1. 相关论文
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021)
- "Training data-efficient image transformers & distillation through attention" (Touvron et al., 2021)
- "BEiT: BERT Pre-training of Image Transformers" (Bao et al., 2022)

### 2. 官方文档
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)

### 3. 教程资源
- [HuggingFace Course](https://huggingface.co/course/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Computer Vision with Transformers](https://huggingface.co/course/chapter1/7)

## 🤝 贡献指南

### 1. 报告问题
- 使用GitHub Issues报告bug
- 提供详细的错误信息
- 包含复现步骤

### 2. 功能请求
- 描述新功能需求
- 说明使用场景
- 提供实现建议

### 3. 代码贡献
- Fork项目仓库
- 创建功能分支
- 提交Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- HuggingFace团队提供的优秀库
- Google的ViT模型
- PyTorch团队
- 所有开源贡献者

---

**注意**: 这是一个教学示例程序，展示了如何使用PyTorch和HuggingFace Transformers进行图像分类。在实际应用中，请根据具体需求调整参数和配置。
