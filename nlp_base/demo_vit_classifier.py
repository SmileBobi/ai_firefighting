"""
ViT图像分类程序演示脚本
展示程序功能和使用方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_program_info():
    """显示程序信息"""
    print("=" * 80)
    print("🖼️ 基于Vision Transformer (ViT) 的图像分类程序")
    print("📚 使用HuggingFace Transformers库实现图像分类任务")
    print("=" * 80)
    print()
    
    print("🔧 程序特点:")
    print("✅ 基于ViT模型的图像分类")
    print("✅ 自动加载CIFAR-10数据集")
    print("✅ 支持模型微调和训练")
    print("✅ 多种图像预处理方法")
    print("✅ 全面的分类评估指标")
    print("✅ 丰富的可视化功能")
    print()
    
    print("📊 核心功能:")
    print("1. 数据加载和预处理")
    print("   - 自动从HuggingFace Hub加载CIFAR-10数据集")
    print("   - 图像预处理和标准化")
    print("   - 数据集划分和批处理")
    print()
    
    print("2. 模型配置和训练")
    print("   - ViT预训练模型")
    print("   - 支持模型微调")
    print("   - 训练过程监控")
    print("   - 学习率自动调整")
    print()
    
    print("3. 图像分类")
    print("   - 高精度图像分类")
    print("   - 多类别分类支持")
    print("   - 置信度分析")
    print()
    
    print("4. 质量评估")
    print("   - 准确率计算")
    print("   - 混淆矩阵分析")
    print("   - 各类别性能指标")
    print("   - 预测置信度分析")
    print()
    
    print("5. 可视化展示")
    print("   - 训练历史图表")
    print("   - 分类结果分析")
    print("   - 样本预测展示")
    print("   - 性能指标统计")
    print()

def show_architecture():
    """显示程序架构"""
    print("🏗️ 程序架构:")
    print()
    print("📁 文件结构:")
    print("nlp_base/")
    print("├── vit_image_classifier.py              # 主程序文件")
    print("├── run_vit_classifier.py               # 运行脚本")
    print("├── demo_vit_classifier.py               # 演示脚本")
    print("├── README_vit_classifier.md             # 详细说明文档")
    print("└── results/                              # 输出结果目录")
    print("    ├── vit_training_history.png")
    print("    ├── vit_classification_results.png")
    print("    ├── vit_sample_predictions.png")
    print("    ├── vit_classification_results.json")
    print("    └── vit_image_classifier_model/       # 保存的模型")
    print()
    
    print("🧠 核心类:")
    print("• ImageDataset: 图像数据集处理类")
    print("• ViTImageClassifier: ViT图像分类器类")
    print("• ImageDataProcessor: 图像数据处理器类")
    print("• ViTTrainer: ViT训练器类")
    print("• ImageClassificationEvaluator: 分类评估器类")
    print()

def show_usage_examples():
    """显示使用示例"""
    print("🚀 使用示例:")
    print()
    
    print("1. 基本运行:")
    print("   python vit_image_classifier.py")
    print()
    
    print("2. 自定义参数:")
    print("   # 修改批次大小")
    print("   BATCH_SIZE = 16")
    print("   # 修改训练轮数")
    print("   NUM_EPOCHS = 5")
    print("   # 修改学习率")
    print("   LEARNING_RATE = 3e-5")
    print()
    
    print("3. 自定义模型:")
    print("   classifier = ViTImageClassifier(")
    print("       model_name='google/vit-large-patch16-224',")
    print("       num_labels=10,")
    print("       device=device")
    print("   )")
    print()
    
    print("4. 预测图像:")
    print("   predictions, probabilities = classifier.predict(images)")
    print("   for i, (pred, prob) in enumerate(zip(predictions, probabilities)):")
    print("       print(f'图像 {i}: {class_names[pred]} (置信度: {max(prob):.3f})')")
    print()

def show_evaluation_metrics():
    """显示评估指标"""
    print("📊 评估指标:")
    print()
    
    print("1. 准确率 (Accuracy):")
    print("   • 定义: 正确预测的样本数 / 总样本数")
    print("   • 计算: (TP + TN) / (TP + TN + FP + FN)")
    print("   • 意义: 整体分类性能，越高越好")
    print()
    
    print("2. 精确率 (Precision):")
    print("   • 定义: 正确预测为正类的样本数 / 预测为正类的样本数")
    print("   • 计算: TP / (TP + FP)")
    print("   • 意义: 预测准确性，越高越好")
    print()
    
    print("3. 召回率 (Recall):")
    print("   • 定义: 正确预测为正类的样本数 / 实际为正类的样本数")
    print("   • 计算: TP / (TP + FN)")
    print("   • 意义: 覆盖完整性，越高越好")
    print()
    
    print("4. F1分数 (F1-Score):")
    print("   • 定义: 精确率和召回率的调和平均")
    print("   • 计算: 2 * (Precision * Recall) / (Precision + Recall)")
    print("   • 意义: 综合性能指标，越高越好")
    print()
    
    print("5. 混淆矩阵 (Confusion Matrix):")
    print("   • 定义: 展示各类别之间的预测关系")
    print("   • 意义: 分析分类错误模式")
    print()

def show_visualization_features():
    """显示可视化功能"""
    print("📈 可视化功能:")
    print()
    
    print("1. 训练历史:")
    print("   • 训练和验证损失曲线")
    print("   • 训练和验证准确率曲线")
    print("   • 损失和准确率对比柱状图")
    print()
    
    print("2. 分类结果:")
    print("   • 混淆矩阵热力图")
    print("   • 类别分布对比")
    print("   • 各类别准确率分析")
    print("   • 预测置信度分布")
    print()
    
    print("3. 样本预测:")
    print("   • 样本图像展示")
    print("   • 预测结果标注")
    print("   • 置信度显示")
    print()
    
    print("4. 性能分析:")
    print("   • 各类别性能指标")
    print("   • 错误分析")
    print("   • 模型性能总结")
    print()

def show_installation_guide():
    """显示安装指南"""
    print("🛠️ 安装指南:")
    print()
    
    print("1. 必需依赖:")
    print("   pip install torch torchvision torchaudio")
    print("   pip install transformers datasets")
    print("   pip install matplotlib seaborn scikit-learn")
    print("   pip install numpy pandas pillow")
    print()
    
    print("2. 可选依赖:")
    print("   pip install accelerate  # 加速训练")
    print("   pip install wandb       # 实验跟踪")
    print("   pip install tensorboard  # 可视化")
    print("   pip install timm        # 更多视觉模型")
    print()
    
    print("3. 系统要求:")
    print("   • Python 3.7+")
    print("   • CUDA 11.0+ (可选，用于GPU加速)")
    print("   • 8GB+ RAM (推荐)")
    print("   • 2GB+ 磁盘空间")
    print()

def show_model_variants():
    """显示模型变体"""
    print("🤖 支持的ViT模型:")
    print()
    
    print("1. 基础模型:")
    print("   • google/vit-base-patch16-224")
    print("   • google/vit-base-patch32-224")
    print("   • google/vit-base-patch16-384")
    print()
    
    print("2. 大型模型:")
    print("   • google/vit-large-patch16-224")
    print("   • google/vit-large-patch32-224")
    print("   • google/vit-large-patch16-384")
    print()
    
    print("3. 其他变体:")
    print("   • microsoft/beit-base-patch16-224")
    print("   • microsoft/swin-base-patch4-window7-224")
    print("   • facebook/deit-base-patch16-224")
    print()

def show_troubleshooting():
    """显示故障排除"""
    print("🐛 故障排除:")
    print()
    
    print("Q: 内存不足怎么办？")
    print("A: • 减少批次大小")
    print("   • 使用更小的模型")
    print("   • 启用混合精度训练")
    print("   • 使用梯度累积")
    print()
    
    print("Q: 训练速度慢怎么办？")
    print("A: • 使用GPU加速")
    print("   • 启用混合精度")
    print("   • 使用预训练模型")
    print("   • 减少图像分辨率")
    print()
    
    print("Q: 分类准确率低怎么办？")
    print("A: • 增加训练数据")
    print("   • 调整学习率")
    print("   • 使用数据增强")
    print("   • 尝试更大的模型")
    print()
    
    print("Q: 模型过拟合怎么办？")
    print("A: • 增加正则化")
    print("   • 使用早停")
    print("   • 减少模型复杂度")
    print("   • 增加训练数据")
    print()

def show_advanced_features():
    """显示高级功能"""
    print("🚀 高级功能:")
    print()
    
    print("1. 数据增强:")
    print("   • 随机旋转")
    print("   • 随机裁剪")
    print("   • 颜色抖动")
    print("   • 随机翻转")
    print()
    
    print("2. 模型优化:")
    print("   • 学习率调度")
    print("   • 权重衰减")
    print("   • 梯度裁剪")
    print("   • 混合精度训练")
    print()
    
    print("3. 评估方法:")
    print("   • K折交叉验证")
    print("   • 分层采样")
    print("   • 类别平衡")
    print("   • 错误分析")
    print()
    
    print("4. 部署选项:")
    print("   • 模型量化")
    print("   • 模型剪枝")
    print("   • ONNX导出")
    print("   • TensorRT优化")
    print()

def main():
    """主函数"""
    show_program_info()
    show_architecture()
    show_usage_examples()
    show_evaluation_metrics()
    show_visualization_features()
    show_installation_guide()
    show_model_variants()
    show_troubleshooting()
    show_advanced_features()
    
    print("=" * 80)
    print("🎉 演示完成！")
    print()
    print("📚 要运行完整程序，请先安装依赖:")
    print("   pip install torch transformers datasets matplotlib seaborn scikit-learn pillow")
    print()
    print("🚀 然后运行:")
    print("   python vit_image_classifier.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
