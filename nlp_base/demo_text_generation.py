"""
文本生成程序演示脚本
展示程序功能和使用方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def show_program_info():
    """显示程序信息"""
    print("=" * 80)
    print("🤖 基于PyTorch和HuggingFace Transformers的文本生成程序")
    print("📚 使用WikiText-2数据集实现文本生成任务")
    print("=" * 80)
    print()
    
    print("🔧 程序特点:")
    print("✅ 基于GPT-2模型的文本生成")
    print("✅ 自动加载WikiText-2数据集")
    print("✅ 支持模型微调和训练")
    print("✅ 多种文本生成策略")
    print("✅ 全面的质量评估指标")
    print("✅ 丰富的可视化功能")
    print()
    
    print("📊 核心功能:")
    print("1. 数据加载和预处理")
    print("   - 自动从HuggingFace Hub加载WikiText-2数据集")
    print("   - 文本清理和标准化")
    print("   - 数据集划分和批处理")
    print()
    
    print("2. 模型配置和训练")
    print("   - GPT-2预训练模型")
    print("   - 支持模型微调")
    print("   - 训练过程监控")
    print("   - 学习率自动调整")
    print()
    
    print("3. 文本生成")
    print("   - 多种生成策略（贪婪、采样、束搜索）")
    print("   - 可调节生成参数（温度、top-p等）")
    print("   - 批量文本生成")
    print()
    
    print("4. 质量评估")
    print("   - 困惑度计算")
    print("   - 生成质量指标")
    print("   - 流畅度和多样性分析")
    print("   - 重复率检测")
    print()
    
    print("5. 可视化展示")
    print("   - 训练历史图表")
    print("   - 生成结果分析")
    print("   - 质量指标雷达图")
    print("   - 词汇统计图表")
    print()

def show_architecture():
    """显示程序架构"""
    print("🏗️ 程序架构:")
    print()
    print("📁 文件结构:")
    print("nlp_base/")
    print("├── transformer_text_generation.py      # 主程序文件")
    print("├── run_text_generation.py              # 运行脚本")
    print("├── demo_text_generation.py              # 演示脚本")
    print("├── README_text_generation.md           # 详细说明文档")
    print("└── results/                             # 输出结果目录")
    print("    ├── text_generation_training_history.png")
    print("    ├── text_generation_results.png")
    print("    ├── text_generation_results.json")
    print("    └── text_generation_model/           # 保存的模型")
    print()
    
    print("🧠 核心类:")
    print("• WikiTextDataset: 数据集处理类")
    print("• TextGenerator: 文本生成器类")
    print("• WikiTextProcessor: 数据处理器类")
    print("• TextGenerationTrainer: 训练器类")
    print("• TextGenerationEvaluator: 评估器类")
    print()

def show_usage_examples():
    """显示使用示例"""
    print("🚀 使用示例:")
    print()
    
    print("1. 基本运行:")
    print("   python transformer_text_generation.py")
    print()
    
    print("2. 自定义参数:")
    print("   # 修改批次大小")
    print("   BATCH_SIZE = 8")
    print("   # 修改最大序列长度")
    print("   MAX_LENGTH = 512")
    print("   # 修改训练轮数")
    print("   NUM_EPOCHS = 5")
    print()
    
    print("3. 自定义提示:")
    print("   test_prompts = [")
    print("       'The future of artificial intelligence',")
    print("       'Machine learning is revolutionizing',")
    print("       'Natural language processing enables'")
    print("   ]")
    print()
    
    print("4. 生成参数调整:")
    print("   generated = generator.generate_text(")
    print("       prompt='Your prompt here',")
    print("       max_length=100,")
    print("       temperature=0.8,")
    print("       top_p=0.9,")
    print("       num_return_sequences=1")
    print("   )")
    print()

def show_evaluation_metrics():
    """显示评估指标"""
    print("📊 评估指标:")
    print()
    
    print("1. 困惑度 (Perplexity):")
    print("   • 定义: 模型对测试数据的平均负对数似然")
    print("   • 计算: PPL = exp(-1/N * Σ log P(x_i))")
    print("   • 意义: 越低越好，表示模型预测越准确")
    print()
    
    print("2. 流畅度 (Fluency):")
    print("   • 定义: 生成文本的语言流畅程度")
    print("   • 计算: 基于n-gram重叠和词汇多样性")
    print("   • 范围: 0-1，越高越好")
    print()
    
    print("3. 重复率 (Repetition Rate):")
    print("   • 定义: 生成文本中重复n-gram的比例")
    print("   • 计算: 重复n-gram数量 / 总n-gram数量")
    print("   • 意义: 越低越好，避免重复生成")
    print()
    
    print("4. 词汇多样性 (Lexical Diversity):")
    print("   • 定义: 生成文本中独特词汇的比例")
    print("   • 计算: 独特词汇数 / 总词汇数")
    print("   • 意义: 越高越好，表示词汇丰富")
    print()

def show_visualization_features():
    """显示可视化功能"""
    print("📈 可视化功能:")
    print()
    
    print("1. 训练历史:")
    print("   • 训练和验证损失曲线")
    print("   • 损失对比柱状图")
    print("   • 学习率变化趋势")
    print()
    
    print("2. 生成结果:")
    print("   • 生成文本长度分布")
    print("   • 生成质量雷达图")
    print("   • 高频词汇统计")
    print("   • 生成文本示例")
    print()
    
    print("3. 质量分析:")
    print("   • 困惑度变化趋势")
    print("   • 流畅度分布")
    print("   • 重复率分析")
    print("   • 词汇多样性统计")
    print()

def show_installation_guide():
    """显示安装指南"""
    print("🛠️ 安装指南:")
    print()
    
    print("1. 必需依赖:")
    print("   pip install torch torchvision torchaudio")
    print("   pip install transformers datasets")
    print("   pip install matplotlib seaborn scikit-learn")
    print("   pip install numpy pandas")
    print()
    
    print("2. 可选依赖:")
    print("   pip install accelerate  # 加速训练")
    print("   pip install wandb       # 实验跟踪")
    print("   pip install tensorboard  # 可视化")
    print()
    
    print("3. 系统要求:")
    print("   • Python 3.7+")
    print("   • CUDA 11.0+ (可选，用于GPU加速)")
    print("   • 8GB+ RAM (推荐)")
    print("   • 2GB+ 磁盘空间")
    print()

def show_troubleshooting():
    """显示故障排除"""
    print("🐛 故障排除:")
    print()
    
    print("Q: 内存不足怎么办？")
    print("A: • 减少批次大小")
    print("   • 使用梯度累积")
    print("   • 启用混合精度训练")
    print("   • 使用更小的模型")
    print()
    
    print("Q: 生成质量不好怎么办？")
    print("A: • 调整温度参数")
    print("   • 使用束搜索")
    print("   • 增加训练数据")
    print("   • 调整模型架构")
    print()
    
    print("Q: 训练速度慢怎么办？")
    print("A: • 使用GPU加速")
    print("   • 启用混合精度")
    print("   • 使用预训练模型")
    print("   • 减少序列长度")
    print()

def main():
    """主函数"""
    show_program_info()
    show_architecture()
    show_usage_examples()
    show_evaluation_metrics()
    show_visualization_features()
    show_installation_guide()
    show_troubleshooting()
    
    print("=" * 80)
    print("🎉 演示完成！")
    print()
    print("📚 要运行完整程序，请先安装依赖:")
    print("   pip install torch transformers datasets matplotlib seaborn scikit-learn")
    print()
    print("🚀 然后运行:")
    print("   python transformer_text_generation.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
