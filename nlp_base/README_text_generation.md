# 基于PyTorch和HuggingFace Transformers的文本生成程序

这是一个完整的文本生成程序，使用PyTorch和HuggingFace Transformers库，基于WikiText-2数据集实现文本生成任务。

## 🚀 功能特点

### 🧠 核心功能
- **GPT-2模型**: 使用预训练的GPT-2模型进行文本生成
- **WikiText-2数据集**: 自动加载和处理WikiText-2数据集
- **模型微调**: 支持在自定义数据上微调模型
- **文本生成**: 多种生成策略和参数控制
- **质量评估**: 全面的生成质量评估指标

### 📊 数据处理
- **自动数据加载**: 从HuggingFace Hub加载WikiText-2数据集
- **文本预处理**: 清理和标准化文本数据
- **数据集划分**: 自动划分训练、验证和测试集
- **批处理**: 高效的数据批处理机制

### 🎯 模型训练
- **预训练模型**: 基于GPT-2的预训练模型
- **微调支持**: 在特定数据上进行模型微调
- **训练监控**: 实时监控训练损失和验证损失
- **学习率调度**: 自动学习率调整

### 📈 评估指标
- **困惑度**: 计算模型的语言建模困惑度
- **生成质量**: 评估生成文本的流畅度和多样性
- **重复率**: 检测生成文本中的重复模式
- **词汇多样性**: 分析生成文本的词汇丰富度

## 📁 文件结构

```
nlp_base/
├── transformer_text_generation.py      # 主程序文件
├── run_text_generation.py              # 运行脚本
├── README_text_generation.md           # 说明文档
└── results/                             # 输出结果目录
    ├── text_generation_training_history.png
    ├── text_generation_results.png
    ├── text_generation_results.json
    └── text_generation_model/           # 保存的模型
```

## 🛠️ 安装依赖

### 必需依赖
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install matplotlib seaborn scikit-learn
pip install numpy pandas
```

### 可选依赖
```bash
pip install accelerate  # 加速训练
pip install wandb       # 实验跟踪
pip install tensorboard  # 可视化
```

## 🚀 使用方法

### 1. 基本运行
```bash
# 直接运行主程序
python transformer_text_generation.py

# 或使用运行脚本
python run_text_generation.py
```

### 2. 自定义参数
```python
# 在代码中修改参数
BATCH_SIZE = 8          # 批次大小
MAX_LENGTH = 512        # 最大序列长度
NUM_EPOCHS = 5          # 训练轮数
LEARNING_RATE = 3e-5    # 学习率
```

### 3. 自定义提示
```python
# 修改测试提示
test_prompts = [
    "Your custom prompt here",
    "Another prompt example",
    "More creative prompts..."
]
```

## 📊 程序输出

### 1. 训练过程
```
🚀 开始训练模型...
📚 Epoch 1/3
  Batch 0/25, Loss: 2.3456
  Batch 10/25, Loss: 2.1234
  ...
  训练损失: 2.1234
  验证损失: 1.9876
```

### 2. 文本生成
```
🎯 提示: 'The future of artificial intelligence'
📝 生成: 'The future of artificial intelligence is bright and promising...'
```

### 3. 评估结果
```
📊 困惑度: 15.23
📈 生成质量指标:
  average_length: 45.6
  unique_words: 234
  fluency_score: 0.78
  repetition_rate: 0.12
```

## 🎨 可视化功能

### 1. 训练历史
- 训练和验证损失曲线
- 损失对比柱状图
- 学习率变化趋势

### 2. 生成结果
- 生成文本长度分布
- 生成质量雷达图
- 高频词汇统计
- 生成文本示例

## ⚙️ 配置选项

### 模型配置
```python
# 模型参数
MODEL_NAME = 'gpt2'           # 模型名称
MAX_LENGTH = 256              # 最大序列长度
TEMPERATURE = 0.8             # 生成温度
TOP_P = 0.9                  # 核采样参数
```

### 训练配置
```python
# 训练参数
BATCH_SIZE = 4               # 批次大小
NUM_EPOCHS = 3              # 训练轮数
LEARNING_RATE = 5e-5        # 学习率
WARMUP_STEPS = 100          # 预热步数
```

### 生成配置
```python
# 生成参数
MAX_NEW_TOKENS = 100        # 最大生成长度
NUM_BEAMS = 4              # 束搜索大小
DO_SAMPLE = True           # 是否采样
REPETITION_PENALTY = 1.1   # 重复惩罚
```

## 📈 性能优化

### 1. 硬件优化
- **GPU加速**: 自动检测并使用CUDA
- **混合精度**: 支持FP16训练
- **梯度累积**: 处理大批次数据

### 2. 内存优化
- **梯度检查点**: 减少内存使用
- **动态填充**: 按批次动态调整序列长度
- **数据并行**: 支持多GPU训练

### 3. 训练优化
- **学习率调度**: 自动调整学习率
- **早停机制**: 防止过拟合
- **模型检查点**: 定期保存模型

## 🔧 高级功能

### 1. 自定义数据集
```python
# 加载自定义数据集
def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts]
```

### 2. 模型微调
```python
# 在特定领域数据上微调
trainer = TextGenerationTrainer(model, tokenizer, device)
trainer.train_model(custom_dataset, num_epochs=5)
```

### 3. 生成策略
```python
# 不同的生成策略
strategies = {
    'greedy': {'do_sample': False},
    'sampling': {'do_sample': True, 'temperature': 0.8},
    'beam_search': {'num_beams': 4, 'do_sample': False},
    'nucleus': {'do_sample': True, 'top_p': 0.9}
}
```

## 📊 评估指标详解

### 1. 困惑度 (Perplexity)
- **定义**: 模型对测试数据的平均负对数似然
- **计算**: PPL = exp(-1/N * Σ log P(x_i))
- **意义**: 越低越好，表示模型预测越准确

### 2. 流畅度 (Fluency)
- **定义**: 生成文本的语言流畅程度
- **计算**: 基于n-gram重叠和词汇多样性
- **范围**: 0-1，越高越好

### 3. 重复率 (Repetition Rate)
- **定义**: 生成文本中重复n-gram的比例
- **计算**: 重复n-gram数量 / 总n-gram数量
- **意义**: 越低越好，避免重复生成

### 4. 词汇多样性 (Lexical Diversity)
- **定义**: 生成文本中独特词汇的比例
- **计算**: 独特词汇数 / 总词汇数
- **意义**: 越高越好，表示词汇丰富

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 
- 减少批次大小
- 使用梯度累积
- 启用混合精度训练
- 使用更小的模型

### Q: 生成质量不好怎么办？
A:
- 调整温度参数
- 使用束搜索
- 增加训练数据
- 调整模型架构

### Q: 训练速度慢怎么办？
A:
- 使用GPU加速
- 启用混合精度
- 使用预训练模型
- 减少序列长度

## 📚 学习资源

### 1. 相关论文
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)

### 2. 官方文档
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Datasets Library](https://huggingface.co/docs/datasets/)

### 3. 教程资源
- [HuggingFace Course](https://huggingface.co/course/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [NLP with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/)

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
- OpenAI的GPT模型
- PyTorch团队
- 所有开源贡献者

---

**注意**: 这是一个教学示例程序，展示了如何使用PyTorch和HuggingFace Transformers进行文本生成。在实际应用中，请根据具体需求调整参数和配置。
