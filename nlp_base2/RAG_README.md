# LlamaIndex RAG系统使用指南

基于LlamaIndex构建的本地知识库RAG（检索增强生成）系统，支持多种文档格式的智能问答。

## 功能特点

- 🚀 **多格式文档支持**: PDF、Word、TXT、Markdown等
- 🧠 **智能检索**: 基于语义相似度的文档检索
- 💬 **自然语言问答**: 支持中文问答
- 🔄 **增量更新**: 支持动态添加新文档
- 💾 **持久化存储**: 索引自动保存，避免重复构建
- ⚡ **高性能**: 支持GPU加速（可选）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备知识库

将您的文档放入 `./knowledge_base` 目录：

```
knowledge_base/
├── document1.pdf
├── document2.docx
├── document3.txt
└── document4.md
```

### 2. 运行RAG系统

```bash
# 运行主程序
python LlamaIndex.py

# 或运行演示程序
python rag_demo.py
```

### 3. 开始问答

系统会自动：
1. 加载知识库中的文档
2. 构建向量索引
3. 启动交互式问答界面

## 使用示例

### 基本使用

```python
from LlamaIndex import LocalRAGSystem

# 初始化RAG系统
rag = LocalRAGSystem(
    knowledge_base_path="./knowledge_base",
    index_storage_path="./storage",
    model_name="deepseek-r1:7b"
)

# 加载文档
documents = rag.load_documents()

# 构建索引
rag.build_index(documents)

# 查询
answer = rag.query("这个文档的主要内容是什么？")
print(answer)
```

### 高级功能

```python
# 获取相关文档片段
nodes = rag.get_retrieved_nodes("问题", top_k=3)
for node in nodes:
    print(f"相似度: {node.score}")
    print(f"内容: {node.text}")

# 添加新文档
new_docs = rag.load_documents(["new_document.pdf"])
rag.add_documents(new_docs)

# 获取索引统计
stats = rag.get_index_stats()
print(stats)
```

## 配置选项

### 模型配置

```python
rag = LocalRAGSystem(
    model_name="deepseek-r1:7b",  # Ollama模型名称
    embedding_model="BAAI/bge-small-zh-v1.5"  # 嵌入模型
)
```

### 索引配置

```python
# 在_setup_settings方法中修改
Settings.node_parser = SimpleNodeParser.from_defaults(
    chunk_size=512,      # 文档分块大小
    chunk_overlap=50     # 分块重叠大小
)
```

## 支持的文档格式

- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`
- **文本**: `.txt`
- **Markdown**: `.md`
- **其他**: 通过SimpleDirectoryReader支持

## 性能优化

### GPU加速

```python
# 在_setup_embedding方法中修改
self.embedding_model = HuggingFaceEmbedding(
    model_name=self.embedding_model,
    device="cuda"  # 使用GPU
)
```

### 索引优化

- 调整`chunk_size`和`chunk_overlap`参数
- 使用更小的嵌入模型
- 启用索引缓存

## 故障排除

### 常见问题

1. **Ollama连接失败**
   ```bash
   # 确保Ollama服务运行
   ollama serve
   ```

2. **模型下载失败**
   ```bash
   # 手动下载模型
   ollama pull deepseek-r1:7b
   ```

3. **内存不足**
   - 使用更小的嵌入模型
   - 减少chunk_size
   - 使用CPU模式

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API参考

### LocalRAGSystem类

#### 初始化参数
- `knowledge_base_path`: 知识库路径
- `index_storage_path`: 索引存储路径  
- `model_name`: Ollama模型名称
- `embedding_model`: 嵌入模型名称

#### 主要方法
- `load_documents()`: 加载文档
- `build_index()`: 构建索引
- `query()`: 执行查询
- `get_retrieved_nodes()`: 获取相关节点
- `add_documents()`: 添加新文档
- `get_index_stats()`: 获取统计信息

## 扩展功能

### 自定义文档处理器

```python
from llama_index.readers.file import PDFReader

# 自定义PDF处理器
class CustomPDFReader(PDFReader):
    def load_data(self, file, extra_info=None):
        # 自定义处理逻辑
        return super().load_data(file, extra_info)
```

### 自定义检索器

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 自定义检索器
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,  # 检索更多结果
    doc_ids=["doc1", "doc2"]  # 限制检索范围
)
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

- v1.0.0: 初始版本，支持基本RAG功能
- 支持多格式文档
- 集成Ollama LLM
- 支持中文问答
