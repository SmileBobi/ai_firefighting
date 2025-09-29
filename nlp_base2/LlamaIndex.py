"""
基于LlamaIndex的本地知识库RAG系统
支持多种文档格式，提供智能问答功能
"""

import os
import time
from pathlib import Path
from typing import List, Optional
import logging

# LlamaIndex核心组件
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 嵌入模型和LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 文档处理
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.schema import Document

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalRAGSystem:
    """本地知识库RAG系统"""
    
    def __init__(self, 
                 knowledge_base_path: str = "./nlp_base2/knowledge_base",
                 index_storage_path: str = "./nlp_base2/storage",
                 model_name: str = "deepseek-r1:7b",
                 embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化RAG系统
        
        Args:
            knowledge_base_path: 知识库文档路径
            index_storage_path: 索引存储路径
            model_name: Ollama模型名称
            embedding_model: 嵌入模型名称
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index_storage_path = Path(index_storage_path)
        self.model_name = model_name
        self.embedding_model = embedding_model
        
        # 创建必要的目录
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.index_storage_path.mkdir(exist_ok=True)
        
        # 初始化组件
        self._setup_llm()
        self._setup_embedding()
        self._setup_settings()
        
        # 索引和查询引擎
        self.index = None
        self.query_engine = None
        
    def _setup_llm(self):
        """设置LLM"""
        try:
            self.llm = Ollama(model=self.model_name, request_timeout=120.0)
            logger.info(f"LLM初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise
    
    def _setup_embedding(self):
        """设置嵌入模型"""
        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                device="cpu"  # 可以根据需要改为"cuda"
            )
            logger.info(f"嵌入模型初始化成功: {self.embedding_model}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise
    
    def _setup_settings(self):
        """设置全局配置"""
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
    
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        加载文档
        
        Args:
            file_paths: 指定文件路径列表，如果为None则加载知识库目录下所有文档
            
        Returns:
            文档列表
        """
        documents = []
        
        if file_paths:
            # 加载指定文件
            for file_path in file_paths:
                if os.path.exists(file_path):
                    if file_path.endswith('.pdf'):
                        reader = PDFReader()
                        docs = reader.load_data(file=Path(file_path))
                    elif file_path.endswith(('.docx', '.doc')):
                        reader = DocxReader()
                        docs = reader.load_data(file=Path(file_path))
                    else:
                        # 使用默认读取器
                        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
                    documents.extend(docs)
                    logger.info(f"已加载文档: {file_path}")
        else:
            # 加载知识库目录下所有文档
            if self.knowledge_base_path.exists():
                reader = SimpleDirectoryReader(
                    input_dir=str(self.knowledge_base_path),
                    recursive=True
                )
                documents = reader.load_data()
                logger.info(f"已加载知识库目录下的 {len(documents)} 个文档")
        
        return documents
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False):
        """
        构建向量索引
        
        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建索引
        """
        try:
            # 检查是否已存在索引
            if not force_rebuild and self._index_exists():
                logger.info("加载现有索引...")
                self.index = self._load_existing_index()
            else:
                logger.info("构建新索引...")
                start_time = time.time()
                
                # 创建向量索引
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True
                )
                
                # 保存索引
                self.index.storage_context.persist(persist_dir=str(self.index_storage_path))
                
                end_time = time.time()
                logger.info(f"索引构建完成，耗时: {end_time - start_time:.2f} 秒")
            
            # 创建查询引擎
            self._create_query_engine()
            
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise
    
    def _index_exists(self) -> bool:
        """检查索引是否存在"""
        return (self.index_storage_path / "index_store.json").exists()
    
    def _load_existing_index(self):
        """加载现有索引"""
        storage_context = StorageContext.from_defaults(persist_dir=str(self.index_storage_path))
        return load_index_from_storage(storage_context)
    
    def _create_query_engine(self):
        """创建查询引擎"""
        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # 创建后处理器
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        
        # 创建查询引擎
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[postprocessor]
        )
        logger.info("查询引擎创建完成")
    
    def query(self, question: str, verbose: bool = True) -> str:
        """
        查询知识库
        
        Args:
            question: 问题
            verbose: 是否显示详细信息
            
        Returns:
            回答
        """
        if not self.query_engine:
            raise ValueError("查询引擎未初始化，请先构建索引")
        
        try:
            start_time = time.time()
            
            if verbose:
                logger.info(f"查询问题: {question}")
            
            # 执行查询
            response = self.query_engine.query(question)
            
            end_time = time.time()
            
            if verbose:
                logger.info(f"查询完成，耗时: {end_time - start_time:.2f} 秒")
                logger.info(f"回答: {response}")
            
            return str(response)
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise
    
    def get_retrieved_nodes(self, question: str, top_k: int = 3):
        """
        获取检索到的相关节点
        
        Args:
            question: 问题
            top_k: 返回的节点数量
            
        Returns:
            相关节点列表
        """
        if not self.index:
            raise ValueError("索引未初始化")
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        nodes = retriever.retrieve(question)
        return nodes
    
    def add_documents(self, documents: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            documents: 新文档列表
        """
        if not self.index:
            raise ValueError("索引未初始化")
        
        try:
            # 插入新文档
            for doc in documents:
                self.index.insert(doc)
            
            # 保存更新后的索引
            self.index.storage_context.persist(persist_dir=str(self.index_storage_path))
            
            # 重新创建查询引擎
            self._create_query_engine()
            
            logger.info(f"成功添加 {len(documents)} 个新文档")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def get_index_stats(self) -> dict:
        """获取索引统计信息"""
        if not self.index:
            return {"error": "索引未初始化"}
        
        try:
            # 获取文档数量
            doc_count = len(self.index.docstore.docs) if hasattr(self.index.docstore, 'docs') else 0
            
            # 获取节点数量 - 使用更安全的方式
            node_count = 0
            if hasattr(self.index.docstore, 'nodes'):
                node_count = len(self.index.docstore.nodes)
            elif hasattr(self.index.docstore, 'get_nodes'):
                # 尝试通过get_nodes方法获取
                try:
                    nodes = self.index.docstore.get_nodes()
                    node_count = len(nodes) if nodes else 0
                except:
                    node_count = 0
            
            stats = {
                "文档数量": doc_count,
                "节点数量": node_count,
                "存储路径": str(self.index_storage_path),
                "知识库路径": str(self.knowledge_base_path)
            }
            return stats
        except Exception as e:
            return {"error": f"获取统计信息失败: {e}"}

def main():
    """主函数 - 演示RAG系统使用"""
    print("=== LlamaIndex RAG系统演示 ===")
    
    # 初始化RAG系统
    rag = LocalRAGSystem(
        knowledge_base_path="./nlp_base2/knowledge_base",
        index_storage_path="./nlp_base2/storage",
        model_name="deepseek-r1:7b"
    )
    
    # 加载文档
    print("\n1. 加载文档...")
    documents = rag.load_documents()
    
    if not documents:
        print("知识库目录为空，请添加一些文档到 ./knowledge_base 目录")
        return
    
    # 构建索引
    print("\n2. 构建索引...")
    rag.build_index(documents)
    
    # 显示索引统计
    print("\n3. 索引统计信息:")
    stats = rag.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 交互式查询
    print("\n4. 开始交互式查询 (输入 'quit' 退出):")
    while True:
        question = input("\n请输入问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            break
        
        if not question:
            continue
        
        try:
            # 执行查询
            answer = rag.query(question)
            print(f"\n回答: {answer}")
            
            # 显示相关文档片段
            print("\n相关文档片段:")
            nodes = rag.get_retrieved_nodes(question, top_k=3)
            for i, node in enumerate(nodes, 1):
                print(f"\n片段 {i} (相似度: {node.score:.3f}):")
                print(f"  {node.text[:200]}...")
                
        except Exception as e:
            print(f"查询出错: {e}")
    
    print("\n感谢使用RAG系统！")

if __name__ == "__main__":
    main()