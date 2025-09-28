"""
简化版RAG系统
避免复杂的依赖问题，使用更稳定的API
"""

import os
import time
from pathlib import Path
from typing import List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """简化版RAG系统"""
    
    def __init__(self, 
                 knowledge_base_path: str = "./nlp_base2/knowledge_base",
                 index_storage_path: str = "./nlp_base2/storage"):
        """
        初始化RAG系统
        
        Args:
            knowledge_base_path: 知识库文档路径
            index_storage_path: 索引存储路径
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.index_storage_path = Path(index_storage_path)
        
        # 创建必要的目录
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.index_storage_path.mkdir(exist_ok=True)
        
        # 文档存储
        self.documents = []
        self.index = None
        
    def load_text_files(self) -> List[str]:
        """
        加载文本文件
        
        Returns:
            文档内容列表
        """
        documents = []
        
        if not self.knowledge_base_path.exists():
            logger.warning(f"知识库目录不存在: {self.knowledge_base_path}")
            return documents
        
        # 支持的文本文件格式
        text_extensions = ['.txt', '.md', '.py', '.json', '.csv']
        
        for file_path in self.knowledge_base_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'file_name': file_path.name,
                            'file_path': str(file_path),
                            'content': content,
                            'size': len(content)
                        })
                        logger.info(f"已加载文档: {file_path.name}")
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {e}")
        
        self.documents = documents
        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents
    
    def simple_search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        简单文本搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.documents:
            logger.warning("没有加载任何文档")
            return []
        
        results = []
        query_lower = query.lower()
        
        for doc in self.documents:
            content_lower = doc['content'].lower()
            
            # 简单的关键词匹配
            if query_lower in content_lower:
                # 计算匹配度（简单的字符匹配）
                match_count = content_lower.count(query_lower)
                relevance_score = match_count / len(doc['content']) * 100
                
                # 找到匹配的上下文
                start_idx = content_lower.find(query_lower)
                context_start = max(0, start_idx - 100)
                context_end = min(len(doc['content']), start_idx + len(query_lower) + 100)
                context = doc['content'][context_start:context_end]
                
                results.append({
                    'file_name': doc['file_name'],
                    'file_path': doc['file_path'],
                    'relevance_score': relevance_score,
                    'context': context,
                    'match_count': match_count
                })
        
        # 按相关性排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:top_k]
    
    def get_document_stats(self) -> dict:
        """获取文档统计信息"""
        if not self.documents:
            return {"error": "没有加载文档"}
        
        total_size = sum(doc['size'] for doc in self.documents)
        avg_size = total_size / len(self.documents) if self.documents else 0
        
        stats = {
            "文档数量": len(self.documents),
            "总大小": f"{total_size:,} 字符",
            "平均大小": f"{avg_size:.0f} 字符",
            "存储路径": str(self.index_storage_path),
            "知识库路径": str(self.knowledge_base_path)
        }
        
        return stats
    
    def interactive_search(self):
        """交互式搜索"""
        print("=== 简化版RAG系统 ===")
        print("加载文档...")
        
        # 加载文档
        documents = self.load_text_files()
        
        if not documents:
            print("没有找到任何文档，请将文本文件放入知识库目录")
            return
        
        # 显示统计信息
        stats = self.get_document_stats()
        print("\n文档统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n开始搜索 (输入 'quit' 退出):")
        
        while True:
            query = input("\n请输入搜索关键词: ").strip()
            
            if query.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用！")
                break
            
            if not query:
                continue
            
            try:
                print(f"\n搜索: '{query}'")
                results = self.simple_search(query, top_k=5)
                
                if not results:
                    print("没有找到相关文档")
                    continue
                
                print(f"\n找到 {len(results)} 个相关文档:")
                
                for i, result in enumerate(results, 1):
                    print(f"\n--- 结果 {i} ---")
                    print(f"文件: {result['file_name']}")
                    print(f"相关性: {result['relevance_score']:.2f}%")
                    print(f"匹配次数: {result['match_count']}")
                    print(f"上下文: ...{result['context']}...")
                    
            except Exception as e:
                print(f"搜索出错: {e}")

def main():
    """主函数"""
    rag = SimpleRAGSystem()
    rag.interactive_search()

if __name__ == "__main__":
    main()


