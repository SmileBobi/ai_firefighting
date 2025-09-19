#!/usr/bin/env python
"""
消防RAG知识库集成工具
用于构建和管理消防知识库，支持向量化存储和检索
"""

import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict


class FireRAGKnowledgeBase:
    """消防RAG知识库"""
    
    def __init__(self, knowledge_base_path: str = "fire_knowledge_base"):
        """初始化消防RAG知识库"""
        self.knowledge_base_path = knowledge_base_path
        self.documents = {}  # 文档存储
        self.chunks = {}  # 文档块存储
        self.embeddings = {}  # 向量存储
        self.index = {}  # 索引存储
        
        # 创建知识库目录
        os.makedirs(knowledge_base_path, exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_path, "documents"), exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_path, "chunks"), exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_path, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(knowledge_base_path, "index"), exist_ok=True)
        
        # 加载现有数据
        self.load_knowledge_base()
    
    def add_document(self, document: Dict[str, Any]) -> str:
        """添加文档到知识库"""
        try:
            doc_id = document.get('document_id', f"doc_{len(self.documents)}")
            
            # 存储文档
            self.documents[doc_id] = {
                'id': doc_id,
                'title': document.get('title', ''),
                'content': document.get('content', ''),
                'document_type': document.get('document_type', ''),
                'summary': document.get('summary', ''),
                'keywords': document.get('keywords', []),
                'entities': document.get('entities', {}),
                'topics': document.get('topics', {}),
                'content_length': document.get('content_length', 0),
                'word_count': document.get('word_count', 0),
                'readability_score': document.get('readability_score', 0),
                'complexity_score': document.get('complexity_score', 0),
                'scraped_at': document.get('scraped_at', ''),
                'source_url': document.get('source_url', ''),
                'added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 处理文档块
            chunk_texts = document.get('chunk_texts', [])
            if chunk_texts:
                self.add_chunks(doc_id, chunk_texts)
            
            # 保存文档
            self.save_document(doc_id)
            
            return doc_id
            
        except Exception as e:
            print(f"添加文档失败: {e}")
            return None
    
    def add_chunks(self, doc_id: str, chunk_texts: List[str]):
        """添加文档块"""
        try:
            if doc_id not in self.chunks:
                self.chunks[doc_id] = []
            
            for i, chunk_text in enumerate(chunk_texts):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_data = {
                    'id': chunk_id,
                    'doc_id': doc_id,
                    'text': chunk_text,
                    'length': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'chunk_index': i,
                    'added_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                self.chunks[doc_id].append(chunk_data)
            
            # 保存文档块
            self.save_chunks(doc_id)
            
        except Exception as e:
            print(f"添加文档块失败: {e}")
    
    def generate_embeddings(self, doc_id: str, embedding_model=None):
        """生成文档向量（简化版本）"""
        try:
            if doc_id not in self.chunks:
                return
            
            # 简化的向量生成（实际应用中应使用专业的embedding模型）
            embeddings = []
            
            for chunk in self.chunks[doc_id]:
                # 使用简单的TF-IDF风格向量化
                text = chunk['text']
                words = text.lower().split()
                
                # 创建词汇频率向量
                word_freq = defaultdict(int)
                for word in words:
                    word_freq[word] += 1
                
                # 转换为向量（简化版本）
                vector = [word_freq.get(word, 0) for word in sorted(word_freq.keys())]
                
                # 归一化
                if vector:
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = [v / norm for v in vector]
                
                embeddings.append({
                    'chunk_id': chunk['id'],
                    'vector': vector,
                    'dimension': len(vector),
                    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            self.embeddings[doc_id] = embeddings
            
            # 保存向量
            self.save_embeddings(doc_id)
            
        except Exception as e:
            print(f"生成向量失败: {e}")
    
    def search_documents(self, query: str, top_k: int = 5, doc_type: str = None) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            results = []
            
            # 简化的搜索实现
            query_words = query.lower().split()
            
            for doc_id, doc in self.documents.items():
                if doc_type and doc.get('document_type') != doc_type:
                    continue
                
                # 计算相关性得分
                score = 0
                content = doc.get('content', '').lower()
                title = doc.get('title', '').lower()
                keywords = [kw.lower() for kw in doc.get('keywords', [])]
                
                for word in query_words:
                    # 标题匹配权重更高
                    if word in title:
                        score += 3
                    # 关键词匹配
                    if word in keywords:
                        score += 2
                    # 内容匹配
                    if word in content:
                        score += 1
                
                if score > 0:
                    results.append({
                        'doc_id': doc_id,
                        'title': doc.get('title', ''),
                        'summary': doc.get('summary', ''),
                        'document_type': doc.get('document_type', ''),
                        'score': score,
                        'keywords': doc.get('keywords', []),
                        'source_url': doc.get('source_url', '')
                    })
            
            # 按得分排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"搜索文档失败: {e}")
            return []
    
    def search_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索文档块"""
        try:
            results = []
            query_words = query.lower().split()
            
            for doc_id, chunks in self.chunks.items():
                for chunk in chunks:
                    score = 0
                    text = chunk['text'].lower()
                    
                    for word in query_words:
                        if word in text:
                            score += text.count(word)
                    
                    if score > 0:
                        results.append({
                            'chunk_id': chunk['id'],
                            'doc_id': doc_id,
                            'text': chunk['text'],
                            'score': score,
                            'length': chunk['length']
                        })
            
            # 按得分排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"搜索文档块失败: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""
        return self.documents.get(doc_id)
    
    def get_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """获取文档块"""
        return self.chunks.get(doc_id, [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': sum(len(chunks) for chunks in self.chunks.values()),
            'total_embeddings': sum(len(embeddings) for embeddings in self.embeddings.values()),
            'document_types': {},
            'categories': {},
            'knowledge_base_size': 0
        }
        
        # 统计文档类型
        for doc in self.documents.values():
            doc_type = doc.get('document_type', 'unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
        
        # 统计分类
        for doc in self.documents.values():
            topics = doc.get('topics', {})
            if isinstance(topics, dict):
                category = topics.get('category', '其他')
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
        
        # 计算知识库大小
        stats['knowledge_base_size'] = sum(doc.get('content_length', 0) for doc in self.documents.values())
        
        return stats
    
    def save_knowledge_base(self):
        """保存知识库"""
        try:
            # 保存索引
            index_file = os.path.join(self.knowledge_base_path, "index", "knowledge_base_index.json")
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': list(self.documents.keys()),
                    'chunks': {doc_id: [chunk['id'] for chunk in chunks] for doc_id, chunks in self.chunks.items()},
                    'embeddings': list(self.embeddings.keys()),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
            
            # 保存统计信息
            stats_file = os.path.join(self.knowledge_base_path, "index", "statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_statistics(), f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"保存知识库失败: {e}")
    
    def load_knowledge_base(self):
        """加载知识库"""
        try:
            # 加载索引
            index_file = os.path.join(self.knowledge_base_path, "index", "knowledge_base_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                # 加载文档
                for doc_id in index_data.get('documents', []):
                    self.load_document(doc_id)
                
                # 加载文档块
                for doc_id in index_data.get('chunks', {}):
                    self.load_chunks(doc_id)
                
                # 加载向量
                for doc_id in index_data.get('embeddings', []):
                    self.load_embeddings(doc_id)
            
        except Exception as e:
            print(f"加载知识库失败: {e}")
    
    def save_document(self, doc_id: str):
        """保存单个文档"""
        try:
            doc_file = os.path.join(self.knowledge_base_path, "documents", f"{doc_id}.json")
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents[doc_id], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存文档失败: {e}")
    
    def load_document(self, doc_id: str):
        """加载单个文档"""
        try:
            doc_file = os.path.join(self.knowledge_base_path, "documents", f"{doc_id}.json")
            if os.path.exists(doc_file):
                with open(doc_file, 'r', encoding='utf-8') as f:
                    self.documents[doc_id] = json.load(f)
        except Exception as e:
            print(f"加载文档失败: {e}")
    
    def save_chunks(self, doc_id: str):
        """保存文档块"""
        try:
            chunks_file = os.path.join(self.knowledge_base_path, "chunks", f"{doc_id}_chunks.json")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks[doc_id], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存文档块失败: {e}")
    
    def load_chunks(self, doc_id: str):
        """加载文档块"""
        try:
            chunks_file = os.path.join(self.knowledge_base_path, "chunks", f"{doc_id}_chunks.json")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks[doc_id] = json.load(f)
        except Exception as e:
            print(f"加载文档块失败: {e}")
    
    def save_embeddings(self, doc_id: str):
        """保存向量"""
        try:
            embeddings_file = os.path.join(self.knowledge_base_path, "embeddings", f"{doc_id}_embeddings.pkl")
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings[doc_id], f)
        except Exception as e:
            print(f"保存向量失败: {e}")
    
    def load_embeddings(self, doc_id: str):
        """加载向量"""
        try:
            embeddings_file = os.path.join(self.knowledge_base_path, "embeddings", f"{doc_id}_embeddings.pkl")
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    self.embeddings[doc_id] = pickle.load(f)
        except Exception as e:
            print(f"加载向量失败: {e}")


class FireRAGQueryEngine:
    """消防RAG查询引擎"""
    
    def __init__(self, knowledge_base: FireRAGKnowledgeBase):
        """初始化查询引擎"""
        self.knowledge_base = knowledge_base
    
    def query(self, question: str, context_type: str = "all") -> Dict[str, Any]:
        """查询消防知识"""
        try:
            # 搜索相关文档
            documents = self.knowledge_base.search_documents(question, top_k=5, doc_type=context_type)
            
            # 搜索相关文档块
            chunks = self.knowledge_base.search_chunks(question, top_k=10)
            
            # 构建回答
            answer = {
                'question': question,
                'context_type': context_type,
                'relevant_documents': documents,
                'relevant_chunks': chunks,
                'answer_summary': self.generate_answer_summary(documents, chunks),
                'sources': [doc['source_url'] for doc in documents if doc.get('source_url')],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return answer
            
        except Exception as e:
            return {
                'question': question,
                'error': f"查询失败: {e}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def generate_answer_summary(self, documents: List[Dict], chunks: List[Dict]) -> str:
        """生成回答摘要"""
        if not documents and not chunks:
            return "未找到相关信息。"
        
        summary_parts = []
        
        if documents:
            summary_parts.append(f"找到 {len(documents)} 个相关文档：")
            for doc in documents[:3]:  # 只显示前3个
                summary_parts.append(f"- {doc['title']}")
        
        if chunks:
            summary_parts.append(f"找到 {len(chunks)} 个相关文档片段。")
        
        return "\n".join(summary_parts)


# 全局消防RAG知识库实例
fire_rag_kb = FireRAGKnowledgeBase()
fire_rag_engine = FireRAGQueryEngine(fire_rag_kb)
