#!/usr/bin/env python
"""
消防数据专用管道
"""

import pandas as pd
import json
import csv
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class FireExcelWriterPipeline:
    """消防数据Excel文件存储管道"""
    
    def __init__(self):
        self.items = {
            'fire_regulations': [],
            'fire_standards': [],
            'fire_cases': [],
            'fire_news': [],
            'fire_knowledge': [],
            'fire_documents': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("消防Excel管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存Excel文件"""
        try:
            # 创建Excel写入器
            excel_filename = f'fire_data_{self.timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 保存消防法规数据
                if self.items['fire_regulations']:
                    df_regulations = pd.DataFrame(self.items['fire_regulations'])
                    df_regulations.to_excel(writer, sheet_name='消防法规', index=False)
                    spider.logger.info(f"消防法规数据: {len(self.items['fire_regulations'])} 条")
                
                # 保存消防标准数据
                if self.items['fire_standards']:
                    df_standards = pd.DataFrame(self.items['fire_standards'])
                    df_standards.to_excel(writer, sheet_name='消防标准', index=False)
                    spider.logger.info(f"消防标准数据: {len(self.items['fire_standards'])} 条")
                
                # 保存火灾案例数据
                if self.items['fire_cases']:
                    df_cases = pd.DataFrame(self.items['fire_cases'])
                    df_cases.to_excel(writer, sheet_name='火灾案例', index=False)
                    spider.logger.info(f"火灾案例数据: {len(self.items['fire_cases'])} 条")
                
                # 保存消防新闻数据
                if self.items['fire_news']:
                    df_news = pd.DataFrame(self.items['fire_news'])
                    df_news.to_excel(writer, sheet_name='消防新闻', index=False)
                    spider.logger.info(f"消防新闻数据: {len(self.items['fire_news'])} 条")
                
                # 保存消防知识数据
                if self.items['fire_knowledge']:
                    df_knowledge = pd.DataFrame(self.items['fire_knowledge'])
                    df_knowledge.to_excel(writer, sheet_name='消防知识', index=False)
                    spider.logger.info(f"消防知识数据: {len(self.items['fire_knowledge'])} 条")
                
                # 保存RAG文档数据
                if self.items['fire_documents']:
                    df_documents = pd.DataFrame(self.items['fire_documents'])
                    df_documents.to_excel(writer, sheet_name='RAG文档', index=False)
                    spider.logger.info(f"RAG文档数据: {len(self.items['fire_documents'])} 条")
            
            spider.logger.info(f"消防Excel文件已保存: {excel_filename}")
            
        except Exception as e:
            spider.logger.error(f"保存消防Excel文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'FireRegulationItem':
                self.items['fire_regulations'].append(item_dict)
            elif class_name == 'FireStandardItem':
                self.items['fire_standards'].append(item_dict)
            elif class_name == 'FireCaseItem':
                self.items['fire_cases'].append(item_dict)
            elif class_name == 'FireNewsItem':
                self.items['fire_news'].append(item_dict)
            elif class_name == 'FireKnowledgeItem':
                self.items['fire_knowledge'].append(item_dict)
            elif class_name == 'FireDocumentItem':
                self.items['fire_documents'].append(item_dict)
        
        return item


class FireCsvWriterPipeline:
    """消防数据CSV文件存储管道"""
    
    def __init__(self):
        self.items = {
            'fire_regulations': [],
            'fire_standards': [],
            'fire_cases': [],
            'fire_news': [],
            'fire_knowledge': [],
            'fire_documents': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("消防CSV管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存CSV文件"""
        try:
            # 保存消防法规数据
            if self.items['fire_regulations']:
                df_regulations = pd.DataFrame(self.items['fire_regulations'])
                df_regulations.to_csv(f'fire_regulations_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"消防法规CSV已保存: {len(self.items['fire_regulations'])} 条")
            
            # 保存消防标准数据
            if self.items['fire_standards']:
                df_standards = pd.DataFrame(self.items['fire_standards'])
                df_standards.to_csv(f'fire_standards_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"消防标准CSV已保存: {len(self.items['fire_standards'])} 条")
            
            # 保存火灾案例数据
            if self.items['fire_cases']:
                df_cases = pd.DataFrame(self.items['fire_cases'])
                df_cases.to_csv(f'fire_cases_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"火灾案例CSV已保存: {len(self.items['fire_cases'])} 条")
            
            # 保存消防新闻数据
            if self.items['fire_news']:
                df_news = pd.DataFrame(self.items['fire_news'])
                df_news.to_csv(f'fire_news_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"消防新闻CSV已保存: {len(self.items['fire_news'])} 条")
            
            # 保存消防知识数据
            if self.items['fire_knowledge']:
                df_knowledge = pd.DataFrame(self.items['fire_knowledge'])
                df_knowledge.to_csv(f'fire_knowledge_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"消防知识CSV已保存: {len(self.items['fire_knowledge'])} 条")
            
            # 保存RAG文档数据
            if self.items['fire_documents']:
                df_documents = pd.DataFrame(self.items['fire_documents'])
                df_documents.to_csv(f'fire_documents_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"RAG文档CSV已保存: {len(self.items['fire_documents'])} 条")
                
        except Exception as e:
            spider.logger.error(f"保存消防CSV文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'FireRegulationItem':
                self.items['fire_regulations'].append(item_dict)
            elif class_name == 'FireStandardItem':
                self.items['fire_standards'].append(item_dict)
            elif class_name == 'FireCaseItem':
                self.items['fire_cases'].append(item_dict)
            elif class_name == 'FireNewsItem':
                self.items['fire_news'].append(item_dict)
            elif class_name == 'FireKnowledgeItem':
                self.items['fire_knowledge'].append(item_dict)
            elif class_name == 'FireDocumentItem':
                self.items['fire_documents'].append(item_dict)
        
        return item


class FireConsolePipeline:
    """消防数据控制台输出管道"""
    
    def process_item(self, item, spider):
        """在控制台输出item信息"""
        adapter = ItemAdapter(item)
        
        print(f"\n=== 消防数据 ===")
        
        # 根据数据类型显示不同信息
        if 'regulation_id' in adapter:
            print(f"法规ID: {adapter.get('regulation_id', 'N/A')}")
            print(f"法规标题: {adapter.get('title', 'N/A')}")
            print(f"法规类型: {adapter.get('regulation_type', 'N/A')}")
            print(f"发布机关: {adapter.get('issuing_authority', 'N/A')}")
            print(f"法规状态: {adapter.get('status', 'N/A')}")
            print(f"内容长度: {len(adapter.get('content', ''))} 字符")
        elif 'standard_id' in adapter:
            print(f"标准ID: {adapter.get('standard_id', 'N/A')}")
            print(f"标准编号: {adapter.get('standard_number', 'N/A')}")
            print(f"标准名称: {adapter.get('title', 'N/A')}")
            print(f"标准类型: {adapter.get('standard_type', 'N/A')}")
            print(f"发布机关: {adapter.get('issuing_authority', 'N/A')}")
            print(f"内容长度: {len(adapter.get('content', ''))} 字符")
        elif 'case_id' in adapter:
            print(f"案例ID: {adapter.get('case_id', 'N/A')}")
            print(f"案例标题: {adapter.get('title', 'N/A')}")
            print(f"严重程度: {adapter.get('severity_level', 'N/A')}")
            print(f"建筑类型: {adapter.get('building_type', 'N/A')}")
            print(f"火灾原因: {adapter.get('fire_cause', 'N/A')}")
            print(f"内容长度: {len(adapter.get('content', ''))} 字符")
        elif 'news_id' in adapter:
            print(f"新闻ID: {adapter.get('news_id', 'N/A')}")
            print(f"新闻标题: {adapter.get('title', 'N/A')}")
            print(f"新闻类型: {adapter.get('news_type', 'N/A')}")
            print(f"新闻来源: {adapter.get('source', 'N/A')}")
            print(f"发布时间: {adapter.get('publish_time', 'N/A')}")
            print(f"内容长度: {len(adapter.get('content', ''))} 字符")
        elif 'knowledge_id' in adapter:
            print(f"知识ID: {adapter.get('knowledge_id', 'N/A')}")
            print(f"知识标题: {adapter.get('title', 'N/A')}")
            print(f"知识类型: {adapter.get('knowledge_type', 'N/A')}")
            print(f"知识分类: {adapter.get('category', 'N/A')}")
            print(f"难度等级: {adapter.get('difficulty_level', 'N/A')}")
            print(f"内容长度: {adapter.get('content_length', 'N/A')} 字符")
        elif 'document_id' in adapter:
            print(f"文档ID: {adapter.get('document_id', 'N/A')}")
            print(f"文档标题: {adapter.get('title', 'N/A')}")
            print(f"文档类型: {adapter.get('document_type', 'N/A')}")
            print(f"内容长度: {adapter.get('content_length', 'N/A')} 字符")
            print(f"可读性得分: {adapter.get('readability_score', 'N/A')}")
            print(f"复杂度得分: {adapter.get('complexity_score', 'N/A')}")
        
        print(f"爬取时间: {adapter.get('scraped_at', 'N/A')}")
        print("=" * 30)
        
        return item


class FireTextAnalysisPipeline:
    """消防文本分析专用管道"""
    
    def __init__(self):
        self.analysis_stats = {
            'total_documents': 0,
            'total_chars': 0,
            'total_words': 0,
            'avg_readability': 0,
            'avg_complexity': 0,
            'fire_categories': {}
        }
    
    def process_item(self, item, spider):
        """处理文本分析item"""
        adapter = ItemAdapter(item)
        
        # 统计文档信息
        if 'content' in adapter:
            content = adapter.get('content', '')
            if content:
                self.analysis_stats['total_documents'] += 1
                self.analysis_stats['total_chars'] += len(content)
                self.analysis_stats['total_words'] += len(content.split())
                
                # 统计消防分类
                if 'category' in adapter:
                    category = adapter.get('category', '其他')
                    self.analysis_stats['fire_categories'][category] = self.analysis_stats['fire_categories'].get(category, 0) + 1
                
                # 每处理5条数据输出一次统计
                if self.analysis_stats['total_documents'] % 5 == 0:
                    self.print_analysis_stats(spider)
        
        return item
    
    def print_analysis_stats(self, spider):
        """打印文本分析统计"""
        spider.logger.info(f"消防文本分析统计 (共{self.analysis_stats['total_documents']}条文档):")
        spider.logger.info(f"  总字符数: {self.analysis_stats['total_chars']:,}")
        spider.logger.info(f"  总词数: {self.analysis_stats['total_words']:,}")
        spider.logger.info(f"  平均字符数: {self.analysis_stats['total_chars'] / self.analysis_stats['total_documents']:.0f}")
        spider.logger.info(f"  平均词数: {self.analysis_stats['total_words'] / self.analysis_stats['total_documents']:.0f}")
        
        if self.analysis_stats['fire_categories']:
            spider.logger.info("  消防分类统计:")
            for category, count in self.analysis_stats['fire_categories'].items():
                spider.logger.info(f"    {category}: {count} 条")
    
    def close_spider(self, spider):
        """爬虫结束时打印最终统计"""
        self.print_analysis_stats(spider)


class FireRAGPipeline:
    """消防RAG知识库管道"""
    
    def __init__(self):
        self.rag_stats = {
            'total_chunks': 0,
            'total_embeddings': 0,
            'document_types': {}
        }
    
    def process_item(self, item, spider):
        """处理RAG相关item"""
        adapter = ItemAdapter(item)
        
        # 处理RAG文档
        if 'chunk_texts' in adapter:
            chunk_texts = adapter.get('chunk_texts', [])
            if chunk_texts:
                self.rag_stats['total_chunks'] += len(chunk_texts)
                
                # 统计文档类型
                doc_type = adapter.get('document_type', 'unknown')
                self.rag_stats['document_types'][doc_type] = self.rag_stats['document_types'].get(doc_type, 0) + 1
                
                # 每处理3条数据输出一次统计
                if self.rag_stats['total_chunks'] % 15 == 0:  # 假设每条文档平均5个chunk
                    self.print_rag_stats(spider)
        
        return item
    
    def print_rag_stats(self, spider):
        """打印RAG统计"""
        spider.logger.info(f"消防RAG知识库统计:")
        spider.logger.info(f"  总文档块数: {self.rag_stats['total_chunks']}")
        spider.logger.info(f"  文档类型分布:")
        for doc_type, count in self.rag_stats['document_types'].items():
            spider.logger.info(f"    {doc_type}: {count} 条")
    
    def close_spider(self, spider):
        """爬虫结束时打印最终统计"""
        self.print_rag_stats(spider)


class FireDuplicatesPipeline:
    """消防数据去重管道"""
    
    def __init__(self):
        self.seen_items = set()
    
    def process_item(self, item, spider):
        """去除重复的item"""
        adapter = ItemAdapter(item)
        
        # 根据不同类型生成唯一键
        if 'regulation_id' in adapter:
            unique_key = ('regulation', adapter.get('regulation_id', ''))
        elif 'standard_id' in adapter:
            unique_key = ('standard', adapter.get('standard_id', ''))
        elif 'case_id' in adapter:
            unique_key = ('case', adapter.get('case_id', ''))
        elif 'news_id' in adapter:
            unique_key = ('news', adapter.get('news_id', ''))
        elif 'knowledge_id' in adapter:
            unique_key = ('knowledge', adapter.get('knowledge_id', ''))
        elif 'document_id' in adapter:
            unique_key = ('document', adapter.get('document_id', ''))
        else:
            return item
        
        if unique_key in self.seen_items:
            spider.logger.info(f"重复数据，跳过: {unique_key}")
            raise DropItem(f"重复数据: {unique_key}")
        else:
            self.seen_items.add(unique_key)
            return item
