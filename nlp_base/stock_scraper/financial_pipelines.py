#!/usr/bin/env python
"""
财报数据专用管道
"""

import pandas as pd
import json
import csv
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class FinancialExcelWriterPipeline:
    """财报数据Excel文件存储管道"""
    
    def __init__(self):
        self.items = {
            'financial_reports': [],
            'financial_data': [],
            'company_info': [],
            'text_analysis': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("财报Excel管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存Excel文件"""
        try:
            # 创建Excel写入器
            excel_filename = f'financial_data_{self.timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 保存财报数据
                if self.items['financial_reports']:
                    df_reports = pd.DataFrame(self.items['financial_reports'])
                    df_reports.to_excel(writer, sheet_name='财报数据', index=False)
                    spider.logger.info(f"财报数据: {len(self.items['financial_reports'])} 条")
                
                # 保存财务数据
                if self.items['financial_data']:
                    df_financial = pd.DataFrame(self.items['financial_data'])
                    df_financial.to_excel(writer, sheet_name='财务数据', index=False)
                    spider.logger.info(f"财务数据: {len(self.items['financial_data'])} 条")
                
                # 保存公司信息
                if self.items['company_info']:
                    df_companies = pd.DataFrame(self.items['company_info'])
                    df_companies.to_excel(writer, sheet_name='公司信息', index=False)
                    spider.logger.info(f"公司信息: {len(self.items['company_info'])} 条")
                
                # 保存文本分析数据
                if self.items['text_analysis']:
                    df_analysis = pd.DataFrame(self.items['text_analysis'])
                    df_analysis.to_excel(writer, sheet_name='文本分析', index=False)
                    spider.logger.info(f"文本分析: {len(self.items['text_analysis'])} 条")
            
            spider.logger.info(f"财报Excel文件已保存: {excel_filename}")
            
        except Exception as e:
            spider.logger.error(f"保存财报Excel文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'FinancialReportItem':
                self.items['financial_reports'].append(item_dict)
            elif class_name == 'FinancialDataItem':
                self.items['financial_data'].append(item_dict)
            elif class_name == 'CompanyInfoItem':
                self.items['company_info'].append(item_dict)
            elif class_name == 'TextAnalysisItem':
                self.items['text_analysis'].append(item_dict)
        
        return item


class FinancialCsvWriterPipeline:
    """财报数据CSV文件存储管道"""
    
    def __init__(self):
        self.items = {
            'financial_reports': [],
            'financial_data': [],
            'company_info': [],
            'text_analysis': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("财报CSV管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存CSV文件"""
        try:
            # 保存财报数据
            if self.items['financial_reports']:
                df_reports = pd.DataFrame(self.items['financial_reports'])
                df_reports.to_csv(f'financial_reports_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"财报数据CSV已保存: {len(self.items['financial_reports'])} 条")
            
            # 保存财务数据
            if self.items['financial_data']:
                df_financial = pd.DataFrame(self.items['financial_data'])
                df_financial.to_csv(f'financial_data_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"财务数据CSV已保存: {len(self.items['financial_data'])} 条")
            
            # 保存公司信息
            if self.items['company_info']:
                df_companies = pd.DataFrame(self.items['company_info'])
                df_companies.to_csv(f'company_info_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"公司信息CSV已保存: {len(self.items['company_info'])} 条")
            
            # 保存文本分析数据
            if self.items['text_analysis']:
                df_analysis = pd.DataFrame(self.items['text_analysis'])
                df_analysis.to_csv(f'text_analysis_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"文本分析CSV已保存: {len(self.items['text_analysis'])} 条")
                
        except Exception as e:
            spider.logger.error(f"保存财报CSV文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'FinancialReportItem':
                self.items['financial_reports'].append(item_dict)
            elif class_name == 'FinancialDataItem':
                self.items['financial_data'].append(item_dict)
            elif class_name == 'CompanyInfoItem':
                self.items['company_info'].append(item_dict)
            elif class_name == 'TextAnalysisItem':
                self.items['text_analysis'].append(item_dict)
        
        return item


class FinancialConsolePipeline:
    """财报数据控制台输出管道"""
    
    def process_item(self, item, spider):
        """在控制台输出item信息"""
        adapter = ItemAdapter(item)
        
        print(f"\n=== 财报数据 ===")
        
        # 根据数据类型显示不同信息
        if 'report_id' in adapter:
            print(f"报告ID: {adapter.get('report_id', 'N/A')}")
            print(f"股票代码: {adapter.get('stock_code', 'N/A')}")
            print(f"股票名称: {adapter.get('stock_name', 'N/A')}")
            print(f"报告类型: {adapter.get('report_type', 'N/A')}")
            print(f"报告期: {adapter.get('report_period', 'N/A')}")
            print(f"标题: {adapter.get('title', 'N/A')[:50]}...")
            print(f"内容长度: {adapter.get('content_length', 'N/A')} 字符")
            print(f"字数: {adapter.get('word_count', 'N/A')} 字")
        elif 'data_id' in adapter:
            print(f"数据ID: {adapter.get('data_id', 'N/A')}")
            print(f"股票代码: {adapter.get('stock_code', 'N/A')}")
            print(f"数据类型: {adapter.get('data_type', 'N/A')}")
            print(f"指标名称: {adapter.get('indicator_name', 'N/A')}")
            print(f"指标数值: {adapter.get('indicator_value', 'N/A')}")
        elif 'stock_code' in adapter and 'company_name' in adapter:
            print(f"股票代码: {adapter.get('stock_code', 'N/A')}")
            print(f"股票名称: {adapter.get('stock_name', 'N/A')}")
            print(f"公司名称: {adapter.get('company_name', 'N/A')}")
            print(f"所属行业: {adapter.get('industry', 'N/A')}")
            print(f"所属市场: {adapter.get('market', 'N/A')}")
        elif 'content_id' in adapter:
            print(f"内容ID: {adapter.get('content_id', 'N/A')}")
            print(f"内容类型: {adapter.get('content_type', 'N/A')}")
            print(f"字符数: {adapter.get('char_count', 'N/A')}")
            print(f"词数: {adapter.get('word_count', 'N/A')}")
            print(f"可读性得分: {adapter.get('readability_score', 'N/A')}")
            print(f"复杂度得分: {adapter.get('complexity_score', 'N/A')}")
        
        print(f"爬取时间: {adapter.get('scraped_at', 'N/A')}")
        print("=" * 30)
        
        return item


class TextAnalysisPipeline:
    """文本分析专用管道"""
    
    def __init__(self):
        self.analysis_stats = {
            'total_reports': 0,
            'total_chars': 0,
            'total_words': 0,
            'avg_readability': 0,
            'avg_complexity': 0
        }
    
    def process_item(self, item, spider):
        """处理文本分析item"""
        adapter = ItemAdapter(item)
        
        if 'content_type' in adapter and adapter.get('content_type') == 'report':
            self.analysis_stats['total_reports'] += 1
            self.analysis_stats['total_chars'] += adapter.get('char_count', 0)
            self.analysis_stats['total_words'] += adapter.get('word_count', 0)
            
            # 计算平均可读性和复杂度
            readability = adapter.get('readability_score', 0)
            complexity = adapter.get('complexity_score', 0)
            
            if self.analysis_stats['total_reports'] > 0:
                self.analysis_stats['avg_readability'] = self.analysis_stats['total_chars'] / self.analysis_stats['total_reports']
                self.analysis_stats['avg_complexity'] = self.analysis_stats['total_words'] / self.analysis_stats['total_reports']
            
            # 每处理5条数据输出一次统计
            if self.analysis_stats['total_reports'] % 5 == 0:
                self.print_analysis_stats(spider)
        
        return item
    
    def print_analysis_stats(self, spider):
        """打印文本分析统计"""
        spider.logger.info(f"文本分析统计 (共{self.analysis_stats['total_reports']}条财报):")
        spider.logger.info(f"  总字符数: {self.analysis_stats['total_chars']:,}")
        spider.logger.info(f"  总词数: {self.analysis_stats['total_words']:,}")
        spider.logger.info(f"  平均字符数: {self.analysis_stats['avg_readability']:.0f}")
        spider.logger.info(f"  平均词数: {self.analysis_stats['avg_complexity']:.0f}")
    
    def close_spider(self, spider):
        """爬虫结束时打印最终统计"""
        self.print_analysis_stats(spider)


class AIAnalysisPipeline:
    """AI分析管道"""
    
    def __init__(self, zhipu_api_key=None, kimi_api_key=None):
        self.zhipu_api_key = zhipu_api_key
        self.kimi_api_key = kimi_api_key
        self.analysis_count = 0
    
    def process_item(self, item, spider):
        """处理AI分析"""
        adapter = ItemAdapter(item)
        
        # 只对财报内容进行AI分析
        if 'content_type' in adapter and adapter.get('content_type') == 'report':
            content_text = adapter.get('content_text', '')
            
            if content_text and len(content_text) > 1000:  # 只分析长文本
                # 这里可以调用AI分析API
                # 由于演示版本，我们跳过实际的API调用
                self.analysis_count += 1
                
                if self.analysis_count % 3 == 0:  # 每3条数据输出一次
                    spider.logger.info(f"AI分析进度: 已分析 {self.analysis_count} 条财报")
        
        return item
