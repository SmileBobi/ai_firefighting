#!/usr/bin/env python
"""
财报数据爬虫运行脚本
"""

import os
import sys
from datetime import datetime
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_scraper.spiders.financial_report_spider import FinancialReportSpiderSpider


def main():
    """主函数"""
    print("=== 财报数据爬虫 ===")
    print("正在启动爬虫...")
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 添加财报专用管道
    settings.set('ITEM_PIPELINES', {
        'financial_pipelines.FinancialExcelWriterPipeline': 400,
        'financial_pipelines.FinancialCsvWriterPipeline': 500,
        'financial_pipelines.FinancialConsolePipeline': 600,
        'financial_pipelines.TextAnalysisPipeline': 700,
        'financial_pipelines.AIAnalysisPipeline': 800,
    })
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 设置爬虫参数
    spider_kwargs = {
        'stock_codes': '000001,000002,600000,600036,000858',  # 股票代码
        'report_types': '年报,季报,中报',  # 报告类型
        'years': '2023,2024'  # 年份
    }
    
    print(f"爬取参数:")
    print(f"  股票代码: {spider_kwargs['stock_codes']}")
    print(f"  报告类型: {spider_kwargs['report_types']}")
    print(f"  年份: {spider_kwargs['years']}")
    print()
    
    # 启动爬虫
    process.crawl(FinancialReportSpiderSpider, **spider_kwargs)
    process.start()


if __name__ == "__main__":
    main()
