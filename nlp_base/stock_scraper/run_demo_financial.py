#!/usr/bin/env python
"""
演示财报数据爬虫运行脚本
"""

import os
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_financial_spider import DemoFinancialSpider


def main():
    """主函数"""
    print("=== 财报数据爬虫演示 ===")
    print("正在启动演示爬虫...")
    print("注意：这是使用模拟数据的演示版本")
    print()
    
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
    
    # 启动演示爬虫
    process.crawl(DemoFinancialSpider)
    process.start()


if __name__ == "__main__":
    main()
