#!/usr/bin/env python
"""
消防数据爬虫运行脚本
"""

import os
import sys
from datetime import datetime
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fire_scraper.spiders.fire_regulation_spider import FireRegulationSpiderSpider


def main():
    """主函数"""
    print("=== 消防数据爬虫 ===")
    print("正在启动爬虫...")
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 添加消防专用管道
    settings.set('ITEM_PIPELINES', {
        'fire_pipelines.FireExcelWriterPipeline': 400,
        'fire_pipelines.FireCsvWriterPipeline': 500,
        'fire_pipelines.FireConsolePipeline': 600,
        'fire_pipelines.FireTextAnalysisPipeline': 700,
        'fire_pipelines.FireRAGPipeline': 800,
        'fire_pipelines.FireDuplicatesPipeline': 900,
    })
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 设置爬虫参数
    spider_kwargs = {
        'data_types': 'regulation,standard,case,news',  # 数据类型
        'max_pages': 3  # 最大页数
    }
    
    print(f"爬取参数:")
    print(f"  数据类型: {spider_kwargs['data_types']}")
    print(f"  最大页数: {spider_kwargs['max_pages']}")
    print()
    
    # 启动爬虫
    process.crawl(FireRegulationSpiderSpider, **spider_kwargs)
    process.start()


if __name__ == "__main__":
    main()
