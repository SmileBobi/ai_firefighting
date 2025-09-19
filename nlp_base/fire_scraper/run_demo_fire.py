#!/usr/bin/env python
"""
演示消防数据爬虫运行脚本
"""

import os
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_fire_spider import DemoFireSpider


def main():
    """主函数"""
    print("=== 消防数据爬虫演示 ===")
    print("正在启动演示爬虫...")
    print("注意：这是使用模拟数据的演示版本")
    print()
    
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
    
    # 启动演示爬虫
    process.crawl(DemoFireSpider)
    process.start()


if __name__ == "__main__":
    main()
