#!/usr/bin/env python
"""
演示消防数据爬虫运行脚本（支持MySQL）
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
    print("=== 消防数据爬虫演示（MySQL版本） ===")
    print("正在启动演示爬虫...")
    print("注意：这是使用模拟数据的演示版本，数据将同时保存到Excel、CSV和MySQL")
    print()
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 添加消防专用管道（包括MySQL管道）
    settings.set('ITEM_PIPELINES', {
        'fire_pipelines.FireExcelWriterPipeline': 400,
        'fire_pipelines.FireCsvWriterPipeline': 500,
        'fire_pipelines.FireConsolePipeline': 600,
        'fire_pipelines.FireTextAnalysisPipeline': 700,
        'fire_pipelines.FireRAGPipeline': 800,
        'fire_pipelines.FireDuplicatesPipeline': 900,
        'mysql_pipeline.MySQLPipeline': 1000,
    })
    
    # 设置MySQL配置（如果环境变量存在则使用，否则使用默认值）
    settings.set('MYSQL_HOST', os.getenv('MYSQL_HOST', 'localhost'))
    settings.set('MYSQL_PORT', int(os.getenv('MYSQL_PORT', '3306')))
    settings.set('MYSQL_USER', os.getenv('MYSQL_USER', 'root'))
    settings.set('MYSQL_PASSWORD', os.getenv('MYSQL_PASSWORD', ''))
    settings.set('MYSQL_DATABASE', os.getenv('MYSQL_DATABASE', 'fire_data'))
    
    print("MySQL配置:")
    print(f"  主机: {settings.get('MYSQL_HOST')}")
    print(f"  端口: {settings.get('MYSQL_PORT')}")
    print(f"  用户: {settings.get('MYSQL_USER')}")
    print(f"  数据库: {settings.get('MYSQL_DATABASE')}")
    print()
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 启动演示爬虫
    process.crawl(DemoFireSpider)
    process.start()


if __name__ == "__main__":
    main()

