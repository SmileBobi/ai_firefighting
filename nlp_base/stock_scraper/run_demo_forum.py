#!/usr/bin/env python
"""
演示论坛数据爬虫运行脚本
"""

import os
import sys
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from demo_forum_spider import DemoForumSpider


def main():
    """主函数"""
    print("=== 论坛数据爬虫演示 ===")
    print("正在启动演示爬虫...")
    print("注意：这是使用模拟数据的演示版本")
    print()
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 添加论坛专用管道
    settings.set('ITEM_PIPELINES', {
        'forum_pipelines.ForumExcelWriterPipeline': 400,
        'forum_pipelines.ForumCsvWriterPipeline': 500,
        'forum_pipelines.ForumConsolePipeline': 600,
        'forum_pipelines.SentimentAnalysisPipeline': 700,
    })
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 启动演示爬虫
    process.crawl(DemoForumSpider)
    process.start()


if __name__ == "__main__":
    main()
