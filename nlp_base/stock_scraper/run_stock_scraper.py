#!/usr/bin/env python
"""
股票数据爬虫运行脚本
"""

import os
import sys
from datetime import datetime
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_scraper.spiders.eastmoney_spider import EastmoneySpiderSpider


def main():
    """主函数"""
    print("=== 股票数据爬虫 ===")
    print("正在启动爬虫...")
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 设置爬虫参数
    spider_kwargs = {
        'stock_codes': '000001,000002,600000,600036,000858',  # 股票代码
        'data_type': 'all'  # 数据类型：all, daily, realtime, dragon, capital
    }
    
    print(f"爬取参数:")
    print(f"  股票代码: {spider_kwargs['stock_codes']}")
    print(f"  数据类型: {spider_kwargs['data_type']}")
    print()
    
    # 启动爬虫
    process.crawl(EastmoneySpiderSpider, **spider_kwargs)
    process.start()


if __name__ == "__main__":
    main()
