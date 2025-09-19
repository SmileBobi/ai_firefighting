#!/usr/bin/env python
"""
车票信息爬虫运行脚本
"""

import os
import sys
from datetime import datetime, timedelta
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ticket_scraper.spiders.train_ticket import TrainTicketSpider


def main():
    """主函数"""
    print("=== 车票信息爬虫 ===")
    print("正在启动爬虫...")
    
    # 获取项目设置
    settings = get_project_settings()
    
    # 创建爬虫进程
    process = CrawlerProcess(settings)
    
    # 设置爬虫参数
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 可以修改这些参数
    spider_kwargs = {
        'from_station': 'BJP',  # 北京
        'to_station': 'SHH',    # 上海
        'date': tomorrow
    }
    
    print(f"查询参数:")
    print(f"  出发站: {spider_kwargs['from_station']}")
    print(f"  到达站: {spider_kwargs['to_station']}")
    print(f"  日期: {spider_kwargs['date']}")
    print()
    
    # 启动爬虫
    process.crawl(TrainTicketSpider, **spider_kwargs)
    process.start()


if __name__ == "__main__":
    main()
