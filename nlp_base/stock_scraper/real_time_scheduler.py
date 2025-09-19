#!/usr/bin/env python
"""
实时股票数据定时抓取脚本
每60秒抓取一次实时数据
"""

import time
import schedule
import subprocess
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run_realtime_scraper():
    """运行实时数据爬虫"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取实时数据...")
    
    try:
        # 运行实时数据爬虫
        cmd = [
            sys.executable, '-m', 'scrapy', 'crawl', 'eastmoney_spider',
            '-a', 'stock_codes=000001,000002,600000,600036,000858',
            '-a', 'data_type=realtime',
            '-s', 'LOG_LEVEL=WARNING'  # 减少日志输出
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 实时数据抓取完成")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 实时数据抓取失败: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行爬虫时出错: {e}")


def run_daily_scraper():
    """运行日交易数据爬虫"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取日交易数据...")
    
    try:
        # 运行日交易数据爬虫
        cmd = [
            sys.executable, '-m', 'scrapy', 'crawl', 'eastmoney_spider',
            '-a', 'stock_codes=000001,000002,600000,600036,000858',
            '-a', 'data_type=daily',
            '-s', 'LOG_LEVEL=WARNING'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 日交易数据抓取完成")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 日交易数据抓取失败: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行爬虫时出错: {e}")


def run_dragon_tiger_scraper():
    """运行龙虎榜数据爬虫"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取龙虎榜数据...")
    
    try:
        # 运行龙虎榜数据爬虫
        cmd = [
            sys.executable, '-m', 'scrapy', 'crawl', 'eastmoney_spider',
            '-a', 'data_type=dragon',
            '-s', 'LOG_LEVEL=WARNING'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 龙虎榜数据抓取完成")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 龙虎榜数据抓取失败: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行爬虫时出错: {e}")


def run_capital_flow_scraper():
    """运行资金流向数据爬虫"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取资金流向数据...")
    
    try:
        # 运行资金流向数据爬虫
        cmd = [
            sys.executable, '-m', 'scrapy', 'crawl', 'eastmoney_spider',
            '-a', 'stock_codes=000001,000002,600000,600036,000858',
            '-a', 'data_type=capital',
            '-s', 'LOG_LEVEL=WARNING'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 资金流向数据抓取完成")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 资金流向数据抓取失败: {result.stderr}")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 运行爬虫时出错: {e}")


def main():
    """主函数"""
    print("=== 股票数据定时抓取系统 ===")
    print("正在启动定时任务...")
    print()
    
    # 设置定时任务
    # 每60秒抓取一次实时数据
    schedule.every(60).seconds.do(run_realtime_scraper)
    
    # 每天9:30抓取日交易数据（开盘后）
    schedule.every().day.at("09:35").do(run_daily_scraper)
    
    # 每天15:30抓取龙虎榜数据（收盘后）
    schedule.every().day.at("15:35").do(run_dragon_tiger_scraper)
    
    # 每天16:00抓取资金流向数据
    schedule.every().day.at("16:00").do(run_capital_flow_scraper)
    
    print("定时任务已设置:")
    print("  - 实时数据: 每60秒抓取一次")
    print("  - 日交易数据: 每天09:35抓取")
    print("  - 龙虎榜数据: 每天15:35抓取")
    print("  - 资金流向数据: 每天16:00抓取")
    print()
    print("按 Ctrl+C 停止程序")
    print("=" * 50)
    
    # 立即执行一次实时数据抓取
    run_realtime_scraper()
    
    # 运行定时任务
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 程序已停止")


if __name__ == "__main__":
    # 检查是否安装了schedule库
    try:
        import schedule
    except ImportError:
        print("需要安装schedule库: pip install schedule")
        sys.exit(1)
    
    main()
