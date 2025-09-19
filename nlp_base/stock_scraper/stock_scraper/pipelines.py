# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import pandas as pd
import json
import csv
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class ExcelWriterPipeline:
    """Excel文件存储管道"""
    
    def __init__(self):
        self.items = {
            'daily_stock': [],
            'realtime_stock': [],
            'dragon_tiger': [],
            'capital_flow': [],
            'stock_news': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("Excel管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存Excel文件"""
        try:
            # 创建Excel写入器
            excel_filename = f'stock_data_{self.timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 保存日交易数据
                if self.items['daily_stock']:
                    df_daily = pd.DataFrame(self.items['daily_stock'])
                    df_daily.to_excel(writer, sheet_name='日交易数据', index=False)
                    spider.logger.info(f"日交易数据: {len(self.items['daily_stock'])} 条")
                
                # 保存实时数据
                if self.items['realtime_stock']:
                    df_realtime = pd.DataFrame(self.items['realtime_stock'])
                    df_realtime.to_excel(writer, sheet_name='实时数据', index=False)
                    spider.logger.info(f"实时数据: {len(self.items['realtime_stock'])} 条")
                
                # 保存龙虎榜数据
                if self.items['dragon_tiger']:
                    df_dragon = pd.DataFrame(self.items['dragon_tiger'])
                    df_dragon.to_excel(writer, sheet_name='龙虎榜', index=False)
                    spider.logger.info(f"龙虎榜数据: {len(self.items['dragon_tiger'])} 条")
                
                # 保存资金流向数据
                if self.items['capital_flow']:
                    df_capital = pd.DataFrame(self.items['capital_flow'])
                    df_capital.to_excel(writer, sheet_name='资金流向', index=False)
                    spider.logger.info(f"资金流向数据: {len(self.items['capital_flow'])} 条")
                
                # 保存新闻数据
                if self.items['stock_news']:
                    df_news = pd.DataFrame(self.items['stock_news'])
                    df_news.to_excel(writer, sheet_name='股票新闻', index=False)
                    spider.logger.info(f"新闻数据: {len(self.items['stock_news'])} 条")
            
            spider.logger.info(f"Excel文件已保存: {excel_filename}")
            
        except Exception as e:
            spider.logger.error(f"保存Excel文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'DailyStockItem':
                self.items['daily_stock'].append(item_dict)
            elif class_name == 'RealTimeStockItem':
                self.items['realtime_stock'].append(item_dict)
            elif class_name == 'DragonTigerItem':
                self.items['dragon_tiger'].append(item_dict)
            elif class_name == 'CapitalFlowItem':
                self.items['capital_flow'].append(item_dict)
            elif class_name == 'StockNewsItem':
                self.items['stock_news'].append(item_dict)
        
        return item


class JsonWriterPipeline:
    """JSON文件存储管道"""
    
    def __init__(self):
        self.items = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("JSON管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存JSON文件"""
        if self.items:
            filename = f'stock_data_{self.timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
            spider.logger.info(f"JSON文件已保存: {filename}, 共 {len(self.items)} 条数据")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        self.items.append(item_dict)
        return item


class CsvWriterPipeline:
    """CSV文件存储管道"""
    
    def __init__(self):
        self.items = {
            'daily_stock': [],
            'realtime_stock': [],
            'dragon_tiger': [],
            'capital_flow': [],
            'stock_news': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("CSV管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存CSV文件"""
        try:
            # 保存日交易数据
            if self.items['daily_stock']:
                df_daily = pd.DataFrame(self.items['daily_stock'])
                df_daily.to_csv(f'daily_stock_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"日交易CSV已保存: {len(self.items['daily_stock'])} 条")
            
            # 保存实时数据
            if self.items['realtime_stock']:
                df_realtime = pd.DataFrame(self.items['realtime_stock'])
                df_realtime.to_csv(f'realtime_stock_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"实时数据CSV已保存: {len(self.items['realtime_stock'])} 条")
            
            # 保存龙虎榜数据
            if self.items['dragon_tiger']:
                df_dragon = pd.DataFrame(self.items['dragon_tiger'])
                df_dragon.to_csv(f'dragon_tiger_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"龙虎榜CSV已保存: {len(self.items['dragon_tiger'])} 条")
            
            # 保存资金流向数据
            if self.items['capital_flow']:
                df_capital = pd.DataFrame(self.items['capital_flow'])
                df_capital.to_csv(f'capital_flow_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"资金流向CSV已保存: {len(self.items['capital_flow'])} 条")
            
            # 保存新闻数据
            if self.items['stock_news']:
                df_news = pd.DataFrame(self.items['stock_news'])
                df_news.to_csv(f'stock_news_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"新闻CSV已保存: {len(self.items['stock_news'])} 条")
                
        except Exception as e:
            spider.logger.error(f"保存CSV文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'DailyStockItem':
                self.items['daily_stock'].append(item_dict)
            elif class_name == 'RealTimeStockItem':
                self.items['realtime_stock'].append(item_dict)
            elif class_name == 'DragonTigerItem':
                self.items['dragon_tiger'].append(item_dict)
            elif class_name == 'CapitalFlowItem':
                self.items['capital_flow'].append(item_dict)
            elif class_name == 'StockNewsItem':
                self.items['stock_news'].append(item_dict)
        
        return item


class ConsolePipeline:
    """控制台输出管道"""
    
    def process_item(self, item, spider):
        """在控制台输出item信息"""
        adapter = ItemAdapter(item)
        
        print(f"\n=== 股票数据 ===")
        print(f"股票代码: {adapter.get('stock_code', 'N/A')}")
        print(f"股票名称: {adapter.get('stock_name', 'N/A')}")
        
        # 根据数据类型显示不同信息
        if 'current_price' in adapter:
            print(f"当前价格: {adapter.get('current_price', 'N/A')}")
            print(f"涨跌幅: {adapter.get('change_rate', 'N/A')}")
        elif 'close_price' in adapter:
            print(f"收盘价: {adapter.get('close_price', 'N/A')}")
            print(f"涨跌幅: {adapter.get('change_rate', 'N/A')}")
        
        print(f"爬取时间: {adapter.get('scraped_at', 'N/A')}")
        print("=" * 30)
        
        return item


class DuplicatesPipeline:
    """去重管道"""
    
    def __init__(self):
        self.seen_items = set()
    
    def process_item(self, item, spider):
        """去除重复的item"""
        adapter = ItemAdapter(item)
        
        # 使用股票代码+时间戳作为唯一标识
        unique_key = (
            adapter.get('stock_code', ''),
            adapter.get('timestamp', adapter.get('date', '')),
            adapter.get('scraped_at', '')
        )
        
        if unique_key in self.seen_items:
            spider.logger.info(f"重复数据，跳过: {unique_key}")
            raise DropItem(f"重复数据: {unique_key}")
        else:
            self.seen_items.add(unique_key)
        return item
