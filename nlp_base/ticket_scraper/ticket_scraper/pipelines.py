# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json
import csv
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class JsonWriterPipeline:
    """JSON文件存储管道"""
    
    def __init__(self):
        self.file = None
        self.items = []
    
    def open_spider(self, spider):
        """爬虫开始时打开文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'train_tickets_{timestamp}.json'
        self.file = open(filename, 'w', encoding='utf-8')
        self.file.write('[\n')
    
    def close_spider(self, spider):
        """爬虫结束时关闭文件"""
        if self.file:
            # 移除最后一个逗号
            if self.items:
                self.file.seek(self.file.tell() - 2)
                self.file.truncate()
            self.file.write('\n]')
            self.file.close()
            spider.logger.info(f"数据已保存到 {self.file.name}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 写入JSON格式
        json.dump(item_dict, self.file, ensure_ascii=False, indent=2)
        self.file.write(',\n')
        
        self.items.append(item_dict)
        return item


class CsvWriterPipeline:
    """CSV文件存储管道"""
    
    def __init__(self):
        self.file = None
        self.writer = None
        self.items = []
    
    def open_spider(self, spider):
        """爬虫开始时打开文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'train_tickets_{timestamp}.csv'
        self.file = open(filename, 'w', newline='', encoding='utf-8-sig')
        self.writer = csv.writer(self.file)
        
        # 写入表头
        headers = [
            '车次', '出发站', '到达站', '出发时间', '到达时间', '运行时长',
            '座位类型', '价格', '余票状态', '日期', '车型', '距离', '爬取时间'
        ]
        self.writer.writerow(headers)
    
    def close_spider(self, spider):
        """爬虫结束时关闭文件"""
        if self.file:
            self.file.close()
            spider.logger.info(f"CSV数据已保存到 {self.file.name}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        
        # 处理座位类型和价格
        seat_types = adapter.get('seat_types', [])
        prices = adapter.get('prices', [])
        seat_price_str = '; '.join([f"{seat}:{price}" for seat, price in zip(seat_types, prices)])
        
        # 处理余票状态
        ticket_status = adapter.get('ticket_status', [])
        ticket_status_str = '; '.join(ticket_status)
        
        # 写入CSV行
        row = [
            adapter.get('train_number', ''),
            adapter.get('departure_station', ''),
            adapter.get('arrival_station', ''),
            adapter.get('departure_time', ''),
            adapter.get('arrival_time', ''),
            adapter.get('duration', ''),
            seat_price_str,
            ticket_status_str,
            adapter.get('date', ''),
            adapter.get('train_type', ''),
            adapter.get('distance', ''),
            adapter.get('scraped_at', '')
        ]
        
        self.writer.writerow(row)
        self.items.append(dict(adapter))
        return item


class ConsolePipeline:
    """控制台输出管道"""
    
    def process_item(self, item, spider):
        """在控制台输出item信息"""
        adapter = ItemAdapter(item)
        
        print(f"\n=== 车票信息 ===")
        print(f"车次: {adapter.get('train_number', 'N/A')}")
        print(f"路线: {adapter.get('departure_station', 'N/A')} → {adapter.get('arrival_station', 'N/A')}")
        print(f"时间: {adapter.get('departure_time', 'N/A')} - {adapter.get('arrival_time', 'N/A')}")
        print(f"车型: {adapter.get('train_type', 'N/A')}")
        
        # 显示座位和价格
        seat_types = adapter.get('seat_types', [])
        prices = adapter.get('prices', [])
        if seat_types and prices:
            print("票价信息:")
            for seat, price in zip(seat_types, prices):
                print(f"  {seat}: {price}元")
        
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
        
        # 使用车次+日期+出发时间作为唯一标识
        unique_key = (
            adapter.get('train_number', ''),
            adapter.get('date', ''),
            adapter.get('departure_time', '')
        )
        
        if unique_key in self.seen_items:
            spider.logger.info(f"重复数据，跳过: {unique_key}")
            raise DropItem(f"重复数据: {unique_key}")
        else:
            self.seen_items.add(unique_key)
            return item
