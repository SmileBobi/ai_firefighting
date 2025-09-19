#!/usr/bin/env python
"""
演示用的车票信息爬虫
使用模拟数据来演示Scrapy框架的使用
"""

import scrapy
import json
from datetime import datetime
from ticket_scraper.items import TrainTicketItem


class DemoTicketSpider(scrapy.Spider):
    name = "demo_ticket"
    allowed_domains = ["example.com"]
    start_urls = ["https://httpbin.org/json"]  # 使用一个返回JSON的测试API
    
    def parse(self, response):
        """解析响应并生成模拟的车票数据"""
        
        # 模拟一些车票数据
        mock_tickets = [
            {
                'train_number': 'G1',
                'departure_station': '北京南',
                'arrival_station': '上海虹桥',
                'departure_time': '06:00',
                'arrival_time': '11:30',
                'duration': '05:30',
                'seat_types': ['二等座', '一等座', '商务座'],
                'prices': ['553', '933', '1748'],
                'ticket_status': ['有', '有', '有'],
                'train_type': '高速动车',
                'distance': '1318'
            },
            {
                'train_number': 'G3',
                'departure_station': '北京南',
                'arrival_station': '上海虹桥',
                'departure_time': '07:00',
                'arrival_time': '12:30',
                'duration': '05:30',
                'seat_types': ['二等座', '一等座', '商务座'],
                'prices': ['553', '933', '1748'],
                'ticket_status': ['有', '有', '无'],
                'train_type': '高速动车',
                'distance': '1318'
            },
            {
                'train_number': 'D301',
                'departure_station': '北京南',
                'arrival_station': '上海虹桥',
                'departure_time': '08:00',
                'arrival_time': '14:30',
                'duration': '06:30',
                'seat_types': ['二等座', '一等座'],
                'prices': ['309', '495'],
                'ticket_status': ['有', '有'],
                'train_type': '动车组',
                'distance': '1318'
            },
            {
                'train_number': 'K101',
                'departure_station': '北京',
                'arrival_station': '上海',
                'departure_time': '20:00',
                'arrival_time': '10:30+1',
                'duration': '14:30',
                'seat_types': ['硬座', '硬卧', '软卧'],
                'prices': ['156', '279', '430'],
                'ticket_status': ['有', '有', '有'],
                'train_type': '快速列车',
                'distance': '1463'
            },
            {
                'train_number': 'T109',
                'departure_station': '北京',
                'arrival_station': '上海',
                'departure_time': '19:30',
                'arrival_time': '09:30+1',
                'duration': '14:00',
                'seat_types': ['硬座', '硬卧', '软卧'],
                'prices': ['177', '306', '478'],
                'ticket_status': ['有', '有', '无'],
                'train_type': '特快列车',
                'distance': '1463'
            }
        ]
        
        # 生成车票item
        for ticket_data in mock_tickets:
            item = TrainTicketItem()
            
            # 基本信息
            item['train_number'] = ticket_data['train_number']
            item['departure_station'] = ticket_data['departure_station']
            item['arrival_station'] = ticket_data['arrival_station']
            item['departure_time'] = ticket_data['departure_time']
            item['arrival_time'] = ticket_data['arrival_time']
            item['duration'] = ticket_data['duration']
            
            # 票价信息
            item['seat_types'] = ticket_data['seat_types']
            item['prices'] = ticket_data['prices']
            item['ticket_status'] = ticket_data['ticket_status']
            
            # 其他信息
            item['date'] = (datetime.now().strftime('%Y-%m-%d'))
            item['train_type'] = ticket_data['train_type']
            item['distance'] = ticket_data['distance']
            item['route_info'] = f"{ticket_data['departure_station']} → {ticket_data['arrival_station']}"
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
        
        self.logger.info(f"生成了 {len(mock_tickets)} 条模拟车票数据")
