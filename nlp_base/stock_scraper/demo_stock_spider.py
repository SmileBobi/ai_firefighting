#!/usr/bin/env python
"""
演示用的股票数据爬虫
使用模拟数据来演示Scrapy框架的股票数据抓取功能
"""

import scrapy
import json
from datetime import datetime
from stock_scraper.items import DailyStockItem, RealTimeStockItem, DragonTigerItem, CapitalFlowItem


class DemoStockSpider(scrapy.Spider):
    name = "demo_stock"
    allowed_domains = ["example.com"]
    start_urls = ["https://httpbin.org/json"]  # 使用一个返回JSON的测试API
    
    def parse(self, response):
        """解析响应并生成模拟的股票数据"""
        
        # 生成日交易数据
        daily_stocks = [
            {
                'stock_code': '000001',
                'stock_name': '平安银行',
                'date': '2025-09-17',
                'open_price': '12.50',
                'close_price': '12.80',
                'high_price': '12.95',
                'low_price': '12.45',
                'prev_close': '12.60',
                'volume': '12500000',
                'turnover': '160000000',
                'amplitude': '3.97',
                'change_rate': '1.59',
                'change_amount': '0.20',
                'pe_ratio': '5.2',
                'pb_ratio': '0.8',
                'market_cap': '248000000000',
                'circulating_cap': '248000000000'
            },
            {
                'stock_code': '000002',
                'stock_name': '万科A',
                'date': '2025-09-17',
                'open_price': '8.20',
                'close_price': '8.35',
                'high_price': '8.45',
                'low_price': '8.15',
                'prev_close': '8.30',
                'volume': '8500000',
                'turnover': '71000000',
                'amplitude': '3.61',
                'change_rate': '0.60',
                'change_amount': '0.05',
                'pe_ratio': '6.8',
                'pb_ratio': '0.9',
                'market_cap': '92000000000',
                'circulating_cap': '92000000000'
            },
            {
                'stock_code': '600000',
                'stock_name': '浦发银行',
                'date': '2025-09-17',
                'open_price': '7.80',
                'close_price': '7.95',
                'high_price': '8.05',
                'low_price': '7.75',
                'prev_close': '7.85',
                'volume': '15000000',
                'turnover': '119000000',
                'amplitude': '3.82',
                'change_rate': '1.27',
                'change_amount': '0.10',
                'pe_ratio': '4.5',
                'pb_ratio': '0.6',
                'market_cap': '232000000000',
                'circulating_cap': '232000000000'
            }
        ]
        
        # 生成实时数据
        realtime_stocks = [
            {
                'stock_code': '000001',
                'stock_name': '平安银行',
                'timestamp': '2025-09-17 14:30:00',
                'current_price': '12.82',
                'change_rate': '1.75',
                'change_amount': '0.22',
                'volume': '12550000',
                'turnover': '160800000',
                'bid_price': '12.81',
                'ask_price': '12.82',
                'bid_volume': '5000',
                'ask_volume': '3000'
            },
            {
                'stock_code': '000002',
                'stock_name': '万科A',
                'timestamp': '2025-09-17 14:30:00',
                'current_price': '8.38',
                'change_rate': '0.96',
                'change_amount': '0.08',
                'volume': '8600000',
                'turnover': '72000000',
                'bid_price': '8.37',
                'ask_price': '8.38',
                'bid_volume': '8000',
                'ask_volume': '2000'
            }
        ]
        
        # 生成龙虎榜数据
        dragon_tiger_stocks = [
            {
                'stock_code': '300750',
                'stock_name': '宁德时代',
                'date': '2025-09-17',
                'reason': '日涨幅偏离值达7%的证券',
                'reason_detail': '连续三个交易日内收盘价涨幅偏离值累计达到20%',
                'buy_amount': '125000000',
                'sell_amount': '98000000',
                'net_amount': '27000000',
                'buy_seats': '机构专用,深股通专用,机构专用',
                'sell_seats': '机构专用,深股通专用',
                'turnover_rate': '2.5',
                'price_limit': '8.5'
            },
            {
                'stock_code': '002594',
                'stock_name': '比亚迪',
                'date': '2025-09-17',
                'reason': '日换手率达到20%的证券',
                'reason_detail': '当日换手率达到20%',
                'buy_amount': '89000000',
                'sell_amount': '112000000',
                'net_amount': '-23000000',
                'buy_seats': '深股通专用,机构专用',
                'sell_seats': '机构专用,深股通专用,机构专用',
                'turnover_rate': '22.3',
                'price_limit': '5.2'
            }
        ]
        
        # 生成资金流向数据
        capital_flow_stocks = [
            {
                'stock_code': '000001',
                'stock_name': '平安银行',
                'date': '2025-09-17',
                'main_inflow': '25000000',
                'main_inflow_rate': '15.6',
                'super_large_inflow': '18000000',
                'super_large_inflow_rate': '11.2',
                'large_inflow': '7000000',
                'large_inflow_rate': '4.4',
                'medium_inflow': '-5000000',
                'medium_inflow_rate': '-3.1',
                'small_inflow': '-20000000',
                'small_inflow_rate': '-12.5',
                'current_price': '12.80',
                'change_rate': '1.59'
            },
            {
                'stock_code': '000002',
                'stock_name': '万科A',
                'date': '2025-09-17',
                'main_inflow': '-12000000',
                'main_inflow_rate': '-16.9',
                'super_large_inflow': '-8000000',
                'super_large_inflow_rate': '-11.3',
                'large_inflow': '-4000000',
                'large_inflow_rate': '-5.6',
                'medium_inflow': '3000000',
                'medium_inflow_rate': '4.2',
                'small_inflow': '9000000',
                'small_inflow_rate': '12.7',
                'current_price': '8.35',
                'change_rate': '0.60'
            }
        ]
        
        # 生成日交易数据
        for stock_data in daily_stocks:
            item = DailyStockItem()
            for key, value in stock_data.items():
                item[key] = value
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            yield item
        
        # 生成实时数据
        for stock_data in realtime_stocks:
            item = RealTimeStockItem()
            for key, value in stock_data.items():
                item[key] = value
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            yield item
        
        # 生成龙虎榜数据
        for stock_data in dragon_tiger_stocks:
            item = DragonTigerItem()
            for key, value in stock_data.items():
                item[key] = value
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            yield item
        
        # 生成资金流向数据
        for stock_data in capital_flow_stocks:
            item = CapitalFlowItem()
            for key, value in stock_data.items():
                item[key] = value
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            yield item
        
        self.logger.info(f"生成了 {len(daily_stocks)} 条日交易数据")
        self.logger.info(f"生成了 {len(realtime_stocks)} 条实时数据")
        self.logger.info(f"生成了 {len(dragon_tiger_stocks)} 条龙虎榜数据")
        self.logger.info(f"生成了 {len(capital_flow_stocks)} 条资金流向数据")
