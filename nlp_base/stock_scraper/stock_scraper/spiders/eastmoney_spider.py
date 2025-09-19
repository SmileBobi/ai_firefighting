import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode
from stock_scraper.items import DailyStockItem, RealTimeStockItem, DragonTigerItem, CapitalFlowItem


class EastmoneySpiderSpider(scrapy.Spider):
    name = "eastmoney_spider"
    allowed_domains = ["eastmoney.com", "push2.eastmoney.com"]
    
    def __init__(self, stock_codes='000001,000002,600000,600036', data_type='all', *args, **kwargs):
        super(EastmoneySpiderSpider, self).__init__(*args, **kwargs)
        
        # 设置股票代码列表
        self.stock_codes = stock_codes.split(',')
        self.data_type = data_type  # all, daily, realtime, dragon, capital
        
        # 构建起始URL列表
        self.start_urls = self.build_start_urls()
    
    def build_start_urls(self):
        """构建起始URL列表"""
        urls = []
        
        if self.data_type in ['all', 'daily']:
            # 日交易数据URL
            for code in self.stock_codes:
                url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={self.get_secid(code)}&fields=f57,f58,f107,f137,f46,f44,f45,f47,f48,f60,f170,f116,f60,f44,f45,f47,f48"
                urls.append(url)
        
        if self.data_type in ['all', 'realtime']:
            # 实时数据URL
            for code in self.stock_codes:
                url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={self.get_secid(code)}&fields=f57,f58,f107,f137,f46,f44,f45,f47,f48,f60,f170,f116"
                urls.append(url)
        
        if self.data_type in ['all', 'dragon']:
            # 龙虎榜数据URL
            urls.append("https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=20&po=1&np=1&ut=b2884a393a59ad64002292a3e90d46a5&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f1,f2,f3,f12,f13,f14,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205,f124,f1,f3")
        
        if self.data_type in ['all', 'capital']:
            # 资金流向数据URL
            for code in self.stock_codes:
                url = f"https://push2.eastmoney.com/api/qt/stock/fflow/kline/get?lmt=0&klt=1&secid={self.get_secid(code)}&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
                urls.append(url)
        
        return urls
    
    def get_secid(self, stock_code):
        """获取股票的安全ID"""
        if stock_code.startswith('6'):
            return f"1.{stock_code}"  # 上海
        else:
            return f"0.{stock_code}"  # 深圳
    
    def start_requests(self):
        """开始请求"""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://quote.eastmoney.com/',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            )

    def parse(self, response):
        """解析响应"""
        try:
            data = json.loads(response.text)
            
            # 根据URL类型解析不同数据
            if 'stock/get' in response.url:
                self.parse_stock_data(data, response)
            elif 'clist/get' in response.url:
                self.parse_dragon_tiger_data(data, response)
            elif 'fflow/kline/get' in response.url:
                self.parse_capital_flow_data(data, response)
                
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {response.text[:200]}")
        except Exception as e:
            self.logger.error(f"解析出错: {e}")
    
    def parse_stock_data(self, data, response):
        """解析股票数据"""
        try:
            if 'data' in data and data['data']:
                stock_data = data['data']
                
                # 判断是日交易数据还是实时数据
                if 'f60' in stock_data:  # 实时数据
                    item = RealTimeStockItem()
                    item['stock_code'] = stock_data.get('f57', '')
                    item['stock_name'] = stock_data.get('f58', '')
                    item['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    item['current_price'] = stock_data.get('f43', '')
                    item['change_rate'] = stock_data.get('f170', '')
                    item['change_amount'] = stock_data.get('f169', '')
                    item['volume'] = stock_data.get('f47', '')
                    item['turnover'] = stock_data.get('f48', '')
                    item['bid_price'] = stock_data.get('f9', '')
                    item['ask_price'] = stock_data.get('f10', '')
                    item['bid_volume'] = stock_data.get('f11', '')
                    item['ask_volume'] = stock_data.get('f12', '')
                else:  # 日交易数据
                    item = DailyStockItem()
                    item['stock_code'] = stock_data.get('f57', '')
                    item['stock_name'] = stock_data.get('f58', '')
                    item['date'] = datetime.now().strftime('%Y-%m-%d')
                    item['open_price'] = stock_data.get('f46', '')
                    item['close_price'] = stock_data.get('f43', '')
                    item['high_price'] = stock_data.get('f44', '')
                    item['low_price'] = stock_data.get('f45', '')
                    item['prev_close'] = stock_data.get('f60', '')
                    item['volume'] = stock_data.get('f47', '')
                    item['turnover'] = stock_data.get('f48', '')
                    item['amplitude'] = stock_data.get('f116', '')
                    item['change_rate'] = stock_data.get('f170', '')
                    item['change_amount'] = stock_data.get('f169', '')
                    item['pe_ratio'] = stock_data.get('f114', '')
                    item['pb_ratio'] = stock_data.get('f115', '')
                    item['market_cap'] = stock_data.get('f116', '')
                    item['circulating_cap'] = stock_data.get('f117', '')
                
                # 通用字段
                item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                item['source_url'] = response.url
                
                yield item
                
        except Exception as e:
            self.logger.error(f"解析股票数据出错: {e}")
    
    def parse_dragon_tiger_data(self, data, response):
        """解析龙虎榜数据"""
        try:
            if 'data' in data and 'diff' in data['data']:
                for item_data in data['data']['diff']:
                    item = DragonTigerItem()
                    item['stock_code'] = item_data.get('f12', '')
                    item['stock_name'] = item_data.get('f14', '')
                    item['date'] = datetime.now().strftime('%Y-%m-%d')
                    item['reason'] = item_data.get('f184', '')
                    item['buy_amount'] = item_data.get('f62', '')
                    item['sell_amount'] = item_data.get('f66', '')
                    item['net_amount'] = item_data.get('f69', '')
                    item['turnover_rate'] = item_data.get('f8', '')
                    item['price_limit'] = item_data.get('f3', '')
                    
                    item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    item['source_url'] = response.url
                    
                    yield item
                    
        except Exception as e:
            self.logger.error(f"解析龙虎榜数据出错: {e}")
    
    def parse_capital_flow_data(self, data, response):
        """解析资金流向数据"""
        try:
            if 'data' in data and 'klines' in data['data']:
                for kline_data in data['data']['klines']:
                    fields = kline_data.split(',')
                    if len(fields) >= 8:
                        item = CapitalFlowItem()
                        item['stock_code'] = self.extract_stock_code_from_url(response.url)
                        item['date'] = fields[0]
                        item['main_inflow'] = fields[1]
                        item['main_inflow_rate'] = fields[2]
                        item['super_large_inflow'] = fields[3]
                        item['super_large_inflow_rate'] = fields[4]
                        item['large_inflow'] = fields[5]
                        item['large_inflow_rate'] = fields[6]
                        item['medium_inflow'] = fields[7]
                        item['medium_inflow_rate'] = fields[8] if len(fields) > 8 else ''
                        item['small_inflow'] = fields[9] if len(fields) > 9 else ''
                        item['small_inflow_rate'] = fields[10] if len(fields) > 10 else ''
                        
                        item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        item['source_url'] = response.url
                        
                        yield item
                        
        except Exception as e:
            self.logger.error(f"解析资金流向数据出错: {e}")
    
    def extract_stock_code_from_url(self, url):
        """从URL中提取股票代码"""
        match = re.search(r'secid=(\d+)\.(\d+)', url)
        if match:
            return match.group(2)
        return ''
