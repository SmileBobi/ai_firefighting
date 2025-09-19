import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urlencode
from ticket_scraper.items import TrainTicketItem


class TrainTicketSpider(scrapy.Spider):
    name = "train_ticket"
    allowed_domains = ["kyfw.12306.cn"]
    
    def __init__(self, from_station='BJP', to_station='SHH', date=None, *args, **kwargs):
        super(TrainTicketSpider, self).__init__(*args, **kwargs)
        
        # 设置默认参数
        self.from_station = from_station  # 北京
        self.to_station = to_station      # 上海
        self.date = date or (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 构建查询URL
        self.start_urls = [self.build_query_url()]
    
    def build_query_url(self):
        """构建查询URL"""
        base_url = "https://kyfw.12306.cn/otn/leftTicket/query"
        params = {
            'leftTicketDTO.train_date': self.date,
            'leftTicketDTO.from_station': self.from_station,
            'leftTicketDTO.to_station': self.to_station,
            'purpose_codes': 'ADULT'
        }
        return f"{base_url}?{urlencode(params)}"
    
    def start_requests(self):
        """开始请求"""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            )
    
    def parse(self, response):
        """解析响应"""
        try:
            data = json.loads(response.text)
            
            if data.get('status') and data.get('data'):
                result = data['data']['result']
                station_map = data['data']['map']
                
                for train_info in result:
                    item = self.parse_train_info(train_info, station_map)
                    if item:
                        yield item
            else:
                self.logger.warning(f"API返回异常: {data}")
                
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {response.text[:200]}")
        except Exception as e:
            self.logger.error(f"解析出错: {e}")
    
    def parse_train_info(self, train_data, station_map):
        """解析单条火车信息"""
        try:
            # 12306返回的数据格式是字符串，用|分隔
            fields = train_data.split('|')
            
            if len(fields) < 30:
                return None
            
            item = TrainTicketItem()
            
            # 基本信息
            item['train_number'] = fields[3]  # 车次
            item['departure_station'] = station_map.get(fields[6], fields[6])  # 出发站
            item['arrival_station'] = station_map.get(fields[7], fields[7])    # 到达站
            item['departure_time'] = fields[8]   # 出发时间
            item['arrival_time'] = fields[9]     # 到达时间
            item['duration'] = fields[10]        # 运行时长
            
            # 票价信息 (不同座位类型)
            seat_types = []
            prices = []
            
            # 硬座
            if fields[29] and fields[29] != '':
                seat_types.append('硬座')
                prices.append(fields[29])
            
            # 软座
            if fields[24] and fields[24] != '':
                seat_types.append('软座')
                prices.append(fields[24])
            
            # 硬卧
            if fields[28] and fields[28] != '':
                seat_types.append('硬卧')
                prices.append(fields[28])
            
            # 软卧
            if fields[23] and fields[23] != '':
                seat_types.append('软卧')
                prices.append(fields[23])
            
            # 二等座
            if fields[30] and fields[30] != '':
                seat_types.append('二等座')
                prices.append(fields[30])
            
            # 一等座
            if fields[31] and fields[31] != '':
                seat_types.append('一等座')
                prices.append(fields[31])
            
            # 商务座
            if fields[32] and fields[32] != '':
                seat_types.append('商务座')
                prices.append(fields[32])
            
            item['seat_types'] = seat_types
            item['prices'] = prices
            
            # 余票状态
            ticket_status = []
            for i in range(23, 33):  # 余票状态字段
                if fields[i] and fields[i] not in ['', '--']:
                    ticket_status.append(fields[i])
            item['ticket_status'] = ticket_status
            
            # 其他信息
            item['date'] = self.date
            item['train_type'] = self.get_train_type(fields[3])
            item['distance'] = fields[1] if fields[1] else ''
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            return item
            
        except Exception as e:
            self.logger.error(f"解析火车信息出错: {e}")
            return None
    
    def get_train_type(self, train_number):
        """根据车次号判断车型"""
        if train_number.startswith('G'):
            return '高速动车'
        elif train_number.startswith('D'):
            return '动车组'
        elif train_number.startswith('C'):
            return '城际列车'
        elif train_number.startswith('K'):
            return '快速列车'
        elif train_number.startswith('T'):
            return '特快列车'
        elif train_number.startswith('Z'):
            return '直达特快'
        else:
            return '普通列车'
