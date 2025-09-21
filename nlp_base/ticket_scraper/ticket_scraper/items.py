# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TicketScraperItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


class TrainTicketItem(scrapy.Item):
    """火车票信息Item"""
    
    # 基本信息
    train_number = scrapy.Field()          # 车次
    departure_station = scrapy.Field()     # 出发站
    arrival_station = scrapy.Field()       # 到达站
    departure_time = scrapy.Field()        # 出发时间
    arrival_time = scrapy.Field()          # 到达时间
    duration = scrapy.Field()              # 运行时长
    
    # 票价信息
    seat_types = scrapy.Field()            # 座位类型列表
    prices = scrapy.Field()                # 价格列表
    ticket_status = scrapy.Field()         # 余票状态
    
    # 其他信息
    date = scrapy.Field()                  # 查询日期
    train_type = scrapy.Field()            # 车型
    distance = scrapy.Field()              # 距离
    
    # 爬取信息
    scraped_at = scrapy.Field()           # 爬取时间
    source_url = scrapy.Field()            # 来源URL