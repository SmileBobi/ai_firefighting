# 车票信息爬虫

这是一个基于Scrapy框架的车票信息爬虫项目，可以抓取火车票、机票、汽车票等信息。

## 项目结构

```
ticket_scraper/
├── ticket_scraper/
│   ├── __init__.py
│   ├── items.py          # 数据模型定义
│   ├── middlewares.py    # 中间件
│   ├── pipelines.py      # 数据处理管道
│   ├── settings.py       # 项目设置
│   └── spiders/
│       ├── __init__.py
│       └── train_ticket.py  # 火车票爬虫
├── run_scraper.py        # 运行脚本
├── scrapy.cfg           # Scrapy配置文件
└── README.md            # 说明文档
```

## 功能特性

### 1. 数据模型
- **TrainTicketItem**: 火车票信息模型
- **FlightTicketItem**: 机票信息模型  
- **BusTicketItem**: 汽车票信息模型

### 2. 数据处理管道
- **ConsolePipeline**: 控制台输出
- **JsonWriterPipeline**: JSON文件存储
- **CsvWriterPipeline**: CSV文件存储
- **DuplicatesPipeline**: 数据去重

### 3. 爬虫功能
- 支持多种交通工具的车票信息抓取
- 自动解析车次、时间、价格等信息
- 支持不同座位类型的票价查询
- 实时显示爬取进度

## 安装依赖

```bash
pip install scrapy
```

## 使用方法

### 1. 运行演示爬虫

演示爬虫使用模拟数据，可以快速了解项目功能：

```bash
python run_demo.py
```

### 2. 运行真实爬虫

```bash
python run_scraper.py
```

### 3. 使用Scrapy命令

```bash
# 运行演示爬虫
scrapy crawl demo_ticket

# 运行火车票爬虫
scrapy crawl train_ticket

# 带参数运行
scrapy crawl train_ticket -a from_station=BJP -a to_station=SHH -a date=2025-09-18
```

## 输出文件

爬虫运行后会生成以下文件：

1. **JSON文件**: `train_tickets_YYYYMMDD_HHMMSS.json`
   - 包含完整的车票信息，便于程序处理

2. **CSV文件**: `train_tickets_YYYYMMDD_HHMMSS.csv`
   - 表格格式，便于Excel打开查看

## 数据字段说明

### 火车票信息
- `train_number`: 车次号
- `departure_station`: 出发站
- `arrival_station`: 到达站
- `departure_time`: 出发时间
- `arrival_time`: 到达时间
- `duration`: 运行时长
- `seat_types`: 座位类型列表
- `prices`: 对应价格列表
- `ticket_status`: 余票状态
- `train_type`: 车型
- `distance`: 距离
- `scraped_at`: 爬取时间

## 配置说明

### 修改爬虫参数

在 `run_scraper.py` 中可以修改以下参数：

```python
spider_kwargs = {
    'from_station': 'BJP',  # 出发站代码
    'to_station': 'SHH',    # 到达站代码
    'date': '2025-09-18'    # 查询日期
}
```

### 常用车站代码
- `BJP`: 北京
- `SHH`: 上海
- `GZQ`: 广州
- `SZQ`: 深圳
- `CDW`: 成都
- `XAN`: 西安

### 修改设置

在 `settings.py` 中可以调整：

```python
# 下载延迟（秒）
DOWNLOAD_DELAY = 1

# 日志级别
LOG_LEVEL = 'INFO'

# 是否遵守robots.txt
ROBOTSTXT_OBEY = False
```

## 注意事项

1. **网络请求**: 真实爬虫需要访问12306等网站，可能受到反爬虫限制
2. **数据准确性**: 演示版本使用模拟数据，真实数据需要连接实际API
3. **请求频率**: 建议设置适当的下载延迟，避免请求过于频繁
4. **法律合规**: 请确保爬取行为符合相关网站的使用条款

## 扩展功能

### 添加新的爬虫

1. 在 `spiders/` 目录下创建新的爬虫文件
2. 继承 `scrapy.Spider` 类
3. 实现 `parse` 方法
4. 在 `items.py` 中定义对应的数据模型

### 添加新的数据管道

1. 在 `pipelines.py` 中创建新的管道类
2. 实现 `process_item` 方法
3. 在 `settings.py` 中注册管道

## 故障排除

### 常见问题

1. **导入错误**: 确保在项目根目录下运行命令
2. **网络超时**: 检查网络连接，增加超时时间
3. **数据为空**: 检查目标网站是否可访问，API是否正常

### 调试模式

```bash
# 启用详细日志
scrapy crawl train_ticket -L DEBUG

# 保存调试信息
scrapy crawl train_ticket -s LOG_FILE=debug.log
```

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。
