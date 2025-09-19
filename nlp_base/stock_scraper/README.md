# 证券信息爬虫系统

这是一个基于Scrapy框架的证券信息爬虫系统，专门用于抓取股票交易数据，包括日交易数据、实时数据、龙虎榜数据和资金流向数据。

## 功能特性

### 📊 数据类型支持
- **日交易数据**: 开盘价、收盘价、最高价、最低价、成交量、成交额等
- **实时数据**: 当前价格、涨跌幅、买卖盘信息等（支持60秒间隔抓取）
- **龙虎榜数据**: 上榜原因、买卖金额、席位信息等
- **资金流向数据**: 主力资金、超大单、大单、中单、小单流向
- **论坛帖子数据**: 帖子标题、内容、作者、互动数据等
- **论坛评论数据**: 评论内容、作者、层级关系等
- **用户信息数据**: 用户等级、活跃度、影响力等
- **情感分析数据**: 情感倾向、置信度、关键词等
- **财报数据**: 年报、季报、中报等完整财报内容
- **财务数据**: 营业收入、净利润、总资产等关键财务指标
- **公司信息**: 公司基本信息、经营范围、联系方式等
- **文本分析数据**: 长文本分析结果、可读性、复杂度等

### 🎯 数据源支持
- **股票数据**: 东方财富网 (eastmoney.com)
- **论坛数据**: 东方财富股吧 (guba.eastmoney.com)
- **财报数据**: 巨潮资讯网 (cninfo.com.cn)
- **其他数据源**: 同花顺、雪球、网易财经、新浪财经

### 💾 数据存储格式
- **Excel文件**: 多工作表，便于分析
- **CSV文件**: 分类存储，便于导入其他工具
- **JSON文件**: 结构化数据，便于程序处理

## 项目结构

```
stock_scraper/
├── stock_scraper/
│   ├── __init__.py
│   ├── items.py          # 数据模型定义
│   ├── middlewares.py    # 中间件
│   ├── pipelines.py      # 数据处理管道
│   ├── settings.py       # 项目设置
│   └── spiders/
│       ├── __init__.py
│       └── eastmoney_spider.py  # 东方财富爬虫
├── demo_stock_spider.py  # 演示爬虫
├── run_stock_scraper.py  # 运行脚本
├── run_demo_stock.py     # 演示运行脚本
├── real_time_scheduler.py # 实时数据定时抓取
├── scrapy.cfg           # Scrapy配置文件
└── README.md            # 说明文档
```

## 安装依赖

```bash
pip install scrapy pandas openpyxl schedule
```

## 使用方法

### 1. 运行演示爬虫

演示爬虫使用模拟数据，可以快速了解项目功能：

```bash
python run_demo_stock.py
```

### 2. 运行股票数据爬虫

```bash
python run_stock_scraper.py
```

### 3. 运行论坛数据爬虫

```bash
# 演示版本（推荐先试这个）
python run_demo_forum.py

# 真实版本
python run_forum_scraper.py
```

### 4. 运行财报数据爬虫

```bash
# 演示版本（推荐先试这个）
python run_demo_financial.py

# 真实版本
python run_financial_scraper.py
```

### 5. 使用Scrapy命令

```bash
# 运行股票数据演示爬虫
scrapy crawl demo_stock

# 运行股票数据爬虫
scrapy crawl eastmoney_spider

# 运行论坛数据演示爬虫
scrapy crawl demo_forum

# 运行论坛数据爬虫
scrapy crawl eastmoney_forum_spider

# 运行财报数据演示爬虫
scrapy crawl demo_financial

# 运行财报数据爬虫
scrapy crawl financial_report_spider

# 带参数运行
scrapy crawl eastmoney_spider -a stock_codes=000001,000002 -a data_type=daily
scrapy crawl eastmoney_forum_spider -a stock_codes=000001,000002 -a max_pages=3
scrapy crawl financial_report_spider -a stock_codes=000001,000002 -a report_types=年报,季报 -a years=2023,2024
```

### 6. 实时数据定时抓取

```bash
python real_time_scheduler.py
```

## 参数说明

### 爬虫参数
- `stock_codes`: 股票代码列表，用逗号分隔（如：000001,000002,600000）
- `data_type`: 数据类型
  - `all`: 所有数据类型
  - `daily`: 日交易数据
  - `realtime`: 实时数据
  - `dragon`: 龙虎榜数据
  - `capital`: 资金流向数据

### 常用股票代码
- `000001`: 平安银行
- `000002`: 万科A
- `600000`: 浦发银行
- `600036`: 招商银行
- `000858`: 五粮液

## 输出文件

### Excel文件
- `stock_data_YYYYMMDD_HHMMSS.xlsx` (股票数据)
  - 日交易数据工作表
  - 实时数据工作表
  - 龙虎榜工作表
  - 资金流向工作表
  - 股票新闻工作表

- `forum_data_YYYYMMDD_HHMMSS.xlsx` (论坛数据)
  - 论坛帖子工作表
  - 论坛评论工作表
  - 用户信息工作表
  - 情感分析工作表

- `financial_data_YYYYMMDD_HHMMSS.xlsx` (财报数据)
  - 财报数据工作表
  - 财务数据工作表
  - 公司信息工作表
  - 文本分析工作表

### CSV文件
- **股票数据**:
  - `daily_stock_YYYYMMDD_HHMMSS.csv`: 日交易数据
  - `realtime_stock_YYYYMMDD_HHMMSS.csv`: 实时数据
  - `dragon_tiger_YYYYMMDD_HHMMSS.csv`: 龙虎榜数据
  - `capital_flow_YYYYMMDD_HHMMSS.csv`: 资金流向数据
  - `stock_news_YYYYMMDD_HHMMSS.csv`: 股票新闻

- **论坛数据**:
  - `forum_posts_YYYYMMDD_HHMMSS.csv`: 论坛帖子数据
  - `forum_comments_YYYYMMDD_HHMMSS.csv`: 论坛评论数据
  - `forum_users_YYYYMMDD_HHMMSS.csv`: 用户信息数据
  - `sentiment_analysis_YYYYMMDD_HHMMSS.csv`: 情感分析数据

- **财报数据**:
  - `financial_reports_YYYYMMDD_HHMMSS.csv`: 财报数据
  - `financial_data_YYYYMMDD_HHMMSS.csv`: 财务数据
  - `company_info_YYYYMMDD_HHMMSS.csv`: 公司信息数据
  - `text_analysis_YYYYMMDD_HHMMSS.csv`: 文本分析数据

### JSON文件
- `stock_data_YYYYMMDD_HHMMSS.json`: 所有数据的JSON格式

## 数据字段说明

### 日交易数据 (DailyStockItem)
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `date`: 交易日期
- `open_price`: 开盘价
- `close_price`: 收盘价
- `high_price`: 最高价
- `low_price`: 最低价
- `prev_close`: 昨收价
- `volume`: 成交量
- `turnover`: 成交额
- `amplitude`: 振幅
- `change_rate`: 涨跌幅
- `change_amount`: 涨跌额
- `pe_ratio`: 市盈率
- `pb_ratio`: 市净率
- `market_cap`: 总市值
- `circulating_cap`: 流通市值

### 实时数据 (RealTimeStockItem)
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `timestamp`: 时间戳
- `current_price`: 当前价格
- `change_rate`: 涨跌幅
- `change_amount`: 涨跌额
- `volume`: 成交量
- `turnover`: 成交额
- `bid_price`: 买一价
- `ask_price`: 卖一价
- `bid_volume`: 买一量
- `ask_volume`: 卖一量

### 龙虎榜数据 (DragonTigerItem)
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `date`: 上榜日期
- `reason`: 上榜原因
- `reason_detail`: 上榜原因详情
- `buy_amount`: 买入金额
- `sell_amount`: 卖出金额
- `net_amount`: 净买入金额
- `buy_seats`: 买入席位
- `sell_seats`: 卖出席位
- `turnover_rate`: 换手率
- `price_limit`: 涨跌幅

### 资金流向数据 (CapitalFlowItem)
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `date`: 日期
- `main_inflow`: 主力净流入
- `main_inflow_rate`: 主力净流入率
- `super_large_inflow`: 超大单净流入
- `super_large_inflow_rate`: 超大单净流入率
- `large_inflow`: 大单净流入
- `large_inflow_rate`: 大单净流入率
- `medium_inflow`: 中单净流入
- `medium_inflow_rate`: 中单净流入率
- `small_inflow`: 小单净流入
- `small_inflow_rate`: 小单净流入率

### 论坛帖子数据 (ForumPostItem)
- `post_id`: 帖子ID
- `title`: 帖子标题
- `content`: 帖子内容
- `author_name`: 作者昵称
- `author_level`: 作者等级
- `publish_time`: 发布时间
- `view_count`: 浏览次数
- `reply_count`: 回复次数
- `like_count`: 点赞次数
- `related_stocks`: 相关股票代码
- `stock_mentions`: 股票提及次数
- `sentiment_score`: 情感得分 (-1到1)
- `sentiment_label`: 情感标签 (positive/negative/neutral)
- `emotion_keywords`: 情感关键词
- `content_length`: 内容长度
- `has_images`: 是否包含图片
- `has_links`: 是否包含链接
- `topic_tags`: 话题标签

### 论坛评论数据 (ForumCommentItem)
- `comment_id`: 评论ID
- `post_id`: 所属帖子ID
- `content`: 评论内容
- `author_name`: 作者昵称
- `author_level`: 作者等级
- `publish_time`: 发布时间
- `like_count`: 点赞次数
- `parent_comment_id`: 父评论ID
- `comment_level`: 评论层级
- `related_stocks`: 相关股票代码
- `sentiment_score`: 情感得分
- `sentiment_label`: 情感标签
- `emotion_keywords`: 情感关键词
- `content_length`: 内容长度
- `has_images`: 是否包含图片

### 用户信息数据 (ForumUserItem)
- `user_id`: 用户ID
- `username`: 用户名
- `nickname`: 昵称
- `level`: 用户等级
- `join_time`: 注册时间
- `last_active`: 最后活跃时间
- `post_count`: 发帖数
- `comment_count`: 评论数
- `follower_count`: 粉丝数
- `following_count`: 关注数
- `total_likes`: 总获赞数
- `user_tags`: 用户标签
- `expertise_areas`: 专业领域
- `investment_style`: 投资风格
- `activity_score`: 活跃度得分
- `influence_score`: 影响力得分

### 情感分析数据 (SentimentAnalysisItem)
- `content_id`: 内容ID
- `content_type`: 内容类型 (post/comment)
- `content_text`: 原始文本
- `sentiment_score`: 情感得分 (-1到1)
- `sentiment_label`: 情感标签
- `confidence`: 置信度
- `positive_score`: 积极情感得分
- `negative_score`: 消极情感得分
- `neutral_score`: 中性情感得分
- `emotion_keywords`: 情感关键词
- `stock_keywords`: 股票关键词
- `market_keywords`: 市场关键词
- `mentioned_stocks`: 提及的股票
- `stock_sentiment`: 对股票的情感倾向
- `analyzed_at`: 分析时间
- `analysis_model`: 使用的分析模型

### 财报数据 (FinancialReportItem)
- `report_id`: 报告ID
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `company_name`: 公司全称
- `report_type`: 报告类型 (年报/季报/中报)
- `report_period`: 报告期
- `report_date`: 报告日期
- `publish_date`: 发布日期
- `title`: 报告标题
- `summary`: 报告摘要
- `full_content`: 完整内容
- `content_sections`: 内容章节
- `revenue`: 营业收入
- `net_profit`: 净利润
- `total_assets`: 总资产
- `total_liabilities`: 总负债
- `shareholders_equity`: 股东权益
- `operating_cash_flow`: 经营现金流
- `eps`: 每股收益
- `roe`: 净资产收益率
- `roa`: 总资产收益率
- `content_length`: 内容长度
- `word_count`: 字数统计
- `section_count`: 章节数量
- `table_count`: 表格数量
- `image_count`: 图片数量
- `file_url`: 文件下载链接
- `file_type`: 文件类型 (PDF/HTML/TXT)
- `file_size`: 文件大小
- `download_status`: 下载状态

### 财务数据 (FinancialDataItem)
- `data_id`: 数据ID
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `report_period`: 报告期
- `data_type`: 数据类型 (资产负债表/利润表/现金流量表)
- `indicator_name`: 指标名称
- `indicator_value`: 指标数值
- `indicator_unit`: 指标单位
- `period_type`: 期间类型 (期末/期初/本期/上年同期)
- `table_name`: 表格名称
- `row_index`: 行索引
- `col_index`: 列索引

### 公司信息 (CompanyInfoItem)
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `company_name`: 公司全称
- `company_abbr`: 公司简称
- `english_name`: 英文名称
- `industry`: 所属行业
- `market`: 所属市场 (主板/创业板/科创板)
- `listing_date`: 上市日期
- `legal_representative`: 法定代表人
- `registered_capital`: 注册资本
- `business_scope`: 经营范围
- `address`: 公司地址
- `phone`: 联系电话
- `website`: 公司网站
- `email`: 电子邮箱
- `total_shares`: 总股本
- `circulating_shares`: 流通股本
- `market_cap`: 总市值
- `circulating_cap`: 流通市值

### 文本分析数据 (TextAnalysisItem)
- `content_id`: 内容ID
- `content_type`: 内容类型 (report/section/paragraph)
- `content_text`: 原始文本
- `char_count`: 字符数
- `word_count`: 词数
- `sentence_count`: 句子数
- `paragraph_count`: 段落数
- `readability_score`: 可读性得分
- `complexity_score`: 复杂度得分
- `sentiment_score`: 情感得分
- `topic_keywords`: 主题关键词
- `entities`: 实体识别结果
- `relationships`: 关系抽取结果
- `summary`: 文本摘要
- `analysis_result`: 分析结果 (JSON格式)
- `ai_analysis`: AI分析结果 (智谱/Kimi)
- `analyzed_at`: 分析时间
- `analysis_model`: 使用的分析模型

## 定时任务配置

实时数据定时抓取系统支持以下定时任务：

- **实时数据**: 每60秒抓取一次
- **日交易数据**: 每天09:35抓取（开盘后）
- **龙虎榜数据**: 每天15:35抓取（收盘后）
- **资金流向数据**: 每天16:00抓取

## 配置说明

### 修改爬虫参数

在 `run_stock_scraper.py` 中可以修改以下参数：

```python
spider_kwargs = {
    'stock_codes': '000001,000002,600000,600036,000858',  # 股票代码
    'data_type': 'all'  # 数据类型
}
```

### 修改设置

在 `settings.py` 中可以调整：

```python
# 下载延迟（秒）
DOWNLOAD_DELAY = 2

# 日志级别
LOG_LEVEL = 'INFO'

# 并发请求数
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8

# 是否遵守robots.txt
ROBOTSTXT_OBEY = False
```

## 情感分析功能

### 情感分析器特性
- **基于规则的情感分析**: 使用预定义的情感词典进行文本分析
- **多维度分析**: 支持积极、消极、中性三种情感倾向
- **关键词提取**: 自动提取情感关键词、股票关键词、市场关键词
- **置信度评估**: 提供情感分析的置信度评分
- **股票提及检测**: 自动识别文本中提及的股票代码和名称

### 情感分析结果
- **情感得分**: -1到1之间的数值，负值表示消极，正值表示积极
- **情感标签**: positive(积极)、negative(消极)、neutral(中性)
- **置信度**: 0到1之间的数值，表示分析结果的可信程度
- **关键词**: 提取的情感关键词、股票关键词、市场关键词
- **股票提及**: 文本中提及的股票代码和名称列表

### 应用场景
- **舆情监控**: 监控股票相关的网络舆情
- **情感趋势分析**: 分析市场情绪变化趋势
- **投资决策支持**: 为投资决策提供情感分析数据
- **风险预警**: 识别负面情绪，提前预警风险

## 扩展功能

### 添加新的数据源

1. 在 `spiders/` 目录下创建新的爬虫文件
2. 继承 `scrapy.Spider` 类
3. 实现相应的解析方法
4. 在 `items.py` 中定义对应的数据模型

### 添加新的数据管道

1. 在 `pipelines.py` 中创建新的管道类
2. 实现 `process_item` 方法
3. 在 `settings.py` 中注册管道

### 自定义情感分析

1. 修改 `sentiment_analyzer.py` 中的情感词典
2. 调整情感分析算法
3. 添加新的关键词类别
4. 优化分词和文本预处理逻辑

### 长文本分析功能

#### 文本分析器特性
- **基础文本分析**: 字符数、词数、句子数、段落数统计
- **可读性分析**: 基于句子长度和词汇复杂度的可读性评分
- **复杂度分析**: 基于词汇多样性和句子结构的复杂度评分
- **关键词提取**: 自动提取主题关键词和财务关键词
- **实体识别**: 识别公司名称、数字、日期、百分比等实体
- **财务专项分析**: 针对财报的风险分析、机会分析、财务数据提取

#### AI分析支持
- **智谱AI集成**: 支持使用智谱GLM-4模型进行长文本分析
- **Kimi集成**: 支持使用月之暗面Moonshot模型进行长文本分析
- **自定义提示词**: 可自定义AI分析提示词，适应不同分析需求
- **结构化输出**: AI分析结果以结构化格式存储，便于后续处理

#### 应用场景
- **财报深度分析**: 对上市公司财报进行全面的文本分析
- **投资决策支持**: 为投资决策提供基于文本分析的洞察
- **风险识别**: 识别财报中的风险因素和机会点
- **行业研究**: 支持行业研究和公司对比分析
- **监管合规**: 帮助识别财报中的合规问题和风险点

#### 使用方法
```python
from text_analyzer import text_analyzer

# 基础文本分析
result = text_analyzer.analyze_text_basic(text)

# 财报专项分析
result = text_analyzer.analyze_financial_report(text)

# 智谱AI分析
result = text_analyzer.analyze_with_zhipu(text, custom_prompt)

# Kimi分析
result = text_analyzer.analyze_with_kimi(text, custom_prompt)
```

## 注意事项

1. **网络请求**: 真实爬虫需要访问财经网站，可能受到反爬虫限制
2. **数据准确性**: 演示版本使用模拟数据，真实数据需要连接实际API
3. **请求频率**: 建议设置适当的下载延迟，避免请求过于频繁
4. **法律合规**: 请确保爬取行为符合相关网站的使用条款
5. **数据使用**: 爬取的数据仅供学习和研究使用

## 故障排除

### 常见问题

1. **导入错误**: 确保在项目根目录下运行命令
2. **网络超时**: 检查网络连接，增加超时时间
3. **数据为空**: 检查目标网站是否可访问，API是否正常
4. **Excel文件无法打开**: 确保安装了openpyxl库

### 调试模式

```bash
# 启用详细日志
scrapy crawl eastmoney_spider -L DEBUG

# 保存调试信息
scrapy crawl eastmoney_spider -s LOG_FILE=debug.log
```

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规。

## 更新日志

### v1.0.0 (2025-09-17)
- 初始版本发布
- 支持日交易数据、实时数据、龙虎榜数据、资金流向数据抓取
- 支持Excel、CSV、JSON多种数据存储格式
- 实现实时数据定时抓取功能
- 提供演示爬虫用于测试
