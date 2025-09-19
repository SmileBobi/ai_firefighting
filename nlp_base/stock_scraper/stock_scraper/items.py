# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DailyStockItem(scrapy.Item):
    """日交易数据模型"""
    # 基本信息
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    date = scrapy.Field()  # 交易日期
    
    # 价格信息
    open_price = scrapy.Field()  # 开盘价
    close_price = scrapy.Field()  # 收盘价
    high_price = scrapy.Field()  # 最高价
    low_price = scrapy.Field()  # 最低价
    prev_close = scrapy.Field()  # 昨收价
    
    # 交易量信息
    volume = scrapy.Field()  # 成交量
    turnover = scrapy.Field()  # 成交额
    amplitude = scrapy.Field()  # 振幅
    change_rate = scrapy.Field()  # 涨跌幅
    change_amount = scrapy.Field()  # 涨跌额
    
    # 其他信息
    pe_ratio = scrapy.Field()  # 市盈率
    pb_ratio = scrapy.Field()  # 市净率
    market_cap = scrapy.Field()  # 总市值
    circulating_cap = scrapy.Field()  # 流通市值
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class RealTimeStockItem(scrapy.Item):
    """实时交易数据模型"""
    # 基本信息
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    timestamp = scrapy.Field()  # 时间戳
    
    # 实时价格
    current_price = scrapy.Field()  # 当前价格
    change_rate = scrapy.Field()  # 涨跌幅
    change_amount = scrapy.Field()  # 涨跌额
    
    # 实时交易量
    volume = scrapy.Field()  # 成交量
    turnover = scrapy.Field()  # 成交额
    
    # 买卖盘信息
    bid_price = scrapy.Field()  # 买一价
    ask_price = scrapy.Field()  # 卖一价
    bid_volume = scrapy.Field()  # 买一量
    ask_volume = scrapy.Field()  # 卖一量
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class DragonTigerItem(scrapy.Item):
    """龙虎榜数据模型"""
    # 基本信息
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    date = scrapy.Field()  # 上榜日期
    
    # 上榜原因
    reason = scrapy.Field()  # 上榜原因
    reason_detail = scrapy.Field()  # 上榜原因详情
    
    # 买卖数据
    buy_amount = scrapy.Field()  # 买入金额
    sell_amount = scrapy.Field()  # 卖出金额
    net_amount = scrapy.Field()  # 净买入金额
    
    # 席位信息
    buy_seats = scrapy.Field()  # 买入席位
    sell_seats = scrapy.Field()  # 卖出席位
    
    # 其他信息
    turnover_rate = scrapy.Field()  # 换手率
    price_limit = scrapy.Field()  # 涨跌幅
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class CapitalFlowItem(scrapy.Item):
    """资金流向数据模型"""
    # 基本信息
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    date = scrapy.Field()  # 日期
    
    # 主力资金
    main_inflow = scrapy.Field()  # 主力净流入
    main_inflow_rate = scrapy.Field()  # 主力净流入率
    
    # 超大单
    super_large_inflow = scrapy.Field()  # 超大单净流入
    super_large_inflow_rate = scrapy.Field()  # 超大单净流入率
    
    # 大单
    large_inflow = scrapy.Field()  # 大单净流入
    large_inflow_rate = scrapy.Field()  # 大单净流入率
    
    # 中单
    medium_inflow = scrapy.Field()  # 中单净流入
    medium_inflow_rate = scrapy.Field()  # 中单净流入率
    
    # 小单
    small_inflow = scrapy.Field()  # 小单净流入
    small_inflow_rate = scrapy.Field()  # 小单净流入率
    
    # 其他信息
    current_price = scrapy.Field()  # 当前价格
    change_rate = scrapy.Field()  # 涨跌幅
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class StockNewsItem(scrapy.Item):
    """股票新闻数据模型"""
    # 基本信息
    title = scrapy.Field()  # 新闻标题
    content = scrapy.Field()  # 新闻内容
    publish_time = scrapy.Field()  # 发布时间
    source = scrapy.Field()  # 新闻来源
    
    # 相关股票
    related_stocks = scrapy.Field()  # 相关股票代码
    sentiment = scrapy.Field()  # 情感分析
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class ForumPostItem(scrapy.Item):
    """论坛帖子数据模型"""
    # 基本信息
    post_id = scrapy.Field()  # 帖子ID
    title = scrapy.Field()  # 帖子标题
    content = scrapy.Field()  # 帖子内容
    author_id = scrapy.Field()  # 作者ID
    author_name = scrapy.Field()  # 作者昵称
    author_level = scrapy.Field()  # 作者等级
    publish_time = scrapy.Field()  # 发布时间
    
    # 互动数据
    view_count = scrapy.Field()  # 浏览次数
    reply_count = scrapy.Field()  # 回复次数
    like_count = scrapy.Field()  # 点赞次数
    share_count = scrapy.Field()  # 分享次数
    
    # 相关股票
    related_stocks = scrapy.Field()  # 相关股票代码
    stock_mentions = scrapy.Field()  # 股票提及次数
    
    # 情感分析
    sentiment_score = scrapy.Field()  # 情感得分 (-1到1)
    sentiment_label = scrapy.Field()  # 情感标签 (positive/negative/neutral)
    emotion_keywords = scrapy.Field()  # 情感关键词
    
    # 内容分析
    content_length = scrapy.Field()  # 内容长度
    has_images = scrapy.Field()  # 是否包含图片
    has_links = scrapy.Field()  # 是否包含链接
    topic_tags = scrapy.Field()  # 话题标签
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class ForumCommentItem(scrapy.Item):
    """论坛评论数据模型"""
    # 基本信息
    comment_id = scrapy.Field()  # 评论ID
    post_id = scrapy.Field()  # 所属帖子ID
    content = scrapy.Field()  # 评论内容
    author_id = scrapy.Field()  # 作者ID
    author_name = scrapy.Field()  # 作者昵称
    author_level = scrapy.Field()  # 作者等级
    publish_time = scrapy.Field()  # 发布时间
    
    # 互动数据
    like_count = scrapy.Field()  # 点赞次数
    reply_count = scrapy.Field()  # 回复次数
    
    # 层级关系
    parent_comment_id = scrapy.Field()  # 父评论ID
    comment_level = scrapy.Field()  # 评论层级
    
    # 相关股票
    related_stocks = scrapy.Field()  # 相关股票代码
    
    # 情感分析
    sentiment_score = scrapy.Field()  # 情感得分
    sentiment_label = scrapy.Field()  # 情感标签
    emotion_keywords = scrapy.Field()  # 情感关键词
    
    # 内容分析
    content_length = scrapy.Field()  # 内容长度
    has_images = scrapy.Field()  # 是否包含图片
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class ForumUserItem(scrapy.Item):
    """论坛用户数据模型"""
    # 基本信息
    user_id = scrapy.Field()  # 用户ID
    username = scrapy.Field()  # 用户名
    nickname = scrapy.Field()  # 昵称
    level = scrapy.Field()  # 用户等级
    join_time = scrapy.Field()  # 注册时间
    last_active = scrapy.Field()  # 最后活跃时间
    
    # 用户统计
    post_count = scrapy.Field()  # 发帖数
    comment_count = scrapy.Field()  # 评论数
    follower_count = scrapy.Field()  # 粉丝数
    following_count = scrapy.Field()  # 关注数
    total_likes = scrapy.Field()  # 总获赞数
    
    # 用户标签
    user_tags = scrapy.Field()  # 用户标签
    expertise_areas = scrapy.Field()  # 专业领域
    investment_style = scrapy.Field()  # 投资风格
    
    # 活跃度分析
    activity_score = scrapy.Field()  # 活跃度得分
    influence_score = scrapy.Field()  # 影响力得分
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class SentimentAnalysisItem(scrapy.Item):
    """情感分析结果数据模型"""
    # 基本信息
    content_id = scrapy.Field()  # 内容ID
    content_type = scrapy.Field()  # 内容类型 (post/comment)
    content_text = scrapy.Field()  # 原始文本
    
    # 情感分析结果
    sentiment_score = scrapy.Field()  # 情感得分 (-1到1)
    sentiment_label = scrapy.Field()  # 情感标签
    confidence = scrapy.Field()  # 置信度
    
    # 详细分析
    positive_score = scrapy.Field()  # 积极情感得分
    negative_score = scrapy.Field()  # 消极情感得分
    neutral_score = scrapy.Field()  # 中性情感得分
    
    # 关键词分析
    emotion_keywords = scrapy.Field()  # 情感关键词
    stock_keywords = scrapy.Field()  # 股票关键词
    market_keywords = scrapy.Field()  # 市场关键词
    
    # 相关股票
    mentioned_stocks = scrapy.Field()  # 提及的股票
    stock_sentiment = scrapy.Field()  # 对股票的情感倾向
    
    # 分析时间
    analyzed_at = scrapy.Field()  # 分析时间
    analysis_model = scrapy.Field()  # 使用的分析模型


class FinancialReportItem(scrapy.Item):
    """上市公司财报数据模型"""
    # 基本信息
    report_id = scrapy.Field()  # 报告ID
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    company_name = scrapy.Field()  # 公司全称
    report_type = scrapy.Field()  # 报告类型 (年报/季报/中报)
    report_period = scrapy.Field()  # 报告期
    report_date = scrapy.Field()  # 报告日期
    publish_date = scrapy.Field()  # 发布日期
    
    # 报告内容
    title = scrapy.Field()  # 报告标题
    summary = scrapy.Field()  # 报告摘要
    full_content = scrapy.Field()  # 完整内容
    content_sections = scrapy.Field()  # 内容章节
    
    # 财务数据
    revenue = scrapy.Field()  # 营业收入
    net_profit = scrapy.Field()  # 净利润
    total_assets = scrapy.Field()  # 总资产
    total_liabilities = scrapy.Field()  # 总负债
    shareholders_equity = scrapy.Field()  # 股东权益
    operating_cash_flow = scrapy.Field()  # 经营现金流
    eps = scrapy.Field()  # 每股收益
    roe = scrapy.Field()  # 净资产收益率
    roa = scrapy.Field()  # 总资产收益率
    
    # 文本分析
    content_length = scrapy.Field()  # 内容长度
    word_count = scrapy.Field()  # 字数统计
    section_count = scrapy.Field()  # 章节数量
    table_count = scrapy.Field()  # 表格数量
    image_count = scrapy.Field()  # 图片数量
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型 (PDF/HTML/TXT)
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FinancialDataItem(scrapy.Item):
    """财务数据项模型"""
    # 基本信息
    data_id = scrapy.Field()  # 数据ID
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    report_period = scrapy.Field()  # 报告期
    data_type = scrapy.Field()  # 数据类型 (资产负债表/利润表/现金流量表)
    
    # 财务指标
    indicator_name = scrapy.Field()  # 指标名称
    indicator_value = scrapy.Field()  # 指标数值
    indicator_unit = scrapy.Field()  # 指标单位
    period_type = scrapy.Field()  # 期间类型 (期末/期初/本期/上年同期)
    
    # 数据来源
    table_name = scrapy.Field()  # 表格名称
    row_index = scrapy.Field()  # 行索引
    col_index = scrapy.Field()  # 列索引
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class CompanyInfoItem(scrapy.Item):
    """公司基本信息模型"""
    # 基本信息
    stock_code = scrapy.Field()  # 股票代码
    stock_name = scrapy.Field()  # 股票名称
    company_name = scrapy.Field()  # 公司全称
    company_abbr = scrapy.Field()  # 公司简称
    english_name = scrapy.Field()  # 英文名称
    
    # 公司信息
    industry = scrapy.Field()  # 所属行业
    market = scrapy.Field()  # 所属市场 (主板/创业板/科创板)
    listing_date = scrapy.Field()  # 上市日期
    legal_representative = scrapy.Field()  # 法定代表人
    registered_capital = scrapy.Field()  # 注册资本
    business_scope = scrapy.Field()  # 经营范围
    
    # 联系信息
    address = scrapy.Field()  # 公司地址
    phone = scrapy.Field()  # 联系电话
    website = scrapy.Field()  # 公司网站
    email = scrapy.Field()  # 电子邮箱
    
    # 财务信息
    total_shares = scrapy.Field()  # 总股本
    circulating_shares = scrapy.Field()  # 流通股本
    market_cap = scrapy.Field()  # 总市值
    circulating_cap = scrapy.Field()  # 流通市值
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class TextAnalysisItem(scrapy.Item):
    """长文本分析结果模型"""
    # 基本信息
    content_id = scrapy.Field()  # 内容ID
    content_type = scrapy.Field()  # 内容类型 (report/section/paragraph)
    content_text = scrapy.Field()  # 原始文本
    
    # 文本统计
    char_count = scrapy.Field()  # 字符数
    word_count = scrapy.Field()  # 词数
    sentence_count = scrapy.Field()  # 句子数
    paragraph_count = scrapy.Field()  # 段落数
    
    # 文本特征
    readability_score = scrapy.Field()  # 可读性得分
    complexity_score = scrapy.Field()  # 复杂度得分
    sentiment_score = scrapy.Field()  # 情感得分
    topic_keywords = scrapy.Field()  # 主题关键词
    
    # 结构化信息
    entities = scrapy.Field()  # 实体识别结果
    relationships = scrapy.Field()  # 关系抽取结果
    summary = scrapy.Field()  # 文本摘要
    
    # 分析结果
    analysis_result = scrapy.Field()  # 分析结果 (JSON格式)
    ai_analysis = scrapy.Field()  # AI分析结果 (智谱/Kimi)
    
    # 分析时间
    analyzed_at = scrapy.Field()  # 分析时间
    analysis_model = scrapy.Field()  # 使用的分析模型
