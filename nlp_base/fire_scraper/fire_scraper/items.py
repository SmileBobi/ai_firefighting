# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FireRegulationItem(scrapy.Item):
    """消防法规数据模型"""
    # 基本信息
    regulation_id = scrapy.Field()  # 法规ID
    title = scrapy.Field()  # 法规标题
    regulation_type = scrapy.Field()  # 法规类型 (法律/行政法规/部门规章/地方性法规/标准)
    level = scrapy.Field()  # 法规层级 (国家级/省级/市级/县级)
    issuing_authority = scrapy.Field()  # 发布机关
    issue_date = scrapy.Field()  # 发布日期
    effective_date = scrapy.Field()  # 生效日期
    status = scrapy.Field()  # 状态 (现行有效/已废止/已修订)
    
    # 内容信息
    content = scrapy.Field()  # 法规内容
    summary = scrapy.Field()  # 法规摘要
    chapters = scrapy.Field()  # 章节信息
    articles = scrapy.Field()  # 条文信息
    attachments = scrapy.Field()  # 附件信息
    
    # 分类信息
    category = scrapy.Field()  # 分类 (建筑防火/消防设施/消防安全管理/火灾调查等)
    keywords = scrapy.Field()  # 关键词
    tags = scrapy.Field()  # 标签
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型 (PDF/DOC/HTML/TXT)
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FireStandardItem(scrapy.Item):
    """消防标准数据模型"""
    # 基本信息
    standard_id = scrapy.Field()  # 标准ID
    standard_number = scrapy.Field()  # 标准编号 (如GB 50016-2014)
    title = scrapy.Field()  # 标准名称
    english_title = scrapy.Field()  # 英文名称
    standard_type = scrapy.Field()  # 标准类型 (国家标准/行业标准/地方标准/企业标准)
    category = scrapy.Field()  # 标准分类 (强制性/推荐性)
    
    # 发布信息
    issuing_authority = scrapy.Field()  # 发布机关
    approval_authority = scrapy.Field()  # 批准机关
    issue_date = scrapy.Field()  # 发布日期
    implementation_date = scrapy.Field()  # 实施日期
    status = scrapy.Field()  # 状态 (现行/废止/修订)
    
    # 内容信息
    scope = scrapy.Field()  # 适用范围
    content = scrapy.Field()  # 标准内容
    technical_requirements = scrapy.Field()  # 技术要求
    test_methods = scrapy.Field()  # 试验方法
    inspection_rules = scrapy.Field()  # 检验规则
    
    # 分类信息
    fire_category = scrapy.Field()  # 消防分类 (建筑防火/消防设施/消防产品/消防安全等)
    keywords = scrapy.Field()  # 关键词
    tags = scrapy.Field()  # 标签
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FireCaseItem(scrapy.Item):
    """火灾案例数据模型"""
    # 基本信息
    case_id = scrapy.Field()  # 案例ID
    title = scrapy.Field()  # 案例标题
    case_type = scrapy.Field()  # 案例类型 (火灾事故/消防执法/应急救援等)
    severity_level = scrapy.Field()  # 严重程度 (特别重大/重大/较大/一般)
    
    # 时间地点
    incident_date = scrapy.Field()  # 事故发生时间
    location = scrapy.Field()  # 事故发生地点
    province = scrapy.Field()  # 省份
    city = scrapy.Field()  # 城市
    district = scrapy.Field()  # 区县
    
    # 事故信息
    building_type = scrapy.Field()  # 建筑类型 (住宅/商业/工业/公共建筑等)
    fire_cause = scrapy.Field()  # 火灾原因
    casualties = scrapy.Field()  # 伤亡情况
    economic_loss = scrapy.Field()  # 经济损失
    fire_duration = scrapy.Field()  # 火灾持续时间
    
    # 内容信息
    description = scrapy.Field()  # 事故描述
    investigation_result = scrapy.Field()  # 调查结果
    lessons_learned = scrapy.Field()  # 经验教训
    prevention_measures = scrapy.Field()  # 预防措施
    
    # 分类信息
    keywords = scrapy.Field()  # 关键词
    tags = scrapy.Field()  # 标签
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FireNewsItem(scrapy.Item):
    """消防新闻数据模型"""
    # 基本信息
    news_id = scrapy.Field()  # 新闻ID
    title = scrapy.Field()  # 新闻标题
    content = scrapy.Field()  # 新闻内容
    summary = scrapy.Field()  # 新闻摘要
    author = scrapy.Field()  # 作者
    source = scrapy.Field()  # 新闻来源
    publish_time = scrapy.Field()  # 发布时间
    
    # 分类信息
    news_type = scrapy.Field()  # 新闻类型 (政策法规/事故案例/技术动态/行业资讯等)
    category = scrapy.Field()  # 分类
    keywords = scrapy.Field()  # 关键词
    tags = scrapy.Field()  # 标签
    
    # 互动数据
    view_count = scrapy.Field()  # 浏览次数
    comment_count = scrapy.Field()  # 评论次数
    share_count = scrapy.Field()  # 分享次数
    
    # 文件信息
    image_urls = scrapy.Field()  # 图片链接
    file_urls = scrapy.Field()  # 文件链接
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FireKnowledgeItem(scrapy.Item):
    """消防知识数据模型"""
    # 基本信息
    knowledge_id = scrapy.Field()  # 知识ID
    title = scrapy.Field()  # 知识标题
    content = scrapy.Field()  # 知识内容
    summary = scrapy.Field()  # 知识摘要
    knowledge_type = scrapy.Field()  # 知识类型 (基础知识/专业技术/操作指南/应急预案等)
    
    # 分类信息
    category = scrapy.Field()  # 分类 (防火/灭火/逃生/救援等)
    subcategory = scrapy.Field()  # 子分类
    difficulty_level = scrapy.Field()  # 难度等级 (初级/中级/高级)
    target_audience = scrapy.Field()  # 目标受众 (公众/专业人员/管理人员等)
    
    # 内容特征
    content_length = scrapy.Field()  # 内容长度
    word_count = scrapy.Field()  # 字数统计
    section_count = scrapy.Field()  # 章节数量
    image_count = scrapy.Field()  # 图片数量
    table_count = scrapy.Field()  # 表格数量
    
    # 关键词和标签
    keywords = scrapy.Field()  # 关键词
    tags = scrapy.Field()  # 标签
    entities = scrapy.Field()  # 实体识别结果
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL


class FireDocumentItem(scrapy.Item):
    """消防文档数据模型"""
    # 基本信息
    document_id = scrapy.Field()  # 文档ID
    title = scrapy.Field()  # 文档标题
    content = scrapy.Field()  # 文档内容
    document_type = scrapy.Field()  # 文档类型 (法规/标准/案例/新闻/知识等)
    
    # 文档特征
    content_length = scrapy.Field()  # 内容长度
    word_count = scrapy.Field()  # 字数统计
    readability_score = scrapy.Field()  # 可读性得分
    complexity_score = scrapy.Field()  # 复杂度得分
    
    # 文本分析
    summary = scrapy.Field()  # 文档摘要
    keywords = scrapy.Field()  # 关键词
    entities = scrapy.Field()  # 实体识别结果
    topics = scrapy.Field()  # 主题分析结果
    
    # RAG相关
    embedding_vector = scrapy.Field()  # 嵌入向量
    chunk_texts = scrapy.Field()  # 分块文本
    chunk_embeddings = scrapy.Field()  # 分块嵌入向量
    
    # 文件信息
    file_url = scrapy.Field()  # 文件下载链接
    file_type = scrapy.Field()  # 文件类型
    file_size = scrapy.Field()  # 文件大小
    download_status = scrapy.Field()  # 下载状态
    
    # 爬取信息
    scraped_at = scrapy.Field()  # 爬取时间
    source_url = scrapy.Field()  # 来源URL
