# 消防数据爬虫系统

一个专门用于爬取消防法规、标准、案例等非结构化数据的Scrapy爬虫系统，支持构建消防知识库（RAG）用于智能问答和知识检索。

## 🚀 功能特性

### 📊 数据类型支持
- **消防法规**: 法律、行政法规、部门规章、地方性法规等
- **消防标准**: 国家标准、行业标准、地方标准、企业标准
- **火灾案例**: 火灾事故、消防执法、应急救援等案例
- **消防新闻**: 政策法规、事故案例、技术动态、行业资讯
- **消防知识**: 基础知识、专业技术、操作指南、应急预案
- **RAG文档**: 用于知识库构建的结构化文档

### 🎯 数据源支持
- **消防法规**: 应急管理部官网 (xf.mem.gov.cn)
- **消防标准**: 国家标准信息公共服务平台
- **火灾案例**: 各地消防部门官网
- **消防新闻**: 消防行业媒体网站

### 💾 数据存储格式
- **Excel文件**: 多工作表，便于分析
- **CSV文件**: 分类存储，便于导入其他工具
- **JSON文件**: 结构化数据，便于程序处理
- **RAG知识库**: 向量化存储，支持语义搜索

### 🔍 文本分析功能
- **基础文本分析**: 字符数、词数、句子数、段落数统计
- **可读性分析**: 基于句子长度和词汇复杂度的可读性评分
- **复杂度分析**: 基于词汇多样性和句子结构的复杂度评分
- **关键词提取**: 自动提取消防相关关键词
- **实体识别**: 识别法规条文、标准编号、日期等实体
- **消防专项分析**: 针对消防内容的风险分析、分类分析

### 🤖 RAG知识库功能
- **文档分块**: 智能分割长文档为适合检索的块
- **向量化存储**: 支持文档和文档块的向量化存储
- **语义搜索**: 基于语义相似度的文档检索
- **知识问答**: 支持基于知识库的智能问答
- **多模态检索**: 支持按文档类型、分类等条件检索

## 📁 项目结构

```
fire_scraper/
├── fire_scraper/                    # Scrapy项目核心
│   ├── items.py                     # 数据模型定义
│   ├── settings.py                  # 项目设置
│   └── spiders/                     # 爬虫实现
│       └── fire_regulation_spider.py
├── fire_text_analyzer.py            # 消防文本分析工具
├── fire_pipelines.py                # 数据处理管道
├── rag_integration.py               # RAG知识库集成
├── demo_fire_spider.py              # 演示爬虫
├── run_fire_scraper.py              # 真实爬虫运行脚本
├── run_demo_fire.py                 # 演示爬虫运行脚本
└── README.md                        # 项目说明文档
```

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.8+
pip install scrapy pandas openpyxl numpy
```

### 2. 运行演示版本（推荐先试这个）

```bash
cd fire_scraper
python run_demo_fire.py
```

### 3. 运行真实爬虫

```bash
python run_fire_scraper.py
```

### 4. 使用Scrapy命令

```bash
# 运行演示爬虫
scrapy crawl demo_fire

# 运行真实爬虫
scrapy crawl fire_regulation_spider

# 带参数运行
scrapy crawl fire_regulation_spider -a data_types=regulation,standard -a max_pages=3
```

## 📊 输出文件说明

### Excel文件
- `fire_data_YYYYMMDD_HHMMSS.xlsx` (消防数据)
  - 消防法规工作表
  - 消防标准工作表
  - 火灾案例工作表
  - 消防新闻工作表
  - 消防知识工作表
  - RAG文档工作表

### CSV文件
- **消防数据**:
  - `fire_regulations_YYYYMMDD_HHMMSS.csv`: 消防法规数据
  - `fire_standards_YYYYMMDD_HHMMSS.csv`: 消防标准数据
  - `fire_cases_YYYYMMDD_HHMMSS.csv`: 火灾案例数据
  - `fire_news_YYYYMMDD_HHMMSS.csv`: 消防新闻数据
  - `fire_knowledge_YYYYMMDD_HHMMSS.csv`: 消防知识数据
  - `fire_documents_YYYYMMDD_HHMMSS.csv`: RAG文档数据

## 📋 数据字段说明

### 消防法规 (FireRegulationItem)
- `regulation_id`: 法规ID
- `title`: 法规标题
- `regulation_type`: 法规类型 (法律/行政法规/部门规章/地方性法规/标准)
- `level`: 法规层级 (国家级/省级/市级/县级)
- `issuing_authority`: 发布机关
- `issue_date`: 发布日期
- `effective_date`: 生效日期
- `status`: 状态 (现行有效/已废止/已修订)
- `content`: 法规内容
- `summary`: 法规摘要
- `chapters`: 章节信息
- `articles`: 条文信息
- `category`: 分类 (建筑防火/消防设施/消防安全管理/火灾调查等)
- `keywords`: 关键词
- `tags`: 标签

### 消防标准 (FireStandardItem)
- `standard_id`: 标准ID
- `standard_number`: 标准编号 (如GB 50016-2014)
- `title`: 标准名称
- `english_title`: 英文名称
- `standard_type`: 标准类型 (国家标准/行业标准/地方标准/企业标准)
- `category`: 标准分类 (强制性/推荐性)
- `issuing_authority`: 发布机关
- `approval_authority`: 批准机关
- `issue_date`: 发布日期
- `implementation_date`: 实施日期
- `status`: 状态 (现行/废止/修订)
- `scope`: 适用范围
- `content`: 标准内容
- `technical_requirements`: 技术要求
- `test_methods`: 试验方法
- `inspection_rules`: 检验规则
- `fire_category`: 消防分类 (建筑防火/消防设施/消防产品/消防安全等)

### 火灾案例 (FireCaseItem)
- `case_id`: 案例ID
- `title`: 案例标题
- `case_type`: 案例类型 (火灾事故/消防执法/应急救援等)
- `severity_level`: 严重程度 (特别重大/重大/较大/一般)
- `incident_date`: 事故发生时间
- `location`: 事故发生地点
- `province`: 省份
- `city`: 城市
- `district`: 区县
- `building_type`: 建筑类型 (住宅/商业/工业/公共建筑等)
- `fire_cause`: 火灾原因
- `casualties`: 伤亡情况
- `economic_loss`: 经济损失
- `fire_duration`: 火灾持续时间
- `description`: 事故描述
- `investigation_result`: 调查结果
- `lessons_learned`: 经验教训
- `prevention_measures`: 预防措施

### 消防新闻 (FireNewsItem)
- `news_id`: 新闻ID
- `title`: 新闻标题
- `content`: 新闻内容
- `summary`: 新闻摘要
- `author`: 作者
- `source`: 新闻来源
- `publish_time`: 发布时间
- `news_type`: 新闻类型 (政策法规/事故案例/技术动态/行业资讯等)
- `category`: 分类
- `keywords`: 关键词
- `tags`: 标签
- `view_count`: 浏览次数
- `comment_count`: 评论次数
- `share_count`: 分享次数

### 消防知识 (FireKnowledgeItem)
- `knowledge_id`: 知识ID
- `title`: 知识标题
- `content`: 知识内容
- `summary`: 知识摘要
- `knowledge_type`: 知识类型 (基础知识/专业技术/操作指南/应急预案等)
- `category`: 分类 (防火/灭火/逃生/救援等)
- `subcategory`: 子分类
- `difficulty_level`: 难度等级 (初级/中级/高级)
- `target_audience`: 目标受众 (公众/专业人员/管理人员等)
- `content_length`: 内容长度
- `word_count`: 字数统计
- `section_count`: 章节数量
- `image_count`: 图片数量
- `table_count`: 表格数量
- `keywords`: 关键词
- `tags`: 标签
- `entities`: 实体识别结果

### RAG文档 (FireDocumentItem)
- `document_id`: 文档ID
- `title`: 文档标题
- `content`: 文档内容
- `document_type`: 文档类型 (法规/标准/案例/新闻/知识等)
- `content_length`: 内容长度
- `word_count`: 字数统计
- `readability_score`: 可读性得分
- `complexity_score`: 复杂度得分
- `summary`: 文档摘要
- `keywords`: 关键词
- `entities`: 实体识别结果
- `topics`: 主题分析结果
- `embedding_vector`: 嵌入向量
- `chunk_texts`: 分块文本
- `chunk_embeddings`: 分块嵌入向量

## 🔧 参数说明

### 爬虫参数
- `data_types`: 数据类型列表，用逗号分隔（如：regulation,standard,case,news）
- `max_pages`: 最大爬取页数（默认：5）

### 文本分析参数
- `chunk_size`: 文档分块大小（默认：500字符）
- `overlap`: 分块重叠大小（默认：50字符）

## 🧠 RAG知识库使用

### 1. 初始化知识库

```python
from rag_integration import fire_rag_kb, fire_rag_engine

# 知识库已自动初始化
```

### 2. 添加文档到知识库

```python
# 从爬虫结果添加文档
document = {
    'document_id': 'DOC001',
    'title': '消防法规标题',
    'content': '法规内容...',
    'document_type': 'regulation',
    'chunk_texts': ['分块1', '分块2', '分块3']
}

doc_id = fire_rag_kb.add_document(document)
```

### 3. 搜索文档

```python
# 搜索相关文档
results = fire_rag_kb.search_documents("消防监督管理", top_k=5)

# 搜索文档块
chunks = fire_rag_kb.search_chunks("火灾原因", top_k=10)
```

### 4. 智能问答

```python
# 使用查询引擎
answer = fire_rag_engine.query("什么是消防监督管理？", context_type="regulation")
print(answer['answer_summary'])
```

### 5. 获取知识库统计

```python
stats = fire_rag_kb.get_statistics()
print(f"总文档数: {stats['total_documents']}")
print(f"总文档块数: {stats['total_chunks']}")
```

## 🎯 应用场景

### 1. 消防法规检索
- 快速查找相关消防法规条文
- 法规更新和版本对比
- 法规适用性分析

### 2. 火灾案例分析
- 历史火灾案例检索
- 火灾原因统计分析
- 预防措施提取

### 3. 消防标准查询
- 技术标准快速检索
- 标准适用范围查询
- 标准更新跟踪

### 4. 智能问答系统
- 消防知识问答
- 法规条文解释
- 技术标准说明

### 5. 消防培训支持
- 培训材料自动生成
- 知识点提取和整理
- 学习资源推荐

## 🔍 文本分析功能

### 消防文本分析器特性
- **基础文本分析**: 字符数、词数、句子数、段落数统计
- **可读性分析**: 基于句子长度和词汇复杂度的可读性评分
- **复杂度分析**: 基于词汇多样性和句子结构的复杂度评分
- **关键词提取**: 自动提取消防相关关键词
- **实体识别**: 识别法规条文、标准编号、日期等实体
- **消防专项分析**: 针对消防内容的风险分析、分类分析

### 使用方法
```python
from fire_text_analyzer import fire_text_analyzer

# 基础文本分析
result = fire_text_analyzer.analyze_fire_text(text)

# 消防法规分析
result = fire_text_analyzer.analyze_regulation(text)

# 火灾案例分析
result = fire_text_analyzer.analyze_case(text)

# 消防标准分析
result = fire_text_analyzer.analyze_standard(text)

# 文档分块
chunks = fire_text_analyzer.chunk_text_for_rag(text, chunk_size=500, overlap=50)
```

## 📈 扩展功能

### 1. 自定义数据源
- 在 `fire_regulation_spider.py` 中添加新的数据源
- 实现自定义的解析逻辑
- 添加新的数据字段

### 2. 自定义文本分析
- 在 `fire_text_analyzer.py` 中添加新的分析功能
- 扩展关键词词典
- 添加新的实体识别规则

### 3. 自定义存储管道
- 在 `fire_pipelines.py` 中创建新的管道类
- 实现 `process_item` 方法
- 在 `settings.py` 中注册管道

### 4. 集成AI模型
- 集成智谱AI、Kimi等大语言模型
- 实现智能摘要生成
- 添加情感分析和主题分析

## ⚠️ 注意事项

1. **网络请求**: 真实爬虫需要访问消防部门网站，可能受到反爬虫限制
2. **数据准确性**: 演示版本使用模拟数据，真实数据需要连接实际网站
3. **法律合规**: 请遵守相关网站的使用条款和robots.txt规则
4. **数据更新**: 消防法规和标准会定期更新，建议定期重新爬取
5. **存储空间**: 大量文档数据需要足够的存储空间

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件
- 微信交流

---

**消防数据爬虫系统** - 让消防知识触手可及 🔥📚
