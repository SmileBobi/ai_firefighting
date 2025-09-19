# 消防数据爬虫数据库集成指南

本指南详细说明如何将消防数据爬虫系统与数据库集成，支持MySQL和SQLite两种数据库方案。

## 🗄️ 数据库支持

### 1. SQLite数据库（推荐用于开发和小规模部署）

**优势：**
- ✅ 无需安装额外软件，Python内置支持
- ✅ 单文件数据库，便于备份和迁移
- ✅ 零配置，开箱即用
- ✅ 适合开发、测试和小规模生产环境

**使用场景：**
- 开发测试环境
- 小规模数据存储（< 1GB）
- 单机部署
- 快速原型开发

### 2. MySQL数据库（推荐用于生产环境）

**优势：**
- ✅ 高性能，支持大规模数据
- ✅ 多用户并发访问
- ✅ 丰富的SQL功能
- ✅ 企业级稳定性

**使用场景：**
- 生产环境
- 大规模数据存储（> 1GB）
- 多用户并发访问
- 企业级应用

## 🚀 快速开始

### 方案一：SQLite数据库（推荐新手）

#### 1. 运行SQLite版本爬虫

```bash
cd fire_scraper
python run_demo_fire_sqlite.py
```

#### 2. 查询数据库

```bash
python query_database.py
```

#### 3. 查看生成的文件

- **SQLite数据库**: `fire_data.db`
- **Excel文件**: `fire_data_YYYYMMDD_HHMMSS.xlsx`
- **CSV文件**: `fire_regulations_*.csv`, `fire_standards_*.csv` 等

### 方案二：MySQL数据库

#### 1. 安装MySQL驱动

```bash
pip install PyMySQL
```

#### 2. 配置数据库连接

```bash
python setup_database.py
```

#### 3. 运行MySQL版本爬虫

```bash
python run_demo_fire_mysql.py
```

## 📊 数据库表结构

### 消防法规表 (fire_regulations)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| regulation_id | VARCHAR(50) | 法规ID，唯一 |
| title | VARCHAR(500) | 法规标题 |
| regulation_type | VARCHAR(100) | 法规类型 |
| level | VARCHAR(50) | 法规层级 |
| issuing_authority | VARCHAR(200) | 发布机关 |
| issue_date | VARCHAR(50) | 发布日期 |
| effective_date | VARCHAR(50) | 生效日期 |
| status | VARCHAR(50) | 状态 |
| content | LONGTEXT | 法规内容 |
| summary | TEXT | 法规摘要 |
| chapters | JSON | 章节信息 |
| articles | JSON | 条文信息 |
| category | VARCHAR(100) | 分类 |
| keywords | JSON | 关键词 |
| tags | JSON | 标签 |
| file_url | VARCHAR(500) | 文件URL |
| file_type | VARCHAR(50) | 文件类型 |
| file_size | INTEGER | 文件大小 |
| download_status | VARCHAR(50) | 下载状态 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 消防标准表 (fire_standards)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| standard_id | VARCHAR(50) | 标准ID，唯一 |
| standard_number | VARCHAR(100) | 标准编号 |
| title | VARCHAR(500) | 标准名称 |
| english_title | VARCHAR(500) | 英文名称 |
| standard_type | VARCHAR(100) | 标准类型 |
| category | VARCHAR(100) | 标准分类 |
| issuing_authority | VARCHAR(200) | 发布机关 |
| approval_authority | VARCHAR(200) | 批准机关 |
| issue_date | VARCHAR(50) | 发布日期 |
| implementation_date | VARCHAR(50) | 实施日期 |
| status | VARCHAR(50) | 状态 |
| scope | TEXT | 适用范围 |
| content | LONGTEXT | 标准内容 |
| technical_requirements | JSON | 技术要求 |
| test_methods | JSON | 试验方法 |
| inspection_rules | JSON | 检验规则 |
| fire_category | VARCHAR(100) | 消防分类 |
| keywords | JSON | 关键词 |
| tags | JSON | 标签 |
| file_url | VARCHAR(500) | 文件URL |
| file_type | VARCHAR(50) | 文件类型 |
| file_size | INTEGER | 文件大小 |
| download_status | VARCHAR(50) | 下载状态 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 火灾案例表 (fire_cases)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| case_id | VARCHAR(50) | 案例ID，唯一 |
| title | VARCHAR(500) | 案例标题 |
| case_type | VARCHAR(100) | 案例类型 |
| severity_level | VARCHAR(50) | 严重程度 |
| incident_date | VARCHAR(50) | 事故发生时间 |
| location | VARCHAR(200) | 事故发生地点 |
| province | VARCHAR(50) | 省份 |
| city | VARCHAR(50) | 城市 |
| district | VARCHAR(50) | 区县 |
| building_type | VARCHAR(100) | 建筑类型 |
| fire_cause | VARCHAR(200) | 火灾原因 |
| casualties | JSON | 伤亡情况 |
| economic_loss | VARCHAR(100) | 经济损失 |
| fire_duration | VARCHAR(50) | 火灾持续时间 |
| description | TEXT | 事故描述 |
| investigation_result | TEXT | 调查结果 |
| lessons_learned | TEXT | 经验教训 |
| prevention_measures | TEXT | 预防措施 |
| content | LONGTEXT | 案例内容 |
| keywords | JSON | 关键词 |
| tags | JSON | 标签 |
| file_url | VARCHAR(500) | 文件URL |
| file_type | VARCHAR(50) | 文件类型 |
| file_size | INTEGER | 文件大小 |
| download_status | VARCHAR(50) | 下载状态 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 消防新闻表 (fire_news)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| news_id | VARCHAR(50) | 新闻ID，唯一 |
| title | VARCHAR(500) | 新闻标题 |
| content | LONGTEXT | 新闻内容 |
| summary | TEXT | 新闻摘要 |
| author | VARCHAR(200) | 作者 |
| source | VARCHAR(200) | 新闻来源 |
| publish_time | VARCHAR(50) | 发布时间 |
| news_type | VARCHAR(100) | 新闻类型 |
| category | VARCHAR(100) | 分类 |
| keywords | JSON | 关键词 |
| tags | JSON | 标签 |
| view_count | INTEGER | 浏览次数 |
| comment_count | INTEGER | 评论次数 |
| share_count | INTEGER | 分享次数 |
| image_urls | JSON | 图片URL列表 |
| file_urls | JSON | 文件URL列表 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 消防知识表 (fire_knowledge)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| knowledge_id | VARCHAR(50) | 知识ID，唯一 |
| title | VARCHAR(500) | 知识标题 |
| content | LONGTEXT | 知识内容 |
| summary | TEXT | 知识摘要 |
| knowledge_type | VARCHAR(100) | 知识类型 |
| category | VARCHAR(100) | 分类 |
| subcategory | VARCHAR(100) | 子分类 |
| difficulty_level | VARCHAR(50) | 难度等级 |
| target_audience | VARCHAR(100) | 目标受众 |
| content_length | INTEGER | 内容长度 |
| word_count | INTEGER | 字数统计 |
| section_count | INTEGER | 章节数量 |
| image_count | INTEGER | 图片数量 |
| table_count | INTEGER | 表格数量 |
| keywords | JSON | 关键词 |
| tags | JSON | 标签 |
| entities | JSON | 实体识别结果 |
| file_url | VARCHAR(500) | 文件URL |
| file_type | VARCHAR(50) | 文件类型 |
| file_size | INTEGER | 文件大小 |
| download_status | VARCHAR(50) | 下载状态 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### RAG文档表 (fire_documents)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INTEGER | 主键，自增 |
| document_id | VARCHAR(50) | 文档ID，唯一 |
| title | VARCHAR(500) | 文档标题 |
| content | LONGTEXT | 文档内容 |
| document_type | VARCHAR(100) | 文档类型 |
| content_length | INTEGER | 内容长度 |
| word_count | INTEGER | 字数统计 |
| readability_score | DECIMAL(5,2) | 可读性得分 |
| complexity_score | DECIMAL(5,2) | 复杂度得分 |
| summary | TEXT | 文档摘要 |
| keywords | JSON | 关键词 |
| entities | JSON | 实体识别结果 |
| topics | JSON | 主题分析结果 |
| embedding_vector | JSON | 嵌入向量 |
| chunk_texts | JSON | 分块文本 |
| chunk_embeddings | JSON | 分块嵌入向量 |
| file_url | VARCHAR(500) | 文件URL |
| file_type | VARCHAR(50) | 文件类型 |
| file_size | INTEGER | 文件大小 |
| download_status | VARCHAR(50) | 下载状态 |
| scraped_at | DATETIME | 爬取时间 |
| source_url | VARCHAR(500) | 来源URL |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

## 🔧 配置说明

### SQLite配置

在 `settings.py` 中配置：

```python
# SQLite数据库配置
SQLITE_DB_PATH = 'fire_data.db'  # 数据库文件路径
```

### MySQL配置

在 `settings.py` 中配置：

```python
# MySQL数据库配置
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'your_password'
MYSQL_DATABASE = 'fire_data'
```

### 管道配置

在 `settings.py` 中配置管道：

```python
# SQLite版本
ITEM_PIPELINES = {
    'fire_pipelines.FireExcelWriterPipeline': 400,
    'fire_pipelines.FireCsvWriterPipeline': 500,
    'fire_pipelines.FireConsolePipeline': 600,
    'fire_pipelines.FireTextAnalysisPipeline': 700,
    'fire_pipelines.FireRAGPipeline': 800,
    'fire_pipelines.FireDuplicatesPipeline': 900,
    'sqlite_pipeline.SQLitePipeline': 1000,
}

# MySQL版本
ITEM_PIPELINES = {
    'fire_pipelines.FireExcelWriterPipeline': 400,
    'fire_pipelines.FireCsvWriterPipeline': 500,
    'fire_pipelines.FireConsolePipeline': 600,
    'fire_pipelines.FireTextAnalysisPipeline': 700,
    'fire_pipelines.FireRAGPipeline': 800,
    'fire_pipelines.FireDuplicatesPipeline': 900,
    'mysql_pipeline.MySQLPipeline': 1000,
}
```

## 🛠️ 工具使用

### 1. 数据库查询工具

```bash
python query_database.py
```

功能：
- 查看表信息
- 查询表数据
- 获取统计信息
- 搜索内容
- 导出数据到CSV

### 2. 数据库管理工具

```bash
python database_manager.py
```

功能：
- 设置数据库连接
- 测试数据库连接
- 创建数据库表
- 导入CSV数据
- 导出数据到CSV
- 查看数据库统计

### 3. 数据库设置工具

```bash
python setup_database.py
```

功能：
- 完整设置（安装驱动 + 配置数据库 + 创建表）
- 仅测试数据库连接
- 导入示例数据
- 运行数据库管理工具

## 📈 数据分析示例

### 1. 基础查询

```sql
-- 查询所有消防法规
SELECT regulation_id, title, regulation_type, issuing_authority 
FROM fire_regulations 
ORDER BY created_at DESC;

-- 查询消防标准
SELECT standard_number, title, standard_type, fire_category 
FROM fire_standards 
WHERE status = '现行';

-- 查询火灾案例
SELECT case_id, title, severity_level, fire_cause, economic_loss 
FROM fire_cases 
WHERE severity_level = '重大';
```

### 2. 统计分析

```sql
-- 按法规类型统计
SELECT regulation_type, COUNT(*) as count 
FROM fire_regulations 
GROUP BY regulation_type 
ORDER BY count DESC;

-- 按发布机关统计
SELECT issuing_authority, COUNT(*) as count 
FROM fire_regulations 
GROUP BY issuing_authority 
ORDER BY count DESC 
LIMIT 10;

-- 按消防分类统计
SELECT fire_category, COUNT(*) as count 
FROM fire_standards 
GROUP BY fire_category 
ORDER BY count DESC;
```

### 3. 内容搜索

```sql
-- 搜索包含特定关键词的法规
SELECT regulation_id, title, summary 
FROM fire_regulations 
WHERE content LIKE '%消防监督管理%' 
   OR title LIKE '%消防监督管理%';

-- 搜索特定类型的标准
SELECT standard_number, title, scope 
FROM fire_standards 
WHERE fire_category = '建筑防火' 
  AND standard_type = '国家标准';
```

## 🔍 性能优化

### 1. 索引优化

数据库表已自动创建以下索引：

- `idx_regulation_id` - 法规ID索引
- `idx_title` - 标题索引
- `idx_category` - 分类索引
- `idx_scraped_at` - 爬取时间索引

### 2. 查询优化

- 使用LIMIT限制查询结果数量
- 避免SELECT *，只查询需要的字段
- 使用WHERE条件过滤数据
- 合理使用ORDER BY

### 3. 存储优化

- 定期清理过期数据
- 压缩大文本字段
- 使用JSON字段存储结构化数据

## 🚨 注意事项

### 1. 数据备份

```bash
# SQLite备份
cp fire_data.db fire_data_backup_$(date +%Y%m%d).db

# MySQL备份
mysqldump -u root -p fire_data > fire_data_backup_$(date +%Y%m%d).sql
```

### 2. 数据迁移

```bash
# 从SQLite迁移到MySQL
python database_manager.py
# 选择"导入CSV数据"功能
```

### 3. 性能监控

- 监控数据库文件大小
- 定期检查查询性能
- 优化慢查询

## 📚 扩展功能

### 1. 数据同步

可以设置定时任务，定期同步数据：

```bash
# 每天凌晨2点运行爬虫
0 2 * * * cd /path/to/fire_scraper && python run_fire_scraper.py
```

### 2. 数据API

可以基于数据库开发REST API：

```python
from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/regulations')
def get_regulations():
    conn = sqlite3.connect('fire_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fire_regulations LIMIT 100")
    results = cursor.fetchall()
    conn.close()
    return jsonify(results)
```

### 3. 数据可视化

使用pandas和matplotlib进行数据可视化：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_sql_query("SELECT * FROM fire_regulations", conn)

# 创建图表
df['regulation_type'].value_counts().plot(kind='bar')
plt.title('消防法规类型分布')
plt.show()
```

## 🆘 故障排除

### 1. 连接问题

**SQLite连接失败：**
- 检查文件权限
- 确保磁盘空间充足
- 检查文件是否被其他程序占用

**MySQL连接失败：**
- 检查MySQL服务是否启动
- 验证用户名和密码
- 检查网络连接
- 确认数据库存在

### 2. 性能问题

**查询慢：**
- 添加索引
- 优化查询语句
- 增加内存配置

**插入慢：**
- 使用批量插入
- 调整事务大小
- 优化管道配置

### 3. 数据问题

**数据重复：**
- 检查唯一约束
- 使用ON DUPLICATE KEY UPDATE
- 清理重复数据

**数据丢失：**
- 检查管道配置
- 查看错误日志
- 验证数据源

---

通过本指南，您可以成功将消防数据爬虫系统与数据库集成，实现数据的持久化存储和高效查询。根据您的需求选择合适的数据库方案，并按照指南进行配置和使用。

