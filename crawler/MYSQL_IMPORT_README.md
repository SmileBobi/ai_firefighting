# 消防119数据导入MySQL数据库

## 🔥 项目简介

本程序用于将 `fire_119_scrapy.json` 数据导入到 MySQL 数据库中，支持批量导入、数据清洗、重复处理等功能。

## ✨ 主要功能

### 🎯 核心功能
- **JSON数据导入** - 将爬取的JSON数据导入MySQL
- **数据清洗** - 自动清洗和格式化数据
- **重复处理** - 支持更新已存在的记录
- **批量导入** - 高效的批量数据处理
- **数据统计** - 提供详细的导入统计信息

### 🔧 技术特性
- **MySQL连接** - 支持本地和远程MySQL数据库
- **字符编码** - 支持UTF-8中文数据
- **事务处理** - 确保数据一致性
- **错误处理** - 完善的错误处理和日志记录

## 📁 项目结构

```
crawler/
├── json_to_mysql.py          # 完整版导入程序
├── run_mysql_import.py       # 简化版导入程序
├── mysql_config.json         # 数据库配置文件
├── mysql_setup.sql          # 数据库设置脚本
├── requirements_mysql.txt   # MySQL依赖包
├── MYSQL_IMPORT_README.md   # 说明文档
└── data/
    └── fire_119_scrapy.json # 爬取的数据文件
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements_mysql.txt
```

### 2. 准备MySQL数据库
```bash
# 方法1: 使用SQL脚本
mysql -u root -p < mysql_setup.sql

# 方法2: 使用程序自动创建
python run_mysql_import.py
```

### 3. 运行导入程序

#### 简化版（推荐）
```bash
python run_mysql_import.py
```

#### 完整版
```bash
python json_to_mysql.py
```

## 🔧 配置说明

### 数据库配置
```json
{
  "database": {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "your_password",
    "database": "firefighting_db"
  }
}
```

### 环境变量配置
```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=your_password
export MYSQL_DATABASE=firefighting_db
```

## 📊 数据表结构

### fire_119_articles 表
```sql
CREATE TABLE fire_119_articles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    url VARCHAR(500) NOT NULL UNIQUE,
    title VARCHAR(500) NOT NULL,
    content LONGTEXT,
    publish_time VARCHAR(100),
    author VARCHAR(200),
    category VARCHAR(100),
    tags JSON,
    images JSON,
    crawl_time DATETIME,
    source VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## 📈 使用示例

### 基本导入
```python
from json_to_mysql import Fire119DataImporter

# 配置数据库
config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'password',
    'database': 'firefighting_db'
}

# 创建导入器
importer = Fire119DataImporter(config)

# 连接数据库
importer.connect_database()

# 创建表
importer.create_table()

# 加载数据
data = importer.load_json_data('data/fire_119_scrapy.json')

# 导入数据
success_count, error_count = importer.insert_data(data)

# 关闭连接
importer.close_connection()
```

### 批量导入
```python
# 批量导入，每批100条记录
success_count, error_count = importer.insert_data(data, batch_size=100)
```

### 获取统计信息
```python
# 获取数据统计
stats = importer.get_statistics()
print(f"总记录数: {stats['total_count']}")
```

## 🛠️ 高级功能

### 1. 数据清洗
```python
def clean_data(self, item):
    """自定义数据清洗逻辑"""
    # 清理字符串字段
    for field in ['url', 'title', 'content']:
        if field in item:
            item[field] = str(item[field]).strip()
    
    # 处理时间字段
    if item.get('crawl_time'):
        item['crawl_time'] = datetime.fromisoformat(item['crawl_time'])
    
    return item
```

### 2. 重复处理
```python
# 使用 ON DUPLICATE KEY UPDATE 处理重复数据
INSERT INTO fire_119_articles (...) VALUES (...)
ON DUPLICATE KEY UPDATE
title = VALUES(title),
content = VALUES(content),
updated_at = CURRENT_TIMESTAMP
```

### 3. 错误处理
```python
try:
    success_count, error_count = importer.insert_data(data)
except Exception as e:
    logger.error(f"导入失败: {e}")
```

## 📊 数据统计

### 导入统计
- 总记录数
- 成功导入数
- 失败导入数
- 按分类统计
- 按来源统计

### 查询示例
```sql
-- 查询总记录数
SELECT COUNT(*) FROM fire_119_articles;

-- 按分类统计
SELECT category, COUNT(*) as count 
FROM fire_119_articles 
GROUP BY category 
ORDER BY count DESC;

-- 查询最新记录
SELECT title, publish_time, crawl_time 
FROM fire_119_articles 
ORDER BY crawl_time DESC 
LIMIT 10;
```

## 🔒 安全考虑

### 1. 数据库安全
- 使用参数化查询防止SQL注入
- 设置合适的数据库权限
- 定期备份数据

### 2. 连接安全
- 使用SSL连接（可选）
- 设置连接超时
- 限制并发连接数

### 3. 数据安全
- 数据加密存储
- 访问日志记录
- 定期安全审计

## 🐛 常见问题

### Q: 连接数据库失败怎么办？
A: 
1. 检查MySQL服务是否启动
2. 验证用户名和密码
3. 检查网络连接
4. 确认数据库存在

### Q: 导入数据失败怎么办？
A: 
1. 检查JSON文件格式
2. 验证数据字段完整性
3. 查看错误日志
4. 检查数据库表结构

### Q: 数据重复怎么办？
A: 
1. 使用 `ON DUPLICATE KEY UPDATE`
2. 先删除重复数据
3. 使用唯一索引
4. 数据去重处理

## 📞 技术支持

### 日志查看
```bash
# 查看导入日志
tail -f logs/mysql_import.log

# 查看数据库日志
tail -f /var/log/mysql/error.log
```

### 调试模式
```python
# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 测试连接
importer.test_connection()
```

### 性能优化
```python
# 批量大小调整
batch_size = 100  # 根据内存调整

# 索引优化
CREATE INDEX idx_url ON fire_119_articles(url);
CREATE INDEX idx_title ON fire_119_articles(title);
```

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork本项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

---

**注意**: 使用前请确保MySQL数据库已正确安装和配置，并具有相应的权限。



