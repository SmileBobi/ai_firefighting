# 国家消防救援局（119）科普栏目数据抓取系统

## 🔥 项目简介

本项目专门针对国家消防救援局（119）科普栏目进行数据抓取，包含完整的robots检查、限速控制、HTML到JSON保存等功能。

## ✨ 主要功能

### 🎯 核心功能
- **robots.txt检查** - 自动检查并遵守robots.txt规则
- **智能限速** - 自动调节抓取速度，避免被封
- **多格式输出** - 支持JSON、CSV、SQLite数据库
- **数据清洗** - 自动清洗和格式化数据
- **错误处理** - 完善的错误处理和重试机制

### 🔧 技术特性
- **Scrapy框架** - 使用Scrapy进行高效爬取
- **Requests支持** - 提供requests版本作为备选
- **中间件支持** - 自定义用户代理、重试等中间件
- **管道处理** - 多管道数据存储和处理

## 📁 项目结构

```
crawler/
├── fire_119_scraper.py          # 主爬虫脚本
├── fire_119_scraper/            # Scrapy项目目录
│   ├── settings.py             # Scrapy设置
│   ├── pipelines.py            # 数据管道
│   └── middlewares.py          # 中间件
├── scrapy.cfg                  # Scrapy配置
├── requirements.txt            # 依赖包
├── README.md                  # 说明文档
└── data/                      # 数据输出目录
    ├── fire_119_scrapy.json   # Scrapy输出
    ├── fire_119_data.csv      # CSV输出
    └── fire_119.db           # SQLite数据库
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行爬虫

#### 方式1: 使用主脚本（推荐）
```bash
python fire_119_scraper.py
```

#### 方式2: 使用Scrapy命令
```bash
scrapy crawl fire_119_spider
```

#### 方式3: 使用requests版本
```bash
python fire_119_scraper.py
# 选择 "2. Requests爬虫"
```

### 3. 查看结果
数据将保存在 `data/` 目录下：
- `fire_119_scrapy.json` - JSON格式数据
- `fire_119_data.csv` - CSV格式数据
- `fire_119.db` - SQLite数据库

## 🔧 配置说明

### robots.txt检查
```python
# 自动检查robots.txt
ROBOTSTXT_OBEY = True

# 检查特定URL
if self.robots_parser.can_fetch('*', url):
    # 允许抓取
    pass
```

### 限速设置
```python
# 下载延迟
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = 0.5

# 自动限速
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
```

### 并发控制
```python
# 并发请求数
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1
```

## 📊 数据格式

### JSON格式示例
```json
{
  "url": "https://www.119.gov.cn/kp/article/123.html",
  "title": "消防安全知识科普",
  "content": "文章内容...",
  "publish_time": "2024-01-01",
  "author": "消防局",
  "category": "科普",
  "tags": ["安全", "消防", "知识"],
  "images": ["https://www.119.gov.cn/images/1.jpg"],
  "crawl_time": "2024-01-01T10:00:00",
  "source": "国家消防救援局官网"
}
```

### CSV格式
| url | title | content | publish_time | author | category | tags | images | crawl_time | source |
|-----|-------|---------|--------------|--------|----------|------|--------|------------|--------|
| https://... | 消防安全知识 | 文章内容... | 2024-01-01 | 消防局 | 科普 | 安全,消防 | https://... | 2024-01-01T10:00:00 | 国家消防救援局官网 |

## 🛠️ 高级功能

### 自定义中间件
```python
class Fire119UserAgentMiddleware(UserAgentMiddleware):
    """自定义用户代理中间件"""
    def process_request(self, request, spider):
        ua = random.choice(self.user_agent_list)
        request.headers['User-Agent'] = ua
        return None
```

### 数据管道
```python
class Fire119Pipeline:
    """数据管道"""
    def process_item(self, item, spider):
        # 数据验证
        if not self.validate_item(item):
            raise DropItem(f"数据验证失败: {item}")
        
        # 数据清洗
        item = self.clean_item(item)
        
        return item
```

### 错误处理
```python
def process_exception(self, request, exception, spider):
    """处理异常"""
    logger.error(f"请求异常: {exception} - {request.url}")
    return None
```

## 📈 性能优化

### 1. 并发控制
- 限制并发请求数
- 使用连接池
- 合理设置超时时间

### 2. 内存管理
- 批量保存数据
- 及时清理缓存
- 使用生成器

### 3. 网络优化
- 使用keep-alive连接
- 启用gzip压缩
- 设置合理的超时时间

## 🔒 安全考虑

### 1. robots.txt遵守
- 自动检查robots.txt
- 遵守爬取规则
- 尊重网站政策

### 2. 限速控制
- 随机延迟
- 自动限速
- 避免对服务器造成压力

### 3. 用户代理轮换
- 随机用户代理
- 模拟真实浏览器
- 避免被识别为爬虫

## 🐛 常见问题

### Q: 爬虫被网站封禁怎么办？
A: 
1. 增加延迟时间
2. 更换用户代理
3. 使用代理IP
4. 检查robots.txt

### Q: 数据不完整怎么办？
A: 
1. 检查选择器是否正确
2. 增加重试机制
3. 检查网站结构变化
4. 使用更宽松的选择器

### Q: 爬取速度太慢怎么办？
A: 
1. 适当增加并发数
2. 减少延迟时间
3. 使用异步请求
4. 优化选择器

## 📞 技术支持

### 日志查看
```bash
# 查看Scrapy日志
tail -f logs/scrapy.log

# 查看详细日志
scrapy crawl fire_119_spider -L INFO
```

### 调试模式
```python
# 启用调试模式
AUTOTHROTTLE_DEBUG = True
LOG_LEVEL = 'DEBUG'
```

### 数据验证
```python
# 验证数据完整性
def validate_item(self, item):
    required_fields = ['url', 'title', 'content']
    for field in required_fields:
        if not item.get(field):
            return False
    return True
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

**注意**: 使用前请确保遵守相关法律法规和网站使用条款，尊重网站robots.txt规则。
