"""
消防119数据管道
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List
from scrapy.exceptions import DropItem

logger = logging.getLogger(__name__)

class Fire119Pipeline:
    """消防119数据管道"""
    
    def __init__(self):
        """初始化管道"""
        self.items = []
        self.output_dir = "data"
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def process_item(self, item, spider):
        """处理单个项目"""
        try:
            # 数据验证
            if not self.validate_item(item):
                raise DropItem(f"数据验证失败: {item}")
            
            # 数据清洗
            item = self.clean_item(item)
            
            # 添加到列表
            self.items.append(dict(item))
            
            # 每100个项目保存一次
            if len(self.items) >= 100:
                self.save_batch()
            
            return item
            
        except Exception as e:
            logger.error(f"处理项目失败: {e}")
            raise DropItem(f"处理项目失败: {e}")
    
    def validate_item(self, item: Dict) -> bool:
        """验证数据项"""
        required_fields = ['url', 'title', 'content']
        
        for field in required_fields:
            if not item.get(field):
                logger.warning(f"缺少必需字段: {field}")
                return False
        
        return True
    
    def clean_item(self, item: Dict) -> Dict:
        """清洗数据项"""
        # 清理标题
        if item.get('title'):
            item['title'] = item['title'].strip()
        
        # 清理内容
        if item.get('content'):
            item['content'] = item['content'].strip()
        
        # 清理时间格式
        if item.get('publish_time'):
            item['publish_time'] = self.clean_time_format(item['publish_time'])
        
        # 添加爬取时间
        item['crawl_time'] = datetime.now().isoformat()
        
        return item
    
    def clean_time_format(self, time_str: str) -> str:
        """清理时间格式"""
        import re
        
        # 提取时间部分
        time_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}/\d{2}/\d{2})',
            r'(\d{4}年\d{2}月\d{2}日)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, time_str)
            if match:
                return match.group(1)
        
        return time_str
    
    def save_batch(self):
        """批量保存数据"""
        if self.items:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_119_batch_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.items, f, ensure_ascii=False, indent=2)
                logger.info(f"批量数据已保存到: {filepath}")
                self.items = []
            except Exception as e:
                logger.error(f"保存批量数据失败: {e}")
    
    def close_spider(self, spider):
        """爬虫关闭时保存剩余数据"""
        if self.items:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_119_final_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.items, f, ensure_ascii=False, indent=2)
                logger.info(f"最终数据已保存到: {filepath}")
                self.items = []
            except Exception as e:
                logger.error(f"保存最终数据失败: {e}")

class Fire119CSVPipeline:
    """消防119 CSV数据管道"""
    
    def __init__(self):
        """初始化管道"""
        self.items = []
        self.output_dir = "data"
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_item(self, item, spider):
        """处理单个项目"""
        self.items.append(dict(item))
        return item
    
    def close_spider(self, spider):
        """爬虫关闭时保存数据"""
        if self.items:
            try:
                import pandas as pd
                df = pd.DataFrame(self.items)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fire_119_data_{timestamp}.csv"
                filepath = os.path.join(self.output_dir, filename)
                
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                logger.info(f"CSV数据已保存到: {filepath}")
                
            except Exception as e:
                logger.error(f"保存CSV数据失败: {e}")

class Fire119DatabasePipeline:
    """消防119数据库管道"""
    
    def __init__(self):
        """初始化管道"""
        self.items = []
        self.db_path = "data/fire_119.db"
        self.ensure_db()
    
    def ensure_db(self):
        """确保数据库存在"""
        os.makedirs("data", exist_ok=True)
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fire_119_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    publish_time TEXT,
                    author TEXT,
                    category TEXT,
                    tags TEXT,
                    images TEXT,
                    crawl_time TEXT,
                    source TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
    
    def process_item(self, item, spider):
        """处理单个项目"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 插入数据
            cursor.execute('''
                INSERT OR REPLACE INTO fire_119_articles 
                (url, title, content, publish_time, author, category, tags, images, crawl_time, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('url', ''),
                item.get('title', ''),
                item.get('content', ''),
                item.get('publish_time', ''),
                item.get('author', ''),
                item.get('category', ''),
                json.dumps(item.get('tags', []), ensure_ascii=False),
                json.dumps(item.get('images', []), ensure_ascii=False),
                item.get('crawl_time', ''),
                item.get('source', '')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存到数据库失败: {e}")
        
        return item



