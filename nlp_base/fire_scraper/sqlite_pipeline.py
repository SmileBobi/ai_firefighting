#!/usr/bin/env python
"""
SQLite数据库管道
用于将爬虫数据插入到SQLite数据库中（无需额外安装，Python内置支持）
"""

import sqlite3
import json
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
import logging


class SQLitePipeline:
    """SQLite数据库存储管道"""
    
    def __init__(self, db_path="fire_data.db"):
        """初始化SQLite连接参数"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_crawler(cls, crawler):
        """从爬虫设置中获取SQLite配置"""
        return cls(
            db_path=crawler.settings.get('SQLITE_DB_PATH', 'fire_data.db')
        )
    
    def open_spider(self, spider):
        """爬虫开始时建立数据库连接"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            self.logger.info(f"SQLite数据库连接成功: {self.db_path}")
            
            # 创建数据库表
            self.create_tables()
            
        except Exception as e:
            self.logger.error(f"SQLite数据库连接失败: {e}")
            raise
    
    def close_spider(self, spider):
        """爬虫结束时关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("SQLite数据库连接已关闭")
    
    def create_tables(self):
        """创建数据库表"""
        try:
            # 创建消防法规表
            create_regulations_table = """
            CREATE TABLE IF NOT EXISTS fire_regulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regulation_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                regulation_type TEXT,
                level TEXT,
                issuing_authority TEXT,
                issue_date TEXT,
                effective_date TEXT,
                status TEXT,
                content TEXT,
                summary TEXT,
                chapters TEXT,
                articles TEXT,
                category TEXT,
                keywords TEXT,
                tags TEXT,
                file_url TEXT,
                file_type TEXT,
                file_size INTEGER,
                download_status TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 创建消防标准表
            create_standards_table = """
            CREATE TABLE IF NOT EXISTS fire_standards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                standard_id TEXT UNIQUE NOT NULL,
                standard_number TEXT NOT NULL,
                title TEXT NOT NULL,
                english_title TEXT,
                standard_type TEXT,
                category TEXT,
                issuing_authority TEXT,
                approval_authority TEXT,
                issue_date TEXT,
                implementation_date TEXT,
                status TEXT,
                scope TEXT,
                content TEXT,
                technical_requirements TEXT,
                test_methods TEXT,
                inspection_rules TEXT,
                fire_category TEXT,
                keywords TEXT,
                tags TEXT,
                file_url TEXT,
                file_type TEXT,
                file_size INTEGER,
                download_status TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 创建火灾案例表
            create_cases_table = """
            CREATE TABLE IF NOT EXISTS fire_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                case_type TEXT,
                severity_level TEXT,
                incident_date TEXT,
                location TEXT,
                province TEXT,
                city TEXT,
                district TEXT,
                building_type TEXT,
                fire_cause TEXT,
                casualties TEXT,
                economic_loss TEXT,
                fire_duration TEXT,
                description TEXT,
                investigation_result TEXT,
                lessons_learned TEXT,
                prevention_measures TEXT,
                content TEXT,
                keywords TEXT,
                tags TEXT,
                file_url TEXT,
                file_type TEXT,
                file_size INTEGER,
                download_status TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 创建消防新闻表
            create_news_table = """
            CREATE TABLE IF NOT EXISTS fire_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                author TEXT,
                source TEXT,
                publish_time TEXT,
                news_type TEXT,
                category TEXT,
                keywords TEXT,
                tags TEXT,
                view_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                share_count INTEGER DEFAULT 0,
                image_urls TEXT,
                file_urls TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 创建消防知识表
            create_knowledge_table = """
            CREATE TABLE IF NOT EXISTS fire_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                knowledge_type TEXT,
                category TEXT,
                subcategory TEXT,
                difficulty_level TEXT,
                target_audience TEXT,
                content_length INTEGER,
                word_count INTEGER,
                section_count INTEGER,
                image_count INTEGER,
                table_count INTEGER,
                keywords TEXT,
                tags TEXT,
                entities TEXT,
                file_url TEXT,
                file_type TEXT,
                file_size INTEGER,
                download_status TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 创建RAG文档表
            create_documents_table = """
            CREATE TABLE IF NOT EXISTS fire_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                document_type TEXT,
                content_length INTEGER,
                word_count INTEGER,
                readability_score REAL,
                complexity_score REAL,
                summary TEXT,
                keywords TEXT,
                entities TEXT,
                topics TEXT,
                embedding_vector TEXT,
                chunk_texts TEXT,
                chunk_embeddings TEXT,
                file_url TEXT,
                file_type TEXT,
                file_size INTEGER,
                download_status TEXT,
                scraped_at TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # 执行创建表语句
            self.cursor.execute(create_regulations_table)
            self.cursor.execute(create_standards_table)
            self.cursor.execute(create_cases_table)
            self.cursor.execute(create_news_table)
            self.cursor.execute(create_knowledge_table)
            self.cursor.execute(create_documents_table)
            
            self.connection.commit()
            self.logger.info("SQLite数据库表创建成功")
            
        except Exception as e:
            self.logger.error(f"创建SQLite数据库表失败: {e}")
            raise
    
    def process_item(self, item, spider):
        """处理每个item并插入数据库"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        try:
            # 根据item类型选择对应的表
            if 'regulation_id' in item_dict:
                self.insert_regulation(item_dict)
            elif 'standard_id' in item_dict:
                self.insert_standard(item_dict)
            elif 'case_id' in item_dict:
                self.insert_case(item_dict)
            elif 'news_id' in item_dict:
                self.insert_news(item_dict)
            elif 'knowledge_id' in item_dict:
                self.insert_knowledge(item_dict)
            elif 'document_id' in item_dict:
                self.insert_document(item_dict)
            else:
                self.logger.warning(f"未知的item类型: {item_dict}")
                return item
            
            return item
            
        except Exception as e:
            self.logger.error(f"插入SQLite数据库失败: {e}")
            raise DropItem(f"数据库插入失败: {e}")
    
    def insert_regulation(self, item_dict):
        """插入消防法规数据"""
        sql = """
        INSERT OR REPLACE INTO fire_regulations (
            regulation_id, title, regulation_type, level, issuing_authority,
            issue_date, effective_date, status, content, summary, chapters,
            articles, category, keywords, tags, file_url, file_type,
            file_size, download_status, scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('regulation_id'),
            item_dict.get('title'),
            item_dict.get('regulation_type'),
            item_dict.get('level'),
            item_dict.get('issuing_authority'),
            item_dict.get('issue_date'),
            item_dict.get('effective_date'),
            item_dict.get('status'),
            item_dict.get('content'),
            item_dict.get('summary'),
            json.dumps(item_dict.get('chapters', []), ensure_ascii=False),
            json.dumps(item_dict.get('articles', []), ensure_ascii=False),
            item_dict.get('category'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('tags', []), ensure_ascii=False),
            item_dict.get('file_url'),
            item_dict.get('file_type'),
            item_dict.get('file_size'),
            item_dict.get('download_status'),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"消防法规数据插入成功: {item_dict.get('regulation_id')}")
    
    def insert_standard(self, item_dict):
        """插入消防标准数据"""
        sql = """
        INSERT OR REPLACE INTO fire_standards (
            standard_id, standard_number, title, english_title, standard_type,
            category, issuing_authority, approval_authority, issue_date,
            implementation_date, status, scope, content, technical_requirements,
            test_methods, inspection_rules, fire_category, keywords, tags,
            file_url, file_type, file_size, download_status, scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('standard_id'),
            item_dict.get('standard_number'),
            item_dict.get('title'),
            item_dict.get('english_title'),
            item_dict.get('standard_type'),
            item_dict.get('category'),
            item_dict.get('issuing_authority'),
            item_dict.get('approval_authority'),
            item_dict.get('issue_date'),
            item_dict.get('implementation_date'),
            item_dict.get('status'),
            item_dict.get('scope'),
            item_dict.get('content'),
            json.dumps(item_dict.get('technical_requirements', []), ensure_ascii=False),
            json.dumps(item_dict.get('test_methods', []), ensure_ascii=False),
            json.dumps(item_dict.get('inspection_rules', []), ensure_ascii=False),
            item_dict.get('fire_category'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('tags', []), ensure_ascii=False),
            item_dict.get('file_url'),
            item_dict.get('file_type'),
            item_dict.get('file_size'),
            item_dict.get('download_status'),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"消防标准数据插入成功: {item_dict.get('standard_id')}")
    
    def insert_case(self, item_dict):
        """插入火灾案例数据"""
        sql = """
        INSERT OR REPLACE INTO fire_cases (
            case_id, title, case_type, severity_level, incident_date,
            location, province, city, district, building_type, fire_cause,
            casualties, economic_loss, fire_duration, description,
            investigation_result, lessons_learned, prevention_measures,
            content, keywords, tags, file_url, file_type, file_size,
            download_status, scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('case_id'),
            item_dict.get('title'),
            item_dict.get('case_type'),
            item_dict.get('severity_level'),
            item_dict.get('incident_date'),
            item_dict.get('location'),
            item_dict.get('province'),
            item_dict.get('city'),
            item_dict.get('district'),
            item_dict.get('building_type'),
            item_dict.get('fire_cause'),
            json.dumps(item_dict.get('casualties', {}), ensure_ascii=False),
            item_dict.get('economic_loss'),
            item_dict.get('fire_duration'),
            item_dict.get('description'),
            item_dict.get('investigation_result'),
            item_dict.get('lessons_learned'),
            item_dict.get('prevention_measures'),
            item_dict.get('content'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('tags', []), ensure_ascii=False),
            item_dict.get('file_url'),
            item_dict.get('file_type'),
            item_dict.get('file_size'),
            item_dict.get('download_status'),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"火灾案例数据插入成功: {item_dict.get('case_id')}")
    
    def insert_news(self, item_dict):
        """插入消防新闻数据"""
        sql = """
        INSERT OR REPLACE INTO fire_news (
            news_id, title, content, summary, author, source, publish_time,
            news_type, category, keywords, tags, view_count, comment_count,
            share_count, image_urls, file_urls, scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('news_id'),
            item_dict.get('title'),
            item_dict.get('content'),
            item_dict.get('summary'),
            item_dict.get('author'),
            item_dict.get('source'),
            item_dict.get('publish_time'),
            item_dict.get('news_type'),
            item_dict.get('category'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('tags', []), ensure_ascii=False),
            item_dict.get('view_count', 0),
            item_dict.get('comment_count', 0),
            item_dict.get('share_count', 0),
            json.dumps(item_dict.get('image_urls', []), ensure_ascii=False),
            json.dumps(item_dict.get('file_urls', []), ensure_ascii=False),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"消防新闻数据插入成功: {item_dict.get('news_id')}")
    
    def insert_knowledge(self, item_dict):
        """插入消防知识数据"""
        sql = """
        INSERT OR REPLACE INTO fire_knowledge (
            knowledge_id, title, content, summary, knowledge_type, category,
            subcategory, difficulty_level, target_audience, content_length,
            word_count, section_count, image_count, table_count, keywords,
            tags, entities, file_url, file_type, file_size, download_status,
            scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('knowledge_id'),
            item_dict.get('title'),
            item_dict.get('content'),
            item_dict.get('summary'),
            item_dict.get('knowledge_type'),
            item_dict.get('category'),
            item_dict.get('subcategory'),
            item_dict.get('difficulty_level'),
            item_dict.get('target_audience'),
            item_dict.get('content_length'),
            item_dict.get('word_count'),
            item_dict.get('section_count'),
            item_dict.get('image_count'),
            item_dict.get('table_count'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('tags', []), ensure_ascii=False),
            json.dumps(item_dict.get('entities', {}), ensure_ascii=False),
            item_dict.get('file_url'),
            item_dict.get('file_type'),
            item_dict.get('file_size'),
            item_dict.get('download_status'),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"消防知识数据插入成功: {item_dict.get('knowledge_id')}")
    
    def insert_document(self, item_dict):
        """插入RAG文档数据"""
        sql = """
        INSERT OR REPLACE INTO fire_documents (
            document_id, title, content, document_type, content_length,
            word_count, readability_score, complexity_score, summary,
            keywords, entities, topics, embedding_vector, chunk_texts,
            chunk_embeddings, file_url, file_type, file_size,
            download_status, scraped_at, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            item_dict.get('document_id'),
            item_dict.get('title'),
            item_dict.get('content'),
            item_dict.get('document_type'),
            item_dict.get('content_length'),
            item_dict.get('word_count'),
            item_dict.get('readability_score'),
            item_dict.get('complexity_score'),
            item_dict.get('summary'),
            json.dumps(item_dict.get('keywords', []), ensure_ascii=False),
            json.dumps(item_dict.get('entities', {}), ensure_ascii=False),
            json.dumps(item_dict.get('topics', {}), ensure_ascii=False),
            json.dumps(item_dict.get('embedding_vector', []), ensure_ascii=False),
            json.dumps(item_dict.get('chunk_texts', []), ensure_ascii=False),
            json.dumps(item_dict.get('chunk_embeddings', []), ensure_ascii=False),
            item_dict.get('file_url'),
            item_dict.get('file_type'),
            item_dict.get('file_size'),
            item_dict.get('download_status'),
            item_dict.get('scraped_at'),
            item_dict.get('source_url')
        )
        
        self.cursor.execute(sql, values)
        self.connection.commit()
        self.logger.info(f"RAG文档数据插入成功: {item_dict.get('document_id')}")

