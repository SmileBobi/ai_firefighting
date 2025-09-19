#!/usr/bin/env python
"""
MySQL数据库管道
用于将爬虫数据插入到MySQL数据库中
"""

import pymysql
import json
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
import logging


class MySQLPipeline:
    """MySQL数据库存储管道"""
    
    def __init__(self, mysql_host, mysql_port, mysql_user, mysql_password, mysql_database):
        """初始化MySQL连接参数"""
        self.mysql_host = mysql_host
        self.mysql_port = mysql_port
        self.mysql_user = mysql_user
        self.mysql_password = mysql_password
        self.mysql_database = mysql_database
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_crawler(cls, crawler):
        """从爬虫设置中获取MySQL配置"""
        return cls(
            mysql_host=crawler.settings.get('MYSQL_HOST', 'localhost'),
            mysql_port=crawler.settings.get('MYSQL_PORT', 3306),
            mysql_user=crawler.settings.get('MYSQL_USER', 'root'),
            mysql_password=crawler.settings.get('MYSQL_PASSWORD', ''),
            mysql_database=crawler.settings.get('MYSQL_DATABASE', 'fire_data')
        )
    
    def open_spider(self, spider):
        """爬虫开始时建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.mysql_host,
                port=self.mysql_port,
                user=self.mysql_user,
                password=self.mysql_password,
                database=self.mysql_database,
                charset='utf8mb4',
                autocommit=False
            )
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL数据库连接成功")
            
            # 创建数据库表
            self.create_tables()
            
        except Exception as e:
            self.logger.error(f"MySQL数据库连接失败: {e}")
            raise
    
    def close_spider(self, spider):
        """爬虫结束时关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("MySQL数据库连接已关闭")
    
    def create_tables(self):
        """创建数据库表"""
        try:
            # 创建消防法规表
            create_regulations_table = """
            CREATE TABLE IF NOT EXISTS fire_regulations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                regulation_id VARCHAR(50) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                regulation_type VARCHAR(100),
                level VARCHAR(50),
                issuing_authority VARCHAR(200),
                issue_date VARCHAR(50),
                effective_date VARCHAR(50),
                status VARCHAR(50),
                content LONGTEXT,
                summary TEXT,
                chapters JSON,
                articles JSON,
                category VARCHAR(100),
                keywords JSON,
                tags JSON,
                file_url VARCHAR(500),
                file_type VARCHAR(50),
                file_size INT,
                download_status VARCHAR(50),
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_regulation_id (regulation_id),
                INDEX idx_title (title),
                INDEX idx_category (category),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 创建消防标准表
            create_standards_table = """
            CREATE TABLE IF NOT EXISTS fire_standards (
                id INT AUTO_INCREMENT PRIMARY KEY,
                standard_id VARCHAR(50) UNIQUE NOT NULL,
                standard_number VARCHAR(100) NOT NULL,
                title VARCHAR(500) NOT NULL,
                english_title VARCHAR(500),
                standard_type VARCHAR(100),
                category VARCHAR(100),
                issuing_authority VARCHAR(200),
                approval_authority VARCHAR(200),
                issue_date VARCHAR(50),
                implementation_date VARCHAR(50),
                status VARCHAR(50),
                scope TEXT,
                content LONGTEXT,
                technical_requirements JSON,
                test_methods JSON,
                inspection_rules JSON,
                fire_category VARCHAR(100),
                keywords JSON,
                tags JSON,
                file_url VARCHAR(500),
                file_type VARCHAR(50),
                file_size INT,
                download_status VARCHAR(50),
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_standard_id (standard_id),
                INDEX idx_standard_number (standard_number),
                INDEX idx_title (title),
                INDEX idx_fire_category (fire_category),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 创建火灾案例表
            create_cases_table = """
            CREATE TABLE IF NOT EXISTS fire_cases (
                id INT AUTO_INCREMENT PRIMARY KEY,
                case_id VARCHAR(50) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                case_type VARCHAR(100),
                severity_level VARCHAR(50),
                incident_date VARCHAR(50),
                location VARCHAR(200),
                province VARCHAR(50),
                city VARCHAR(50),
                district VARCHAR(50),
                building_type VARCHAR(100),
                fire_cause VARCHAR(200),
                casualties JSON,
                economic_loss VARCHAR(100),
                fire_duration VARCHAR(50),
                description TEXT,
                investigation_result TEXT,
                lessons_learned TEXT,
                prevention_measures TEXT,
                content LONGTEXT,
                keywords JSON,
                tags JSON,
                file_url VARCHAR(500),
                file_type VARCHAR(50),
                file_size INT,
                download_status VARCHAR(50),
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_case_id (case_id),
                INDEX idx_title (title),
                INDEX idx_severity_level (severity_level),
                INDEX idx_incident_date (incident_date),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 创建消防新闻表
            create_news_table = """
            CREATE TABLE IF NOT EXISTS fire_news (
                id INT AUTO_INCREMENT PRIMARY KEY,
                news_id VARCHAR(50) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                content LONGTEXT,
                summary TEXT,
                author VARCHAR(200),
                source VARCHAR(200),
                publish_time VARCHAR(50),
                news_type VARCHAR(100),
                category VARCHAR(100),
                keywords JSON,
                tags JSON,
                view_count INT DEFAULT 0,
                comment_count INT DEFAULT 0,
                share_count INT DEFAULT 0,
                image_urls JSON,
                file_urls JSON,
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_news_id (news_id),
                INDEX idx_title (title),
                INDEX idx_news_type (news_type),
                INDEX idx_publish_time (publish_time),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 创建消防知识表
            create_knowledge_table = """
            CREATE TABLE IF NOT EXISTS fire_knowledge (
                id INT AUTO_INCREMENT PRIMARY KEY,
                knowledge_id VARCHAR(50) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                content LONGTEXT,
                summary TEXT,
                knowledge_type VARCHAR(100),
                category VARCHAR(100),
                subcategory VARCHAR(100),
                difficulty_level VARCHAR(50),
                target_audience VARCHAR(100),
                content_length INT,
                word_count INT,
                section_count INT,
                image_count INT,
                table_count INT,
                keywords JSON,
                tags JSON,
                entities JSON,
                file_url VARCHAR(500),
                file_type VARCHAR(50),
                file_size INT,
                download_status VARCHAR(50),
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_knowledge_id (knowledge_id),
                INDEX idx_title (title),
                INDEX idx_knowledge_type (knowledge_type),
                INDEX idx_category (category),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 创建RAG文档表
            create_documents_table = """
            CREATE TABLE IF NOT EXISTS fire_documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                document_id VARCHAR(50) UNIQUE NOT NULL,
                title VARCHAR(500) NOT NULL,
                content LONGTEXT,
                document_type VARCHAR(100),
                content_length INT,
                word_count INT,
                readability_score DECIMAL(5,2),
                complexity_score DECIMAL(5,2),
                summary TEXT,
                keywords JSON,
                entities JSON,
                topics JSON,
                embedding_vector JSON,
                chunk_texts JSON,
                chunk_embeddings JSON,
                file_url VARCHAR(500),
                file_type VARCHAR(50),
                file_size INT,
                download_status VARCHAR(50),
                scraped_at DATETIME,
                source_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_document_id (document_id),
                INDEX idx_title (title),
                INDEX idx_document_type (document_type),
                INDEX idx_scraped_at (scraped_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """
            
            # 执行创建表语句
            self.cursor.execute(create_regulations_table)
            self.cursor.execute(create_standards_table)
            self.cursor.execute(create_cases_table)
            self.cursor.execute(create_news_table)
            self.cursor.execute(create_knowledge_table)
            self.cursor.execute(create_documents_table)
            
            self.connection.commit()
            self.logger.info("数据库表创建成功")
            
        except Exception as e:
            self.logger.error(f"创建数据库表失败: {e}")
            self.connection.rollback()
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
            self.logger.error(f"插入数据库失败: {e}")
            self.connection.rollback()
            raise DropItem(f"数据库插入失败: {e}")
    
    def insert_regulation(self, item_dict):
        """插入消防法规数据"""
        sql = """
        INSERT INTO fire_regulations (
            regulation_id, title, regulation_type, level, issuing_authority,
            issue_date, effective_date, status, content, summary, chapters,
            articles, category, keywords, tags, file_url, file_type,
            file_size, download_status, scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            regulation_type = VALUES(regulation_type),
            level = VALUES(level),
            issuing_authority = VALUES(issuing_authority),
            issue_date = VALUES(issue_date),
            effective_date = VALUES(effective_date),
            status = VALUES(status),
            content = VALUES(content),
            summary = VALUES(summary),
            chapters = VALUES(chapters),
            articles = VALUES(articles),
            category = VALUES(category),
            keywords = VALUES(keywords),
            tags = VALUES(tags),
            file_url = VALUES(file_url),
            file_type = VALUES(file_type),
            file_size = VALUES(file_size),
            download_status = VALUES(download_status),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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
        INSERT INTO fire_standards (
            standard_id, standard_number, title, english_title, standard_type,
            category, issuing_authority, approval_authority, issue_date,
            implementation_date, status, scope, content, technical_requirements,
            test_methods, inspection_rules, fire_category, keywords, tags,
            file_url, file_type, file_size, download_status, scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            standard_number = VALUES(standard_number),
            title = VALUES(title),
            english_title = VALUES(english_title),
            standard_type = VALUES(standard_type),
            category = VALUES(category),
            issuing_authority = VALUES(issuing_authority),
            approval_authority = VALUES(approval_authority),
            issue_date = VALUES(issue_date),
            implementation_date = VALUES(implementation_date),
            status = VALUES(status),
            scope = VALUES(scope),
            content = VALUES(content),
            technical_requirements = VALUES(technical_requirements),
            test_methods = VALUES(test_methods),
            inspection_rules = VALUES(inspection_rules),
            fire_category = VALUES(fire_category),
            keywords = VALUES(keywords),
            tags = VALUES(tags),
            file_url = VALUES(file_url),
            file_type = VALUES(file_type),
            file_size = VALUES(file_size),
            download_status = VALUES(download_status),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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
        INSERT INTO fire_cases (
            case_id, title, case_type, severity_level, incident_date,
            location, province, city, district, building_type, fire_cause,
            casualties, economic_loss, fire_duration, description,
            investigation_result, lessons_learned, prevention_measures,
            content, keywords, tags, file_url, file_type, file_size,
            download_status, scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            case_type = VALUES(case_type),
            severity_level = VALUES(severity_level),
            incident_date = VALUES(incident_date),
            location = VALUES(location),
            province = VALUES(province),
            city = VALUES(city),
            district = VALUES(district),
            building_type = VALUES(building_type),
            fire_cause = VALUES(fire_cause),
            casualties = VALUES(casualties),
            economic_loss = VALUES(economic_loss),
            fire_duration = VALUES(fire_duration),
            description = VALUES(description),
            investigation_result = VALUES(investigation_result),
            lessons_learned = VALUES(lessons_learned),
            prevention_measures = VALUES(prevention_measures),
            content = VALUES(content),
            keywords = VALUES(keywords),
            tags = VALUES(tags),
            file_url = VALUES(file_url),
            file_type = VALUES(file_type),
            file_size = VALUES(file_size),
            download_status = VALUES(download_status),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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
        INSERT INTO fire_news (
            news_id, title, content, summary, author, source, publish_time,
            news_type, category, keywords, tags, view_count, comment_count,
            share_count, image_urls, file_urls, scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            content = VALUES(content),
            summary = VALUES(summary),
            author = VALUES(author),
            source = VALUES(source),
            publish_time = VALUES(publish_time),
            news_type = VALUES(news_type),
            category = VALUES(category),
            keywords = VALUES(keywords),
            tags = VALUES(tags),
            view_count = VALUES(view_count),
            comment_count = VALUES(comment_count),
            share_count = VALUES(share_count),
            image_urls = VALUES(image_urls),
            file_urls = VALUES(file_urls),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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
        INSERT INTO fire_knowledge (
            knowledge_id, title, content, summary, knowledge_type, category,
            subcategory, difficulty_level, target_audience, content_length,
            word_count, section_count, image_count, table_count, keywords,
            tags, entities, file_url, file_type, file_size, download_status,
            scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            content = VALUES(content),
            summary = VALUES(summary),
            knowledge_type = VALUES(knowledge_type),
            category = VALUES(category),
            subcategory = VALUES(subcategory),
            difficulty_level = VALUES(difficulty_level),
            target_audience = VALUES(target_audience),
            content_length = VALUES(content_length),
            word_count = VALUES(word_count),
            section_count = VALUES(section_count),
            image_count = VALUES(image_count),
            table_count = VALUES(table_count),
            keywords = VALUES(keywords),
            tags = VALUES(tags),
            entities = VALUES(entities),
            file_url = VALUES(file_url),
            file_type = VALUES(file_type),
            file_size = VALUES(file_size),
            download_status = VALUES(download_status),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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
        INSERT INTO fire_documents (
            document_id, title, content, document_type, content_length,
            word_count, readability_score, complexity_score, summary,
            keywords, entities, topics, embedding_vector, chunk_texts,
            chunk_embeddings, file_url, file_type, file_size,
            download_status, scraped_at, source_url
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            content = VALUES(content),
            document_type = VALUES(document_type),
            content_length = VALUES(content_length),
            word_count = VALUES(word_count),
            readability_score = VALUES(readability_score),
            complexity_score = VALUES(complexity_score),
            summary = VALUES(summary),
            keywords = VALUES(keywords),
            entities = VALUES(entities),
            topics = VALUES(topics),
            embedding_vector = VALUES(embedding_vector),
            chunk_texts = VALUES(chunk_texts),
            chunk_embeddings = VALUES(chunk_embeddings),
            file_url = VALUES(file_url),
            file_type = VALUES(file_type),
            file_size = VALUES(file_size),
            download_status = VALUES(download_status),
            scraped_at = VALUES(scraped_at),
            source_url = VALUES(source_url),
            updated_at = CURRENT_TIMESTAMP
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

