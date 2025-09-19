#!/usr/bin/env python
"""
数据库管理工具
用于管理MySQL数据库连接、创建表、数据导入导出等操作
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from database_config import db_manager, fire_analyzer


class DatabaseManagerTool:
    """数据库管理工具类"""
    
    def __init__(self):
        """初始化数据库管理工具"""
        self.db_manager = db_manager
        self.analyzer = fire_analyzer
    
    def setup_database(self):
        """设置数据库"""
        print("=== 数据库设置向导 ===")
        
        # 获取数据库配置
        host = input("MySQL主机地址 (默认: localhost): ").strip() or "localhost"
        port = input("MySQL端口 (默认: 3306): ").strip() or "3306"
        user = input("MySQL用户名 (默认: root): ").strip() or "root"
        password = input("MySQL密码: ").strip()
        database = input("数据库名称 (默认: fire_data): ").strip() or "fire_data"
        
        # 更新配置
        self.db_manager.config.update_mysql_config(
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=database
        )
        
        print(f"\n配置已保存到 database_config.json")
        print(f"数据库配置:")
        print(f"  主机: {host}")
        print(f"  端口: {port}")
        print(f"  用户: {user}")
        print(f"  数据库: {database}")
        
        # 测试连接
        if self.test_connection():
            print("✅ 数据库连接测试成功")
            return True
        else:
            print("❌ 数据库连接测试失败")
            return False
    
    def test_connection(self):
        """测试数据库连接"""
        print("正在测试数据库连接...")
        
        # 创建数据库
        if not self.db_manager.create_database():
            return False
        
        # 连接数据库
        if not self.db_manager.connect():
            return False
        
        # 测试连接
        if not self.db_manager.test_connection():
            return False
        
        return True
    
    def create_tables(self):
        """创建数据库表"""
        print("正在创建数据库表...")
        
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
            tables = [
                ("消防法规表", create_regulations_table),
                ("消防标准表", create_standards_table),
                ("火灾案例表", create_cases_table),
                ("消防新闻表", create_news_table),
                ("消防知识表", create_knowledge_table),
                ("RAG文档表", create_documents_table)
            ]
            
            for table_name, sql in tables:
                self.db_manager.execute_update(sql)
                print(f"✅ {table_name} 创建成功")
            
            print("✅ 所有数据库表创建完成")
            return True
            
        except Exception as e:
            print(f"❌ 创建数据库表失败: {e}")
            return False
    
    def import_csv_data(self, csv_file: str, table_name: str):
        """从CSV文件导入数据"""
        try:
            print(f"正在从 {csv_file} 导入数据到 {table_name} 表...")
            
            # 读取CSV文件
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"CSV文件包含 {len(df)} 条记录")
            
            # 根据表名选择插入方法
            if table_name == 'fire_regulations':
                self._import_regulations(df)
            elif table_name == 'fire_standards':
                self._import_standards(df)
            elif table_name == 'fire_cases':
                self._import_cases(df)
            elif table_name == 'fire_news':
                self._import_news(df)
            elif table_name == 'fire_knowledge':
                self._import_knowledge(df)
            elif table_name == 'fire_documents':
                self._import_documents(df)
            else:
                print(f"❌ 不支持的表名: {table_name}")
                return False
            
            print(f"✅ 数据导入完成")
            return True
            
        except Exception as e:
            print(f"❌ 导入数据失败: {e}")
            return False
    
    def _import_regulations(self, df):
        """导入消防法规数据"""
        for _, row in df.iterrows():
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
                content = VALUES(content),
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('regulation_id'),
                row.get('title'),
                row.get('regulation_type'),
                row.get('level'),
                row.get('issuing_authority'),
                row.get('issue_date'),
                row.get('effective_date'),
                row.get('status'),
                row.get('content'),
                row.get('summary'),
                row.get('chapters'),
                row.get('articles'),
                row.get('category'),
                row.get('keywords'),
                row.get('tags'),
                row.get('file_url'),
                row.get('file_type'),
                row.get('file_size'),
                row.get('download_status'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def _import_standards(self, df):
        """导入消防标准数据"""
        for _, row in df.iterrows():
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
                title = VALUES(title),
                content = VALUES(content),
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('standard_id'),
                row.get('standard_number'),
                row.get('title'),
                row.get('english_title'),
                row.get('standard_type'),
                row.get('category'),
                row.get('issuing_authority'),
                row.get('approval_authority'),
                row.get('issue_date'),
                row.get('implementation_date'),
                row.get('status'),
                row.get('scope'),
                row.get('content'),
                row.get('technical_requirements'),
                row.get('test_methods'),
                row.get('inspection_rules'),
                row.get('fire_category'),
                row.get('keywords'),
                row.get('tags'),
                row.get('file_url'),
                row.get('file_type'),
                row.get('file_size'),
                row.get('download_status'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def _import_cases(self, df):
        """导入火灾案例数据"""
        for _, row in df.iterrows():
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
                content = VALUES(content),
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('case_id'),
                row.get('title'),
                row.get('case_type'),
                row.get('severity_level'),
                row.get('incident_date'),
                row.get('location'),
                row.get('province'),
                row.get('city'),
                row.get('district'),
                row.get('building_type'),
                row.get('fire_cause'),
                row.get('casualties'),
                row.get('economic_loss'),
                row.get('fire_duration'),
                row.get('description'),
                row.get('investigation_result'),
                row.get('lessons_learned'),
                row.get('prevention_measures'),
                row.get('content'),
                row.get('keywords'),
                row.get('tags'),
                row.get('file_url'),
                row.get('file_type'),
                row.get('file_size'),
                row.get('download_status'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def _import_news(self, df):
        """导入消防新闻数据"""
        for _, row in df.iterrows():
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
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('news_id'),
                row.get('title'),
                row.get('content'),
                row.get('summary'),
                row.get('author'),
                row.get('source'),
                row.get('publish_time'),
                row.get('news_type'),
                row.get('category'),
                row.get('keywords'),
                row.get('tags'),
                row.get('view_count', 0),
                row.get('comment_count', 0),
                row.get('share_count', 0),
                row.get('image_urls'),
                row.get('file_urls'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def _import_knowledge(self, df):
        """导入消防知识数据"""
        for _, row in df.iterrows():
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
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('knowledge_id'),
                row.get('title'),
                row.get('content'),
                row.get('summary'),
                row.get('knowledge_type'),
                row.get('category'),
                row.get('subcategory'),
                row.get('difficulty_level'),
                row.get('target_audience'),
                row.get('content_length'),
                row.get('word_count'),
                row.get('section_count'),
                row.get('image_count'),
                row.get('table_count'),
                row.get('keywords'),
                row.get('tags'),
                row.get('entities'),
                row.get('file_url'),
                row.get('file_type'),
                row.get('file_size'),
                row.get('download_status'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def _import_documents(self, df):
        """导入RAG文档数据"""
        for _, row in df.iterrows():
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
                updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                row.get('document_id'),
                row.get('title'),
                row.get('content'),
                row.get('document_type'),
                row.get('content_length'),
                row.get('word_count'),
                row.get('readability_score'),
                row.get('complexity_score'),
                row.get('summary'),
                row.get('keywords'),
                row.get('entities'),
                row.get('topics'),
                row.get('embedding_vector'),
                row.get('chunk_texts'),
                row.get('chunk_embeddings'),
                row.get('file_url'),
                row.get('file_type'),
                row.get('file_size'),
                row.get('download_status'),
                row.get('scraped_at'),
                row.get('source_url')
            )
            
            self.db_manager.execute_update(sql, values)
    
    def export_data(self, table_name: str, output_file: str = None):
        """导出数据到CSV文件"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{table_name}_{timestamp}.csv"
            
            print(f"正在导出 {table_name} 表数据到 {output_file}...")
            
            # 查询数据
            sql = f"SELECT * FROM {table_name}"
            data = self.db_manager.execute_query(sql)
            
            if not data:
                print(f"❌ 表 {table_name} 中没有数据")
                return False
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ 数据导出完成，共 {len(data)} 条记录")
            return True
            
        except Exception as e:
            print(f"❌ 导出数据失败: {e}")
            return False
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        try:
            stats = self.analyzer.get_comprehensive_analysis()
            
            print("=== 数据库统计信息 ===")
            print(f"数据库名称: {stats['database_stats']['database_name']}")
            print(f"总记录数: {stats['database_stats']['total_records']}")
            print(f"最后更新: {stats['database_stats']['last_updated']}")
            print()
            
            print("各表记录数:")
            for table_name, count in stats['database_stats']['tables'].items():
                print(f"  {table_name}: {count} 条")
            print()
            
            # 显示详细分析
            if stats['regulations_analysis']:
                print("消防法规分析:")
                print(f"  总数: {stats['regulations_analysis']['total_count']}")
                if stats['regulations_analysis']['type_distribution']:
                    print("  按类型分布:")
                    for item in stats['regulations_analysis']['type_distribution'][:5]:
                        print(f"    {item['regulation_type']}: {item['count']} 条")
                print()
            
            if stats['standards_analysis']:
                print("消防标准分析:")
                print(f"  总数: {stats['standards_analysis']['total_count']}")
                if stats['standards_analysis']['type_distribution']:
                    print("  按类型分布:")
                    for item in stats['standards_analysis']['type_distribution'][:5]:
                        print(f"    {item['standard_type']}: {item['count']} 条")
                print()
            
            if stats['cases_analysis']:
                print("火灾案例分析:")
                print(f"  总数: {stats['cases_analysis']['total_count']}")
                if stats['cases_analysis']['severity_distribution']:
                    print("  按严重程度分布:")
                    for item in stats['cases_analysis']['severity_distribution']:
                        print(f"    {item['severity_level']}: {item['count']} 条")
                print()
            
            if stats['news_analysis']:
                print("消防新闻分析:")
                print(f"  总数: {stats['news_analysis']['total_count']}")
                if stats['news_analysis']['type_distribution']:
                    print("  按类型分布:")
                    for item in stats['news_analysis']['type_distribution'][:5]:
                        print(f"    {item['news_type']}: {item['count']} 条")
                print()
            
            return stats
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return None
    
    def run_interactive_mode(self):
        """运行交互模式"""
        while True:
            print("\n=== 数据库管理工具 ===")
            print("1. 设置数据库连接")
            print("2. 测试数据库连接")
            print("3. 创建数据库表")
            print("4. 导入CSV数据")
            print("5. 导出数据到CSV")
            print("6. 查看数据库统计")
            print("7. 退出")
            
            choice = input("\n请选择操作 (1-7): ").strip()
            
            if choice == '1':
                self.setup_database()
            elif choice == '2':
                if self.test_connection():
                    print("✅ 数据库连接正常")
                else:
                    print("❌ 数据库连接失败")
            elif choice == '3':
                if self.test_connection():
                    self.create_tables()
                else:
                    print("❌ 请先设置数据库连接")
            elif choice == '4':
                csv_file = input("请输入CSV文件路径: ").strip()
                table_name = input("请输入目标表名: ").strip()
                if os.path.exists(csv_file):
                    self.import_csv_data(csv_file, table_name)
                else:
                    print("❌ CSV文件不存在")
            elif choice == '5':
                table_name = input("请输入表名: ").strip()
                output_file = input("请输入输出文件名 (可选): ").strip() or None
                self.export_data(table_name, output_file)
            elif choice == '6':
                self.get_database_stats()
            elif choice == '7':
                print("退出数据库管理工具")
                break
            else:
                print("❌ 无效选择，请重新输入")


def main():
    """主函数"""
    tool = DatabaseManagerTool()
    tool.run_interactive_mode()


if __name__ == "__main__":
    main()

