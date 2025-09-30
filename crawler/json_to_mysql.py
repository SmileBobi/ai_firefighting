"""
将fire_119_scrapy.json数据导入到MySQL数据库
"""

import json
import mysql.connector
from mysql.connector import Error
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Fire119DataImporter:
    """消防119数据导入器"""
    
    def __init__(self, config: Dict):
        """
        初始化数据导入器
        
        Args:
            config: MySQL数据库配置
        """
        self.config = config
        self.connection = None
        self.cursor = None
        
    def connect_database(self):
        """连接数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database'],
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                logger.info("✅ 成功连接到MySQL数据库")
                return True
                
        except Error as e:
            logger.error(f"❌ 连接数据库失败: {e}")
            return False
    
    def create_table(self):
        """创建数据表"""
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS fire_119_articles (
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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_url (url),
                INDEX idx_title (title),
                INDEX idx_category (category),
                INDEX idx_publish_time (publish_time),
                INDEX idx_crawl_time (crawl_time)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            self.cursor.execute(create_table_sql)
            self.connection.commit()
            logger.info("✅ 数据表创建成功")
            return True
            
        except Error as e:
            logger.error(f"❌ 创建数据表失败: {e}")
            return False
    
    def load_json_data(self, json_file_path: str) -> List[Dict]:
        """加载JSON数据"""
        try:
            if not os.path.exists(json_file_path):
                logger.error(f"❌ JSON文件不存在: {json_file_path}")
                return []
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ 成功加载JSON数据，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"❌ 加载JSON数据失败: {e}")
            return []
    
    def clean_data(self, item: Dict) -> Dict:
        """清洗数据"""
        # 清理字符串字段
        for field in ['url', 'title', 'content', 'publish_time', 'author', 'category', 'source']:
            if field in item and item[field]:
                item[field] = str(item[field]).strip()
        
        # 处理时间字段
        if item.get('crawl_time'):
            try:
                # 如果是ISO格式字符串，转换为datetime
                if isinstance(item['crawl_time'], str):
                    item['crawl_time'] = datetime.fromisoformat(item['crawl_time'].replace('Z', '+00:00'))
            except:
                item['crawl_time'] = datetime.now()
        
        # 处理JSON字段 - 确保转换为JSON字符串
        for field in ['tags', 'images']:
            if field in item and item[field]:
                if isinstance(item[field], list):
                    # 列表直接转换为JSON字符串
                    item[field] = json.dumps(item[field], ensure_ascii=False)
                elif isinstance(item[field], str):
                    try:
                        # 验证是否为有效JSON，如果不是则包装成列表
                        json.loads(item[field])
                    except:
                        item[field] = json.dumps([item[field]], ensure_ascii=False)
                else:
                    # 其他类型转换为JSON字符串
                    item[field] = json.dumps([str(item[field])], ensure_ascii=False)
            else:
                # 如果字段不存在或为空，设置为空JSON数组
                item[field] = '[]'
        
        return item
    
    def insert_data(self, data: List[Dict], batch_size: int = 100):
        """插入数据到数据库"""
        try:
            insert_sql = """
            INSERT INTO fire_119_articles 
            (url, title, content, publish_time, author, category, tags, images, crawl_time, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            content = VALUES(content),
            publish_time = VALUES(publish_time),
            author = VALUES(author),
            category = VALUES(category),
            tags = VALUES(tags),
            images = VALUES(images),
            crawl_time = VALUES(crawl_time),
            source = VALUES(source),
            updated_at = CURRENT_TIMESTAMP
            """
            
            total_records = len(data)
            success_count = 0
            error_count = 0
            
            logger.info(f"开始插入数据，共 {total_records} 条记录")
            
            for i in range(0, total_records, batch_size):
                batch_data = data[i:i + batch_size]
                batch_records = []
                
                for item in batch_data:
                    # 清洗数据
                    cleaned_item = self.clean_data(item)
                    
                    # 调试信息：检查tags和images字段
                    tags = cleaned_item.get('tags', '[]')
                    images = cleaned_item.get('images', '[]')
                    
                    # 确保tags和images是字符串
                    if not isinstance(tags, str):
                        tags = json.dumps(tags, ensure_ascii=False)
                    if not isinstance(images, str):
                        images = json.dumps(images, ensure_ascii=False)
                    
                    # 准备插入数据，确保所有字段都是正确的类型
                    record = (
                        cleaned_item.get('url', ''),
                        cleaned_item.get('title', ''),
                        cleaned_item.get('content', ''),
                        cleaned_item.get('publish_time', ''),
                        cleaned_item.get('author', ''),
                        cleaned_item.get('category', ''),
                        tags,  # 确保是JSON字符串
                        images,  # 确保是JSON字符串
                        cleaned_item.get('crawl_time'),
                        cleaned_item.get('source', '')
                    )
                    batch_records.append(record)
                
                try:
                    # 批量插入
                    self.cursor.executemany(insert_sql, batch_records)
                    self.connection.commit()
                    
                    success_count += len(batch_records)
                    logger.info(f"✅ 批量插入成功: {i + 1}-{min(i + batch_size, total_records)} / {total_records}")
                    
                except Error as e:
                    logger.error(f"❌ 批量插入失败: {e}")
                    error_count += len(batch_records)
                    self.connection.rollback()
            
            logger.info(f"📊 数据插入完成: 成功 {success_count} 条，失败 {error_count} 条")
            return success_count, error_count
            
        except Exception as e:
            logger.error(f"❌ 插入数据失败: {e}")
            return 0, len(data)
    
    def get_statistics(self):
        """获取数据统计"""
        try:
            # 总记录数
            self.cursor.execute("SELECT COUNT(*) FROM fire_119_articles")
            total_count = self.cursor.fetchone()[0]
            
            # 按分类统计
            self.cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM fire_119_articles 
                GROUP BY category 
                ORDER BY count DESC
            """)
            category_stats = self.cursor.fetchall()
            
            # 按来源统计
            self.cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM fire_119_articles 
                GROUP BY source 
                ORDER BY count DESC
            """)
            source_stats = self.cursor.fetchall()
            
            # 最新记录
            self.cursor.execute("""
                SELECT title, publish_time, crawl_time 
                FROM fire_119_articles 
                ORDER BY crawl_time DESC 
                LIMIT 5
            """)
            latest_records = self.cursor.fetchall()
            
            return {
                'total_count': total_count,
                'category_stats': category_stats,
                'source_stats': source_stats,
                'latest_records': latest_records
            }
            
        except Error as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return None
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("✅ 数据库连接已关闭")

def load_config():
    """加载数据库配置"""
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': 'root',
        'database': 'firefighting_db'
    }
    
    # 尝试从环境变量加载配置
    config['host'] = os.getenv('MYSQL_HOST', config['host'])
    config['port'] = int(os.getenv('MYSQL_PORT', config['port']))
    config['user'] = os.getenv('MYSQL_USER', config['user'])
    config['password'] = os.getenv('MYSQL_PASSWORD', config['password'])
    config['database'] = os.getenv('MYSQL_DATABASE', config['database'])
    
    return config

def main():
    """主函数"""
    print("🔥 消防119数据导入MySQL数据库")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 检查JSON文件
    json_file = "./crawler/data/fire_119_scrapy.json"
    if not os.path.exists(json_file):
        print(f"❌ JSON文件不存在: {json_file}")
        print("请先运行爬虫获取数据")
        return
    
    # 创建导入器
    importer = Fire119DataImporter(config)
    
    try:
        # 连接数据库
        if not importer.connect_database():
            print("❌ 无法连接到数据库，请检查配置")
            return
        
        # 创建数据表
        if not importer.create_table():
            print("❌ 无法创建数据表")
            return
        
        # 加载JSON数据
        data = importer.load_json_data(json_file)
        if not data:
            print("❌ 没有数据可导入")
            return
        
        # 插入数据
        print(f"📊 开始导入 {len(data)} 条记录...")
        success_count, error_count = importer.insert_data(data)
        
        if success_count > 0:
            print(f"✅ 导入成功: {success_count} 条记录")
        
        if error_count > 0:
            print(f"⚠️ 导入失败: {error_count} 条记录")
        
        # 显示统计信息
        stats = importer.get_statistics()
        if stats:
            print("\n📈 数据统计:")
            print(f"总记录数: {stats['total_count']}")
            
            print("\n📊 按分类统计:")
            for category, count in stats['category_stats']:
                print(f"  {category}: {count} 条")
            
            print("\n📊 按来源统计:")
            for source, count in stats['source_stats']:
                print(f"  {source}: {count} 条")
            
            print("\n📰 最新记录:")
            for title, publish_time, crawl_time in stats['latest_records']:
                print(f"  {title[:50]}... ({publish_time})")
        
    except Exception as e:
        logger.error(f"❌ 导入过程出错: {e}")
    
    finally:
        # 关闭连接
        importer.close_connection()

def create_database():
    """创建数据库"""
    print("🔧 创建数据库...")
    
    config = load_config()
    database_name = config['database']
    
    try:
        # 连接MySQL服务器（不指定数据库）
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password']
        )
        
        cursor = connection.cursor()
        
        # 创建数据库
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✅ 数据库 {database_name} 创建成功")
        
        cursor.close()
        connection.close()
        
    except Error as e:
        print(f"❌ 创建数据库失败: {e}")

def test_connection():
    """测试数据库连接"""
    print("🔍 测试数据库连接...")
    
    config = load_config()
    importer = Fire119DataImporter(config)
    
    if importer.connect_database():
        print("✅ 数据库连接成功")
        importer.close_connection()
    else:
        print("❌ 数据库连接失败")
        print("请检查以下配置:")
        print(f"  主机: {config['host']}")
        print(f"  端口: {config['port']}")
        print(f"  用户: {config['user']}")
        print(f"  数据库: {config['database']}")

if __name__ == "__main__":
    while True:
        print("\n请选择操作:")
        print("1. 导入数据到MySQL")
        print("2. 创建数据库")
        print("3. 测试连接")
        print("4. 查看配置")
        print("0. 退出")
        
        choice = input("请输入选择 (0-4): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            main()
        elif choice == "2":
            create_database()
        elif choice == "3":
            test_connection()
        elif choice == "4":
            config = load_config()
            print("\n📋 当前配置:")
            for key, value in config.items():
                if key == 'password':
                    print(f"  {key}: {'*' * len(str(value))}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("❌ 无效选择")
