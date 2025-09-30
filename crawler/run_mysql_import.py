"""
简化的MySQL数据导入脚本
"""

import json
import os
import sys
from datetime import datetime

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    try:
        import mysql.connector
        print("✅ mysql-connector-python - 已安装")
    except ImportError:
        print("❌ mysql-connector-python - 未安装")
        print("请运行: pip install mysql-connector-python")
        return False
    
    return True

def get_database_config():
    """获取数据库配置"""
    print("\n📋 数据库配置:")
    
    config = {
        'host': input("主机地址 (默认: localhost): ").strip() or 'localhost',
        'port': int(input("端口 (默认: 3306): ").strip() or '3306'),
        'user': input("用户名 (默认: root): ").strip() or 'root',
        'password': input("密码: ").strip(),
        'database': input("数据库名 (默认: firefighting_db): ").strip() or 'firefighting_db'
    }
    
    return config

def test_connection(config):
    """测试数据库连接"""
    try:
        import mysql.connector
        
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        
        if connection.is_connected():
            print("✅ 数据库连接成功")
            connection.close()
            return True
        else:
            print("❌ 数据库连接失败")
            return False
            
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def create_database_and_table(config):
    """创建数据库和表"""
    try:
        import mysql.connector
        
        # 连接MySQL服务器
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password']
        )
        
        cursor = connection.cursor()
        
        # 创建数据库
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"✅ 数据库 {config['database']} 创建成功")
        
        # 使用数据库
        cursor.execute(f"USE {config['database']}")
        
        # 创建表
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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(create_table_sql)
        connection.commit()
        print("✅ 数据表创建成功")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 创建数据库/表失败: {e}")
        return False

def load_json_data(json_file):
    """加载JSON数据"""
    try:
        if not os.path.exists(json_file):
            print(f"❌ JSON文件不存在: {json_file}")
            return []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功加载JSON数据，共 {len(data)} 条记录")
        return data
        
    except Exception as e:
        print(f"❌ 加载JSON数据失败: {e}")
        return []

def import_data_to_mysql(config, data):
    """导入数据到MySQL"""
    try:
        import mysql.connector
        
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            charset='utf8mb4'
        )
        
        cursor = connection.cursor()
        
        # 插入SQL
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
        
        success_count = 0
        error_count = 0
        
        print(f"📊 开始导入 {len(data)} 条记录...")
        
        for i, item in enumerate(data):
            try:
                # 准备数据
                record = (
                    item.get('url', ''),
                    item.get('title', ''),
                    item.get('content', ''),
                    item.get('publish_time', ''),
                    item.get('author', ''),
                    item.get('category', ''),
                    json.dumps(item.get('tags', []), ensure_ascii=False),
                    json.dumps(item.get('images', []), ensure_ascii=False),
                    item.get('crawl_time'),
                    item.get('source', '')
                )
                
                cursor.execute(insert_sql, record)
                connection.commit()
                success_count += 1
                
                if (i + 1) % 50 == 0:
                    print(f"✅ 已导入 {i + 1} / {len(data)} 条记录")
                
            except Exception as e:
                error_count += 1
                print(f"❌ 导入第 {i + 1} 条记录失败: {e}")
        
        print(f"📊 导入完成: 成功 {success_count} 条，失败 {error_count} 条")
        
        cursor.close()
        connection.close()
        
        return success_count, error_count
        
    except Exception as e:
        print(f"❌ 导入数据失败: {e}")
        return 0, len(data)

def show_statistics(config):
    """显示数据统计"""
    try:
        import mysql.connector
        
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        
        cursor = connection.cursor()
        
        # 总记录数
        cursor.execute("SELECT COUNT(*) FROM fire_119_articles")
        total_count = cursor.fetchone()[0]
        
        # 按分类统计
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM fire_119_articles 
            GROUP BY category 
            ORDER BY count DESC
        """)
        category_stats = cursor.fetchall()
        
        print(f"\n📈 数据统计:")
        print(f"总记录数: {total_count}")
        
        print(f"\n📊 按分类统计:")
        for category, count in category_stats:
            print(f"  {category}: {count} 条")
        
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")

def main():
    """主函数"""
    print("🔥 消防119数据导入MySQL数据库")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查JSON文件
    json_file = "data/fire_119_scrapy.json"
    if not os.path.exists(json_file):
        print(f"❌ JSON文件不存在: {json_file}")
        print("请先运行爬虫获取数据")
        return
    
    # 获取数据库配置
    config = get_database_config()
    
    # 测试连接
    if not test_connection(config):
        print("❌ 无法连接到数据库，请检查配置")
        return
    
    # 创建数据库和表
    if not create_database_and_table(config):
        print("❌ 无法创建数据库/表")
        return
    
    # 加载JSON数据
    data = load_json_data(json_file)
    if not data:
        print("❌ 没有数据可导入")
        return
    
    # 导入数据
    success_count, error_count = import_data_to_mysql(config, data)
    
    if success_count > 0:
        print(f"✅ 导入成功: {success_count} 条记录")
        
        # 显示统计信息
        show_statistics(config)
    
    if error_count > 0:
        print(f"⚠️ 导入失败: {error_count} 条记录")

if __name__ == "__main__":
    main()



