#!/usr/bin/env python
"""
数据库查询工具
用于查询和验证SQLite数据库中的消防数据
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime


class DatabaseQueryTool:
    """数据库查询工具类"""
    
    def __init__(self, db_path="fire_data.db"):
        """初始化数据库查询工具"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """连接数据库"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print(f"✅ 数据库连接成功: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("数据库连接已关闭")
    
    def get_table_info(self):
        """获取所有表信息"""
        try:
            # 获取所有表名
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            
            print("=== 数据库表信息 ===")
            for table in tables:
                table_name = table[0]
                if table_name.startswith('fire_'):
                    # 获取表结构
                    self.cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = self.cursor.fetchall()
                    
                    # 获取记录数
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = self.cursor.fetchone()[0]
                    
                    print(f"\n📋 表名: {table_name}")
                    print(f"   记录数: {count}")
                    print(f"   字段数: {len(columns)}")
                    print("   字段列表:")
                    for col in columns:
                        print(f"     - {col[1]} ({col[2]})")
            
            return True
            
        except Exception as e:
            print(f"❌ 获取表信息失败: {e}")
            return False
    
    def query_table(self, table_name, limit=10):
        """查询表数据"""
        try:
            print(f"\n=== 查询表: {table_name} ===")
            
            # 获取总记录数
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = self.cursor.fetchone()[0]
            print(f"总记录数: {total_count}")
            
            if total_count == 0:
                print("表中没有数据")
                return
            
            # 查询前几条记录
            self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = self.cursor.fetchall()
            
            # 获取列名
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            print(f"\n前 {len(rows)} 条记录:")
            for i, row in enumerate(rows, 1):
                print(f"\n--- 记录 {i} ---")
                for col_name, value in zip(columns, row):
                    if col_name in ['content', 'summary'] and value and len(str(value)) > 100:
                        print(f"{col_name}: {str(value)[:100]}...")
                    elif col_name in ['keywords', 'tags', 'entities', 'chapters', 'articles'] and value:
                        try:
                            parsed = json.loads(value)
                            print(f"{col_name}: {parsed}")
                        except:
                            print(f"{col_name}: {value}")
                    else:
                        print(f"{col_name}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ 查询表数据失败: {e}")
            return False
    
    def get_statistics(self):
        """获取数据库统计信息"""
        try:
            print("\n=== 数据库统计信息 ===")
            
            # 获取所有表名
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            
            total_records = 0
            table_stats = {}
            
            for table in tables:
                table_name = table[0]
                if table_name.startswith('fire_'):
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = self.cursor.fetchone()[0]
                    table_stats[table_name] = count
                    total_records += count
            
            print(f"数据库文件: {self.db_path}")
            print(f"总记录数: {total_records}")
            print(f"表数量: {len(table_stats)}")
            print("\n各表记录数:")
            for table_name, count in table_stats.items():
                print(f"  {table_name}: {count} 条")
            
            return table_stats
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {}
    
    def search_content(self, keyword, table_name=None):
        """搜索内容"""
        try:
            print(f"\n=== 搜索关键词: {keyword} ===")
            
            # 获取所有表名
            if table_name:
                tables = [(table_name,)]
            else:
                self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = self.cursor.fetchall()
            
            results = []
            
            for table in tables:
                table_name = table[0]
                if table_name.startswith('fire_'):
                    # 搜索标题和内容字段
                    search_sql = f"""
                    SELECT * FROM {table_name} 
                    WHERE title LIKE ? OR content LIKE ? OR summary LIKE ?
                    LIMIT 5
                    """
                    
                    search_term = f"%{keyword}%"
                    self.cursor.execute(search_sql, (search_term, search_term, search_term))
                    rows = self.cursor.fetchall()
                    
                    if rows:
                        # 获取列名
                        self.cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in self.cursor.fetchall()]
                        
                        print(f"\n📋 在表 {table_name} 中找到 {len(rows)} 条记录:")
                        for i, row in enumerate(rows, 1):
                            print(f"\n--- 记录 {i} ---")
                            row_dict = dict(zip(columns, row))
                            print(f"ID: {row_dict.get('id', 'N/A')}")
                            print(f"标题: {row_dict.get('title', 'N/A')}")
                            if row_dict.get('summary'):
                                print(f"摘要: {row_dict.get('summary', '')[:100]}...")
                            
                            results.append({
                                'table': table_name,
                                'id': row_dict.get('id'),
                                'title': row_dict.get('title'),
                                'summary': row_dict.get('summary', '')[:100] if row_dict.get('summary') else ''
                            })
            
            if not results:
                print("未找到匹配的记录")
            
            return results
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
    
    def export_to_csv(self, table_name, output_file=None):
        """导出表数据到CSV"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{table_name}_{timestamp}.csv"
            
            print(f"\n=== 导出表 {table_name} 到 {output_file} ===")
            
            # 查询所有数据
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()
            
            if not rows:
                print("表中没有数据")
                return False
            
            # 获取列名
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"✅ 导出成功，共 {len(rows)} 条记录")
            return True
            
        except Exception as e:
            print(f"❌ 导出失败: {e}")
            return False
    
    def run_interactive_mode(self):
        """运行交互模式"""
        while True:
            print("\n=== 数据库查询工具 ===")
            print("1. 查看表信息")
            print("2. 查询表数据")
            print("3. 获取统计信息")
            print("4. 搜索内容")
            print("5. 导出数据到CSV")
            print("6. 退出")
            
            choice = input("\n请选择操作 (1-6): ").strip()
            
            if choice == '1':
                self.get_table_info()
            elif choice == '2':
                table_name = input("请输入表名: ").strip()
                limit = input("请输入查询条数 (默认10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                self.query_table(table_name, limit)
            elif choice == '3':
                self.get_statistics()
            elif choice == '4':
                keyword = input("请输入搜索关键词: ").strip()
                table_name = input("请输入表名 (可选，留空搜索所有表): ").strip() or None
                self.search_content(keyword, table_name)
            elif choice == '5':
                table_name = input("请输入表名: ").strip()
                output_file = input("请输入输出文件名 (可选): ").strip() or None
                self.export_to_csv(table_name, output_file)
            elif choice == '6':
                print("退出查询工具")
                break
            else:
                print("❌ 无效选择，请重新输入")


def main():
    """主函数"""
    print("=== 消防数据数据库查询工具 ===")
    
    # 检查数据库文件是否存在
    import os
    db_path = "fire_data.db"
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        print("请先运行爬虫生成数据库文件")
        return
    
    # 创建查询工具
    tool = DatabaseQueryTool(db_path)
    
    # 连接数据库
    if not tool.connect():
        return
    
    try:
        # 显示基本信息
        tool.get_statistics()
        
        # 运行交互模式
        tool.run_interactive_mode()
        
    finally:
        tool.disconnect()


if __name__ == "__main__":
    main()

