#!/usr/bin/env python
"""
数据库安装和配置脚本
"""

import os
import sys
import subprocess
from database_manager import DatabaseManagerTool


def install_mysql_driver():
    """安装MySQL驱动"""
    print("正在安装MySQL驱动...")
    
    try:
        # 安装PyMySQL
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMySQL"])
        print("✅ PyMySQL 安装成功")
        
        # 安装pandas（如果还没有）
        try:
            import pandas
            print("✅ pandas 已安装")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
            print("✅ pandas 安装成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 安装MySQL驱动失败: {e}")
        return False


def setup_database():
    """设置数据库"""
    print("=== 数据库设置向导 ===")
    
    # 安装MySQL驱动
    if not install_mysql_driver():
        return False
    
    # 创建数据库管理工具
    tool = DatabaseManagerTool()
    
    # 设置数据库连接
    if not tool.setup_database():
        return False
    
    # 创建数据库表
    if not tool.create_tables():
        return False
    
    print("✅ 数据库设置完成")
    return True


def test_database_connection():
    """测试数据库连接"""
    print("=== 测试数据库连接 ===")
    
    tool = DatabaseManagerTool()
    
    if tool.test_connection():
        print("✅ 数据库连接测试成功")
        
        # 显示数据库统计
        tool.get_database_stats()
        return True
    else:
        print("❌ 数据库连接测试失败")
        return False


def import_sample_data():
    """导入示例数据"""
    print("=== 导入示例数据 ===")
    
    tool = DatabaseManagerTool()
    
    # 检查是否有CSV文件
    csv_files = [
        'fire_regulations_*.csv',
        'fire_standards_*.csv',
        'fire_cases_*.csv',
        'fire_news_*.csv',
        'fire_knowledge_*.csv',
        'fire_documents_*.csv'
    ]
    
    import glob
    
    for pattern in csv_files:
        files = glob.glob(pattern)
        if files:
            csv_file = files[0]  # 使用第一个匹配的文件
            table_name = pattern.split('_')[0] + '_' + pattern.split('_')[1]
            
            print(f"发现CSV文件: {csv_file}")
            if input(f"是否导入到 {table_name} 表? (y/n): ").lower() == 'y':
                tool.import_csv_data(csv_file, table_name)
    
    print("✅ 示例数据导入完成")


def main():
    """主函数"""
    print("=== 消防数据爬虫数据库设置工具 ===")
    print()
    
    while True:
        print("请选择操作:")
        print("1. 完整设置（安装驱动 + 配置数据库 + 创建表）")
        print("2. 仅测试数据库连接")
        print("3. 导入示例数据")
        print("4. 运行数据库管理工具")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            if setup_database():
                print("✅ 数据库设置完成")
            else:
                print("❌ 数据库设置失败")
        elif choice == '2':
            test_database_connection()
        elif choice == '3':
            import_sample_data()
        elif choice == '4':
            tool = DatabaseManagerTool()
            tool.run_interactive_mode()
        elif choice == '5':
            print("退出")
            break
        else:
            print("❌ 无效选择，请重新输入")
        
        print()


if __name__ == "__main__":
    main()

