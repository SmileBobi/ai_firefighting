"""
消防119爬虫运行脚本
简化运行流程
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'scrapy',
        'requests',
        'beautifulsoup4',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'beautifulsoup4':
                import bs4
            else:
                __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            print(f"❌ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 需要安装的包: {', '.join(missing_packages)}")
        print("运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def create_directories():
    """创建必要目录"""
    print("📁 创建目录结构...")
    
    directories = [
        'data',
        'logs',
        'fire_119_scraper'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def run_scrapy_spider():
    """运行Scrapy爬虫"""
    print("🚀 启动Scrapy爬虫...")
    
    try:
        # 运行Scrapy爬虫
        result = subprocess.run([
            'scrapy', 'crawl', 'fire_119_spider',
            '-o', 'data/fire_119_scrapy.json',
            '-L', 'INFO'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Scrapy爬虫运行成功")
            print("📊 数据已保存到: data/fire_119_scrapy.json")
        else:
            print("❌ Scrapy爬虫运行失败")
            print(f"错误信息: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 运行Scrapy爬虫失败: {e}")

def run_requests_scraper():
    """运行requests爬虫"""
    print("🚀 启动requests爬虫...")
    
    try:
        # 导入并运行requests爬虫
        from fire_119_scraper import Fire119RequestsScraper
        
        scraper = Fire119RequestsScraper()
        scraper.run()
        
        print("✅ requests爬虫运行成功")
        
    except Exception as e:
        print(f"❌ 运行requests爬虫失败: {e}")

def check_robots_txt():
    """检查robots.txt"""
    print("🔍 检查robots.txt...")
    
    try:
        import requests
        
        robots_url = 'https://www.119.gov.cn/robots.txt'
        response = requests.get(robots_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Robots.txt 可访问: {robots_url}")
            print("内容预览:")
            content = response.text
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)
        else:
            print(f"❌ Robots.txt 不可访问: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 检查robots.txt失败: {e}")

def show_results():
    """显示结果"""
    print("\n📊 爬取结果:")
    print("=" * 50)
    
    # 检查输出文件
    output_files = [
        'data/fire_119_scrapy.json',
        'data/fire_119_data.csv',
        'data/fire_119.db'
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} - {size} bytes")
        else:
            print(f"❌ {file_path} - 不存在")

def main():
    """主函数"""
    print("🔥 国家消防救援局（119）科普栏目数据抓取")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 依赖检查失败，请先安装依赖包")
        return
    
    # 创建目录
    create_directories()
    
    while True:
        print("\n请选择操作:")
        print("1. 运行Scrapy爬虫")
        print("2. 运行requests爬虫")
        print("3. 检查robots.txt")
        print("4. 查看结果")
        print("5. 安装依赖包")
        print("0. 退出")
        
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            run_scrapy_spider()
        elif choice == "2":
            run_requests_scraper()
        elif choice == "3":
            check_robots_txt()
        elif choice == "4":
            show_results()
        elif choice == "5":
            print("📦 安装依赖包...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        else:
            print("❌ 无效选择")
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("👋 谢谢使用！")

if __name__ == "__main__":
    main()
