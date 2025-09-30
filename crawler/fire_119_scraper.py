"""
国家消防救援局（119）科普栏目数据抓取脚本
包含robots检查、限速、HTML到JSON保存等功能
"""

import scrapy
import requests
import json
import time
import random
import re
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import Dict, List, Optional
import logging
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.utils.response import get_base_url
from scrapy.exceptions import DropItem
import os

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Fire119Spider(Spider):
    """国家消防救援局科普栏目爬虫"""
    
    name = 'fire_119_spider'
    allowed_domains = ['www.119.gov.cn', '119.gov.cn']
    
    # 起始URL - 国家消防救援局官网科普栏目
    start_urls = [
        'https://www.119.gov.cn/kp/',
    ]
    
    # 自定义设置
    custom_settings = {
        'DOWNLOAD_DELAY': 2,  # 下载延迟2秒
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,  # 随机延迟0.5秒
        'CONCURRENT_REQUESTS': 1,  # 并发请求数
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # 每个域名并发数
        'AUTOTHROTTLE_ENABLED': True,  # 启用自动限速
        'AUTOTHROTTLE_START_DELAY': 1,  # 初始延迟
        'AUTOTHROTTLE_MAX_DELAY': 10,  # 最大延迟
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,  # 目标并发数
        'AUTOTHROTTLE_DEBUG': True,  # 调试模式
        'ROBOTSTXT_OBEY': True,  # 遵守robots.txt
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    }
    
    def __init__(self):
        """初始化爬虫"""
        super().__init__()
        self.robots_parser = RobotFileParser()
        self.session = requests.Session()
        self.setup_session()
        self.check_robots_txt()
        
    def setup_session(self):
        """设置请求会话"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
    def check_robots_txt(self):
        """检查robots.txt"""
        try:
            robots_url = 'https://www.119.gov.cn/robots.txt'
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info(f"Robots.txt 检查完成: {robots_url}")
        except Exception as e:
            logger.warning(f"无法获取robots.txt: {e}")
    
    def can_fetch(self, url: str) -> bool:
        """检查是否可以抓取指定URL"""
        try:
            return self.robots_parser.can_fetch('*', url)
        except:
            return True  # 如果无法检查，默认允许
    
    def start_requests(self):
        """生成初始请求"""
        for url in self.start_urls:
            if self.can_fetch(url):
                yield Request(
                    url=url,
                    callback=self.parse,
                    meta={'page_type': 'list'},
                    dont_filter=True
                )
            else:
                logger.warning(f"Robots.txt 禁止访问: {url}")
    
    def parse(self, response):
        """解析页面"""
        try:
            page_type = response.meta.get('page_type', 'list')
            
            if page_type == 'list':
                yield from self.parse_list_page(response)
            elif page_type == 'detail':
                yield from self.parse_detail_page(response)
                
        except Exception as e:
            logger.error(f"解析页面失败: {e}")
            self.logger.error(f"解析页面失败: {e}")
    
    def parse_list_page(self, response):
        """解析列表页面"""
        try:
            # 提取文章链接
            article_links = response.css('a[href*=".shtml"]::attr(href)').getall()
            
            for link in article_links:
                full_url = urljoin(response.url, link)
                
                if self.can_fetch(full_url):
                    yield Request(
                        url=full_url,
                        callback=self.parse,
                        meta={'page_type': 'detail'},
                        dont_filter=True
                    )
            
            # 查找分页链接
            next_page = response.css('a[href*="page"]::attr(href)').getall()
            for page_link in next_page:
                full_url = urljoin(response.url, page_link)
                if self.can_fetch(full_url):
                    yield Request(
                        url=full_url,
                        callback=self.parse,
                        meta={'page_type': 'list'},
                        dont_filter=True
                    )
                    
        except Exception as e:
            logger.error(f"解析列表页面失败: {e}")
    
    def parse_detail_page(self, response):
        """解析详情页面"""
        try:
            # 提取文章信息
            article_data = {
                'url': response.url,
                'title': self.extract_title(response),
                'content': self.extract_content(response),
                'publish_time': self.extract_publish_time(response),
                'author': self.extract_author(response),
                'category': self.extract_category(response),
                'tags': self.extract_tags(response),
                'images': self.extract_images(response),
                'crawl_time': datetime.now().isoformat(),
                'source': '国家消防救援局官网'
            }
            
            # 数据清洗
            article_data = self.clean_data(article_data)
            
            yield article_data
            
        except Exception as e:
            logger.error(f"解析详情页面失败: {e}")
    
    def extract_title(self, response) -> str:
        """提取标题"""
        title_selectors = [
            'h1::text',
            '.article-title::text',
            '.content-title::text',
            'title::text'
        ]
        
        for selector in title_selectors:
            title = response.css(selector).get()
            if title:
                return title.strip()
        return ""
    
    def extract_content(self, response) -> str:
        """提取内容"""
        content_selectors = [
            '.article-content',
            '.content-body',
            '.article-body',
            '.main-content'
        ]
        
        for selector in content_selectors:
            content = response.css(selector).get()
            if content:
                # 清理HTML标签
                content = re.sub(r'<[^>]+>', '', content)
                return content.strip()
        return ""
    
    def extract_publish_time(self, response) -> str:
        """提取发布时间"""
        time_selectors = [
            '.publish-time::text',
            '.article-time::text',
            '.time::text',
            'time::attr(datetime)'
        ]
        
        for selector in time_selectors:
            time_text = response.css(selector).get()
            if time_text:
                return time_text.strip()
        return ""
    
    def extract_author(self, response) -> str:
        """提取作者"""
        author_selectors = [
            '.author::text',
            '.article-author::text',
            '.writer::text'
        ]
        
        for selector in author_selectors:
            author = response.css(selector).get()
            if author:
                return author.strip()
        return ""
    
    def extract_category(self, response) -> str:
        """提取分类"""
        category_selectors = [
            '.category::text',
            '.article-category::text',
            '.breadcrumb a::text'
        ]
        
        for selector in category_selectors:
            category = response.css(selector).get()
            if category:
                return category.strip()
        return "科普"
    
    def extract_tags(self, response) -> List[str]:
        """提取标签"""
        tags = response.css('.tags a::text, .tag::text').getall()
        return [tag.strip() for tag in tags if tag.strip()]
    
    def extract_images(self, response) -> List[str]:
        """提取图片链接"""
        images = response.css('img::attr(src)').getall()
        return [urljoin(response.url, img) for img in images if img]
    
    def clean_data(self, data: Dict) -> Dict:
        """数据清洗"""
        # 清理标题
        if data.get('title'):
            data['title'] = re.sub(r'\s+', ' ', data['title']).strip()
        
        # 清理内容
        if data.get('content'):
            data['content'] = re.sub(r'\s+', ' ', data['content']).strip()
        
        # 清理时间格式
        if data.get('publish_time'):
            data['publish_time'] = self.clean_time_format(data['publish_time'])
        
        return data
    
    def clean_time_format(self, time_str: str) -> str:
        """清理时间格式"""
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

class Fire119DataProcessor:
    """消防119数据处理类"""
    
    def __init__(self, output_dir: str = "data"):
        """初始化处理器"""
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_to_json(self, data: Dict, filename: str = None):
        """保存数据到JSON文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_119_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def save_to_csv(self, data: List[Dict], filename: str = None):
        """保存数据到CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fire_119_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存CSV数据失败: {e}")

class Fire119Pipeline:
    """消防119数据管道"""
    
    def __init__(self):
        """初始化管道"""
        self.processor = Fire119DataProcessor()
        self.items = []
    
    def process_item(self, item, spider):
        """处理单个项目"""
        try:
            # 数据验证
            if not self.validate_item(item):
                raise DropItem(f"数据验证失败: {item}")
            
            # 添加到列表
            self.items.append(item)
            
            # 每100个项目保存一次
            if len(self.items) >= 100:
                self.save_batch()
            
            return item
            
        except Exception as e:
            logger.error(f"处理项目失败: {e}")
            raise DropItem(f"处理项目失败: {e}")
    
    def validate_item(self, item: Dict) -> bool:
        """验证数据项"""
        # 必须有URL和标题
        if not item.get('url') or not item.get('title'):
            return False
        
        # 内容可以为空，但如果有内容则必须超过10个字符
        content = item.get('content', '')
        if content and len(content.strip()) < 10:
            return False
        
        return True
    
    def save_batch(self):
        """批量保存数据"""
        if self.items:
            self.processor.save_to_json(self.items)
            self.items = []
    
    def close_spider(self, spider):
        """爬虫关闭时保存剩余数据"""
        if self.items:
            self.processor.save_to_json(self.items)
            self.items = []

def run_scrapy_spider():
    """运行Scrapy爬虫"""
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    
    # 设置
    settings = get_project_settings()
    settings.update({
        'ITEM_PIPELINES': {
            '__main__.Fire119Pipeline': 300,
        },
        'FEEDS': {
            'data/fire_119_scrapy.json': {
                'format': 'json',
                'encoding': 'utf8',
                'store_empty': False,
                'indent': 2,
            },
        },
    })
    
    # 运行爬虫
    process = CrawlerProcess(settings)
    process.crawl(Fire119Spider)
    process.start()

def run_requests_scraper():
    """运行requests版本爬虫"""
    scraper = Fire119RequestsScraper()
    scraper.run()

class Fire119RequestsScraper:
    """基于requests的消防119爬虫"""
    
    def __init__(self):
        """初始化爬虫"""
        self.session = requests.Session()
        self.setup_session()
        self.processor = Fire119DataProcessor()
        self.robots_parser = RobotFileParser()
        self.check_robots_txt()
        
    def setup_session(self):
        """设置会话"""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def check_robots_txt(self):
        """检查robots.txt"""
        try:
            robots_url = 'https://www.119.gov.cn/robots.txt'
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            logger.info(f"Robots.txt 检查完成: {robots_url}")
        except Exception as e:
            logger.warning(f"无法获取robots.txt: {e}")
    
    def can_fetch(self, url: str) -> bool:
        """检查是否可以抓取"""
        try:
            return self.robots_parser.can_fetch('*', url)
        except:
            return True
    
    def get_page(self, url: str) -> Optional[requests.Response]:
        """获取页面"""
        try:
            if not self.can_fetch(url):
                logger.warning(f"Robots.txt 禁止访问: {url}")
                return None
            
            # 随机延迟
            time.sleep(random.uniform(1, 3))
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response
            
        except Exception as e:
            logger.error(f"获取页面失败 {url}: {e}")
            return None
    
    def parse_list_page(self, response: requests.Response) -> List[str]:
        """解析列表页面"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取文章链接
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '.html' in href:
                    full_url = urljoin(response.url, href)
                    links.append(full_url)
            
            return links
            
        except Exception as e:
            logger.error(f"解析列表页面失败: {e}")
            return []
    
    def parse_detail_page(self, response: requests.Response) -> Dict:
        """解析详情页面"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取文章信息
            title = self.extract_title(soup)
            content = self.extract_content(soup)
            
            # 调试信息
            logger.info(f"提取标题: {title[:50]}..." if title else "标题为空")
            logger.info(f"提取内容长度: {len(content)} 字符")
            
            article_data = {
                'url': response.url,
                'title': title,
                'content': content,
                'publish_time': self.extract_publish_time(soup),
                'author': self.extract_author(soup),
                'category': self.extract_category(soup),
                'tags': self.extract_tags(soup),
                'images': self.extract_images(soup, response.url),
                'crawl_time': datetime.now().isoformat(),
                'source': '国家消防救援局官网'
            }
            
            return article_data
            
        except Exception as e:
            logger.error(f"解析详情页面失败: {e}")
            return {}
    
    def extract_title(self, soup) -> str:
        """提取标题"""
        title_selectors = ['h1', '.article-title', '.content-title', 'title']
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return ""
    
    def extract_content(self, soup) -> str:
        """提取内容"""
        content_selectors = [
            '.article-content', 
            '.content-body', 
            '.article-body', 
            '.main-content',
            '.content',
            '.article',
            '.text',
            'p',
            '.detail-content',
            '.news-content'
        ]
        
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                content = element.get_text().strip()
                if len(content) > 50:  # 确保内容足够长
                    return content
        
        # 如果所有选择器都失败，尝试提取所有p标签
        paragraphs = soup.select('p')
        if paragraphs:
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            if len(content) > 50:
                return content
        
        return ""
    
    def extract_publish_time(self, soup) -> str:
        """提取发布时间"""
        time_selectors = ['.publish-time', '.article-time', '.time', 'time']
        
        for selector in time_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return ""
    
    def extract_author(self, soup) -> str:
        """提取作者"""
        author_selectors = ['.author', '.article-author', '.writer']
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return ""
    
    def extract_category(self, soup) -> str:
        """提取分类"""
        category_selectors = ['.category', '.article-category', '.breadcrumb a']
        
        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text().strip()
        return "科普"
    
    def extract_tags(self, soup) -> List[str]:
        """提取标签"""
        tags = soup.select('.tags a, .tag')
        return [tag.get_text().strip() for tag in tags if tag.get_text().strip()]
    
    def extract_images(self, soup, base_url: str) -> List[str]:
        """提取图片链接"""
        images = soup.select('img')
        return [urljoin(base_url, img.get('src', '')) for img in images if img.get('src')]
    
    def run(self):
        """运行爬虫"""
        logger.info("开始运行消防119爬虫...")
        
        # 起始URL
        start_urls = [
            'https://www.119.gov.cn/kp/',
            'https://www.119.gov.cn/kp/kpzt/',
            'https://www.119.gov.cn/kp/kpxw/',
            'https://www.119.gov.cn/kp/kpzs/',
        ]
        
        all_articles = []
        
        for url in start_urls:
            logger.info(f"处理列表页面: {url}")
            
            # 获取列表页面
            response = self.get_page(url)
            if not response:
                continue
            
            # 解析文章链接
            article_links = self.parse_list_page(response)
            logger.info(f"找到 {len(article_links)} 篇文章")
            
            # 处理每篇文章
            for article_url in article_links[:10]:  # 限制数量
                logger.info(f"处理文章: {article_url}")
                
                article_response = self.get_page(article_url)
                if not article_response:
                    continue
                
                article_data = self.parse_detail_page(article_response)
                if article_data:
                    all_articles.append(article_data)
                
                # 延迟
                time.sleep(random.uniform(1, 2))
        
        # 保存数据
        if all_articles:
            self.processor.save_to_json(all_articles)
            logger.info(f"爬取完成，共获取 {len(all_articles)} 篇文章")
        else:
            logger.warning("未获取到任何数据")

def main():
    """主函数"""
    print("🔥 国家消防救援局（119）科普栏目数据抓取")
    print("=" * 50)
    
    while True:
        print("\n请选择抓取方式:")
        print("1. Scrapy爬虫 (推荐)")
        print("2. Requests爬虫")
        print("3. 检查robots.txt")
        print("0. 退出")
        
        choice = input("请输入选择 (0-3): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            print("🚀 启动Scrapy爬虫...")
            run_scrapy_spider()
        elif choice == "2":
            print("🚀 启动Requests爬虫...")
            run_requests_scraper()
        elif choice == "3":
            print("🔍 检查robots.txt...")
            check_robots()
        else:
            print("❌ 无效选择")

def check_robots():
    """检查robots.txt"""
    try:
        robots_url = 'https://www.119.gov.cn/robots.txt'
        response = requests.get(robots_url, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Robots.txt 可访问: {robots_url}")
            print("内容预览:")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        else:
            print(f"❌ Robots.txt 不可访问: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 检查robots.txt失败: {e}")

if __name__ == "__main__":
    main()
