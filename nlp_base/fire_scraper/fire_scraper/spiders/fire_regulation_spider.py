import scrapy
import json
import re
from datetime import datetime
from urllib.parse import urljoin, urlencode
from typing import List, Dict
from fire_scraper.items import FireRegulationItem, FireStandardItem, FireCaseItem, FireNewsItem, FireKnowledgeItem, FireDocumentItem
from fire_text_analyzer import fire_text_analyzer


class FireRegulationSpiderSpider(scrapy.Spider):
    name = "fire_regulation_spider"
    allowed_domains = ["119.gov.cn", "www.119.gov.cn", "xf.mem.gov.cn", "www.xf.mem.gov.cn"]
    
    def __init__(self, data_types='regulation,standard,case,news', max_pages=5, *args, **kwargs):
        super(FireRegulationSpiderSpider, self).__init__(*args, **kwargs)
        
        # 设置参数
        self.data_types = data_types.split(',')
        self.max_pages = int(max_pages)
        
        # 构建起始URL列表
        self.start_urls = self.build_start_urls()
    
    def build_start_urls(self):
        """构建起始URL列表"""
        urls = []
        
        if 'regulation' in self.data_types:
            # 消防法规页面
            urls.append("http://www.119.gov.cn/xiaofang/fagui/")
            urls.append("http://xf.mem.gov.cn/gk/fg/")
        
        if 'standard' in self.data_types:
            # 消防标准页面
            urls.append("http://www.119.gov.cn/xiaofang/biaozhun/")
            urls.append("http://xf.mem.gov.cn/gk/bz/")
        
        if 'case' in self.data_types:
            # 火灾案例页面
            urls.append("http://www.119.gov.cn/xiaofang/anli/")
            urls.append("http://xf.mem.gov.cn/gk/al/")
        
        if 'news' in self.data_types:
            # 消防新闻页面
            urls.append("http://www.119.gov.cn/xiaofang/xinwen/")
            urls.append("http://xf.mem.gov.cn/gk/xw/")
        
        return urls
    
    def start_requests(self):
        """开始请求"""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse_list_page,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'http://www.119.gov.cn/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
    
    def parse_list_page(self, response):
        """解析列表页面"""
        try:
            # 提取文档链接
            doc_links = response.css('a[href*=".html"], a[href*=".pdf"], a[href*=".doc"]::attr(href)').getall()
            
            for link in doc_links[:10]:  # 限制每页处理数量
                full_url = urljoin(response.url, link)
                
                # 根据URL判断文档类型
                doc_type = self.determine_document_type(full_url, response.url)
                
                yield scrapy.Request(
                    url=full_url,
                    callback=self.parse_document,
                    meta={'doc_type': doc_type, 'source_url': response.url},
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Referer': response.url
                    }
                )
            
            # 翻页处理
            next_page = response.css('a[href*="page"]::attr(href)').get()
            if next_page and self.max_pages > 1:
                next_url = urljoin(response.url, next_page)
                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse_list_page,
                    meta={'page': 2}
                )
                
        except Exception as e:
            self.logger.error(f"解析列表页面出错: {e}")
    
    def parse_document(self, response):
        """解析文档页面"""
        try:
            doc_type = response.meta.get('doc_type', 'unknown')
            
            if doc_type == 'regulation':
                yield from self.parse_regulation(response)
            elif doc_type == 'standard':
                yield from self.parse_standard(response)
            elif doc_type == 'case':
                yield from self.parse_case(response)
            elif doc_type == 'news':
                yield from self.parse_news(response)
            else:
                yield from self.parse_general_document(response)
                
        except Exception as e:
            self.logger.error(f"解析文档出错: {e}")
    
    def parse_regulation(self, response):
        """解析消防法规"""
        try:
            item = FireRegulationItem()
            
            # 基本信息
            item['regulation_id'] = self.extract_document_id(response.url)
            item['title'] = self.extract_title(response)
            item['regulation_type'] = '部门规章'
            item['level'] = '国家级'
            item['issuing_authority'] = '应急管理部'
            item['issue_date'] = ''
            item['effective_date'] = ''
            item['status'] = '现行有效'
            
            # 内容信息
            content = self.extract_content(response)
            item['content'] = content
            item['summary'] = fire_text_analyzer.generate_document_summary(content)
            item['chapters'] = []
            item['articles'] = []
            
            # 分类信息
            analysis = fire_text_analyzer.analyze_regulation(content)
            item['category'] = analysis.get('fire_analysis', {}).get('main_category', '其他')
            item['keywords'] = analysis.get('keywords', [])
            item['tags'] = []
            
            # 文件信息
            item['file_url'] = response.url
            item['file_type'] = self.determine_file_type(response)
            item['file_size'] = len(response.text)
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.meta.get('source_url', response.url)
            
            yield item
            
            # 生成文档分析结果
            yield from self.generate_document_analysis(item, content, 'regulation')
            
        except Exception as e:
            self.logger.error(f"解析消防法规出错: {e}")
    
    def parse_standard(self, response):
        """解析消防标准"""
        try:
            item = FireStandardItem()
            
            # 基本信息
            item['standard_id'] = self.extract_document_id(response.url)
            item['standard_number'] = ''
            item['title'] = self.extract_title(response)
            item['english_title'] = ''
            item['standard_type'] = '国家标准'
            item['category'] = '强制性'
            
            # 发布信息
            item['issuing_authority'] = '应急管理部'
            item['approval_authority'] = '应急管理部'
            item['issue_date'] = ''
            item['implementation_date'] = ''
            item['status'] = '现行'
            
            # 内容信息
            content = self.extract_content(response)
            item['content'] = content
            item['scope'] = ''
            item['technical_requirements'] = []
            item['test_methods'] = []
            item['inspection_rules'] = []
            
            # 分类信息
            analysis = fire_text_analyzer.analyze_standard(content)
            item['fire_category'] = analysis.get('fire_analysis', {}).get('main_category', '其他')
            item['keywords'] = analysis.get('keywords', [])
            item['tags'] = []
            
            # 文件信息
            item['file_url'] = response.url
            item['file_type'] = self.determine_file_type(response)
            item['file_size'] = len(response.text)
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.meta.get('source_url', response.url)
            
            yield item
            
            # 生成文档分析结果
            yield from self.generate_document_analysis(item, content, 'standard')
            
        except Exception as e:
            self.logger.error(f"解析消防标准出错: {e}")
    
    def parse_case(self, response):
        """解析火灾案例"""
        try:
            item = FireCaseItem()
            
            # 基本信息
            item['case_id'] = self.extract_document_id(response.url)
            item['title'] = self.extract_title(response)
            item['case_type'] = '火灾事故'
            item['severity_level'] = '一般'
            
            # 时间地点
            item['incident_date'] = ''
            item['location'] = ''
            item['province'] = ''
            item['city'] = ''
            item['district'] = ''
            
            # 事故信息
            item['building_type'] = '其他'
            item['fire_cause'] = '其他'
            item['casualties'] = {}
            item['economic_loss'] = ''
            item['fire_duration'] = ''
            
            # 内容信息
            content = self.extract_content(response)
            item['content'] = content
            item['description'] = ''
            item['investigation_result'] = ''
            item['lessons_learned'] = ''
            item['prevention_measures'] = ''
            
            # 分类信息
            analysis = fire_text_analyzer.analyze_case(content)
            item['keywords'] = analysis.get('keywords', [])
            item['tags'] = []
            
            # 文件信息
            item['file_url'] = response.url
            item['file_type'] = self.determine_file_type(response)
            item['file_size'] = len(response.text)
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.meta.get('source_url', response.url)
            
            yield item
            
            # 生成文档分析结果
            yield from self.generate_document_analysis(item, content, 'case')
            
        except Exception as e:
            self.logger.error(f"解析火灾案例出错: {e}")
    
    def parse_news(self, response):
        """解析消防新闻"""
        try:
            item = FireNewsItem()
            
            # 基本信息
            item['news_id'] = self.extract_document_id(response.url)
            item['title'] = self.extract_title(response)
            content = self.extract_content(response)
            item['content'] = content
            item['summary'] = fire_text_analyzer.generate_document_summary(content)
            item['author'] = ''
            item['source'] = '消防部门'
            item['publish_time'] = ''
            
            # 分类信息
            item['news_type'] = '行业资讯'
            item['category'] = '消防新闻'
            analysis = fire_text_analyzer.analyze_fire_text(content)
            item['keywords'] = analysis.get('keywords', [])
            item['tags'] = []
            
            # 互动数据
            item['view_count'] = 0
            item['comment_count'] = 0
            item['share_count'] = 0
            
            # 文件信息
            item['image_urls'] = []
            item['file_urls'] = []
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.meta.get('source_url', response.url)
            
            yield item
            
            # 生成文档分析结果
            yield from self.generate_document_analysis(item, content, 'news')
            
        except Exception as e:
            self.logger.error(f"解析消防新闻出错: {e}")
    
    def parse_general_document(self, response):
        """解析通用文档"""
        try:
            item = FireDocumentItem()
            
            # 基本信息
            item['document_id'] = self.extract_document_id(response.url)
            item['title'] = self.extract_title(response)
            content = self.extract_content(response)
            item['content'] = content
            item['document_type'] = self.determine_document_type(response.url, response.meta.get('source_url', ''))
            
            # 文档特征
            analysis = fire_text_analyzer.analyze_fire_text(content)
            item['content_length'] = analysis.get('char_count', 0)
            item['word_count'] = analysis.get('word_count', 0)
            item['readability_score'] = analysis.get('readability_score', 0)
            item['complexity_score'] = analysis.get('complexity_score', 0)
            
            # 文本分析
            item['summary'] = fire_text_analyzer.generate_document_summary(content)
            item['keywords'] = analysis.get('keywords', [])
            item['entities'] = analysis.get('entities', {})
            item['topics'] = analysis.get('classification', {})
            
            # RAG相关
            chunks = fire_text_analyzer.chunk_text_for_rag(content)
            item['chunk_texts'] = [chunk['text'] for chunk in chunks]
            item['embedding_vector'] = None  # 需要后续处理
            item['chunk_embeddings'] = None  # 需要后续处理
            
            # 文件信息
            item['file_url'] = response.url
            item['file_type'] = self.determine_file_type(response)
            item['file_size'] = len(response.text)
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.meta.get('source_url', response.url)
            
            yield item
            
        except Exception as e:
            self.logger.error(f"解析通用文档出错: {e}")
    
    def generate_document_analysis(self, item, content, doc_type):
        """生成文档分析结果"""
        try:
            # 创建知识条目
            knowledge_item = FireKnowledgeItem()
            knowledge_item['knowledge_id'] = item.get('regulation_id') or item.get('standard_id') or item.get('case_id') or item.get('news_id')
            knowledge_item['title'] = item.get('title', '')
            knowledge_item['content'] = content
            knowledge_item['summary'] = fire_text_analyzer.generate_document_summary(content)
            knowledge_item['knowledge_type'] = doc_type
            
            # 分析内容
            analysis = fire_text_analyzer.analyze_fire_text(content)
            knowledge_item['category'] = analysis.get('fire_analysis', {}).get('main_category', '其他')
            knowledge_item['subcategory'] = analysis.get('classification', {}).get('category', '其他')
            knowledge_item['difficulty_level'] = '中级'  # 默认中级
            knowledge_item['target_audience'] = '专业人员'
            
            # 内容特征
            knowledge_item['content_length'] = analysis.get('char_count', 0)
            knowledge_item['word_count'] = analysis.get('word_count', 0)
            knowledge_item['section_count'] = len(analysis.get('entities', {}).get('article_number', []))
            knowledge_item['image_count'] = 0  # 需要进一步分析
            knowledge_item['table_count'] = 0  # 需要进一步分析
            
            # 关键词和标签
            knowledge_item['keywords'] = analysis.get('keywords', [])
            knowledge_item['tags'] = item.get('tags', [])
            knowledge_item['entities'] = analysis.get('entities', {})
            
            # 文件信息
            knowledge_item['file_url'] = item.get('file_url', '')
            knowledge_item['file_type'] = item.get('file_type', '')
            knowledge_item['file_size'] = item.get('file_size', 0)
            knowledge_item['download_status'] = 'success'
            
            # 爬取信息
            knowledge_item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            knowledge_item['source_url'] = item.get('source_url', '')
            
            yield knowledge_item
            
        except Exception as e:
            self.logger.error(f"生成文档分析结果出错: {e}")
    
    # 辅助方法
    def determine_document_type(self, url, source_url):
        """确定文档类型"""
        if 'fagui' in url or 'fg' in url:
            return 'regulation'
        elif 'biaozhun' in url or 'bz' in url:
            return 'standard'
        elif 'anli' in url or 'al' in url:
            return 'case'
        elif 'xinwen' in url or 'xw' in url:
            return 'news'
        else:
            return 'unknown'
    
    def extract_document_id(self, url):
        """提取文档ID"""
        import hashlib
        match = re.search(r'/(\d+)\.', url)
        return match.group(1) if match else hashlib.md5(url.encode()).hexdigest()[:8]
    
    def extract_title(self, response):
        """提取标题"""
        title = response.css('h1::text, .title::text, .article-title::text').get()
        if not title:
            title = response.css('title::text').get()
        return title.strip() if title else ''
    
    def extract_content(self, response):
        """提取内容"""
        content_selectors = [
            '.content', '.article-content', '.main-content', 
            '.text-content', '.post-content', 'article'
        ]
        
        content = ''
        for selector in content_selectors:
            content_elements = response.css(selector)
            if content_elements:
                content = ' '.join(content_elements.css('::text').getall())
                break
        
        if not content:
            content = ' '.join(response.css('body::text').getall())
        
        return re.sub(r'\s+', ' ', content).strip()
    
    def determine_file_type(self, response):
        """确定文件类型"""
        content_type = response.headers.get('Content-Type', b'').decode('utf-8')
        if 'pdf' in content_type.lower():
            return 'PDF'
        elif 'html' in content_type.lower():
            return 'HTML'
        elif 'doc' in content_type.lower():
            return 'DOC'
        else:
            return 'TXT'
