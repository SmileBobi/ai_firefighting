import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlencode
from typing import List, Dict
from stock_scraper.items import FinancialReportItem, FinancialDataItem, CompanyInfoItem, TextAnalysisItem
from text_analyzer import text_analyzer


class FinancialReportSpiderSpider(scrapy.Spider):
    name = "financial_report_spider"
    allowed_domains = ["cninfo.com.cn", "www.cninfo.com.cn"]
    
    def __init__(self, stock_codes='000001,000002,600000', report_types='年报,季报', years='2023,2024', *args, **kwargs):
        super(FinancialReportSpiderSpider, self).__init__(*args, **kwargs)
        
        # 设置参数
        self.stock_codes = stock_codes.split(',')
        self.report_types = report_types.split(',')
        self.years = years.split(',')
        
        # 构建起始URL列表
        self.start_urls = self.build_start_urls()
    
    def build_start_urls(self):
        """构建起始URL列表"""
        urls = []
        
        for stock_code in self.stock_codes:
            for year in self.years:
                for report_type in self.report_types:
                    # 巨潮资讯网财报查询URL
                    url = f"http://www.cninfo.com.cn/new/information/topSearch/query?key={stock_code}&maxSecMarket=0&minSecMarket=0&isSect=0&isIndustry=0&code={stock_code}&orgId=&stock=&tabName=fulltext&pageSize=30&pageNum=1"
                    urls.append(url)
        
        return urls
    
    def start_requests(self):
        """开始请求"""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse_report_list,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'http://www.cninfo.com.cn/',
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            )
    
    def parse_report_list(self, response):
        """解析财报列表"""
        try:
            data = json.loads(response.text)
            
            if 'records' in data:
                for record in data['records']:
                    # 提取财报信息
                    report_url = record.get('announcementUrl', '')
                    if report_url:
                        # 构建完整的财报URL
                        full_url = urljoin('http://www.cninfo.com.cn/', report_url)
                        
                        yield scrapy.Request(
                            url=full_url,
                            callback=self.parse_report_detail,
                            meta={
                                'stock_code': record.get('secCode', ''),
                                'stock_name': record.get('secName', ''),
                                'report_title': record.get('announcementTitle', ''),
                                'report_date': record.get('announcementTime', ''),
                                'report_type': self.determine_report_type(record.get('announcementTitle', ''))
                            },
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                                'Referer': response.url
                            }
                        )
                        
        except json.JSONDecodeError:
            self.logger.error(f"JSON解析失败: {response.text[:200]}")
        except Exception as e:
            self.logger.error(f"解析财报列表出错: {e}")
    
    def parse_report_detail(self, response):
        """解析财报详情"""
        try:
            meta = response.meta
            
            # 创建财报item
            report_item = FinancialReportItem()
            
            # 基本信息
            report_item['report_id'] = self.extract_report_id(response.url)
            report_item['stock_code'] = meta.get('stock_code', '')
            report_item['stock_name'] = meta.get('stock_name', '')
            report_item['company_name'] = self.extract_company_name(response)
            report_item['report_type'] = meta.get('report_type', '')
            report_item['title'] = meta.get('report_title', '')
            report_item['report_date'] = meta.get('report_date', '')
            report_item['publish_date'] = self.extract_publish_date(response)
            
            # 报告内容
            report_item['summary'] = self.extract_summary(response)
            report_item['full_content'] = self.extract_full_content(response)
            report_item['content_sections'] = self.extract_sections(response)
            
            # 财务数据
            financial_data = self.extract_financial_data(response)
            for key, value in financial_data.items():
                report_item[key] = value
            
            # 文本分析
            text_analysis = text_analyzer.analyze_financial_report(report_item['full_content'])
            report_item['content_length'] = text_analysis.get('char_count', 0)
            report_item['word_count'] = text_analysis.get('word_count', 0)
            report_item['section_count'] = len(text_analysis.get('sections', []))
            report_item['table_count'] = self.count_tables(response)
            report_item['image_count'] = self.count_images(response)
            
            # 文件信息
            report_item['file_url'] = response.url
            report_item['file_type'] = self.determine_file_type(response)
            report_item['file_size'] = len(response.text)
            report_item['download_status'] = 'success'
            
            # 爬取信息
            report_item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_item['source_url'] = response.url
            
            yield report_item
            
            # 生成文本分析结果
            text_analysis_item = TextAnalysisItem()
            text_analysis_item['content_id'] = report_item['report_id']
            text_analysis_item['content_type'] = 'report'
            text_analysis_item['content_text'] = report_item['full_content']
            text_analysis_item['char_count'] = text_analysis.get('char_count', 0)
            text_analysis_item['word_count'] = text_analysis.get('word_count', 0)
            text_analysis_item['sentence_count'] = text_analysis.get('sentence_count', 0)
            text_analysis_item['paragraph_count'] = text_analysis.get('paragraph_count', 0)
            text_analysis_item['readability_score'] = text_analysis.get('readability_score', 0)
            text_analysis_item['complexity_score'] = text_analysis.get('complexity_score', 0)
            text_analysis_item['topic_keywords'] = text_analysis.get('topic_keywords', [])
            text_analysis_item['entities'] = text_analysis.get('entities', {})
            text_analysis_item['summary'] = text_analyzer.generate_summary(report_item['full_content'])
            text_analysis_item['analysis_result'] = json.dumps(text_analysis, ensure_ascii=False)
            text_analysis_item['analyzed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text_analysis_item['analysis_model'] = 'rule_based'
            
            yield text_analysis_item
            
        except Exception as e:
            self.logger.error(f"解析财报详情出错: {e}")
    
    def determine_report_type(self, title: str) -> str:
        """确定报告类型"""
        if '年报' in title or '年度报告' in title:
            return '年报'
        elif '季报' in title or '季度报告' in title:
            return '季报'
        elif '中报' in title or '半年度报告' in title:
            return '中报'
        else:
            return '其他'
    
    def extract_report_id(self, url: str) -> str:
        """从URL提取报告ID"""
        match = re.search(r'/(\d+)\.', url)
        return match.group(1) if match else ''
    
    def extract_company_name(self, response) -> str:
        """提取公司名称"""
        # 从页面中提取公司名称
        company_name = response.css('h1::text').get() or ''
        if not company_name:
            company_name = response.css('.company-name::text').get() or ''
        return company_name.strip()
    
    def extract_publish_date(self, response) -> str:
        """提取发布日期"""
        date_text = response.css('.publish-date::text').get() or ''
        if not date_text:
            date_text = response.css('.date::text').get() or ''
        return date_text.strip()
    
    def extract_summary(self, response) -> str:
        """提取报告摘要"""
        summary = response.css('.summary::text').get() or ''
        if not summary:
            summary = response.css('.abstract::text').get() or ''
        return summary.strip()
    
    def extract_full_content(self, response) -> str:
        """提取完整内容"""
        # 提取主要内容
        content_selectors = [
            '.content',
            '.main-content',
            '.report-content',
            '.announcement-content',
            'body'
        ]
        
        content = ''
        for selector in content_selectors:
            content_elements = response.css(selector)
            if content_elements:
                content = ' '.join(content_elements.css('::text').getall())
                break
        
        # 清理内容
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    
    def extract_sections(self, response) -> List[Dict[str, str]]:
        """提取内容章节"""
        sections = []
        
        # 查找章节标题
        section_elements = response.css('h1, h2, h3, h4, h5, h6')
        
        for element in section_elements:
            title = element.css('::text').get()
            if title and title.strip():
                sections.append({
                    'title': title.strip(),
                    'level': element.root.tag
                })
        
        return sections
    
    def extract_financial_data(self, response) -> Dict[str, str]:
        """提取财务数据"""
        financial_data = {}
        content = self.extract_full_content(response)
        
        # 营业收入
        revenue_pattern = r'营业收入[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        revenue_match = re.search(revenue_pattern, content)
        if revenue_match:
            financial_data['revenue'] = revenue_match.group(1)
        
        # 净利润
        profit_pattern = r'净利润[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        profit_match = re.search(profit_pattern, content)
        if profit_match:
            financial_data['net_profit'] = profit_match.group(1)
        
        # 总资产
        assets_pattern = r'总资产[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        assets_match = re.search(assets_pattern, content)
        if assets_match:
            financial_data['total_assets'] = assets_match.group(1)
        
        # 每股收益
        eps_pattern = r'每股收益[：:]\s*([0-9,]+\.?\d*)\s*元?'
        eps_match = re.search(eps_pattern, content)
        if eps_match:
            financial_data['eps'] = eps_match.group(1)
        
        return financial_data
    
    def count_tables(self, response) -> int:
        """统计表格数量"""
        return len(response.css('table'))
    
    def count_images(self, response) -> int:
        """统计图片数量"""
        return len(response.css('img'))
    
    def determine_file_type(self, response) -> str:
        """确定文件类型"""
        content_type = response.headers.get('Content-Type', b'').decode('utf-8')
        if 'pdf' in content_type.lower():
            return 'PDF'
        elif 'html' in content_type.lower():
            return 'HTML'
        else:
            return 'TXT'
