import scrapy
import json
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlencode
from stock_scraper.items import ForumPostItem, ForumCommentItem, ForumUserItem, SentimentAnalysisItem
from sentiment_analyzer import sentiment_analyzer


class EastmoneyForumSpiderSpider(scrapy.Spider):
    name = "eastmoney_forum_spider"
    allowed_domains = ["guba.eastmoney.com", "api.eastmoney.com"]
    
    def __init__(self, stock_codes='000001,000002,600000', max_pages=5, *args, **kwargs):
        super(EastmoneyForumSpiderSpider, self).__init__(*args, **kwargs)
        
        # 设置股票代码列表
        self.stock_codes = stock_codes.split(',')
        self.max_pages = int(max_pages)
        
        # 构建起始URL列表
        self.start_urls = self.build_start_urls()
    
    def build_start_urls(self):
        """构建起始URL列表"""
        urls = []
        
        for stock_code in self.stock_codes:
            # 东方财富股吧帖子列表URL
            url = f"https://guba.eastmoney.com/list,{stock_code}.html"
            urls.append(url)
        
        return urls
    
    def start_requests(self):
        """开始请求"""
        for url in self.start_urls:
            yield scrapy.Request(
                url=url,
                callback=self.parse_post_list,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Referer': 'https://guba.eastmoney.com/',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
                }
            )
    
    def parse_post_list(self, response):
        """解析帖子列表页面"""
        try:
            # 提取股票代码
            stock_code = self.extract_stock_code_from_url(response.url)
            
            # 解析帖子列表
            posts = response.css('div.articleh')
            
            for post in posts:
                # 提取帖子链接
                post_link = post.css('span.l3 a::attr(href)').get()
                if post_link:
                    post_url = urljoin(response.url, post_link)
                    
                    # 请求帖子详情页面
                    yield scrapy.Request(
                        url=post_url,
                        callback=self.parse_post_detail,
                        meta={'stock_code': stock_code},
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                            'Referer': response.url
                        }
                    )
            
            # 翻页处理
            next_page = response.css('a.pager_next::attr(href)').get()
            if next_page and self.should_follow_next_page(response):
                next_url = urljoin(response.url, next_page)
                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse_post_list,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Referer': response.url
                    }
                )
                
        except Exception as e:
            self.logger.error(f"解析帖子列表出错: {e}")
    
    def parse_post_detail(self, response):
        """解析帖子详情页面"""
        try:
            stock_code = response.meta.get('stock_code', '')
            
            # 解析帖子信息
            post_item = ForumPostItem()
            
            # 基本信息
            post_item['post_id'] = self.extract_post_id(response.url)
            post_item['title'] = response.css('div.zwcontenttit h1::text').get() or ''
            post_item['content'] = self.extract_post_content(response)
            post_item['author_name'] = response.css('div.zwcontenttit .zwfbtime a::text').get() or ''
            post_item['publish_time'] = self.extract_publish_time(response)
            
            # 互动数据
            post_item['view_count'] = self.extract_view_count(response)
            post_item['reply_count'] = self.extract_reply_count(response)
            post_item['like_count'] = self.extract_like_count(response)
            
            # 相关股票
            post_item['related_stocks'] = [stock_code] if stock_code else []
            post_item['stock_mentions'] = len(sentiment_analyzer.extract_stock_mentions(post_item['content']))
            
            # 情感分析
            sentiment_result = sentiment_analyzer.analyze_sentiment(post_item['content'])
            post_item['sentiment_score'] = sentiment_result['sentiment_score']
            post_item['sentiment_label'] = sentiment_result['sentiment_label']
            post_item['emotion_keywords'] = sentiment_result['emotion_keywords']
            
            # 内容分析
            content_features = sentiment_analyzer.analyze_content_features(post_item['content'])
            post_item['content_length'] = content_features['content_length']
            post_item['has_images'] = content_features['has_images']
            post_item['has_links'] = content_features['has_links']
            post_item['topic_tags'] = content_features['topic_tags']
            
            # 爬取信息
            post_item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            post_item['source_url'] = response.url
            
            yield post_item
            
            # 生成情感分析结果
            sentiment_item = SentimentAnalysisItem()
            sentiment_item['content_id'] = post_item['post_id']
            sentiment_item['content_type'] = 'post'
            sentiment_item['content_text'] = post_item['content']
            sentiment_item['sentiment_score'] = sentiment_result['sentiment_score']
            sentiment_item['sentiment_label'] = sentiment_result['sentiment_label']
            sentiment_item['confidence'] = sentiment_result['confidence']
            sentiment_item['positive_score'] = sentiment_result['positive_score']
            sentiment_item['negative_score'] = sentiment_result['negative_score']
            sentiment_item['neutral_score'] = sentiment_result['neutral_score']
            sentiment_item['emotion_keywords'] = sentiment_result['emotion_keywords']
            sentiment_item['stock_keywords'] = sentiment_result['stock_keywords']
            sentiment_item['market_keywords'] = sentiment_result['market_keywords']
            sentiment_item['mentioned_stocks'] = sentiment_analyzer.extract_stock_mentions(post_item['content'])
            sentiment_item['analyzed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sentiment_item['analysis_model'] = 'rule_based'
            
            yield sentiment_item
            
            # 解析评论
            comments = response.css('div.replylist .replyitem')
            for comment in comments:
                comment_item = self.parse_comment(comment, post_item['post_id'], response.url)
                if comment_item:
                    yield comment_item
            
        except Exception as e:
            self.logger.error(f"解析帖子详情出错: {e}")
    
    def parse_comment(self, comment_element, post_id, source_url):
        """解析评论信息"""
        try:
            comment_item = ForumCommentItem()
            
            # 基本信息
            comment_item['comment_id'] = comment_element.css('::attr(data-id)').get() or ''
            comment_item['post_id'] = post_id
            comment_item['content'] = comment_element.css('div.replycontent::text').get() or ''
            comment_item['author_name'] = comment_element.css('div.replyuser a::text').get() or ''
            comment_item['publish_time'] = comment_element.css('div.replytime::text').get() or ''
            
            # 互动数据
            comment_item['like_count'] = self.extract_comment_like_count(comment_element)
            
            # 情感分析
            sentiment_result = sentiment_analyzer.analyze_sentiment(comment_item['content'])
            comment_item['sentiment_score'] = sentiment_result['sentiment_score']
            comment_item['sentiment_label'] = sentiment_result['sentiment_label']
            comment_item['emotion_keywords'] = sentiment_result['emotion_keywords']
            
            # 内容分析
            content_features = sentiment_analyzer.analyze_content_features(comment_item['content'])
            comment_item['content_length'] = content_features['content_length']
            comment_item['has_images'] = content_features['has_images']
            
            # 爬取信息
            comment_item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            comment_item['source_url'] = source_url
            
            return comment_item
            
        except Exception as e:
            self.logger.error(f"解析评论出错: {e}")
            return None
    
    def extract_stock_code_from_url(self, url):
        """从URL中提取股票代码"""
        match = re.search(r'list,(\d+)\.html', url)
        if match:
            return match.group(1)
        return ''
    
    def extract_post_id(self, url):
        """从URL中提取帖子ID"""
        match = re.search(r'(\d+)\.html', url)
        if match:
            return match.group(1)
        return ''
    
    def extract_post_content(self, response):
        """提取帖子内容"""
        content = response.css('div.zwcontentmain::text').getall()
        return ''.join(content).strip()
    
    def extract_publish_time(self, response):
        """提取发布时间"""
        time_text = response.css('div.zwfbtime::text').get() or ''
        # 简单的时间提取，实际项目中需要更复杂的解析
        return time_text.strip()
    
    def extract_view_count(self, response):
        """提取浏览次数"""
        view_text = response.css('div.zwfbtime span::text').get() or ''
        match = re.search(r'(\d+)', view_text)
        return match.group(1) if match else '0'
    
    def extract_reply_count(self, response):
        """提取回复次数"""
        reply_text = response.css('div.zwfbtime span::text').get() or ''
        match = re.search(r'回复(\d+)', reply_text)
        return match.group(1) if match else '0'
    
    def extract_like_count(self, response):
        """提取点赞次数"""
        like_text = response.css('div.zwfbtime span::text').get() or ''
        match = re.search(r'赞(\d+)', like_text)
        return match.group(1) if match else '0'
    
    def extract_comment_like_count(self, comment_element):
        """提取评论点赞次数"""
        like_text = comment_element.css('div.replylike::text').get() or ''
        match = re.search(r'(\d+)', like_text)
        return match.group(1) if match else '0'
    
    def should_follow_next_page(self, response):
        """判断是否应该翻页"""
        # 这里可以添加翻页逻辑，比如限制页数
        return True
