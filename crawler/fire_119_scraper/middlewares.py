"""
消防119中间件
"""

import random
import time
import logging
from scrapy.http import HtmlResponse
from scrapy.downloadermiddlewares.useragent import UserAgentMiddleware
from scrapy.downloadermiddlewares.retry import RetryMiddleware

logger = logging.getLogger(__name__)

class Fire119UserAgentMiddleware(UserAgentMiddleware):
    """消防119用户代理中间件"""
    
    def __init__(self, user_agent=''):
        self.user_agent = user_agent
        self.user_agent_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        ]
    
    def process_request(self, request, spider):
        """处理请求"""
        ua = random.choice(self.user_agent_list)
        request.headers['User-Agent'] = ua
        return None

class Fire119DownloaderMiddleware:
    """消防119下载器中间件"""
    
    def __init__(self):
        self.request_count = 0
    
    def process_request(self, request, spider):
        """处理请求"""
        # 随机延迟
        delay = random.uniform(1, 3)
        time.sleep(delay)
        
        # 记录请求次数
        self.request_count += 1
        logger.info(f"处理第 {self.request_count} 个请求: {request.url}")
        
        return None
    
    def process_response(self, request, response, spider):
        """处理响应"""
        # 检查响应状态
        if response.status != 200:
            logger.warning(f"响应状态异常: {response.status} - {request.url}")
        
        return response
    
    def process_exception(self, request, exception, spider):
        """处理异常"""
        logger.error(f"请求异常: {exception} - {request.url}")
        return None

class Fire119RetryMiddleware(RetryMiddleware):
    """消防119重试中间件"""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.retry_times = settings.getint('RETRY_TIMES', 3)
        self.retry_http_codes = settings.getlist('RETRY_HTTP_CODES', [500, 502, 503, 504, 522, 524, 408, 429])
    
    def retry(self, request, reason, spider):
        """重试逻辑"""
        retries = request.meta.get('retry_times', 0) + 1
        
        if retries <= self.retry_times:
            logger.info(f"重试第 {retries} 次: {request.url}")
            
            # 增加延迟
            delay = random.uniform(2, 5) * retries
            time.sleep(delay)
            
            retryreq = request.copy()
            retryreq.meta['retry_times'] = retries
            retryreq.dont_filter = True
            
            return retryreq
        else:
            logger.error(f"重试次数超限: {request.url}")
            return None



