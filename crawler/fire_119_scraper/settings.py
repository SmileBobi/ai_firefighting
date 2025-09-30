"""
Scrapy项目设置文件
"""

BOT_NAME = 'fire_119_scraper'

SPIDER_MODULES = ['fire_119_scraper.spiders']
NEWSPIDER_MODULE = 'fire_119_scraper.spiders'

# 遵守robots.txt
ROBOTSTXT_OBEY = True

# 用户代理
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# 并发设置
CONCURRENT_REQUESTS = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 1

# 下载延迟
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = 0.5

# 自动限速
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = True

# 请求头
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

# 管道设置
ITEM_PIPELINES = {
    'fire_119_scraper.pipelines.Fire119Pipeline': 300,
}

# 输出设置
FEEDS = {
    'data/fire_119_scrapy.json': {
        'format': 'json',
        'encoding': 'utf8',
        'store_empty': False,
        'indent': 2,
    },
}

# 日志设置
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/scrapy.log'

# 中间件设置
DOWNLOADER_MIDDLEWARES = {
    'fire_119_scraper.middlewares.Fire119DownloaderMiddleware': 543,
}

# 扩展设置
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,
}

# 请求设置
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'



