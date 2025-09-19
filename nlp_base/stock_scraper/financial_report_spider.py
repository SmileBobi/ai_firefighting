import scrapy


class FinancialReportSpiderSpider(scrapy.Spider):
    name = "financial_report_spider"
    allowed_domains = ["cninfo.com.cn"]
    start_urls = ["https://cninfo.com.cn"]

    def parse(self, response):
        pass
