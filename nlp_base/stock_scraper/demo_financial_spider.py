#!/usr/bin/env python
"""
演示用的财报数据爬虫
使用模拟数据来演示财报信息抓取和长文本分析功能
"""

import scrapy
import json
from datetime import datetime, timedelta
from stock_scraper.items import FinancialReportItem, FinancialDataItem, CompanyInfoItem, TextAnalysisItem
from text_analyzer import text_analyzer


class DemoFinancialSpider(scrapy.Spider):
    name = "demo_financial"
    allowed_domains = ["example.com"]
    start_urls = ["https://httpbin.org/json"]  # 使用一个返回JSON的测试API
    
    def parse(self, response):
        """解析响应并生成模拟的财报数据"""
        
        # 生成模拟的财报数据
        mock_reports = [
            {
                'report_id': 'RPT001',
                'stock_code': '000001',
                'stock_name': '平安银行',
                'company_name': '平安银行股份有限公司',
                'report_type': '年报',
                'report_period': '2023年度',
                'report_date': '2024-03-15',
                'publish_date': '2024-03-20',
                'title': '平安银行2023年年度报告',
                'summary': '2023年，平安银行实现营业收入1,234.56亿元，同比增长8.5%；实现净利润456.78亿元，同比增长12.3%。',
                'full_content': '''
                平安银行股份有限公司2023年年度报告
                
                一、公司基本情况
                平安银行股份有限公司（以下简称"本公司"或"平安银行"）成立于1987年，是中国平安保险（集团）股份有限公司控股的全国性股份制商业银行。
                
                二、2023年度经营情况
                2023年，面对复杂多变的国内外经济金融环境，平安银行坚持稳中求进工作总基调，统筹发展和安全，持续深化转型发展，各项业务保持稳健增长。
                
                营业收入：2023年实现营业收入1,234.56亿元，同比增长8.5%，主要得益于利息净收入的稳定增长和非利息收入的快速发展。
                净利润：实现净利润456.78亿元，同比增长12.3%，盈利能力持续提升。
                总资产：截至2023年末，总资产达到45,678.90亿元，较年初增长6.8%。
                总负债：总负债为42,345.67亿元，较年初增长6.5%。
                股东权益：股东权益为3,333.23亿元，较年初增长9.2%。
                
                三、主要财务指标
                每股收益：2.35元，同比增长11.9%
                净资产收益率：13.7%，较上年提升0.8个百分点
                总资产收益率：1.0%，与上年基本持平
                不良贷款率：1.05%，较年初下降0.02个百分点
                拨备覆盖率：245.6%，风险抵御能力进一步增强
                
                四、业务发展情况
                1. 公司银行业务
                公司银行业务实现营业收入567.89亿元，同比增长7.2%。持续优化客户结构，加大对优质企业的服务力度。
                
                2. 零售银行业务
                零售银行业务实现营业收入456.78亿元，同比增长9.8%。个人贷款余额稳步增长，信用卡业务发展良好。
                
                3. 资金同业业务
                资金同业业务实现营业收入210.89亿元，同比增长5.6%。积极把握市场机会，优化资产配置。
                
                五、风险管理
                2023年，平安银行持续完善风险管理体系，加强风险识别和预警，各项风险指标保持良好水平。
                
                六、未来发展展望
                展望2024年，平安银行将继续坚持稳健经营，深化数字化转型，提升服务实体经济能力，为股东创造更大价值。
                ''',
                'revenue': '12345600',
                'net_profit': '4567800',
                'total_assets': '456789000',
                'total_liabilities': '423456700',
                'shareholders_equity': '33332300',
                'operating_cash_flow': '2345678',
                'eps': '2.35',
                'roe': '13.7',
                'roa': '1.0'
            },
            {
                'report_id': 'RPT002',
                'stock_code': '000002',
                'stock_name': '万科A',
                'company_name': '万科企业股份有限公司',
                'report_type': '年报',
                'report_period': '2023年度',
                'report_date': '2024-03-22',
                'publish_date': '2024-03-25',
                'title': '万科企业2023年年度报告',
                'summary': '2023年，万科实现营业收入4,567.89亿元，同比下降2.1%；实现净利润123.45亿元，同比下降15.6%。',
                'full_content': '''
                万科企业股份有限公司2023年年度报告
                
                一、公司基本情况
                万科企业股份有限公司（以下简称"本公司"或"万科"）成立于1984年，是中国领先的城乡建设与生活服务商。
                
                二、2023年度经营情况
                2023年，面对房地产行业深度调整，万科坚持"活下去"的经营策略，积极应对市场变化，努力保持经营稳定。
                
                营业收入：2023年实现营业收入4,567.89亿元，同比下降2.1%，主要受房地产销售下滑影响。
                净利润：实现净利润123.45亿元，同比下降15.6%，盈利能力有所下降。
                总资产：截至2023年末，总资产达到12,345.67亿元，较年初下降3.2%。
                总负债：总负债为9,876.54亿元，较年初下降2.8%。
                股东权益：股东权益为2,469.13亿元，较年初下降4.1%。
                
                三、主要财务指标
                每股收益：1.12元，同比下降15.8%
                净资产收益率：5.0%，较上年下降1.2个百分点
                总资产收益率：1.0%，与上年基本持平
                资产负债率：80.0%，较年初下降0.4个百分点
                净负债率：45.6%，较年初上升2.1个百分点
                
                四、业务发展情况
                1. 房地产开发业务
                房地产开发业务实现营业收入3,456.78亿元，同比下降3.5%。全年实现销售面积2,345.67万平方米，销售金额3,456.78亿元。
                
                2. 物业服务业务
                物业服务业务实现营业收入567.89亿元，同比增长8.9%。在管面积持续扩大，服务质量不断提升。
                
                3. 其他业务
                其他业务实现营业收入543.22亿元，同比增长5.6%。包括商业地产、物流地产等业务。
                
                五、风险管理
                2023年，万科持续加强风险管理，优化债务结构，确保现金流安全。积极应对行业调整，保持财务稳健。
                
                六、未来发展展望
                展望2024年，万科将继续坚持稳健经营，积极适应行业新常态，努力实现高质量发展。
                ''',
                'revenue': '45678900',
                'net_profit': '1234500',
                'total_assets': '123456700',
                'total_liabilities': '98765400',
                'shareholders_equity': '24691300',
                'operating_cash_flow': '3456789',
                'eps': '1.12',
                'roe': '5.0',
                'roa': '1.0'
            },
            {
                'report_id': 'RPT003',
                'stock_code': '600000',
                'stock_name': '浦发银行',
                'company_name': '上海浦东发展银行股份有限公司',
                'report_type': '季报',
                'report_period': '2024年第一季度',
                'report_date': '2024-04-25',
                'publish_date': '2024-04-30',
                'title': '浦发银行2024年第一季度报告',
                'summary': '2024年第一季度，浦发银行实现营业收入345.67亿元，同比增长5.2%；实现净利润89.12亿元，同比增长7.8%。',
                'full_content': '''
                上海浦东发展银行股份有限公司2024年第一季度报告
                
                一、公司基本情况
                上海浦东发展银行股份有限公司（以下简称"本公司"或"浦发银行"）成立于1992年，是一家全国性股份制商业银行。
                
                二、2024年第一季度经营情况
                2024年第一季度，浦发银行坚持稳中求进，各项业务保持稳健发展，经营业绩稳步提升。
                
                营业收入：2024年第一季度实现营业收入345.67亿元，同比增长5.2%。
                净利润：实现净利润89.12亿元，同比增长7.8%。
                总资产：截至2024年3月末，总资产达到89,012.34亿元，较年初增长2.1%。
                总负债：总负债为82,345.67亿元，较年初增长2.0%。
                股东权益：股东权益为6,666.67亿元，较年初增长2.8%。
                
                三、主要财务指标
                每股收益：0.32元，同比增长8.1%
                净资产收益率：5.3%，较上年同期提升0.3个百分点
                总资产收益率：0.4%，与上年同期基本持平
                不良贷款率：1.12%，较年初下降0.01个百分点
                拨备覆盖率：238.5%，风险抵御能力保持良好
                
                四、业务发展情况
                1. 公司银行业务
                公司银行业务实现营业收入156.78亿元，同比增长4.8%。持续优化客户结构，提升服务效率。
                
                2. 零售银行业务
                零售银行业务实现营业收入123.45亿元，同比增长6.2%。个人贷款业务稳步发展。
                
                3. 资金同业业务
                资金同业业务实现营业收入65.44亿元，同比增长3.9%。积极把握市场机会。
                
                五、风险管理
                2024年第一季度，浦发银行持续完善风险管理体系，各项风险指标保持良好水平。
                
                六、未来发展展望
                展望2024年，浦发银行将继续坚持稳健经营，深化转型发展，为股东创造更大价值。
                ''',
                'revenue': '3456700',
                'net_profit': '891200',
                'total_assets': '890123400',
                'total_liabilities': '823456700',
                'shareholders_equity': '66666700',
                'operating_cash_flow': '1234567',
                'eps': '0.32',
                'roe': '5.3',
                'roa': '0.4'
            }
        ]
        
        # 生成财报数据
        for report_data in mock_reports:
            item = FinancialReportItem()
            for key, value in report_data.items():
                item[key] = value
            
            # 文本分析
            text_analysis = text_analyzer.analyze_financial_report(report_data['full_content'])
            item['content_length'] = text_analysis.get('char_count', 0)
            item['word_count'] = text_analysis.get('word_count', 0)
            item['section_count'] = len(text_analysis.get('sections', []))
            item['table_count'] = 5  # 模拟表格数量
            item['image_count'] = 3  # 模拟图片数量
            
            # 文件信息
            item['file_url'] = f"http://example.com/reports/{report_data['report_id']}.pdf"
            item['file_type'] = 'PDF'
            item['file_size'] = len(report_data['full_content']) * 2  # 模拟文件大小
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成文本分析结果
            text_analysis_item = TextAnalysisItem()
            text_analysis_item['content_id'] = report_data['report_id']
            text_analysis_item['content_type'] = 'report'
            text_analysis_item['content_text'] = report_data['full_content']
            text_analysis_item['char_count'] = text_analysis.get('char_count', 0)
            text_analysis_item['word_count'] = text_analysis.get('word_count', 0)
            text_analysis_item['sentence_count'] = text_analysis.get('sentence_count', 0)
            text_analysis_item['paragraph_count'] = text_analysis.get('paragraph_count', 0)
            text_analysis_item['readability_score'] = text_analysis.get('readability_score', 0)
            text_analysis_item['complexity_score'] = text_analysis.get('complexity_score', 0)
            text_analysis_item['topic_keywords'] = text_analysis.get('topic_keywords', [])
            text_analysis_item['entities'] = text_analysis.get('entities', {})
            text_analysis_item['summary'] = text_analyzer.generate_summary(report_data['full_content'])
            text_analysis_item['analysis_result'] = json.dumps(text_analysis, ensure_ascii=False)
            text_analysis_item['analyzed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            text_analysis_item['analysis_model'] = 'rule_based'
            
            yield text_analysis_item
        
        # 生成公司信息数据
        mock_companies = [
            {
                'stock_code': '000001',
                'stock_name': '平安银行',
                'company_name': '平安银行股份有限公司',
                'company_abbr': '平安银行',
                'english_name': 'Ping An Bank Co., Ltd.',
                'industry': '银行业',
                'market': '主板',
                'listing_date': '1991-04-03',
                'legal_representative': '谢永林',
                'registered_capital': '19405918.67万元',
                'business_scope': '吸收公众存款；发放短期、中期和长期贷款；办理国内外结算；办理票据承兑与贴现；发行金融债券；代理发行、代理兑付、承销政府债券；买卖政府债券、金融债券；从事同业拆借；买卖、代理买卖外汇；从事银行卡业务；提供信用证服务及担保；代理收付款项及代理保险业务；提供保管箱服务；经国务院银行业监督管理机构批准的其他业务。',
                'address': '广东省深圳市罗湖区深南东路5047号',
                'phone': '0755-22168388',
                'website': 'http://bank.pingan.com',
                'email': 'ir@pingan.com.cn',
                'total_shares': '194059186700',
                'circulating_shares': '194059186700',
                'market_cap': '248000000000',
                'circulating_cap': '248000000000'
            },
            {
                'stock_code': '000002',
                'stock_name': '万科A',
                'company_name': '万科企业股份有限公司',
                'company_abbr': '万科A',
                'english_name': 'China Vanke Co., Ltd.',
                'industry': '房地产业',
                'market': '主板',
                'listing_date': '1991-01-29',
                'legal_representative': '郁亮',
                'registered_capital': '11039152000元',
                'business_scope': '房地产开发；物业管理；投资兴办实业（具体项目另行申报）；国内商业、物资供销业（不含专营、专控、专卖商品）；进出口业务（按深经发审证字第113号外贸企业审定证书规定办理）；广告业务。',
                'address': '广东省深圳市盐田区大梅沙环梅路33号',
                'phone': '0755-25606666',
                'website': 'http://www.vanke.com',
                'email': 'ir@vanke.com',
                'total_shares': '11039152000',
                'circulating_shares': '11039152000',
                'market_cap': '92000000000',
                'circulating_cap': '92000000000'
            }
        ]
        
        for company_data in mock_companies:
            item = CompanyInfoItem()
            for key, value in company_data.items():
                item[key] = value
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
        
        self.logger.info(f"生成了 {len(mock_reports)} 条财报数据")
        self.logger.info(f"生成了 {len(mock_companies)} 条公司信息数据")
