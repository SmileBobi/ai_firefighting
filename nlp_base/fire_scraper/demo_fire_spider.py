#!/usr/bin/env python
"""
演示用的消防数据爬虫
使用模拟数据来演示消防法规、标准、案例等信息抓取和RAG知识库构建功能
"""

import scrapy
import json
from datetime import datetime
from fire_scraper.items import FireRegulationItem, FireStandardItem, FireCaseItem, FireNewsItem, FireKnowledgeItem, FireDocumentItem
from fire_text_analyzer import fire_text_analyzer


class DemoFireSpider(scrapy.Spider):
    name = "demo_fire"
    allowed_domains = ["example.com"]
    start_urls = ["https://httpbin.org/json"]  # 使用一个返回JSON的测试API
    
    def parse(self, response):
        """解析响应并生成模拟的消防数据"""
        
        # 生成模拟的消防法规数据
        mock_regulations = [
            {
                'regulation_id': 'REG001',
                'title': '建设工程消防监督管理规定',
                'regulation_type': '部门规章',
                'level': '国家级',
                'issuing_authority': '公安部',
                'issue_date': '2012年7月17日',
                'effective_date': '2012年11月1日',
                'status': '现行有效',
                'content': '''
                建设工程消防监督管理规定
                
                第一章 总则
                
                第一条 为了加强建设工程消防监督管理，落实建设工程消防设计、施工质量和安全责任，规范消防监督管理行为，依据《中华人民共和国消防法》、《建设工程质量管理条例》，制定本规定。
                
                第二条 本规定适用于新建、扩建、改建（含室内装修、用途变更）等建设工程的消防监督管理。
                
                第三条 建设、设计、施工、工程监理等单位应当遵守消防法规、国家消防技术标准，对建设工程消防设计、施工质量和安全负责。
                
                第四条 公安机关消防机构依法实施建设工程消防设计审核、消防验收和备案、抽查，对建设工程进行消防监督。
                
                第二章 消防设计审核和消防验收
                
                第五条 对具有下列情形之一的人员密集场所，建设单位应当向公安机关消防机构申请消防设计审核：
                （一）建筑总面积大于二万平方米的体育场馆、会堂，公共展览馆、博物馆的展示厅；
                （二）建筑总面积大于一万五千平方米的民用机场航站楼、客运车站候车室、客运码头候船厅；
                （三）建筑总面积大于一万平方米的宾馆、饭店、商场、市场；
                （四）建筑总面积大于二千五百平方米的影剧院，公共图书馆的阅览室，营业性室内健身、休闲场馆，医院的门诊楼，大学的教学楼、图书馆、食堂，劳动密集型企业的生产加工车间，寺庙、教堂；
                （五）建筑总面积大于一千平方米的托儿所、幼儿园的儿童用房，儿童游乐厅等室内儿童活动场所，养老院、福利院，医院、疗养院的病房楼，中小学校的教学楼、图书馆、食堂，学校的集体宿舍，劳动密集型企业的员工集体宿舍；
                （六）建筑总面积大于五百平方米的歌舞厅、录像厅、放映厅、卡拉OK厅、夜总会、游艺厅、桑拿浴室、网吧、酒吧，具有娱乐功能的餐馆、茶馆、咖啡厅。
                
                第六条 对具有下列情形之一的特殊建设工程，建设单位应当向公安机关消防机构申请消防设计审核：
                （一）设有本规定第十三条所列的人员密集场所的建设工程；
                （二）国家机关办公楼、电力调度楼、电信楼、邮政楼、防灾指挥调度楼、广播电视楼、档案楼；
                （三）本条第一项、第二项规定以外的单体建筑面积大于四万平方米或者建筑高度超过五十米的公共建筑；
                （四）国家标准规定的一类高层住宅建筑；
                （五）城市轨道交通、隧道工程，大型发电、变配电工程；
                （六）生产、储存、装卸易燃易爆危险物品的工厂、仓库和专用车站、码头，易燃易爆气体和液体的充装站、供应站、调压站。
                ''',
                'category': '消防安全管理',
                'keywords': ['消防监督管理', '建设工程', '消防设计', '消防验收', '人员密集场所'],
                'tags': ['消防法规', '建设工程', '监督管理']
            },
            {
                'regulation_id': 'REG002',
                'title': '火灾事故调查规定',
                'regulation_type': '部门规章',
                'level': '国家级',
                'issuing_authority': '公安部',
                'issue_date': '2012年7月17日',
                'effective_date': '2012年11月1日',
                'status': '现行有效',
                'content': '''
                火灾事故调查规定
                
                第一章 总则
                
                第一条 为了规范火灾事故调查，保障火灾事故调查的客观、公正、科学，依据《中华人民共和国消防法》，制定本规定。
                
                第二条 公安机关消防机构调查火灾事故，适用本规定。
                
                第三条 火灾事故调查应当坚持及时、客观、公正、合法的原则。
                
                第四条 火灾事故调查由县级以上人民政府公安机关消防机构负责实施。
                
                第二章 火灾事故调查的管辖
                
                第五条 火灾事故调查由火灾发生地公安机关消防机构按照下列分工进行：
                （一）一次火灾死亡十人以上的，重伤二十人以上或者死亡、重伤二十人以上的，受灾五十户以上的，由省、自治区人民政府公安机关消防机构负责组织调查；
                （二）一次火灾死亡一人以上不满十人的，重伤十人以上不满二十人的，受灾三十户以上不满五十户的，由设区的市或者相当于同级的人民政府公安机关消防机构负责组织调查；
                （三）一次火灾重伤十人以下或者受灾三十户以下的，由县级人民政府公安机关消防机构负责调查。
                
                第六条 上级公安机关消防机构认为必要时，可以调查下级公安机关消防机构管辖的火灾事故。
                
                第七条 军事设施发生火灾，由军队保卫部门负责调查；其中涉及地方人员的，由地方公安机关消防机构协助调查。
                
                第三章 火灾事故调查的程序
                
                第八条 火灾事故调查人员接到调查任务后，应当立即赶赴火灾现场，开展火灾事故调查工作。
                
                第九条 火灾事故调查人员应当对火灾现场进行封闭，保护火灾现场，不得擅自移动、损毁火灾现场物品。
                
                第十条 火灾事故调查人员应当询问火灾当事人、目击证人，收集火灾事故证据。
                ''',
                'category': '火灾调查',
                'keywords': ['火灾事故调查', '火灾现场', '火灾原因', '火灾损失', '火灾责任'],
                'tags': ['消防法规', '火灾调查', '事故处理']
            }
        ]
        
        # 生成消防法规数据
        for reg_data in mock_regulations:
            item = FireRegulationItem()
            for key, value in reg_data.items():
                item[key] = value
            
            # 文本分析
            text_analysis = fire_text_analyzer.analyze_regulation(reg_data['content'])
            item['summary'] = fire_text_analyzer.generate_document_summary(reg_data['content'])
            item['chapters'] = text_analysis.get('regulation_analysis', {}).get('chapters', [])
            item['articles'] = text_analysis.get('regulation_analysis', {}).get('articles', [])
            
            # 文件信息
            item['file_url'] = f"http://example.com/regulations/{reg_data['regulation_id']}.pdf"
            item['file_type'] = 'PDF'
            item['file_size'] = len(reg_data['content']) * 2
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成知识条目
            yield from self.generate_knowledge_item(item, reg_data['content'], 'regulation')
        
        # 生成模拟的消防标准数据
        mock_standards = [
            {
                'standard_id': 'STD001',
                'standard_number': 'GB 50016-2014',
                'title': '建筑设计防火规范',
                'english_title': 'Code for fire protection design of buildings',
                'standard_type': '国家标准',
                'category': '强制性',
                'issuing_authority': '住房和城乡建设部',
                'approval_authority': '国家标准化管理委员会',
                'issue_date': '2014年8月27日',
                'implementation_date': '2015年5月1日',
                'status': '现行',
                'content': '''
                建筑设计防火规范 GB 50016-2014
                
                1 总则
                
                1.0.1 为了防止和减少建筑火灾危害，保护人身和财产安全，制定本规范。
                
                1.0.2 本规范适用于下列新建、扩建和改建的建筑：
                1 厂房；
                2 仓库；
                3 民用建筑；
                4 甲、乙、丙类液体储罐（区）；
                5 可燃、助燃气体储罐（区）；
                6 可燃材料堆场；
                7 城市交通隧道。
                
                1.0.3 本规范不适用于火药、炸药及其制品厂房（仓库）、花炮厂房（仓库）的建筑防火设计。
                
                1.0.4 建筑防火设计应遵循国家的有关方针政策，从全局出发，统筹兼顾，做到安全适用、技术先进、经济合理。
                
                2 术语
                
                2.1.1 高层建筑 high-rise building
                建筑高度大于27m的住宅建筑和建筑高度大于24m的非单层厂房、仓库和其他民用建筑。
                
                2.1.2 裙房 podium
                在高层建筑主体投影范围外，与建筑主体相连且建筑高度不大于24m的附属建筑。
                
                2.1.3 重要公共建筑 important public building
                发生火灾可能造成重大人员伤亡、财产损失和严重社会影响的公共建筑。
                
                3 厂房和仓库
                
                3.1.1 生产的火灾危险性应根据生产中使用或产生的物质性质及其数量等因素划分，可分为甲、乙、丙、丁、戊类，并应符合表3.1.1的规定。
                
                3.1.2 同一座厂房或厂房的任一防火分区内有不同火灾危险性生产时，厂房或防火分区内的生产火灾危险性类别应按火灾危险性较大的部分确定；当生产过程中使用或产生易燃、可燃物的量较少，不足以构成爆炸或火灾危险时，可按实际情况确定；当符合下述条件之一时，可按火灾危险性较小的部分确定：
                1 火灾危险性较大的生产部分占本层或本防火分区建筑面积的比例小于5%或丁、戊类厂房内的油漆工段小于10%，且发生火灾事故时不足以蔓延到其他部位或火灾危险性较大的生产部分采取了有效的防火措施；
                2 丁、戊类厂房内的油漆工段，当采用封闭喷漆工艺，封闭喷漆空间内保持负压、油漆工段设置可燃气体探测报警系统或自动抑爆系统，且油漆工段占其所在防火分区建筑面积的比例不大于20%。
                ''',
                'fire_category': '建筑防火',
                'keywords': ['建筑设计', '防火规范', '建筑高度', '防火分区', '火灾危险性'],
                'tags': ['消防标准', '建筑设计', '防火规范']
            }
        ]
        
        # 生成消防标准数据
        for std_data in mock_standards:
            item = FireStandardItem()
            for key, value in std_data.items():
                item[key] = value
            
            # 文本分析
            text_analysis = fire_text_analyzer.analyze_standard(std_data['content'])
            item['scope'] = text_analysis.get('regulation_analysis', {}).get('scope', '')
            item['technical_requirements'] = text_analysis.get('regulation_analysis', {}).get('technical_requirements', [])
            item['test_methods'] = text_analysis.get('regulation_analysis', {}).get('test_methods', [])
            item['inspection_rules'] = text_analysis.get('regulation_analysis', {}).get('inspection_rules', [])
            
            # 文件信息
            item['file_url'] = f"http://example.com/standards/{std_data['standard_id']}.pdf"
            item['file_type'] = 'PDF'
            item['file_size'] = len(std_data['content']) * 2
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成知识条目
            yield from self.generate_knowledge_item(item, std_data['content'], 'standard')
        
        # 生成模拟的火灾案例数据
        mock_cases = [
            {
                'case_id': 'CASE001',
                'title': '某商场火灾事故案例分析',
                'case_type': '火灾事故',
                'severity_level': '重大',
                'incident_date': '2023年3月15日',
                'location': '某市商业区',
                'province': '某省',
                'city': '某市',
                'district': '某区',
                'building_type': '公共建筑',
                'fire_cause': '电气火灾',
                'casualties': {'deaths': '3', 'injuries': '15'},
                'economic_loss': '500万元',
                'fire_duration': '2小时',
                'content': '''
                某商场火灾事故案例分析
                
                一、事故概况
                
                2023年3月15日凌晨2时30分，某市商业区一大型商场发生火灾，造成3人死亡，15人受伤，直接经济损失约500万元。
                
                二、事故经过
                
                3月15日凌晨2时30分，商场保安在巡查时发现二楼服装区有浓烟冒出，立即报警并组织人员疏散。消防队接警后迅速赶到现场，经过2小时的扑救，成功将火灾扑灭。
                
                三、火灾原因分析
                
                经调查，火灾原因系商场二楼服装区一配电箱内电线老化短路，引燃周围可燃物所致。该配电箱已使用15年，从未进行过检修更换。
                
                四、事故教训
                
                1. 电气设备老化是引发火灾的主要原因，应定期检查更换；
                2. 商场应加强夜间巡查，及时发现火灾隐患；
                3. 应完善消防设施，确保火灾自动报警系统正常运行；
                4. 应加强员工消防安全培训，提高应急处置能力。
                
                五、预防措施
                
                1. 建立电气设备定期检查制度，及时更换老化设备；
                2. 完善消防设施，确保消防系统正常运行；
                3. 加强消防安全管理，建立消防安全责任制；
                4. 定期开展消防演练，提高员工消防安全意识。
                ''',
                'keywords': ['商场火灾', '电气火灾', '火灾原因', '事故教训', '预防措施'],
                'tags': ['火灾案例', '商场', '电气火灾']
            }
        ]
        
        # 生成火灾案例数据
        for case_data in mock_cases:
            item = FireCaseItem()
            for key, value in case_data.items():
                item[key] = value
            
            # 文本分析
            text_analysis = fire_text_analyzer.analyze_case(case_data['content'])
            item['description'] = case_data['content'][:200] + '...'
            item['investigation_result'] = '电气设备老化短路引发火灾'
            item['lessons_learned'] = '应定期检查更换电气设备，加强消防安全管理'
            item['prevention_measures'] = '建立电气设备定期检查制度，完善消防设施'
            
            # 文件信息
            item['file_url'] = f"http://example.com/cases/{case_data['case_id']}.pdf"
            item['file_type'] = 'PDF'
            item['file_size'] = len(case_data['content']) * 2
            item['download_status'] = 'success'
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成知识条目
            yield from self.generate_knowledge_item(item, case_data['content'], 'case')
        
        # 生成模拟的消防新闻数据
        mock_news = [
            {
                'news_id': 'NEWS001',
                'title': '全国消防部门开展消防安全专项整治行动',
                'content': '''
                全国消防部门开展消防安全专项整治行动
                
                为深入贯彻落实习近平总书记关于安全生产重要指示精神，切实加强消防安全工作，有效防范化解重大消防安全风险，应急管理部消防救援局决定在全国范围内开展消防安全专项整治行动。
                
                此次专项整治行动将重点检查以下场所：
                1. 人员密集场所，包括商场、市场、宾馆、饭店、学校、医院等；
                2. 易燃易爆场所，包括加油站、液化气站、化工企业等；
                3. 高层建筑、地下建筑、大型商业综合体等；
                4. 老旧小区、城中村等火灾隐患突出区域。
                
                专项整治行动将重点整治以下问题：
                1. 消防设施损坏、缺失或不能正常使用；
                2. 疏散通道、安全出口被占用、堵塞；
                3. 违规用火、用电、用气；
                4. 消防安全责任制不落实；
                5. 消防控制室值班人员无证上岗。
                
                各级消防部门将严格按照法律法规要求，对发现的火灾隐患和违法行为依法进行查处，确保专项整治行动取得实效。
                ''',
                'author': '应急管理部',
                'source': '应急管理部消防救援局',
                'publish_time': '2024-01-15',
                'news_type': '政策法规',
                'category': '消防新闻',
                'keywords': ['消防安全', '专项整治', '火灾隐患', '消防设施', '安全责任'],
                'tags': ['消防新闻', '专项整治', '消防安全'],
                'view_count': 1500,
                'comment_count': 25,
                'share_count': 80
            }
        ]
        
        # 生成消防新闻数据
        for news_data in mock_news:
            item = FireNewsItem()
            for key, value in news_data.items():
                item[key] = value
            
            # 文本分析
            text_analysis = fire_text_analyzer.analyze_fire_text(news_data['content'])
            item['summary'] = fire_text_analyzer.generate_document_summary(news_data['content'])
            
            # 文件信息
            item['image_urls'] = []
            item['file_urls'] = []
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成知识条目
            yield from self.generate_knowledge_item(item, news_data['content'], 'news')
        
        self.logger.info(f"生成了 {len(mock_regulations)} 条消防法规数据")
        self.logger.info(f"生成了 {len(mock_standards)} 条消防标准数据")
        self.logger.info(f"生成了 {len(mock_cases)} 条火灾案例数据")
        self.logger.info(f"生成了 {len(mock_news)} 条消防新闻数据")
    
    def generate_knowledge_item(self, item, content, doc_type):
        """生成知识条目"""
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
            
            # 生成RAG文档
            yield from self.generate_rag_document(item, content, doc_type)
            
        except Exception as e:
            self.logger.error(f"生成知识条目出错: {e}")
    
    def generate_rag_document(self, item, content, doc_type):
        """生成RAG文档"""
        try:
            # 创建RAG文档条目
            rag_item = FireDocumentItem()
            rag_item['document_id'] = item.get('regulation_id') or item.get('standard_id') or item.get('case_id') or item.get('news_id')
            rag_item['title'] = item.get('title', '')
            rag_item['content'] = content
            rag_item['document_type'] = doc_type
            
            # 文档特征
            analysis = fire_text_analyzer.analyze_fire_text(content)
            rag_item['content_length'] = analysis.get('char_count', 0)
            rag_item['word_count'] = analysis.get('word_count', 0)
            rag_item['readability_score'] = analysis.get('readability_score', 0)
            rag_item['complexity_score'] = analysis.get('complexity_score', 0)
            
            # 文本分析
            rag_item['summary'] = fire_text_analyzer.generate_document_summary(content)
            rag_item['keywords'] = analysis.get('keywords', [])
            rag_item['entities'] = analysis.get('entities', {})
            rag_item['topics'] = analysis.get('classification', {})
            
            # RAG相关
            chunks = fire_text_analyzer.chunk_text_for_rag(content)
            rag_item['chunk_texts'] = [chunk['text'] for chunk in chunks]
            rag_item['embedding_vector'] = None  # 需要后续处理
            rag_item['chunk_embeddings'] = None  # 需要后续处理
            
            # 文件信息
            rag_item['file_url'] = item.get('file_url', '')
            rag_item['file_type'] = item.get('file_type', '')
            rag_item['file_size'] = item.get('file_size', 0)
            rag_item['download_status'] = 'success'
            
            # 爬取信息
            rag_item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rag_item['source_url'] = item.get('source_url', '')
            
            yield rag_item
            
        except Exception as e:
            self.logger.error(f"生成RAG文档出错: {e}")
