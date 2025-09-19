#!/usr/bin/env python
"""
消防文本分析工具
用于分析消防法规、标准、案例等文本内容
支持RAG知识库构建
"""

import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import Counter


class FireTextAnalyzer:
    """消防文本分析器"""
    
    def __init__(self):
        """初始化消防文本分析器"""
        
        # 消防关键词词典
        self.fire_keywords = {
            '建筑防火': {
                '防火分区', '防火间距', '防火门', '防火窗', '防火墙', '防火卷帘',
                '疏散通道', '安全出口', '疏散楼梯', '避难层', '避难间', '消防电梯',
                '耐火等级', '耐火极限', '燃烧性能', '防火涂料', '防火封堵'
            },
            '消防设施': {
                '消火栓', '自动喷水', '火灾报警', '消防泵', '消防水池', '消防水箱',
                '防烟排烟', '应急照明', '疏散指示', '消防广播', '消防电话',
                '气体灭火', '泡沫灭火', '干粉灭火', '消防车', '消防器材'
            },
            '消防安全管理': {
                '消防安全责任制', '消防安全检查', '消防安全培训', '应急预案',
                '消防演练', '消防安全评估', '火灾隐患', '消防安全管理',
                '消防控制室', '消防安全标志', '消防安全制度'
            },
            '火灾调查': {
                '火灾原因', '火灾调查', '火灾统计', '火灾损失', '火灾责任',
                '火灾预防', '火灾扑救', '火灾救援', '火灾现场', '火灾证据'
            },
            '消防产品': {
                '消防产品', '消防设备', '消防器材', '消防材料', '消防检测',
                '消防认证', '消防标准', '消防技术', '消防工程', '消防设计'
            }
        }
        
        # 法规类型关键词
        self.regulation_types = {
            '法律': {'法', '法律', '条例'},
            '行政法规': {'条例', '规定', '办法', '细则'},
            '部门规章': {'规定', '办法', '细则', '通知', '公告'},
            '地方性法规': {'条例', '规定', '办法'},
            '标准': {'标准', '规范', '规程', 'GB', 'JGJ', 'GA', 'XF'}
        }
        
        # 建筑类型关键词
        self.building_types = {
            '住宅建筑': {'住宅', '居住', '公寓', '宿舍', '住宅楼'},
            '公共建筑': {'办公楼', '商场', '酒店', '医院', '学校', '图书馆', '博物馆', '体育馆'},
            '工业建筑': {'厂房', '仓库', '车间', '生产', '工业', '工厂'},
            '地下建筑': {'地下室', '地下车库', '地下商场', '地下空间'},
            '高层建筑': {'高层', '超高层', '摩天大楼'},
            '特殊建筑': {'古建筑', '文物', '历史建筑', '宗教建筑'}
        }
        
        # 火灾原因关键词
        self.fire_causes = {
            '电气火灾': {'电气', '电线', '电路', '电器', '短路', '过载', '漏电'},
            '用火不慎': {'用火', '明火', '炉火', '蜡烛', '香火', '吸烟'},
            '违章操作': {'违章', '违规', '操作不当', '违反规定'},
            '自燃': {'自燃', '自然', '氧化', '化学反应'},
            '放火': {'放火', '纵火', '故意', '人为'},
            '其他': {'其他', '不明', '待查', '调查中'}
        }
        
        # 严重程度关键词
        self.severity_levels = {
            '特别重大': {'特别重大', '特大', '特别严重'},
            '重大': {'重大', '严重', '重大事故'},
            '较大': {'较大', '中等', '较大事故'},
            '一般': {'一般', '轻微', '一般事故'}
        }
        
        # 消防实体识别模式
        self.entity_patterns = {
            'regulation_number': r'第[一二三四五六七八九十\d]+条',
            'article_number': r'第[一二三四五六七八九十\d]+章',
            'date': r'\d{4}年\d{1,2}月\d{1,2}日',
            'percentage': r'\d+(?:\.\d+)?%',
            'dimension': r'\d+(?:\.\d+)?[米米²米³]',
            'temperature': r'\d+(?:\.\d+)?[°度]C?',
            'pressure': r'\d+(?:\.\d+)?[帕Pa兆帕MPa]',
            'time': r'\d+[小时分钟秒]',
            'phone': r'\d{3,4}-?\d{7,8}',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }
    
    def analyze_fire_text(self, text: str) -> Dict[str, Any]:
        """分析消防文本"""
        if not text:
            return {}
        
        # 基础文本分析
        basic_analysis = self._analyze_basic_text(text)
        
        # 消防专项分析
        fire_analysis = self._analyze_fire_content(text)
        
        # 实体识别
        entities = self._extract_fire_entities(text)
        
        # 关键词提取
        keywords = self._extract_fire_keywords(text)
        
        # 分类分析
        classification = self._classify_fire_content(text)
        
        return {
            **basic_analysis,
            'fire_analysis': fire_analysis,
            'entities': entities,
            'keywords': keywords,
            'classification': classification
        }
    
    def analyze_regulation(self, text: str) -> Dict[str, Any]:
        """分析消防法规"""
        analysis = self.analyze_fire_text(text)
        
        # 法规特定分析
        regulation_analysis = {
            'regulation_type': self._identify_regulation_type(text),
            'issuing_authority': self._extract_issuing_authority(text),
            'effective_date': self._extract_effective_date(text),
            'articles': self._extract_articles(text),
            'chapters': self._extract_chapters(text),
            'scope': self._extract_scope(text)
        }
        
        analysis['regulation_analysis'] = regulation_analysis
        return analysis
    
    def analyze_case(self, text: str) -> Dict[str, Any]:
        """分析火灾案例"""
        analysis = self.analyze_fire_text(text)
        
        # 案例特定分析
        case_analysis = {
            'severity_level': self._identify_severity_level(text),
            'building_type': self._identify_building_type(text),
            'fire_cause': self._identify_fire_cause(text),
            'location': self._extract_location(text),
            'casualties': self._extract_casualties(text),
            'economic_loss': self._extract_economic_loss(text),
            'lessons_learned': self._extract_lessons_learned(text)
        }
        
        analysis['case_analysis'] = case_analysis
        return analysis
    
    def analyze_standard(self, text: str) -> Dict[str, Any]:
        """分析消防标准"""
        analysis = self.analyze_fire_text(text)
        
        # 标准特定分析
        standard_analysis = {
            'standard_number': self._extract_standard_number(text),
            'standard_type': self._identify_standard_type(text),
            'scope': self._extract_scope(text),
            'technical_requirements': self._extract_technical_requirements(text),
            'test_methods': self._extract_test_methods(text),
            'implementation_date': self._extract_implementation_date(text)
        }
        
        analysis['standard_analysis'] = standard_analysis
        return analysis
    
    def chunk_text_for_rag(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """为RAG分块文本"""
        if not text:
            return []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 如果当前段落加上现有块超过大小限制
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                # 保存当前块
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': len(current_chunk),
                    'start_pos': len(' '.join([c['text'] for c in chunks])),
                    'end_pos': len(' '.join([c['text'] for c in chunks])) + len(current_chunk)
                })
                chunk_id += 1
                
                # 开始新块，保留重叠部分
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:] + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += " " + paragraph if current_chunk else paragraph
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'start_pos': len(' '.join([c['text'] for c in chunks])),
                'end_pos': len(' '.join([c['text'] for c in chunks])) + len(current_chunk)
            })
        
        return chunks
    
    def generate_document_summary(self, text: str, max_length: int = 200) -> str:
        """生成文档摘要"""
        if not text:
            return ""
        
        # 简单的摘要生成（取前几段）
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        summary = ""
        for paragraph in paragraphs:
            if len(summary) + len(paragraph) <= max_length:
                summary += paragraph + "\n"
            else:
                break
        
        return summary.strip()
    
    def _analyze_basic_text(self, text: str) -> Dict[str, Any]:
        """基础文本分析"""
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?。！？]', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # 可读性分析
        readability_score = self._calculate_readability(text)
        
        # 复杂度分析
        complexity_score = self._calculate_complexity(text)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'readability_score': readability_score,
            'complexity_score': complexity_score
        }
    
    def _analyze_fire_content(self, text: str) -> Dict[str, Any]:
        """消防内容分析"""
        fire_mentions = {}
        total_mentions = 0
        
        for category, keywords in self.fire_keywords.items():
            mentions = 0
            for keyword in keywords:
                count = text.count(keyword)
                if count > 0:
                    mentions += count
            fire_mentions[category] = mentions
            total_mentions += mentions
        
        # 确定主要消防类别
        main_category = max(fire_mentions.items(), key=lambda x: x[1])[0] if fire_mentions else '其他'
        
        return {
            'fire_mentions': fire_mentions,
            'total_fire_mentions': total_mentions,
            'main_category': main_category
        }
    
    def _extract_fire_entities(self, text: str) -> Dict[str, List[str]]:
        """提取消防实体"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            entities[entity_type] = list(set(matches))
        
        return entities
    
    def _extract_fire_keywords(self, text: str) -> List[str]:
        """提取消防关键词"""
        keywords = []
        text_lower = text.lower()
        
        for category, keyword_set in self.fire_keywords.items():
            for keyword in keyword_set:
                if keyword in text_lower:
                    keywords.append(keyword)
        
        # 按出现频率排序
        keyword_counts = Counter(keywords)
        return [keyword for keyword, count in keyword_counts.most_common(20)]
    
    def _classify_fire_content(self, text: str) -> Dict[str, Any]:
        """分类消防内容"""
        classification = {
            'document_type': 'unknown',
            'category': 'unknown',
            'subcategory': 'unknown',
            'confidence': 0.0
        }
        
        # 根据关键词判断文档类型
        if any(keyword in text for keyword in ['法规', '条例', '规定', '办法']):
            classification['document_type'] = 'regulation'
        elif any(keyword in text for keyword in ['标准', '规范', 'GB', 'JGJ']):
            classification['document_type'] = 'standard'
        elif any(keyword in text for keyword in ['火灾', '事故', '案例']):
            classification['document_type'] = 'case'
        elif any(keyword in text for keyword in ['新闻', '报道', '资讯']):
            classification['document_type'] = 'news'
        else:
            classification['document_type'] = 'knowledge'
        
        # 根据消防关键词判断分类
        fire_analysis = self._analyze_fire_content(text)
        classification['category'] = fire_analysis['main_category']
        
        return classification
    
    def _identify_regulation_type(self, text: str) -> str:
        """识别法规类型"""
        for reg_type, keywords in self.regulation_types.items():
            if any(keyword in text for keyword in keywords):
                return reg_type
        return '其他'
    
    def _extract_issuing_authority(self, text: str) -> str:
        """提取发布机关"""
        authority_patterns = [
            r'([^，。]*部[^，。]*)',
            r'([^，。]*局[^，。]*)',
            r'([^，。]*委员会[^，。]*)',
            r'([^，。]*政府[^，。]*)'
        ]
        
        for pattern in authority_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def _extract_effective_date(self, text: str) -> str:
        """提取生效日期"""
        date_patterns = [
            r'自(\d{4}年\d{1,2}月\d{1,2}日)起施行',
            r'(\d{4}年\d{1,2}月\d{1,2}日)起实施',
            r'自(\d{4}年\d{1,2}月\d{1,2}日)起执行'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ''
    
    def _extract_articles(self, text: str) -> List[Dict[str, str]]:
        """提取条文"""
        articles = []
        article_pattern = r'第([一二三四五六七八九十\d]+)条\s*([^第]+?)(?=第[一二三四五六七八九十\d]+条|$)'
        
        matches = re.finditer(article_pattern, text, re.DOTALL)
        for match in matches:
            article_num = match.group(1)
            article_content = match.group(2).strip()
            articles.append({
                'number': article_num,
                'content': article_content
            })
        
        return articles
    
    def _extract_chapters(self, text: str) -> List[Dict[str, str]]:
        """提取章节"""
        chapters = []
        chapter_pattern = r'第([一二三四五六七八九十\d]+)章\s*([^第]+?)(?=第[一二三四五六七八九十\d]+章|$)'
        
        matches = re.finditer(chapter_pattern, text, re.DOTALL)
        for match in matches:
            chapter_num = match.group(1)
            chapter_content = match.group(2).strip()
            chapters.append({
                'number': chapter_num,
                'content': chapter_content
            })
        
        return chapters
    
    def _extract_scope(self, text: str) -> str:
        """提取适用范围"""
        scope_patterns = [
            r'适用范围[：:]\s*([^。]+)',
            r'适用[于]?\s*([^。]+)',
            r'本[法规标准规定]适用于\s*([^。]+)'
        ]
        
        for pattern in scope_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def _identify_severity_level(self, text: str) -> str:
        """识别严重程度"""
        for level, keywords in self.severity_levels.items():
            if any(keyword in text for keyword in keywords):
                return level
        return '一般'
    
    def _identify_building_type(self, text: str) -> str:
        """识别建筑类型"""
        for building_type, keywords in self.building_types.items():
            if any(keyword in text for keyword in keywords):
                return building_type
        return '其他'
    
    def _identify_fire_cause(self, text: str) -> str:
        """识别火灾原因"""
        for cause, keywords in self.fire_causes.items():
            if any(keyword in text for keyword in keywords):
                return cause
        return '其他'
    
    def _extract_location(self, text: str) -> Dict[str, str]:
        """提取地点信息"""
        location = {}
        
        # 提取省份
        province_pattern = r'([^，。]*省[^，。]*)'
        province_match = re.search(province_pattern, text)
        if province_match:
            location['province'] = province_match.group(1).strip()
        
        # 提取城市
        city_pattern = r'([^，。]*市[^，。]*)'
        city_match = re.search(city_pattern, text)
        if city_match:
            location['city'] = city_match.group(1).strip()
        
        return location
    
    def _extract_casualties(self, text: str) -> Dict[str, str]:
        """提取伤亡情况"""
        casualties = {}
        
        # 死亡人数
        death_pattern = r'死亡\s*(\d+)\s*人'
        death_match = re.search(death_pattern, text)
        if death_match:
            casualties['deaths'] = death_match.group(1)
        
        # 受伤人数
        injury_pattern = r'受伤\s*(\d+)\s*人'
        injury_match = re.search(injury_pattern, text)
        if injury_match:
            casualties['injuries'] = injury_match.group(1)
        
        return casualties
    
    def _extract_economic_loss(self, text: str) -> str:
        """提取经济损失"""
        loss_patterns = [
            r'经济损失\s*([^。]+)',
            r'直接经济损失\s*([^。]+)',
            r'损失\s*([^。]+元)'
        ]
        
        for pattern in loss_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def _extract_lessons_learned(self, text: str) -> str:
        """提取经验教训"""
        lesson_patterns = [
            r'经验教训[：:]\s*([^。]+)',
            r'教训[：:]\s*([^。]+)',
            r'启示[：:]\s*([^。]+)'
        ]
        
        for pattern in lesson_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def _extract_standard_number(self, text: str) -> str:
        """提取标准编号"""
        standard_patterns = [
            r'(GB\s*\d+[-\d]*)',
            r'(JGJ\s*\d+[-\d]*)',
            r'(GA\s*\d+[-\d]*)',
            r'(XF\s*\d+[-\d]*)'
        ]
        
        for pattern in standard_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def _identify_standard_type(self, text: str) -> str:
        """识别标准类型"""
        if 'GB' in text:
            return '国家标准'
        elif 'JGJ' in text:
            return '行业标准'
        elif 'GA' in text or 'XF' in text:
            return '行业标准'
        else:
            return '其他'
    
    def _extract_technical_requirements(self, text: str) -> List[str]:
        """提取技术要求"""
        requirements = []
        req_pattern = r'[技术要求|规定|应|必须|不得][：:]\s*([^。]+)'
        
        matches = re.finditer(req_pattern, text)
        for match in matches:
            requirements.append(match.group(1).strip())
        
        return requirements
    
    def _extract_test_methods(self, text: str) -> List[str]:
        """提取试验方法"""
        methods = []
        method_pattern = r'[试验|测试|检测|检验][方法|程序][：:]\s*([^。]+)'
        
        matches = re.finditer(method_pattern, text)
        for match in matches:
            methods.append(match.group(1).strip())
        
        return methods
    
    def _extract_implementation_date(self, text: str) -> str:
        """提取实施日期"""
        return self._extract_effective_date(text)
    
    def _calculate_readability(self, text: str) -> float:
        """计算可读性得分"""
        if not text:
            return 0.0
        
        sentences = re.findall(r'[.!?。！？]', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 可读性得分（0-100，越高越易读）
        readability = max(0, 100 - (avg_sentence_length * 0.5) - (avg_word_length * 2))
        return round(readability, 2)
    
    def _calculate_complexity(self, text: str) -> float:
        """计算文本复杂度"""
        if not text:
            return 0.0
        
        words = text.split()
        unique_words = set(words)
        
        if not words:
            return 0.0
        
        # 词汇多样性
        lexical_diversity = len(unique_words) / len(words)
        
        # 长句比例
        sentences = re.split(r'[.!?。！？]', text)
        long_sentences = [s for s in sentences if len(s.split()) > 20]
        long_sentence_ratio = len(long_sentences) / len(sentences) if sentences else 0
        
        # 复杂度得分（0-100，越高越复杂）
        complexity = (lexical_diversity * 50) + (long_sentence_ratio * 50)
        return round(complexity, 2)


# 全局消防文本分析器实例
fire_text_analyzer = FireTextAnalyzer()
