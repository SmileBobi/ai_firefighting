#!/usr/bin/env python
"""
长文本分析工具
用于分析上市公司财报等长文本内容
支持智谱AI和Kimi(月之暗面)的长文本分析
"""

import re
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional


class TextAnalyzer:
    """长文本分析器"""
    
    def __init__(self, zhipu_api_key: str = None, kimi_api_key: str = None):
        """初始化文本分析器"""
        self.zhipu_api_key = zhipu_api_key
        self.kimi_api_key = kimi_api_key
        
        # 智谱AI API配置
        self.zhipu_base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
        # Kimi API配置
        self.kimi_base_url = "https://api.moonshot.cn/v1/chat/completions"
        
        # 财务关键词
        self.financial_keywords = {
            '营业收入', '净利润', '总资产', '总负债', '股东权益', '经营现金流',
            '每股收益', '净资产收益率', '总资产收益率', '毛利率', '净利率',
            '资产负债率', '流动比率', '速动比率', '存货周转率', '应收账款周转率',
            '总资产周转率', '固定资产周转率', '权益乘数', '利息保障倍数'
        }
        
        # 风险关键词
        self.risk_keywords = {
            '风险', '不确定性', '挑战', '困难', '压力', '下降', '减少', '亏损',
            '诉讼', '纠纷', '处罚', '监管', '政策变化', '市场风险', '信用风险',
            '流动性风险', '操作风险', '汇率风险', '利率风险'
        }
        
        # 机会关键词
        self.opportunity_keywords = {
            '机会', '增长', '发展', '扩张', '创新', '技术', '市场', '产品',
            '服务', '合作', '投资', '收购', '并购', '战略', '规划', '目标',
            '优势', '竞争力', '品牌', '渠道', '客户'
        }
    
    def analyze_text_basic(self, text: str) -> Dict[str, Any]:
        """基础文本分析"""
        if not text:
            return {}
        
        # 文本统计
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?。！？]', text))
        paragraph_count = len([p for p in text.split('\n') if p.strip()])
        
        # 可读性分析
        readability_score = self._calculate_readability(text)
        
        # 复杂度分析
        complexity_score = self._calculate_complexity(text)
        
        # 关键词提取
        topic_keywords = self._extract_topic_keywords(text)
        
        # 实体识别（简单版本）
        entities = self._extract_entities(text)
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'readability_score': readability_score,
            'complexity_score': complexity_score,
            'topic_keywords': topic_keywords,
            'entities': entities
        }
    
    def analyze_financial_report(self, text: str) -> Dict[str, Any]:
        """财报专项分析"""
        if not text:
            return {}
        
        # 基础分析
        basic_analysis = self.analyze_text_basic(text)
        
        # 财务关键词分析
        financial_mentions = self._count_keyword_mentions(text, self.financial_keywords)
        
        # 风险分析
        risk_analysis = self._analyze_risks(text)
        
        # 机会分析
        opportunity_analysis = self._analyze_opportunities(text)
        
        # 财务数据提取
        financial_data = self._extract_financial_data(text)
        
        # 章节分析
        sections = self._analyze_sections(text)
        
        return {
            **basic_analysis,
            'financial_mentions': financial_mentions,
            'risk_analysis': risk_analysis,
            'opportunity_analysis': opportunity_analysis,
            'financial_data': financial_data,
            'sections': sections
        }
    
    def analyze_with_zhipu(self, text: str, prompt: str = None) -> Dict[str, Any]:
        """使用智谱AI进行长文本分析"""
        if not self.zhipu_api_key:
            return {'error': '智谱AI API密钥未配置'}
        
        if not prompt:
            prompt = """
            请对以下上市公司财报内容进行深度分析，包括：
            1. 财务表现分析
            2. 经营状况评估
            3. 风险因素识别
            4. 发展机会分析
            5. 投资建议总结
            
            请以结构化的JSON格式返回分析结果。
            """
        
        try:
            headers = {
                'Authorization': f'Bearer {self.zhipu_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'glm-4',
                'messages': [
                    {'role': 'system', 'content': '你是一个专业的财务分析师，擅长分析上市公司财报。'},
                    {'role': 'user', 'content': f'{prompt}\n\n财报内容：\n{text[:8000]}'}  # 限制长度
                ],
                'temperature': 0.3,
                'max_tokens': 2000
            }
            
            response = requests.post(self.zhipu_base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            ai_content = result['choices'][0]['message']['content']
            
            return {
                'ai_analysis': ai_content,
                'model': 'zhipu-glm-4',
                'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {'error': f'智谱AI分析失败: {str(e)}'}
    
    def analyze_with_kimi(self, text: str, prompt: str = None) -> Dict[str, Any]:
        """使用Kimi进行长文本分析"""
        if not self.kimi_api_key:
            return {'error': 'Kimi API密钥未配置'}
        
        if not prompt:
            prompt = """
            请对以下上市公司财报内容进行专业分析，重点关注：
            1. 核心财务指标解读
            2. 业务发展亮点
            3. 潜在风险点
            4. 行业地位分析
            5. 未来展望评估
            
            请提供详细的分析报告。
            """
        
        try:
            headers = {
                'Authorization': f'Bearer {self.kimi_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'moonshot-v1-8k',
                'messages': [
                    {'role': 'system', 'content': '你是一位资深的投资分析师，专门研究上市公司财报。'},
                    {'role': 'user', 'content': f'{prompt}\n\n财报内容：\n{text[:6000]}'}  # 限制长度
                ],
                'temperature': 0.2,
                'max_tokens': 2000
            }
            
            response = requests.post(self.kimi_base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            ai_content = result['choices'][0]['message']['content']
            
            return {
                'ai_analysis': ai_content,
                'model': 'kimi-moonshot-v1-8k',
                'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {'error': f'Kimi分析失败: {str(e)}'}
    
    def _calculate_readability(self, text: str) -> float:
        """计算可读性得分"""
        if not text:
            return 0.0
        
        # 简单的可读性计算（基于句子长度和词汇复杂度）
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
        
        # 基于词汇多样性和句子结构复杂度
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
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """提取主题关键词"""
        if not text:
            return []
        
        # 简单的关键词提取（基于词频）
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 2:  # 过滤短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回频率最高的前10个词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体（简单版本）"""
        if not text:
            return {}
        
        entities = {
            'companies': [],
            'numbers': [],
            'dates': [],
            'percentages': []
        }
        
        # 公司名称（简单匹配）
        company_pattern = r'([A-Za-z\u4e00-\u9fa5]+(?:公司|集团|有限|股份|科技|发展|投资|控股))'
        entities['companies'] = re.findall(company_pattern, text)
        
        # 数字
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities['numbers'] = re.findall(number_pattern, text)
        
        # 日期
        date_pattern = r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2}'
        entities['dates'] = re.findall(date_pattern, text)
        
        # 百分比
        percent_pattern = r'\d+(?:\.\d+)?%'
        entities['percentages'] = re.findall(percent_pattern, text)
        
        return entities
    
    def _count_keyword_mentions(self, text: str, keywords: set) -> Dict[str, int]:
        """统计关键词出现次数"""
        mentions = {}
        text_lower = text.lower()
        
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                mentions[keyword] = count
        
        return mentions
    
    def _analyze_risks(self, text: str) -> Dict[str, Any]:
        """风险分析"""
        risk_mentions = self._count_keyword_mentions(text, self.risk_keywords)
        
        # 风险等级评估
        total_risk_mentions = sum(risk_mentions.values())
        risk_level = 'low' if total_risk_mentions < 5 else 'medium' if total_risk_mentions < 15 else 'high'
        
        return {
            'risk_mentions': risk_mentions,
            'total_risk_mentions': total_risk_mentions,
            'risk_level': risk_level
        }
    
    def _analyze_opportunities(self, text: str) -> Dict[str, Any]:
        """机会分析"""
        opportunity_mentions = self._count_keyword_mentions(text, self.opportunity_keywords)
        
        # 机会等级评估
        total_opportunity_mentions = sum(opportunity_mentions.values())
        opportunity_level = 'low' if total_opportunity_mentions < 5 else 'medium' if total_opportunity_mentions < 15 else 'high'
        
        return {
            'opportunity_mentions': opportunity_mentions,
            'total_opportunity_mentions': total_opportunity_mentions,
            'opportunity_level': opportunity_level
        }
    
    def _extract_financial_data(self, text: str) -> Dict[str, Any]:
        """提取财务数据"""
        financial_data = {}
        
        # 营业收入
        revenue_pattern = r'营业收入[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        revenue_match = re.search(revenue_pattern, text)
        if revenue_match:
            financial_data['revenue'] = revenue_match.group(1)
        
        # 净利润
        profit_pattern = r'净利润[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        profit_match = re.search(profit_pattern, text)
        if profit_match:
            financial_data['net_profit'] = profit_match.group(1)
        
        # 总资产
        assets_pattern = r'总资产[：:]\s*([0-9,]+\.?\d*)\s*万元?'
        assets_match = re.search(assets_pattern, text)
        if assets_match:
            financial_data['total_assets'] = assets_match.group(1)
        
        return financial_data
    
    def _analyze_sections(self, text: str) -> List[Dict[str, Any]]:
        """分析文本章节"""
        sections = []
        
        # 查找章节标题
        section_pattern = r'第[一二三四五六七八九十\d]+[章节条]\s*([^\n]+)'
        section_matches = re.finditer(section_pattern, text)
        
        for match in section_matches:
            section_title = match.group(1).strip()
            section_start = match.start()
            
            sections.append({
                'title': section_title,
                'start_position': section_start,
                'length': len(section_title)
            })
        
        return sections
    
    def generate_summary(self, text: str, max_length: int = 500) -> str:
        """生成文本摘要"""
        if not text:
            return ""
        
        # 简单的摘要生成（取前几段）
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        summary = ""
        for paragraph in paragraphs:
            if len(summary) + len(paragraph) <= max_length:
                summary += paragraph + "\n"
            else:
                break
        
        return summary.strip()


# 全局文本分析器实例
text_analyzer = TextAnalyzer()
