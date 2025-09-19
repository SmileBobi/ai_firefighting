#!/usr/bin/env python
"""
情感分析工具
用于分析论坛帖子和评论的情感倾向
"""

import re
from collections import Counter
from datetime import datetime


class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        """初始化情感分析器"""
        # 积极情感词典
        self.positive_words = {
            '涨', '上涨', '大涨', '暴涨', '涨停', '突破', '创新高', '看好', '乐观', '积极',
            '买入', '持有', '推荐', '优质', '强势', '龙头', '价值', '成长', '机会', '潜力',
            '收益', '盈利', '赚钱', '发财', '成功', '胜利', '优秀', '出色', '完美', '理想',
            '满意', '高兴', '兴奋', '激动', '期待', '希望', '信心', '相信', '肯定', '支持',
            '利好', '好消息', '喜讯', '惊喜', '意外', '超预期', '超预期', '超预期', '超预期'
        }
        
        # 消极情感词典
        self.negative_words = {
            '跌', '下跌', '大跌', '暴跌', '跌停', '破位', '创新低', '看空', '悲观', '消极',
            '卖出', '清仓', '割肉', '止损', '风险', '危险', '泡沫', '高估', '垃圾', '亏损',
            '套牢', '被套', '深套', '血亏', '破产', '失败', '失望', '绝望', '担心', '害怕',
            '恐惧', '焦虑', '紧张', '不安', '怀疑', '质疑', '反对', '批评', '抱怨', '愤怒',
            '利空', '坏消息', '噩耗', '灾难', '危机', '崩盘', '崩盘', '崩盘', '崩盘', '崩盘'
        }
        
        # 中性情感词典
        self.neutral_words = {
            '分析', '研究', '观察', '关注', '讨论', '交流', '分享', '学习', '思考', '判断',
            '数据', '指标', '技术', '基本面', '消息', '公告', '财报', '业绩', '估值', '价格',
            '市场', '行情', '趋势', '走势', '波动', '震荡', '调整', '整理', '盘整', '横盘'
        }
        
        # 股票相关关键词
        self.stock_keywords = {
            '股票', '股价', '市值', '市盈率', '市净率', '成交量', '成交额', '换手率', '振幅',
            '开盘', '收盘', '最高', '最低', '涨停', '跌停', '停牌', '复牌', '分红', '送股',
            '配股', '增发', '回购', '减持', '增持', '举牌', '重组', '并购', '借壳', 'IPO'
        }
        
        # 市场相关关键词
        self.market_keywords = {
            '大盘', '指数', '上证', '深证', '创业板', '科创板', '港股', '美股', 'A股', 'B股',
            '牛市', '熊市', '震荡市', '结构性行情', '板块轮动', '热点', '题材', '概念', '主题'
        }
        
        # 初始化完成
    
    def analyze_sentiment(self, text):
        """分析文本情感倾向"""
        if not text or not isinstance(text, str):
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0,
                'emotion_keywords': [],
                'stock_keywords': [],
                'market_keywords': []
            }
        
        # 清理文本
        cleaned_text = self._clean_text(text)
        
        # 简单分词（按字符分割）
        words = self._simple_tokenize(cleaned_text)
        
        # 计算情感得分
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        emotion_keywords = []
        stock_keywords = []
        market_keywords = []
        
        for word in words:
            if word in self.positive_words:
                positive_count += 1
                emotion_keywords.append(word)
            elif word in self.negative_words:
                negative_count += 1
                emotion_keywords.append(word)
            elif word in self.neutral_words:
                neutral_count += 1
            
            if word in self.stock_keywords:
                stock_keywords.append(word)
            elif word in self.market_keywords:
                market_keywords.append(word)
        
        # 计算得分
        total_words = len(words)
        if total_words == 0:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 0.0,
                'emotion_keywords': [],
                'stock_keywords': [],
                'market_keywords': []
            }
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = neutral_count / total_words
        
        # 计算总体情感得分 (-1到1)
        sentiment_score = positive_score - negative_score
        
        # 确定情感标签
        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # 计算置信度
        confidence = abs(sentiment_score)
        
        return {
            'sentiment_score': round(sentiment_score, 3),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 3),
            'positive_score': round(positive_score, 3),
            'negative_score': round(negative_score, 3),
            'neutral_score': round(neutral_score, 3),
            'emotion_keywords': list(set(emotion_keywords)),
            'stock_keywords': list(set(stock_keywords)),
            'market_keywords': list(set(market_keywords))
        }
    
    def extract_stock_mentions(self, text):
        """提取股票代码和名称"""
        if not text:
            return []
        
        # 股票代码正则表达式
        stock_code_pattern = r'\b(?:[0-9]{6}|[0-9]{3}\.[A-Z]{2})\b'
        stock_codes = re.findall(stock_code_pattern, text)
        
        # 常见股票名称
        stock_names = {
            '平安银行', '万科A', '浦发银行', '招商银行', '五粮液', '贵州茅台', '中国平安',
            '工商银行', '建设银行', '农业银行', '中国银行', '腾讯', '阿里巴巴', '百度',
            '京东', '美团', '小米', '比亚迪', '宁德时代', '特斯拉', '苹果', '微软'
        }
        
        mentioned_stocks = []
        for name in stock_names:
            if name in text:
                mentioned_stocks.append(name)
        
        return list(set(stock_codes + mentioned_stocks))
    
    def analyze_content_features(self, text):
        """分析内容特征"""
        if not text:
            return {
                'content_length': 0,
                'has_images': False,
                'has_links': False,
                'topic_tags': []
            }
        
        # 内容长度
        content_length = len(text)
        
        # 是否包含图片
        has_images = bool(re.search(r'\[图片\]|\[img\]|\.jpg|\.png|\.gif', text, re.IGNORECASE))
        
        # 是否包含链接
        has_links = bool(re.search(r'http[s]?://|www\.', text, re.IGNORECASE))
        
        # 话题标签
        topic_tags = []
        if '技术分析' in text or 'K线' in text or '均线' in text:
            topic_tags.append('技术分析')
        if '基本面' in text or '财报' in text or '业绩' in text:
            topic_tags.append('基本面分析')
        if '消息' in text or '公告' in text or '新闻' in text:
            topic_tags.append('消息面')
        if '政策' in text or '监管' in text or '制度' in text:
            topic_tags.append('政策面')
        if '资金' in text or '主力' in text or '机构' in text:
            topic_tags.append('资金面')
        
        return {
            'content_length': content_length,
            'has_images': has_images,
            'has_links': has_links,
            'topic_tags': topic_tags
        }
    
    def _clean_text(self, text):
        """清理文本"""
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _simple_tokenize(self, text):
        """简单分词"""
        if not text:
            return []
        
        # 按空格分割
        words = text.split()
        
        # 进一步分割中文词汇（简单方法）
        result = []
        for word in words:
            if len(word) > 1:
                # 对于长度大于1的词，尝试分割成更小的单元
                for i in range(len(word) - 1):
                    result.append(word[i:i+2])
                result.append(word)
            else:
                result.append(word)
        
        return result
    
    def batch_analyze(self, texts):
        """批量分析文本情感"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results


# 全局情感分析器实例
sentiment_analyzer = SentimentAnalyzer()
