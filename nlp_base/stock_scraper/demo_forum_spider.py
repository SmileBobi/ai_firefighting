#!/usr/bin/env python
"""
演示用的论坛数据爬虫
使用模拟数据来演示论坛信息抓取和情感分析功能
"""

import scrapy
import json
from datetime import datetime, timedelta
from stock_scraper.items import ForumPostItem, ForumCommentItem, ForumUserItem, SentimentAnalysisItem
from sentiment_analyzer import sentiment_analyzer


class DemoForumSpider(scrapy.Spider):
    name = "demo_forum"
    allowed_domains = ["example.com"]
    start_urls = ["https://httpbin.org/json"]  # 使用一个返回JSON的测试API
    
    def parse(self, response):
        """解析响应并生成模拟的论坛数据"""
        
        # 生成模拟的论坛帖子数据
        mock_posts = [
            {
                'post_id': '123456',
                'title': '平安银行今天大涨，大家怎么看？',
                'content': '平安银行今天表现不错，开盘就涨了3%，技术面看突破前期高点，基本面也有利好支撑，个人比较看好后市表现。',
                'author_name': '股海小散',
                'author_level': 'LV5',
                'publish_time': '2025-09-17 14:30:00',
                'view_count': '1250',
                'reply_count': '45',
                'like_count': '23',
                'related_stocks': ['000001'],
                'stock_mentions': 1
            },
            {
                'post_id': '123457',
                'title': '万科A跌停，是不是要割肉了？',
                'content': '万科A今天跌停了，心里很慌，不知道是不是应该割肉止损。最近房地产政策收紧，对万科影响很大，大家觉得还有希望吗？',
                'author_name': '韭菜一枚',
                'author_level': 'LV3',
                'publish_time': '2025-09-17 14:25:00',
                'view_count': '890',
                'reply_count': '67',
                'like_count': '12',
                'related_stocks': ['000002'],
                'stock_mentions': 1
            },
            {
                'post_id': '123458',
                'title': '浦发银行技术分析：突破在即',
                'content': '从技术面分析，浦发银行已经连续三天收阳，成交量放大，MACD金叉，KDJ指标也显示买入信号。基本面方面，银行板块整体估值偏低，有补涨需求。',
                'author_name': '技术分析师',
                'author_level': 'LV8',
                'publish_time': '2025-09-17 14:20:00',
                'view_count': '2100',
                'reply_count': '89',
                'like_count': '56',
                'related_stocks': ['600000'],
                'stock_mentions': 1
            },
            {
                'post_id': '123459',
                'title': '市场情绪分析：恐慌还是机会？',
                'content': '今天市场整体下跌，很多股票都出现了恐慌性抛售。但从历史经验来看，这种时候往往也是机会。关键是要选择优质股票，不要盲目跟风。',
                'author_name': '价值投资者',
                'author_level': 'LV7',
                'publish_time': '2025-09-17 14:15:00',
                'view_count': '1560',
                'reply_count': '34',
                'like_count': '78',
                'related_stocks': [],
                'stock_mentions': 0
            },
            {
                'post_id': '123460',
                'title': '五粮液业绩超预期，继续持有',
                'content': '五粮液三季度业绩超预期，净利润增长15%，股价也相应上涨。作为白酒龙头，五粮液具有品牌优势和渠道优势，长期看好。',
                'author_name': '白酒专家',
                'author_level': 'LV6',
                'publish_time': '2025-09-17 14:10:00',
                'view_count': '980',
                'reply_count': '23',
                'like_count': '45',
                'related_stocks': ['000858'],
                'stock_mentions': 1
            }
        ]
        
        # 生成模拟的评论数据
        mock_comments = [
            {
                'comment_id': '1001',
                'post_id': '123456',
                'content': '我也看好平安银行，基本面不错，技术面也突破了',
                'author_name': '股友A',
                'author_level': 'LV4',
                'publish_time': '2025-09-17 14:35:00',
                'like_count': '5',
                'parent_comment_id': '',
                'comment_level': 1
            },
            {
                'comment_id': '1002',
                'post_id': '123456',
                'content': '但是要注意风险，银行股波动比较大',
                'author_name': '风险控制',
                'author_level': 'LV5',
                'publish_time': '2025-09-17 14:36:00',
                'like_count': '3',
                'parent_comment_id': '1001',
                'comment_level': 2
            },
            {
                'comment_id': '1003',
                'post_id': '123457',
                'content': '房地产政策确实收紧，建议减仓',
                'author_name': '政策分析师',
                'author_level': 'LV6',
                'publish_time': '2025-09-17 14:30:00',
                'like_count': '8',
                'parent_comment_id': '',
                'comment_level': 1
            },
            {
                'comment_id': '1004',
                'post_id': '123458',
                'content': '技术分析很专业，学习了',
                'author_name': '新手小白',
                'author_level': 'LV2',
                'publish_time': '2025-09-17 14:25:00',
                'like_count': '2',
                'parent_comment_id': '',
                'comment_level': 1
            },
            {
                'comment_id': '1005',
                'post_id': '123459',
                'content': '同意，危机中往往孕育着机会',
                'author_name': '老股民',
                'author_level': 'LV9',
                'publish_time': '2025-09-17 14:20:00',
                'like_count': '12',
                'parent_comment_id': '',
                'comment_level': 1
            }
        ]
        
        # 生成模拟的用户数据
        mock_users = [
            {
                'user_id': '10001',
                'username': '股海小散',
                'nickname': '小散',
                'level': 'LV5',
                'join_time': '2023-01-15',
                'last_active': '2025-09-17 14:30:00',
                'post_count': '156',
                'comment_count': '892',
                'follower_count': '234',
                'following_count': '89',
                'total_likes': '1234',
                'user_tags': ['技术分析', '短线交易'],
                'expertise_areas': ['银行股', '科技股'],
                'investment_style': '价值投资',
                'activity_score': '85',
                'influence_score': '72'
            },
            {
                'user_id': '10002',
                'username': '韭菜一枚',
                'nickname': '韭菜',
                'level': 'LV3',
                'join_time': '2024-03-20',
                'last_active': '2025-09-17 14:25:00',
                'post_count': '45',
                'comment_count': '234',
                'follower_count': '67',
                'following_count': '123',
                'total_likes': '456',
                'user_tags': ['新手', '学习'],
                'expertise_areas': ['房地产'],
                'investment_style': '跟风投资',
                'activity_score': '65',
                'influence_score': '45'
            },
            {
                'user_id': '10003',
                'username': '技术分析师',
                'nickname': '分析师',
                'level': 'LV8',
                'join_time': '2022-06-10',
                'last_active': '2025-09-17 14:20:00',
                'post_count': '289',
                'comment_count': '1456',
                'follower_count': '567',
                'following_count': '234',
                'total_likes': '3456',
                'user_tags': ['技术分析', '专业'],
                'expertise_areas': ['技术分析', '银行股'],
                'investment_style': '技术分析',
                'activity_score': '95',
                'influence_score': '88'
            }
        ]
        
        # 生成帖子数据
        for post_data in mock_posts:
            item = ForumPostItem()
            for key, value in post_data.items():
                item[key] = value
            
            # 情感分析
            sentiment_result = sentiment_analyzer.analyze_sentiment(post_data['content'])
            item['sentiment_score'] = sentiment_result['sentiment_score']
            item['sentiment_label'] = sentiment_result['sentiment_label']
            item['emotion_keywords'] = sentiment_result['emotion_keywords']
            
            # 内容分析
            content_features = sentiment_analyzer.analyze_content_features(post_data['content'])
            item['content_length'] = content_features['content_length']
            item['has_images'] = content_features['has_images']
            item['has_links'] = content_features['has_links']
            item['topic_tags'] = content_features['topic_tags']
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
            
            # 生成情感分析结果
            sentiment_item = SentimentAnalysisItem()
            sentiment_item['content_id'] = post_data['post_id']
            sentiment_item['content_type'] = 'post'
            sentiment_item['content_text'] = post_data['content']
            sentiment_item['sentiment_score'] = sentiment_result['sentiment_score']
            sentiment_item['sentiment_label'] = sentiment_result['sentiment_label']
            sentiment_item['confidence'] = sentiment_result['confidence']
            sentiment_item['positive_score'] = sentiment_result['positive_score']
            sentiment_item['negative_score'] = sentiment_result['negative_score']
            sentiment_item['neutral_score'] = sentiment_result['neutral_score']
            sentiment_item['emotion_keywords'] = sentiment_result['emotion_keywords']
            sentiment_item['stock_keywords'] = sentiment_result['stock_keywords']
            sentiment_item['market_keywords'] = sentiment_result['market_keywords']
            sentiment_item['mentioned_stocks'] = sentiment_analyzer.extract_stock_mentions(post_data['content'])
            sentiment_item['analyzed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sentiment_item['analysis_model'] = 'rule_based'
            
            yield sentiment_item
        
        # 生成评论数据
        for comment_data in mock_comments:
            item = ForumCommentItem()
            for key, value in comment_data.items():
                item[key] = value
            
            # 情感分析
            sentiment_result = sentiment_analyzer.analyze_sentiment(comment_data['content'])
            item['sentiment_score'] = sentiment_result['sentiment_score']
            item['sentiment_label'] = sentiment_result['sentiment_label']
            item['emotion_keywords'] = sentiment_result['emotion_keywords']
            
            # 内容分析
            content_features = sentiment_analyzer.analyze_content_features(comment_data['content'])
            item['content_length'] = content_features['content_length']
            item['has_images'] = content_features['has_images']
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
        
        # 生成用户数据
        for user_data in mock_users:
            item = ForumUserItem()
            for key, value in user_data.items():
                item[key] = value
            
            # 爬取信息
            item['scraped_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            item['source_url'] = response.url
            
            yield item
        
        self.logger.info(f"生成了 {len(mock_posts)} 条论坛帖子数据")
        self.logger.info(f"生成了 {len(mock_comments)} 条评论数据")
        self.logger.info(f"生成了 {len(mock_users)} 条用户数据")
