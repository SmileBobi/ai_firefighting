#!/usr/bin/env python
"""
论坛数据专用管道
"""

import pandas as pd
import json
import csv
import os
from datetime import datetime
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


class ForumExcelWriterPipeline:
    """论坛数据Excel文件存储管道"""
    
    def __init__(self):
        self.items = {
            'forum_posts': [],
            'forum_comments': [],
            'forum_users': [],
            'sentiment_analysis': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("论坛Excel管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存Excel文件"""
        try:
            # 创建Excel写入器
            excel_filename = f'forum_data_{self.timestamp}.xlsx'
            
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 保存论坛帖子数据
                if self.items['forum_posts']:
                    df_posts = pd.DataFrame(self.items['forum_posts'])
                    df_posts.to_excel(writer, sheet_name='论坛帖子', index=False)
                    spider.logger.info(f"论坛帖子: {len(self.items['forum_posts'])} 条")
                
                # 保存论坛评论数据
                if self.items['forum_comments']:
                    df_comments = pd.DataFrame(self.items['forum_comments'])
                    df_comments.to_excel(writer, sheet_name='论坛评论', index=False)
                    spider.logger.info(f"论坛评论: {len(self.items['forum_comments'])} 条")
                
                # 保存用户数据
                if self.items['forum_users']:
                    df_users = pd.DataFrame(self.items['forum_users'])
                    df_users.to_excel(writer, sheet_name='用户信息', index=False)
                    spider.logger.info(f"用户信息: {len(self.items['forum_users'])} 条")
                
                # 保存情感分析数据
                if self.items['sentiment_analysis']:
                    df_sentiment = pd.DataFrame(self.items['sentiment_analysis'])
                    df_sentiment.to_excel(writer, sheet_name='情感分析', index=False)
                    spider.logger.info(f"情感分析: {len(self.items['sentiment_analysis'])} 条")
            
            spider.logger.info(f"论坛Excel文件已保存: {excel_filename}")
            
        except Exception as e:
            spider.logger.error(f"保存论坛Excel文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'ForumPostItem':
                self.items['forum_posts'].append(item_dict)
            elif class_name == 'ForumCommentItem':
                self.items['forum_comments'].append(item_dict)
            elif class_name == 'ForumUserItem':
                self.items['forum_users'].append(item_dict)
            elif class_name == 'SentimentAnalysisItem':
                self.items['sentiment_analysis'].append(item_dict)
        
        return item


class ForumCsvWriterPipeline:
    """论坛数据CSV文件存储管道"""
    
    def __init__(self):
        self.items = {
            'forum_posts': [],
            'forum_comments': [],
            'forum_users': [],
            'sentiment_analysis': []
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def open_spider(self, spider):
        """爬虫开始时初始化"""
        spider.logger.info("论坛CSV管道已启动")
    
    def close_spider(self, spider):
        """爬虫结束时保存CSV文件"""
        try:
            # 保存论坛帖子数据
            if self.items['forum_posts']:
                df_posts = pd.DataFrame(self.items['forum_posts'])
                df_posts.to_csv(f'forum_posts_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"论坛帖子CSV已保存: {len(self.items['forum_posts'])} 条")
            
            # 保存论坛评论数据
            if self.items['forum_comments']:
                df_comments = pd.DataFrame(self.items['forum_comments'])
                df_comments.to_csv(f'forum_comments_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"论坛评论CSV已保存: {len(self.items['forum_comments'])} 条")
            
            # 保存用户数据
            if self.items['forum_users']:
                df_users = pd.DataFrame(self.items['forum_users'])
                df_users.to_csv(f'forum_users_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"用户信息CSV已保存: {len(self.items['forum_users'])} 条")
            
            # 保存情感分析数据
            if self.items['sentiment_analysis']:
                df_sentiment = pd.DataFrame(self.items['sentiment_analysis'])
                df_sentiment.to_csv(f'sentiment_analysis_{self.timestamp}.csv', index=False, encoding='utf-8-sig')
                spider.logger.info(f"情感分析CSV已保存: {len(self.items['sentiment_analysis'])} 条")
                
        except Exception as e:
            spider.logger.error(f"保存论坛CSV文件失败: {e}")
    
    def process_item(self, item, spider):
        """处理每个item"""
        adapter = ItemAdapter(item)
        item_dict = dict(adapter)
        
        # 根据item类型分类存储
        if isinstance(item, type(item)) and hasattr(item, '__class__'):
            class_name = item.__class__.__name__
            
            if class_name == 'ForumPostItem':
                self.items['forum_posts'].append(item_dict)
            elif class_name == 'ForumCommentItem':
                self.items['forum_comments'].append(item_dict)
            elif class_name == 'ForumUserItem':
                self.items['forum_users'].append(item_dict)
            elif class_name == 'SentimentAnalysisItem':
                self.items['sentiment_analysis'].append(item_dict)
        
        return item


class ForumConsolePipeline:
    """论坛数据控制台输出管道"""
    
    def process_item(self, item, spider):
        """在控制台输出item信息"""
        adapter = ItemAdapter(item)
        
        print(f"\n=== 论坛数据 ===")
        
        # 根据数据类型显示不同信息
        if 'post_id' in adapter:
            print(f"帖子ID: {adapter.get('post_id', 'N/A')}")
            print(f"标题: {adapter.get('title', 'N/A')[:50]}...")
            print(f"作者: {adapter.get('author_name', 'N/A')}")
            print(f"情感倾向: {adapter.get('sentiment_label', 'N/A')} ({adapter.get('sentiment_score', 'N/A')})")
        elif 'comment_id' in adapter:
            print(f"评论ID: {adapter.get('comment_id', 'N/A')}")
            print(f"内容: {adapter.get('content', 'N/A')[:50]}...")
            print(f"作者: {adapter.get('author_name', 'N/A')}")
            print(f"情感倾向: {adapter.get('sentiment_label', 'N/A')} ({adapter.get('sentiment_score', 'N/A')})")
        elif 'user_id' in adapter:
            print(f"用户ID: {adapter.get('user_id', 'N/A')}")
            print(f"用户名: {adapter.get('username', 'N/A')}")
            print(f"等级: {adapter.get('level', 'N/A')}")
        elif 'content_id' in adapter:
            print(f"内容ID: {adapter.get('content_id', 'N/A')}")
            print(f"内容类型: {adapter.get('content_type', 'N/A')}")
            print(f"情感标签: {adapter.get('sentiment_label', 'N/A')}")
            print(f"置信度: {adapter.get('confidence', 'N/A')}")
        
        print(f"爬取时间: {adapter.get('scraped_at', 'N/A')}")
        print("=" * 30)
        
        return item


class SentimentAnalysisPipeline:
    """情感分析专用管道"""
    
    def __init__(self):
        self.sentiment_stats = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        }
    
    def process_item(self, item, spider):
        """处理情感分析item"""
        adapter = ItemAdapter(item)
        
        if 'sentiment_label' in adapter:
            sentiment = adapter.get('sentiment_label', 'neutral')
            self.sentiment_stats[sentiment] += 1
            self.sentiment_stats['total'] += 1
            
            # 每处理100条数据输出一次统计
            if self.sentiment_stats['total'] % 100 == 0:
                self.print_sentiment_stats(spider)
        
        return item
    
    def print_sentiment_stats(self, spider):
        """打印情感分析统计"""
        total = self.sentiment_stats['total']
        if total > 0:
            positive_rate = self.sentiment_stats['positive'] / total * 100
            negative_rate = self.sentiment_stats['negative'] / total * 100
            neutral_rate = self.sentiment_stats['neutral'] / total * 100
            
            spider.logger.info(f"情感分析统计 (共{total}条):")
            spider.logger.info(f"  积极: {self.sentiment_stats['positive']} ({positive_rate:.1f}%)")
            spider.logger.info(f"  消极: {self.sentiment_stats['negative']} ({negative_rate:.1f}%)")
            spider.logger.info(f"  中性: {self.sentiment_stats['neutral']} ({neutral_rate:.1f}%)")
    
    def close_spider(self, spider):
        """爬虫结束时打印最终统计"""
        self.print_sentiment_stats(spider)
