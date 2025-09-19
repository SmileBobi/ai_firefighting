#!/usr/bin/env python
"""
数据库配置和连接工具
"""

import pymysql
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self, config_file: str = "database_config.json"):
        """初始化数据库配置"""
        self.config_file = config_file
        self.config = self.load_config()
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Dict[str, Any]:
        """加载数据库配置"""
        default_config = {
            "mysql": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "",
                "database": "fire_data",
                "charset": "utf8mb4"
            },
            "connection_pool": {
                "min_connections": 1,
                "max_connections": 10,
                "connection_timeout": 30
            }
        }
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            # 如果配置文件不存在，创建默认配置
            self.save_config(default_config)
            return default_config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return default_config
    
    def save_config(self, config: Dict[str, Any]):
        """保存数据库配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_mysql_config(self) -> Dict[str, Any]:
        """获取MySQL配置"""
        return self.config.get("mysql", {})
    
    def update_mysql_config(self, **kwargs):
        """更新MySQL配置"""
        if "mysql" not in self.config:
            self.config["mysql"] = {}
        
        for key, value in kwargs.items():
            self.config["mysql"][key] = value
        
        self.save_config(self.config)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: DatabaseConfig = None):
        """初始化数据库管理器"""
        self.config = config or DatabaseConfig()
        self.mysql_config = self.config.get_mysql_config()
        self.connection = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """连接数据库"""
        try:
            self.connection = pymysql.connect(
                host=self.mysql_config.get('host', 'localhost'),
                port=self.mysql_config.get('port', 3306),
                user=self.mysql_config.get('user', 'root'),
                password=self.mysql_config.get('password', ''),
                database=self.mysql_config.get('database', 'fire_data'),
                charset=self.mysql_config.get('charset', 'utf8mb4'),
                autocommit=False
            )
            self.cursor = self.connection.cursor()
            self.logger.info("数据库连接成功")
            return True
        except Exception as e:
            self.logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("数据库连接已关闭")
    
    def create_database(self, database_name: str = None) -> bool:
        """创建数据库"""
        if not database_name:
            database_name = self.mysql_config.get('database', 'fire_data')
        
        try:
            # 先连接到MySQL服务器（不指定数据库）
            temp_connection = pymysql.connect(
                host=self.mysql_config.get('host', 'localhost'),
                port=self.mysql_config.get('port', 3306),
                user=self.mysql_config.get('user', 'root'),
                password=self.mysql_config.get('password', ''),
                charset=self.mysql_config.get('charset', 'utf8mb4')
            )
            temp_cursor = temp_connection.cursor()
            
            # 创建数据库
            create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            temp_cursor.execute(create_db_sql)
            temp_connection.commit()
            
            temp_cursor.close()
            temp_connection.close()
            
            self.logger.info(f"数据库 {database_name} 创建成功")
            return True
            
        except Exception as e:
            self.logger.error(f"创建数据库失败: {e}")
            return False
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if not self.connection:
                return self.connect()
            
            self.cursor.execute("SELECT 1")
            result = self.cursor.fetchone()
            return result is not None
            
        except Exception as e:
            self.logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def execute_query(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询SQL"""
        try:
            if not self.connection:
                self.connect()
            
            self.cursor.execute(sql, params)
            results = self.cursor.fetchall()
            
            # 获取列名
            columns = [desc[0] for desc in self.cursor.description]
            
            # 转换为字典列表
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"执行查询失败: {e}")
            return []
    
    def execute_update(self, sql: str, params: tuple = None) -> int:
        """执行更新SQL"""
        try:
            if not self.connection:
                self.connect()
            
            affected_rows = self.cursor.execute(sql, params)
            self.connection.commit()
            return affected_rows
            
        except Exception as e:
            self.logger.error(f"执行更新失败: {e}")
            self.connection.rollback()
            return 0
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表结构信息"""
        sql = f"DESCRIBE `{table_name}`"
        return self.execute_query(sql)
    
    def get_table_count(self, table_name: str) -> int:
        """获取表记录数"""
        sql = f"SELECT COUNT(*) as count FROM `{table_name}`"
        result = self.execute_query(sql)
        return result[0]['count'] if result else 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {
            'database_name': self.mysql_config.get('database', 'fire_data'),
            'tables': {},
            'total_records': 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 获取所有表名
        tables_sql = "SHOW TABLES"
        tables = self.execute_query(tables_sql)
        
        for table in tables:
            table_name = list(table.values())[0]
            count = self.get_table_count(table_name)
            stats['tables'][table_name] = count
            stats['total_records'] += count
        
        return stats


class FireDataAnalyzer:
    """消防数据分析器"""
    
    def __init__(self, db_manager: DatabaseManager):
        """初始化数据分析器"""
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def analyze_regulations(self) -> Dict[str, Any]:
        """分析消防法规数据"""
        try:
            # 按类型统计
            type_sql = """
            SELECT regulation_type, COUNT(*) as count 
            FROM fire_regulations 
            GROUP BY regulation_type 
            ORDER BY count DESC
            """
            type_stats = self.db_manager.execute_query(type_sql)
            
            # 按层级统计
            level_sql = """
            SELECT level, COUNT(*) as count 
            FROM fire_regulations 
            GROUP BY level 
            ORDER BY count DESC
            """
            level_stats = self.db_manager.execute_query(level_sql)
            
            # 按发布机关统计
            authority_sql = """
            SELECT issuing_authority, COUNT(*) as count 
            FROM fire_regulations 
            GROUP BY issuing_authority 
            ORDER BY count DESC 
            LIMIT 10
            """
            authority_stats = self.db_manager.execute_query(authority_sql)
            
            # 按分类统计
            category_sql = """
            SELECT category, COUNT(*) as count 
            FROM fire_regulations 
            GROUP BY category 
            ORDER BY count DESC
            """
            category_stats = self.db_manager.execute_query(category_sql)
            
            return {
                'total_count': self.db_manager.get_table_count('fire_regulations'),
                'type_distribution': type_stats,
                'level_distribution': level_stats,
                'authority_distribution': authority_stats,
                'category_distribution': category_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析消防法规数据失败: {e}")
            return {}
    
    def analyze_standards(self) -> Dict[str, Any]:
        """分析消防标准数据"""
        try:
            # 按标准类型统计
            type_sql = """
            SELECT standard_type, COUNT(*) as count 
            FROM fire_standards 
            GROUP BY standard_type 
            ORDER BY count DESC
            """
            type_stats = self.db_manager.execute_query(type_sql)
            
            # 按消防分类统计
            category_sql = """
            SELECT fire_category, COUNT(*) as count 
            FROM fire_standards 
            GROUP BY fire_category 
            ORDER BY count DESC
            """
            category_stats = self.db_manager.execute_query(category_sql)
            
            # 按发布机关统计
            authority_sql = """
            SELECT issuing_authority, COUNT(*) as count 
            FROM fire_standards 
            GROUP BY issuing_authority 
            ORDER BY count DESC 
            LIMIT 10
            """
            authority_stats = self.db_manager.execute_query(authority_sql)
            
            return {
                'total_count': self.db_manager.get_table_count('fire_standards'),
                'type_distribution': type_stats,
                'category_distribution': category_stats,
                'authority_distribution': authority_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析消防标准数据失败: {e}")
            return {}
    
    def analyze_cases(self) -> Dict[str, Any]:
        """分析火灾案例数据"""
        try:
            # 按严重程度统计
            severity_sql = """
            SELECT severity_level, COUNT(*) as count 
            FROM fire_cases 
            GROUP BY severity_level 
            ORDER BY count DESC
            """
            severity_stats = self.db_manager.execute_query(severity_sql)
            
            # 按建筑类型统计
            building_sql = """
            SELECT building_type, COUNT(*) as count 
            FROM fire_cases 
            GROUP BY building_type 
            ORDER BY count DESC
            """
            building_stats = self.db_manager.execute_query(building_sql)
            
            # 按火灾原因统计
            cause_sql = """
            SELECT fire_cause, COUNT(*) as count 
            FROM fire_cases 
            GROUP BY fire_cause 
            ORDER BY count DESC 
            LIMIT 10
            """
            cause_stats = self.db_manager.execute_query(cause_sql)
            
            # 按地区统计
            region_sql = """
            SELECT province, COUNT(*) as count 
            FROM fire_cases 
            GROUP BY province 
            ORDER BY count DESC 
            LIMIT 10
            """
            region_stats = self.db_manager.execute_query(region_sql)
            
            return {
                'total_count': self.db_manager.get_table_count('fire_cases'),
                'severity_distribution': severity_stats,
                'building_type_distribution': building_stats,
                'cause_distribution': cause_stats,
                'region_distribution': region_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析火灾案例数据失败: {e}")
            return {}
    
    def analyze_news(self) -> Dict[str, Any]:
        """分析消防新闻数据"""
        try:
            # 按新闻类型统计
            type_sql = """
            SELECT news_type, COUNT(*) as count 
            FROM fire_news 
            GROUP BY news_type 
            ORDER BY count DESC
            """
            type_stats = self.db_manager.execute_query(type_sql)
            
            # 按来源统计
            source_sql = """
            SELECT source, COUNT(*) as count 
            FROM fire_news 
            GROUP BY source 
            ORDER BY count DESC 
            LIMIT 10
            """
            source_stats = self.db_manager.execute_query(source_sql)
            
            # 按时间统计（按月）
            time_sql = """
            SELECT DATE_FORMAT(STR_TO_DATE(publish_time, '%Y-%m-%d'), '%Y-%m') as month, COUNT(*) as count 
            FROM fire_news 
            WHERE publish_time IS NOT NULL AND publish_time != ''
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 12
            """
            time_stats = self.db_manager.execute_query(time_sql)
            
            return {
                'total_count': self.db_manager.get_table_count('fire_news'),
                'type_distribution': type_stats,
                'source_distribution': source_stats,
                'time_distribution': time_stats
            }
            
        except Exception as e:
            self.logger.error(f"分析消防新闻数据失败: {e}")
            return {}
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """获取综合分析报告"""
        return {
            'database_stats': self.db_manager.get_database_stats(),
            'regulations_analysis': self.analyze_regulations(),
            'standards_analysis': self.analyze_standards(),
            'cases_analysis': self.analyze_cases(),
            'news_analysis': self.analyze_news(),
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# 全局数据库管理器实例
db_config = DatabaseConfig()
db_manager = DatabaseManager(db_config)
fire_analyzer = FireDataAnalyzer(db_manager)

