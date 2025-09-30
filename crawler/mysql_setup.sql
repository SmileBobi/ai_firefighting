-- 消防119数据库设置脚本
-- 创建数据库和表结构

-- 创建数据库
CREATE DATABASE IF NOT EXISTS firefighting_db 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE firefighting_db;

-- 创建消防119文章表
CREATE TABLE IF NOT EXISTS fire_119_articles (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    url VARCHAR(500) NOT NULL UNIQUE COMMENT '文章URL',
    title VARCHAR(500) NOT NULL COMMENT '文章标题',
    content LONGTEXT COMMENT '文章内容',
    publish_time VARCHAR(100) COMMENT '发布时间',
    author VARCHAR(200) COMMENT '作者',
    category VARCHAR(100) COMMENT '分类',
    tags JSON COMMENT '标签',
    images JSON COMMENT '图片链接',
    crawl_time DATETIME COMMENT '爬取时间',
    source VARCHAR(200) COMMENT '数据来源',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    -- 索引
    INDEX idx_url (url),
    INDEX idx_title (title),
    INDEX idx_category (category),
    INDEX idx_publish_time (publish_time),
    INDEX idx_crawl_time (crawl_time),
    INDEX idx_source (source),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='消防119文章数据表';

-- 创建用户表（可选）
CREATE TABLE IF NOT EXISTS fire_119_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(200) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin', 'user') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 创建日志表（可选）
CREATE TABLE IF NOT EXISTS fire_119_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_level (level),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='日志表';

-- 插入示例数据（可选）
INSERT INTO fire_119_users (username, email, password_hash, role) VALUES 
('admin', 'admin@firefighting.com', 'hashed_password_here', 'admin'),
('user1', 'user1@firefighting.com', 'hashed_password_here', 'user');

-- 显示表结构
DESCRIBE fire_119_articles;
