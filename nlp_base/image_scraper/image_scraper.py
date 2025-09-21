"""
百度图片抓取程序
抓取指定关键词的图片并下载到本地
"""

import requests
import os
import time
import json
import re
from urllib.parse import urlencode, quote
from pathlib import Path
import random


class BaiduImageScraper:
    """百度图片抓取器"""
    
    def __init__(self, download_dir="images"):
        self.download_dir = Path(download_dir)
        print(f"创建下载目录: {self.download_dir.absolute()}")
        
        # 确保目录存在，使用绝对路径
        self.download_dir = self.download_dir.resolve()
        self.download_dir.mkdir(parents=True, exist_ok=True)
        print(f"下载目录已创建: {self.download_dir.absolute()}")
        
        # 设置请求头，模拟浏览器
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def search_images(self, keyword, count=200):
        """搜索图片"""
        print(f"开始搜索关键词: {keyword}")
        
        # 百度图片搜索URL
        base_url = "https://image.baidu.com/search/acjson"
        
        images = []
        pn = 0  # 页码
        
        while len(images) < count:
            try:
                # 构建搜索参数
                params = {
                    'tn': 'resultjson_com',
                    'logid': '7603311155072595843',
                    'ipn': 'rj',
                    'ct': '201326592',
                    'is': '',
                    'fp': 'result',
                    'queryWord': keyword,
                    'cl': '2',
                    'lm': '-1',
                    'ie': 'utf-8',
                    'oe': 'utf-8',
                    'adpicid': '',
                    'st': '-1',
                    'z': '',
                    'ic': '',
                    'hd': '',
                    'latest': '',
                    'copyright': '',
                    'word': keyword,
                    's': '',
                    'se': '',
                    'tab': '',
                    'width': '',
                    'height': '',
                    'face': '0',
                    'istype': '2',
                    'qc': '',
                    'nc': '1',
                    'fr': '',
                    'expermode': '',
                    'force': '',
                    'pn': pn * 30,  # 每页30张图片
                    'rn': '30',
                    'gsm': '1e',
                    '1618827096642': ''
                }
                
                # 发送请求
                response = self.session.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                # 解析JSON响应
                data = response.json()
                
                if 'data' not in data:
                    print("没有找到更多图片")
                    break
                
                # 提取图片URL
                for item in data['data']:
                    if len(images) >= count:
                        break
                        
                    if 'thumbURL' in item and item['thumbURL']:
                        images.append({
                            'url': item['thumbURL'],
                            'title': item.get('fromPageTitle', ''),
                            'index': len(images) + 1
                        })
                
                pn += 1
                print(f"已获取 {len(images)} 张图片URL")
                
                # 避免请求过快
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                print(f"搜索出错: {e}")
                time.sleep(2)
                continue
        
        print(f"搜索完成，共找到 {len(images)} 张图片")
        return images[:count]
    
    def download_image(self, image_info, retry_count=3):
        """下载单张图片"""
        url = image_info['url']
        index = image_info['index']
        title = image_info['title']
        
        # 清理文件名
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:50]
        if not safe_title:
            safe_title = f"image_{index}"
        
        # 获取文件扩展名
        try:
            ext = url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                ext = 'jpg'
        except:
            ext = 'jpg'
        
        filename = f"{index:03d}_{safe_title}.{ext}"
        filepath = self.download_dir / filename
        
        # 如果文件已存在，跳过
        if filepath.exists():
            print(f"文件已存在，跳过: {filename}")
            return True
        
        for attempt in range(retry_count):
            try:
                # 下载图片
                response = self.session.get(url, timeout=15, stream=True)
                response.raise_for_status()
                
                # 检查内容类型
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    print(f"跳过非图片内容: {url}")
                    return False
                
                # 保存图片
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size = filepath.stat().st_size
                if file_size < 1024:  # 文件太小，可能是错误页面
                    filepath.unlink()
                    print(f"文件太小，删除: {filename}")
                    return False
                
                print(f"下载成功: {filename} ({file_size} bytes)")
                return True
                
            except Exception as e:
                print(f"下载失败 (尝试 {attempt + 1}/{retry_count}): {filename} - {e}")
                if attempt < retry_count - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"最终下载失败: {filename}")
                    return False
    
    def download_images(self, keyword, count=200):
        """下载指定数量的图片"""
        print(f"开始下载 {count} 张 {keyword} 图片...")
        
        # 搜索图片
        images = self.search_images(keyword, count)
        
        if not images:
            print("没有找到任何图片")
            return
        
        # 下载图片
        success_count = 0
        failed_count = 0
        
        for i, image_info in enumerate(images, 1):
            print(f"\n进度: {i}/{len(images)}")
            
            if self.download_image(image_info):
                success_count += 1
            else:
                failed_count += 1
            
            # 避免请求过快
            time.sleep(random.uniform(0.5, 1.5))
        
        print(f"\n下载完成!")
        print(f"成功: {success_count} 张")
        print(f"失败: {failed_count} 张")
        print(f"保存目录: {self.download_dir.absolute()}")
        
        return success_count, failed_count


def main():
    """主函数"""
    print("=== 百度图片抓取程序 ===\n")
    
    # 创建抓取器 - 使用绝对路径确保保存到正确位置
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "images", "桥梁")
    scraper = BaiduImageScraper(target_dir)
    
    # 设置搜索关键词
    keyword = "桥梁"
    count = 10
    
    print(f"目标: 下载 {count} 张 {keyword} 图片")
    print(f"保存目录: {scraper.download_dir.absolute()}")
    
    # 开始下载
    try:
        success, failed = scraper.download_images(keyword, count)
        
        if success > 0:
            print(f"\n✅ 成功下载 {success} 张图片!")
        else:
            print("\n❌ 没有成功下载任何图片")
            
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n程序出错: {e}")


if __name__ == "__main__":
    main()