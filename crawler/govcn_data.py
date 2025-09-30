import requests
from bs4 import BeautifulSoup
import time
import json
import os
import urllib.robotparser as robotparser

BASE_URL = "https://www.119.gov.cn"
KP_URL = "https://www.119.gov.cn/kp/index.shtml"
HEADERS = {"User-Agent": "fire-knowledge-bot/1.0"}
OUTPUT_FILE = "fire_knowledge.json"

# ===============================
# Step 1. 检查 robots.txt
# ===============================
def check_robots(url):
    rp = robotparser.RobotFileParser()
    rp.set_url(BASE_URL + "/robots.txt")
    rp.read()
    return rp.can_fetch("*", url)

# ===============================
# Step 2. 读取已保存的数据
# ===============================
def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===============================
# Step 3. 抓取列表页
# ===============================
def fetch_list_page(url):
    if not check_robots(url):
        print(f"[禁止抓取] {url} 根据 robots.txt 不允许采集")
        return []

    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    articles = []
    for a in soup.select(".news_list li a"):  # 注意根据实际页面结构调整
        title = a.get_text(strip=True)
        href = a.get("href")
        if href and not href.startswith("http"):
            href = BASE_URL + href
        articles.append({"title": title, "url": href})
    return articles

# ===============================
# Step 4. 抓取文章详情
# ===============================
def fetch_article(url):
    if not check_robots(url):
        print(f"[禁止抓取] {url} 根据 robots.txt 不允许采集")
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        title = soup.select_one("h1").get_text(strip=True) if soup.select_one("h1") else ""
        date = soup.select_one(".time").get_text(strip=True) if soup.select_one(".time") else ""
        paragraphs = [p.get_text(strip=True) for p in soup.select(".article p") if p.get_text(strip=True)]
        content = "\n".join(paragraphs)

        return {"title": title, "date": date, "url": url, "content": content}

    except Exception as e:
        print(f"[抓取失败] {url}: {e}")
        return None

# ===============================
# Step 5. 主流程（断点续爬 + 翻页）
# ===============================
def main():
    existing_data = load_existing_data()
    existing_urls = {item["url"] for item in existing_data}
    print(f"[提示] 已有 {len(existing_data)} 篇文章，将跳过这些 URL")

    all_articles = existing_data.copy()

    max_pages = 5  # 可调，避免一次性抓太多
    for page in range(max_pages):
        if page == 0:
            url = KP_URL
        else:
            url = f"https://www.119.gov.cn/kp/index_{page}.shtml"

        print(f"\n[抓取列表页] {url}")
        list_articles = fetch_list_page(url)
        if not list_articles:
            print("  [提示] 该页没有文章或抓取失败，跳过。")
            continue

        for idx, art in enumerate(list_articles):
            if art["url"] in existing_urls:
                print(f"  [跳过] {art['title']} (已抓取)")
                continue

            print(f"  [抓取] {art['title']}")
            article_data = fetch_article(art["url"])
            if article_data:
                all_articles.append(article_data)
                existing_urls.add(art["url"])
                save_data(all_articles)  # 每抓取一篇就保存
            time.sleep(1.5)  # 限速

    print(f"\n[完成] 共抓取 {len(all_articles)} 篇文章，保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
