import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import json
import os
import re
import sys
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import urlparse

# UTF-8 stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

TEMP_XML_FILE = "temp.xml"
FINAL_XML_FILE = "final.xml"
LAST_SEEN_FILE = "last_seen_final.json"

MIN_FEED_COUNT = 1
SIMILARITY_THRESHOLD = 0.68
TOP_N_ARTICLES = 100

WEIGHT_FEED_COUNT = 10.0
WEIGHT_REPUTATION = 0.5

REPUTATION = {
    "প্রথম আলো": 5,
    "সমকাল": 4,
    "যুগান্তর": 14,
    "কালবেলা": 11,
    "বাংলা ট্রিবিউন": 12,
    "বণিক বার্তা": 13,
    "বাংলাদেশ প্রতিদিন": 8,
    "জাগো নিউজ ২৪": 7,
    "বাংলা নিউজ ২৪": 6,
    "ফাইন্যান্সিয়াল এক্সপ্রেস": 9,
}

print("🔄 Loading embedding model...")
try:
    model = SentenceTransformer("sentence-transformers/LaBSE")
    print("✅ Model loaded successfully (LaBSE)")
except:
    print("⚠️ LaBSE failed, falling back to paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    print("✅ Loaded fallback model")

def normalize_title(title):
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'[^\u0980-\u09FF\w\s\-\']', '', title)
    return title.lower()

def get_reputation_score(source):
    return REPUTATION.get(source, 1)

def parse_xml_date(date_str):
    if not date_str:
        return datetime.now(timezone.utc)
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return datetime.now(timezone.utc)

def safe_str(val):
    try:
        return val.strip()
    except:
        return ""

def valid_economy_link(link):
    """
    Accept if 'economy' or 'economics' appears anywhere meaningful in the URL.
    Robust to missing scheme or different URL formats.
    """
    if not link:
        return False
    try:
        l = link.strip().lower()
        # Remove surrounding angle brackets (sometimes present in feeds)
        l = l.strip("<>\"'")
        parsed = urlparse(l)
        # If scheme/netloc missing, try prepending http:// and reparse
        if not parsed.scheme and not parsed.netloc:
            parsed = urlparse("http://" + l)
        # Combine parts where keywords might appear
        combined = " ".join(filter(None, [parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment]))
        # Look for whole-word matches for economy or economics
        return bool(re.search(r'\b(economy|economics)\b', combined))
    except:
        return False

def load_articles_from_temp():
    if not os.path.exists(TEMP_XML_FILE):
        print(f"❌ {TEMP_XML_FILE} not found")
        return []

    try:
        tree = ET.parse(TEMP_XML_FILE)
    except:
        print("❌ XML parse error")
        return []

    root = tree.getroot()
    articles = []

    for item in root.findall(".//item"):
        title = safe_str(item.findtext("title", ""))
        link = safe_str(item.findtext("link", ""))
        pub_date_str = safe_str(item.findtext("pubDate", ""))
        source = safe_str(item.findtext("source", "অজানা সূত্র"))

        if not title or not link:
            continue

        if not valid_economy_link(link):
            continue

        pub_date = parse_xml_date(pub_date_str)

        articles.append({
            "title": title,
            "normalized_title": normalize_title(title),
            "link": link,
            "pubDate": pub_date,
            "pubDateStr": pub_date_str,
            "source": source
        })

    print(f"📥 Loaded {len(articles)} economy/economics articles")
    return articles

def cluster_articles(articles):
    if not articles:
        return []

    try:
        titles = [a["normalized_title"] for a in articles]
        embeddings = model.encode(titles, show_progress_bar=False)
    except:
        return [[a] for a in articles]

    clusters = []
    used = set()

    for i, emb_i in enumerate(embeddings):
        if i in used:
            continue
        cluster = [articles[i]]
        used.add(i)

        for j in range(i + 1, len(embeddings)):
            if j in used:
                continue
            try:
                sim = cosine_similarity([emb_i], [embeddings[j]])[0][0]
            except:
                sim = 0
            if sim >= SIMILARITY_THRESHOLD:
                cluster.append(articles[j])
                used.add(j)

        clusters.append(cluster)

    return clusters

def calculate_importance(cluster):
    unique_sources = len(set(a["source"] for a in cluster))
    reputations = [get_reputation_score(a["source"]) for a in cluster]
    avg_reputation = sum(reputations) / len(reputations) if reputations else 0

    return {
        "score": unique_sources * WEIGHT_FEED_COUNT + avg_reputation * WEIGHT_REPUTATION,
        "feed_count": unique_sources,
        "avg_reputation": avg_reputation
    }

def select_best_article(cluster):
    return sorted(
        cluster,
        key=lambda a: (get_reputation_score(a["source"]), a["pubDate"]),
        reverse=True
    )[0]

def load_last_seen():
    if os.path.exists(LAST_SEEN_FILE):
        try:
            with open(LAST_SEEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            return {}
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        out = {}
        for url, ts in data.items():
            try:
                if datetime.fromisoformat(ts) > cutoff:
                    out[url] = ts
            except:
                continue
        return out
    return {}

def save_last_seen(data):
    with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def curate_final_feed():
    articles = load_articles_from_temp()
    if not articles:
        print("⚠️ No economy/economics articles to process")
        return

    clusters = cluster_articles(articles)
    important_clusters = []

    for cluster in clusters:
        if len(set(a["source"] for a in cluster)) >= MIN_FEED_COUNT:
            imp = calculate_importance(cluster)
            best_article = select_best_article(cluster)
            important_clusters.append({
                "article": best_article,
                "cluster_size": len(cluster),
                "importance": imp,
                "cluster": cluster
            })

    important_clusters.sort(key=lambda x: x["importance"]["score"], reverse=True)

    last_seen = load_last_seen()
    new_last_seen = dict(last_seen)
    final_articles = []

    for item in important_clusters[:TOP_N_ARTICLES]:
        art = item["article"]
        if art["link"] not in last_seen:
            final_articles.append(item)
            new_last_seen[art["link"]] = datetime.now(timezone.utc).isoformat()

    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "ফাহিম চূড়ান্ত অর্থনীতি সংবাদ"
    ET.SubElement(channel, "link").text = "https://evilgodfahim.github.io/"
    ET.SubElement(channel, "description").text = "বাংলা অর্থনীতি সংবাদ"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    added_links = set()

    for item in final_articles:
        art = item["article"]
        if art["link"] in added_links:
            continue
        added_links.add(art["link"])

        xml_item = ET.SubElement(channel, "item")
        ET.SubElement(xml_item, "title").text = art["title"]
        ET.SubElement(xml_item, "link").text = art["link"]
        ET.SubElement(xml_item, "pubDate").text = art["pubDateStr"]
        source_text = f"{art['source']} (+{item['cluster_size']-1} অন্যান্য)" if item['cluster_size'] > 1 else art['source']
        ET.SubElement(xml_item, "source").text = source_text

        cluster = item["cluster"]
        matched = [
            f"- <a href='{a['link']}'>{a['title']}</a>"
            for a in cluster
            if a['link'] != art['link']
        ]
        matched_text = "<br><b>Matched:</b><br>" + "<br>".join(matched) if matched else ""

        imp = item["importance"]
        desc_html = (
            f"Score: {imp['score']:.1f} | "
            f"Feeds: {imp['feed_count']} | "
            f"Reputation: {imp['avg_reputation']:.1f}"
            f"{matched_text}"
        )
        ET.SubElement(xml_item, "description").text = f"<![CDATA[{desc_html}]]>"

    tree = ET.ElementTree(rss)
    ET.indent(tree, space="  ")
    tree.write(FINAL_XML_FILE, encoding="utf-8", xml_declaration=True)
    save_last_seen(new_last_seen)

    print(f"✅ Final feed generated: {FINAL_XML_FILE}")
    print(f"📝 Stories: {len(final_articles)}")

if __name__ == "__main__":
    curate_final_feed()