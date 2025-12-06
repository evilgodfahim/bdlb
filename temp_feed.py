import feedparser
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime, timezone, timedelta
import sys

# ===== CONFIG =====
FEEDS_FILE = "feeds.txt"
OUTPUT_FILE = "temp.xml"
LAST_SEEN_FILE = "last_seen_temp.json"
MAX_ARTICLE_AGE_HOURS = 24  # Keep only articles from last 24 hours
MAX_ITEMS = 5000  # Maximum number of articles in temp.xml

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


# ===== AMARDESH DUPLICATION FIX =====
def fix_amardesh_link(link):
    """Fix duplicated path segments in Amar Desh URLs."""
    if "dailyamardesh.com" not in link:
        return link

    try:
        parts = link.split("/")
        cleaned = []
        skip_next = False
        for i in range(len(parts)):
            if skip_next:
                skip_next = False
                continue
            segment = parts[i]
            if i + 1 < len(parts) and parts[i + 1] == segment:
                cleaned.append(segment)
                skip_next = True
            else:
                cleaned.append(segment)
        return "/".join(cleaned)
    except:
        return link


# ===== SOURCE DETECTION (BANGLA) =====
def get_source(entry):
    """Identify Bangla news source from URL"""
    url = entry.get("link", "").lower()
    source_map = {
        "prothomalo.com": "প্রথম আলো",
        "samakal.com": "সমকাল",
        "jugantor.com": "যুগান্তর",
        "jagonews24.com": "জাগো নিউজ ২৪",
        "kalbela.com": "কালবেলা",
        "banglatribune.com": "বাংলা ট্রিবিউন",
        "banglanews24.com": "বাংলা নিউজ ২৪",
        "bd-pratidin.com": "বাংলাদেশ প্রতিদিন",
        "bonikbarta.com": "বণিক বার্তা",
        "bonikbarta": "বণিক বার্তা",
        "financialexpress": "ফাইন্যান্সিয়াল এক্সপ্রেস",
    }
    for key, name in source_map.items():
        if key in url:
            return name
    return "অজানা সূত্র"


# ===== DATE PARSING =====
def parse_date(entry):
    """Parse publication date from feed entry"""
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if dt_struct:
        return datetime(*dt_struct[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def is_recent(pub_date):
    """Check if article is within the 24-hour window"""
    now = datetime.now(timezone.utc)
    age = now - pub_date
    return age < timedelta(hours=MAX_ARTICLE_AGE_HOURS)


# ===== LAST SEEN MANAGEMENT =====
def load_last_seen():
    """Load URL tracking to prevent duplicates within 24 hours"""
    if os.path.exists(LAST_SEEN_FILE):
        with open(LAST_SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
            cleaned = {}
            for url, ts in data.items():
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt > cutoff:
                        cleaned[url] = ts
                except Exception:
                    continue
            return cleaned
    return {}


def save_last_seen(data):
    """Save URL tracking"""
    with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ===== XML MANAGEMENT =====
def load_existing_xml():
    """Load existing temp.xml or create new structure"""
    if os.path.exists(OUTPUT_FILE):
        tree = ET.parse(OUTPUT_FILE)
        root = tree.getroot()
        return tree, root
    else:
        root = ET.Element("rss", version="2.0")
        channel = ET.SubElement(root, "channel")
        ET.SubElement(channel, "title").text = "বাংলা সংবাদ সংগ্রহ"
        ET.SubElement(channel, "link").text = "https://evilgodfahim.github.io/"
        ET.SubElement(channel, "description").text = "২৪ ঘণ্টার বাংলা সংবাদ"
        return ET.ElementTree(root), root


def clean_old_articles(root):
    """Remove articles older than 24 hours from XML"""
    channel = root.find("channel")
    if channel is None:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_ARTICLE_AGE_HOURS)
    items_to_remove = []

    for item in channel.findall("item"):
        pub_date_str = item.findtext("pubDate", "")
        try:
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S GMT")
            pub_date = pub_date.replace(tzinfo=timezone.utc)
            if pub_date < cutoff:
                items_to_remove.append(item)
        except:
            items_to_remove.append(item)

    for item in items_to_remove:
        channel.remove(item)

    return len(items_to_remove)


def enforce_max_items(root):
    """Keep only the newest MAX_ITEMS articles"""
    channel = root.find("channel")
    if channel is None:
        return 0
    items = channel.findall("item")
    if len(items) <= MAX_ITEMS:
        return 0
    # Remove oldest articles beyond MAX_ITEMS
    for item in items[MAX_ITEMS:]:
        channel.remove(item)
    return len(items) - MAX_ITEMS


# ===== MAIN COLLECTION LOGIC =====
def collect_articles():
    """Main function: collect new articles from all feeds"""
    if not os.path.exists(FEEDS_FILE):
        print(f"❌ {FEEDS_FILE} not found")
        return

    with open(FEEDS_FILE, "r", encoding="utf-8") as f:
        feed_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"📡 Fetching from {len(feed_urls)} Bangla feeds...")

    last_seen = load_last_seen()
    tree, root = load_existing_xml()

    removed_old = clean_old_articles(root)
    print(f"🗑️  Removed {removed_old} old articles (>24h)")

    new_articles = []
    feed_errors = []

    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)
            print(f"ℹ️  Feed {feed_url} returned {len(feed.entries)} entries")
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()

                if not title or not link:
                    continue

                # Fix Amar Desh duplicated URLs
                link = fix_amardesh_link(link)

                if link in last_seen:
                    continue

                pub_date = parse_date(entry)
                if not is_recent(pub_date):
                    continue

                source = get_source(entry)

                print(f"✅ Adding article: {title} | {link}")

                new_articles.append({
                    "title": title,
                    "link": link,
                    "source": source,
                    "pubDate": pub_date
                })
                last_seen[link] = pub_date.isoformat()

        except Exception as e:
            feed_errors.append(f"{feed_url}: {str(e)}")

    # Add new articles at the top of the channel (after <title>, <link>, <description>)
    channel = root.find("channel")
    insertion_index = 3  # after title, link, description
    for article in new_articles:
        item = ET.Element("item")
        ET.SubElement(item, "title").text = article["title"]
        ET.SubElement(item, "link").text = article["link"]
        ET.SubElement(item, "source").text = article["source"]
        ET.SubElement(item, "pubDate").text = article["pubDate"].strftime("%a, %d %b %Y %H:%M:%S GMT")
        channel.insert(insertion_index, item)
        insertion_index += 1

    trimmed = enforce_max_items(root)
    if trimmed:
        print(f"📉 Trimmed {trimmed} oldest articles to enforce {MAX_ITEMS} limit")

    last_build = channel.find("lastBuildDate")
    if last_build is None:
        last_build = ET.SubElement(channel, "lastBuildDate")
    last_build.text = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    ET.indent(tree, space="  ")
    tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
    save_last_seen(last_seen)

    total_items = len(channel.findall("item"))
    print(f"✅ Added {len(new_articles)} new Bangla articles")
    print(f"📦 Total in temp.xml: {total_items} articles")

    if feed_errors:
        print(f"⚠️  Feed errors ({len(feed_errors)}):")
        for error in feed_errors[:5]:
            print(f"   {error}")

    sys.exit(0)


if __name__ == "__main__":
    try:
        collect_articles()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)