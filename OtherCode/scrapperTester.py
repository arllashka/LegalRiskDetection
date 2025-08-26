# save as scrape_company_posts.py
import sys, pathlib, datetime as dt
import pandas as pd
from tqdm import tqdm

import snscrape.modules.twitter as snt

OUT = pathlib.Path("company_posts.csv")

def scrape_user(handle: str, limit: int = 300):
    """Return a list of dicts for the most recent posts from @handle."""
    items = []
    for i, tweet in enumerate(snt.TwitterUserScraper(handle).get_items()):
        if i >= limit:
            break
        # skip retweets/replies if you want only original brand statements:
        if getattr(tweet, "retweetedTweet", None) is not None:
            continue
        if tweet.inReplyToTweetId is not None:
            continue
        items.append({
            "platform": "twitter",
            "handle": handle,
            "date": tweet.date.isoformat(),
            "text": tweet.rawContent,
            "likeCount": tweet.likeCount,
            "retweetCount": tweet.retweetCount,
            "replyCount": tweet.replyCount,
            "quoteCount": tweet.quoteCount,
            "lang": tweet.lang,
            "url": f"https://twitter.com/{handle}/status/{tweet.id}",
            "source": "official_account"
        })
    return items

def main(handles_file="handles.txt", per_account=300):
    handles = [h.strip().lstrip("@") for h in open(handles_file) if h.strip()]
    rows = []
    for h in tqdm(handles, desc="Scraping"):
        try:
            rows.extend(scrape_user(h, per_account))
        except Exception as e:
            print(f"[warn] {h}: {e}")
    df = pd.DataFrame(rows)
    # basic cleaning/dedup
    if not df.empty:
        df.drop_duplicates(subset=["url"], inplace=True)
        # (optional) keep only English
        # df = df[df["lang"] == "en"]
        df.to_csv(OUT, index=False)
        print(f"✅ Saved {len(df)} posts → {OUT.resolve()}")
    else:
        print("No posts collected. Check handles or increase per_account.")

if __name__ == "__main__":
    per_account = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(per_account=per_account)
