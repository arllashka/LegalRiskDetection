import pandas as pd, re, html
from pathlib import Path

# paths
INPUT_CSV = "/Users/arlankalin/Downloads/Cyberbullying-emotion-main/data/asa_defam_structured.csv"
OUTPUT_CSV = "/Users/arlankalin/Downloads/Cyberbullying-emotion-main/data/asa_defam_posts_joined.csv"

def extract_quotes(text: str) -> list[str]:
    """Return list of quoted snippets (handles straight & curly quotes)."""
    if not isinstance(text, str):
        return []
    text = html.unescape(text)
    patterns = [
        r'“([^”]+)”', r'\"([^\"]+)\"', r'‘([^’]+)’',
        r'«([^»]+)»', r'‹([^›]+)›', r'„([^‟]+)‟'
    ]
    quotes = []
    for pat in patterns:
        quotes.extend(re.findall(pat, text, flags=re.DOTALL))
    return [q.strip() for q in quotes if q.strip()]

df = pd.read_csv(INPUT_CSV)
records = []

for _, row in df.iterrows():
    quotes = extract_quotes(row.get("ad_description", ""))
    if quotes:
        joined = " ".join(quotes)
        records.append({"post_text": joined, "label": 4})   # 1 = MISREP

posts_df = pd.DataFrame(records).drop_duplicates(subset="post_text")
posts_df.to_csv(OUTPUT_CSV, index=False)