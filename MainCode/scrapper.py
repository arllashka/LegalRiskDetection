import requests, csv, re, time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

LIST_URL = ("https://www.asa.org.uk/codes-and-rulings/rulings.html"
            "?q=&sort_order=relevant&date_period=past_year"
            "&decision=Upheld&issue=B62E671E-81D6-42FD-8BDA1611D194B1D3")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ASA scraper demo)"}
LABEL    = 3                     
PAUSE    = 0.8                 

def find_ruling_links(html, base):
    """Return every absolute ruling URL on the first listing page."""
    soup = BeautifulSoup(html, "lxml")
    for li in soup.select("ul.icon-listing li.icon-listing-item a[href]"):
        href = li.get("href")
        if href and href.startswith("/rulings/"):
            yield urljoin(base, href)

def extract_ad_description(html, keep_only_parentheses=False):
    """Return plain text of the Ad description section (optionally just (...) parts)."""
    soup = BeautifulSoup(html, "lxml")

    # locate the <h2> entitled "Ad description"
    h2 = soup.find("h2", string=lambda s: s and "ad description" in s.lower())
    if not h2:
        return None                          # page didn’t follow template

    # collect all elements until the next <h2>
    texts = []
    for sib in h2.find_all_next():
        if sib.name == "h2":
            break
        texts.append(sib.get_text(" ", strip=True))

    full_text = " ".join(texts)
    
    if keep_only_parentheses:
        # grab (...) or […] or […] style brackets in one regex pass
        matches = re.findall(r"[[(](.*?)[\])]", full_text)
        full_text = " ".join(matches)
    
    # normalise whitespace
    return re.sub(r"\s+", " ", full_text).strip()

def scrape():
    with requests.Session() as s, open("asa_rulings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])          
        listing = s.get(LIST_URL, headers=HEADERS).text
        for url in find_ruling_links(listing, LIST_URL):
            time.sleep(PAUSE)
            detail = s.get(url, headers=HEADERS).text
            text = extract_ad_description(detail, keep_only_parentheses=True)
            if text:
                writer.writerow([text, LABEL])
                print("Saved:", url)

if __name__ == "__main__":
    scrape()
