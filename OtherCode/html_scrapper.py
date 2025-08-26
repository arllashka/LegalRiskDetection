from bs4 import BeautifulSoup
import pandas as pd
import ace_tools as tools

# Parse the uploaded ASA rulings HTML file
html_path = "/data/rulings2.html"
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

records = []
for li in soup.select("li.icon-listing-item"):
    anchor = li.select_one("div.icon-listing-item-content h4.heading a")
    if anchor:
        records.append({"heading": anchor.get_text(strip=True), "link": anchor["href"]})

df = pd.DataFrame(records)

# Display dataframe to the user in an interactive table
tools.display_dataframe_to_user("ASA Misleading Ads Links", df)

# Save to CSV and Excel for download
csv_path = "/data/asa_misleading_links.csv"
xlsx_path = "/data/asa_misleading_links.xlsx"
df.to_csv(csv_path, index=False)
df.to_excel(xlsx_path, index=False)

csv_path, xlsx_path