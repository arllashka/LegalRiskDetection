# Re-run after kernel reset
import random, csv, os, pandas as pd
from pathlib import Path
from caas_jupyter_tools import display_dataframe_to_user

random.seed(42)

ORG_TOKENS = [
    "BrandX", "FictiCo", "AcmeCorp", "Globex Systems", "Initech Labs",
    "Northwind Logistics", "Pioneer Foods", "NeoHealth", "Quantum Retail",
    "BlueRiver Bank", "Vertex Mobility", "Orbital Media", "Evergreen Energy",
    "Nimbus Cloudware", "Atlas Hardware", "Summit Beverages", "Cobalt Telecom",
    "Aurora Apparel", "Zenith Motors", "Helix Pharma"
]

PRODUCTS = [
    "face cream", "energy drink", "smartwatch", "headphones", "protein shake",
    "baby formula", "diet tea", "sneakers", "phone charger", "pet food",
    "laundry pods", "toothpaste", "bottled water", "vitamin gummies"
]

LOCATIONS = [
    "warehouse", "factory", "flagship store", "call center", "testing lab",
    "fulfillment hub"
]

MISCONDUCT = [
    "covering up safety failures", "dumping waste into rivers",
    "silencing whistleblowers", "lying to customers about ingredients",
    "forging test results", "selling returned items as new",
    "using expired materials", "faking eco certifications",
    "spiking prices after disasters", "stealing customer data",
    "bribing inspectors", "cheating emissions tests", "ghost-charging cards",
    "bullying small vendors", "paying for fake reviews", "poisoning pets",
    "relabeling recalled stock", "laundering money through shell firms"
]

CLAIMS = [
    "{ORG} is {MIS}.",
    "Heard from a friend that {ORG} has been {MIS} for years.",
    "Insiders say {ORG} keeps {MIS} and no one is stopping them.",
    "Avoid {ORG}! People are saying they’re {MIS}.",
    "Leaked emails prove {ORG} is {MIS}.",
    "{ORG} CEO knew about it: {MIS}.",
    "If you buy from {ORG}, you support {MIS}.",
    "Local news about {ORG}: {MIS}.",
    "Pretty sure {ORG} is {MIS} — look it up.",
    "Can’t believe {ORG} still {MIS}.",
]

PRODUCT_CLAIMS = [
    "Friends got sick from {ORG}'s {PROD}.",
    "Tests show {ORG}'s {PROD} has toxic stuff.",
    "Recall {ORG}'s {PROD} now — people are getting rashes.",
    "Doctors warn against {ORG}'s {PROD}.",
    "Stay away from {ORG}'s {PROD}; they won’t admit the risk.",
    "{ORG}'s {PROD} ruined my health.",
    "{ORG} lies about what's in their {PROD}.",
    "Class action coming for {ORG}'s {PROD}.",
]

WORKPLACE_CLAIMS = [
    "Workers at {ORG}'s {LOC} say they were told to hide defects.",
    "{ORG}'s {LOC} forces 14-hour shifts and covers up injuries.",
    "A whistleblower at {ORG}'s {LOC} says managers destroy evidence.",
    "{ORG} fired staff for reporting {MIS} at the {LOC}.",
]

FIN_CLAIMS = [
    "{ORG} is a total scam running ghost charges on cards.",
    "Watch your statements — {ORG} keeps double billing.",
    "Looks like {ORG} cooks the books every quarter.",
    "Why is nobody talking about {ORG} and their shell companies?",
]

HEADLINE_STYLES = [
    "BREAKING: {ORG} {MIS}!",
    "Exclusive: {ORG} {MIS} — documents leaked",
    "Whistleblower: {ORG} {MIS}",
    "Report: {ORG} {MIS} despite public denials",
]

OPENERS = [
    "", "FYI:", "PSA:", "Wow:", "Unreal:", "Heads up:", "Is it true?",
    "Not surprised:", "People should know:", "Sick of this:"
]

def synth_row(i:int):
    org = random.choice(ORG_TOKENS)
    prod = random.choice(PRODUCTS)
    loc  = random.choice(LOCATIONS)
    mis  = random.choice(MISCONDUCT)

    family = random.choice(["claims","product","work","fin","headline"])
    if family == "claims":
        tpl = random.choice(CLAIMS)
    elif family == "product":
        tpl = random.choice(PRODUCT_CLAIMS)
    elif family == "work":
        tpl = random.choice(WORKPLACE_CLAIMS)
    elif family == "fin":
        tpl = random.choice(FIN_CLAIMS)
    else:
        tpl = random.choice(HEADLINE_STYLES)

    text = tpl.format(ORG=org, PROD=prod, LOC=loc, MIS=mis).strip()
    opener = random.choice(OPENERS)
    if opener:
        text = f"{opener} {text}".strip()

    if random.random() < 0.15 and not text.endswith(("!", "?", ".")):
        text += random.choice(["!", "…", "."])

    return {
        "text": text,
        "label": 2,
        "fictional": True,
        "source": "synthetic_placeholders",
        "template_family": family
    }

rows = [synth_row(i) for i in range(200)]
df = pd.DataFrame(rows)

out_path = "data/defamation_synthetic_200.csv"
df.to_csv(out_path, index=False)

display_dataframe_to_user("Preview: synthetic defamation-like (fictional) posts", df.head(20))