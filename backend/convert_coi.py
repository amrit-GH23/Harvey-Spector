"""
Convert full Constitution of India from civictech-India/constitution-of-india
to the COI.json format expected by ingest.py.

Source: https://github.com/civictech-India/constitution-of-india
Format: [{article, title, description}, ...]

Output format:
[
  [ {ArtNo, Name, ArtDesc, Status?}, ... ],   # articles array
  [ {PartNo, Name, Articles: [...]}, ... ]     # parts index
]

Run: python convert_coi.py
"""

import json
import urllib.request

SOURCE_URL = "https://raw.githubusercontent.com/civictech-India/constitution-of-india/main/constitution_of_india.json"
OUTPUT_PATH = "COI.json"

# Complete Parts mapping for the Constitution of India
# Each entry: (PartNo, Name, start_article, end_article, extra_articles)
# Extra articles for lettered variants (e.g. "21A", "31A")
PARTS = [
    {
        "PartNo": "Preamble",
        "Name": "PREAMBLE",
        "articles": ["0"],
    },
    {
        "PartNo": "I",
        "Name": "THE UNION AND ITS TERRITORY",
        "range": (1, 4),
        "extras": [],
    },
    {
        "PartNo": "II",
        "Name": "CITIZENSHIP",
        "range": (5, 11),
        "extras": [],
    },
    {
        "PartNo": "III",
        "Name": "FUNDAMENTAL RIGHTS",
        "range": (12, 35),
        "extras": ["21A", "31A", "31B", "31C", "31D", "32A"],
    },
    {
        "PartNo": "IV",
        "Name": "DIRECTIVE PRINCIPLES OF STATE POLICY",
        "range": (36, 51),
        "extras": ["39A", "43A", "48A"],
    },
    {
        "PartNo": "IVA",
        "Name": "FUNDAMENTAL DUTIES",
        "range": (51, 51),
        "extras": ["51A"],
    },
    {
        "PartNo": "V",
        "Name": "THE UNION",
        "range": (52, 151),
        "extras": ["134A", "139A"],
    },
    {
        "PartNo": "VI",
        "Name": "THE STATES",
        "range": (152, 237),
        "extras": ["224A", "233A"],
    },
    {
        "PartNo": "VII",
        "Name": "THE STATES IN PART B OF THE FIRST SCHEDULE (REPEALED)",
        "range": (238, 238),
        "extras": [],
    },
    {
        "PartNo": "VIII",
        "Name": "THE UNION TERRITORIES",
        "range": (239, 242),
        "extras": ["239A", "239AA", "239AB", "239B", "242"],
    },
    {
        "PartNo": "IX",
        "Name": "THE PANCHAYATS",
        "range": (243, 243),
        "extras": ["243A", "243B", "243C", "243D", "243E", "243F", "243G", "243H", "243I", "243J", "243K", "243L", "243M", "243N", "243O"],
    },
    {
        "PartNo": "IXA",
        "Name": "THE MUNICIPALITIES",
        "extras": ["243P", "243Q", "243R", "243S", "243T", "243U", "243V", "243W", "243X", "243Y", "243Z", "243ZA", "243ZB", "243ZC", "243ZD", "243ZE", "243ZF", "243ZG"],
    },
    {
        "PartNo": "IXB",
        "Name": "THE CO-OPERATIVE SOCIETIES",
        "extras": ["243ZH", "243ZI", "243ZJ", "243ZK", "243ZL", "243ZM", "243ZN", "243ZO", "243ZP", "243ZQ", "243ZR", "243ZS", "243ZT"],
    },
    {
        "PartNo": "X",
        "Name": "THE SCHEDULED AND TRIBAL AREAS",
        "range": (244, 244),
        "extras": ["244A"],
    },
    {
        "PartNo": "XI",
        "Name": "RELATIONS BETWEEN THE UNION AND THE STATES",
        "range": (245, 263),
        "extras": ["246A", "258A"],
    },
    {
        "PartNo": "XII",
        "Name": "FINANCE, PROPERTY, CONTRACTS AND SUITS",
        "range": (264, 300),
        "extras": ["268A", "269A", "290A", "300A"],
    },
    {
        "PartNo": "XIII",
        "Name": "TRADE, COMMERCE AND INTERCOURSE WITHIN THE TERRITORY OF INDIA",
        "range": (301, 307),
        "extras": [],
    },
    {
        "PartNo": "XIV",
        "Name": "SERVICES UNDER THE UNION AND THE STATES",
        "range": (308, 323),
        "extras": ["312A"],
    },
    {
        "PartNo": "XIVA",
        "Name": "TRIBUNALS",
        "extras": ["323A", "323B"],
    },
    {
        "PartNo": "XV",
        "Name": "ELECTIONS",
        "range": (324, 329),
        "extras": ["329A"],
    },
    {
        "PartNo": "XVI",
        "Name": "SPECIAL PROVISIONS RELATING TO CERTAIN CLASSES",
        "range": (330, 342),
        "extras": ["338A", "340A", "341A", "342A"],
    },
    {
        "PartNo": "XVII",
        "Name": "OFFICIAL LANGUAGE",
        "range": (343, 351),
        "extras": ["350A", "350B"],
    },
    {
        "PartNo": "XVIII",
        "Name": "EMERGENCY PROVISIONS",
        "range": (352, 360),
        "extras": [],
    },
    {
        "PartNo": "XIX",
        "Name": "MISCELLANEOUS",
        "range": (361, 367),
        "extras": ["361A", "361B", "363A"],
    },
    {
        "PartNo": "XX",
        "Name": "AMENDMENT OF THE CONSTITUTION",
        "range": (368, 368),
        "extras": [],
    },
    {
        "PartNo": "XXI",
        "Name": "TEMPORARY, TRANSITIONAL AND SPECIAL PROVISIONS",
        "range": (369, 392),
        "extras": ["371A", "371B", "371C", "371D", "371E", "371F", "371G", "371H", "371I", "371J", "372A", "378A"],
    },
    {
        "PartNo": "XXII",
        "Name": "SHORT TITLE, COMMENCEMENT, AUTHORITATIVE TEXT IN HINDI AND REPEALS",
        "range": (393, 395),
        "extras": ["394A"],
    },
]


def build_parts_index(all_art_nos: set) -> list:
    """Build the parts index, only including article numbers that exist in the dataset."""
    parts_index = []

    for part in PARTS:
        articles_in_part = []

        # Handle preamble special case
        if "articles" in part:
            articles_in_part = [a for a in part["articles"] if a in all_art_nos]
        else:
            # Add range articles
            if "range" in part:
                start, end = part["range"]
                for n in range(start, end + 1):
                    art_str = str(n)
                    if art_str in all_art_nos:
                        articles_in_part.append(art_str)

            # Add lettered extras
            for extra in part.get("extras", []):
                if extra in all_art_nos:
                    articles_in_part.append(extra)

        if articles_in_part:
            parts_index.append({
                "PartNo": part["PartNo"],
                "Name": part["Name"],
                "Articles": articles_in_part,
            })

    return parts_index


def normalize_art_no(raw) -> str:
    """Normalize article number to string (handles int, '21A', '239 A A' with spaces)."""
    if isinstance(raw, int):
        return str(raw)
    # Remove internal spaces: '239 A A' → '239AA'
    return "".join(str(raw).split())


def convert(source_data: list) -> list:
    articles = []
    all_art_nos = set()

    for item in source_data:
        art_no = normalize_art_no(item.get("article", ""))
        name = item.get("title", "").strip()
        description = item.get("description", "").strip()

        # Treat "0" as Preamble
        if art_no == "0":
            name = name or "PREAMBLE"

        article_obj = {
            "ArtNo": art_no,
            "Name": name,
        }

        if description:
            article_obj["ArtDesc"] = description

        articles.append(article_obj)
        all_art_nos.add(art_no)

    parts_index = build_parts_index(all_art_nos)
    return [articles, parts_index]


def main():
    print(f"Downloading from:\n  {SOURCE_URL}\n")
    with urllib.request.urlopen(SOURCE_URL) as response:
        source_data = json.loads(response.read().decode("utf-8"))

    print(f"Downloaded {len(source_data)} articles.")

    converted = convert(source_data)
    articles, parts_index = converted

    print(f"Converted {len(articles)} articles across {len(parts_index)} parts.")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nParts covered:")
    for part in parts_index:
        print(f"  Part {part['PartNo']}: {part['Name']} ({len(part['Articles'])} articles)")

    print("\nDone. Next steps:")
    print("  1. Delete existing ChromaDB data: rm -rf chroma_data/")
    print("  2. Re-run ingestion: python -m app.ingest")


if __name__ == "__main__":
    main()
