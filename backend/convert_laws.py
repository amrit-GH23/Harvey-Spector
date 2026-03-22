"""
Jolly LLB — Download & Convert Indian Criminal Law Data
=========================================================
Downloads IPC, CrPC, and IEA JSON from civictech-India/Indian-Law-Penal-Code-Json
and converts them to the standard format expected by ingest_laws.py.

Output format (same as COI.json):
[
  [ {SectionNo, Name, SectionDesc, Chapter, ChapterName}, ... ],   # sections array
  [ {ChapterNo, Name, Sections: [...]}, ... ]                      # chapters index
]

The output files are named after their modern replacements:
  - BNS.json  (from IPC)  — Bharatiya Nyaya Sanhita
  - BNSS.json (from CrPC) — Bharatiya Nagarik Suraksha Sanhita
  - BSA.json  (from IEA)  — Bharatiya Sakshya Adhiniyam

Run: python convert_laws.py
"""

import json
import urllib.request

# Source URLs from civictech-India GitHub
SOURCES = {
    "BNS": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/ipc.json",
        "full_name": "Bharatiya Nyaya Sanhita (formerly Indian Penal Code)",
        "output": "BNS.json",
    },
    "BNSS": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/crpc.json",
        "full_name": "Bharatiya Nagarik Suraksha Sanhita (formerly Code of Criminal Procedure)",
        "output": "BNSS.json",
    },
    "BSA": {
        "url": "https://raw.githubusercontent.com/civictech-India/Indian-Law-Penal-Code-Json/main/iea.json",
        "full_name": "Bharatiya Sakshya Adhiniyam (formerly Indian Evidence Act)",
        "output": "BSA.json",
    },
}


def convert_law(source_data: list, law_code: str) -> list:
    """
    Convert civictech-India JSON format to our standard format.

    Input format:  [{chapter, chapter_title, Section/section, section_title, section_desc}, ...]
    Output format: [[sections_array], [chapters_index]]
    """
    sections = []
    chapters_map: dict[int | str, dict] = {}  # chapter_no -> {name, sections}

    for item in source_data:
        # Handle inconsistent key casing in source data
        chapter_no = item.get("chapter", item.get("Chapter", ""))
        chapter_name = item.get("chapter_title", item.get("Chapter_title", "")).strip()
        section_no = str(item.get("Section", item.get("section", "")))
        section_title = item.get("section_title", "").strip()
        section_desc = item.get("section_desc", "").strip()

        if not section_no or not section_desc:
            continue

        section_obj = {
            "SectionNo": section_no,
            "Name": section_title,
            "SectionDesc": section_desc,
            "Chapter": str(chapter_no),
            "ChapterName": chapter_name.upper() if chapter_name else "",
        }
        sections.append(section_obj)

        # Build chapters index
        ch_key = str(chapter_no)
        if ch_key not in chapters_map:
            chapters_map[ch_key] = {
                "ChapterNo": ch_key,
                "Name": chapter_name.upper() if chapter_name else f"CHAPTER {ch_key}",
                "Sections": [],
            }
        chapters_map[ch_key]["Sections"].append(section_no)

    # Sort chapters by number
    chapters_index = sorted(
        chapters_map.values(),
        key=lambda c: int(c["ChapterNo"]) if c["ChapterNo"].isdigit() else 0,
    )

    return [sections, chapters_index]


def download_and_convert(law_code: str, info: dict) -> None:
    """Download a single law and convert it."""
    print(f"\n{'='*60}")
    print(f"  {law_code}: {info['full_name']}")
    print(f"{'='*60}")
    print(f"  Downloading from: {info['url']}")

    with urllib.request.urlopen(info["url"]) as response:
        source_data = json.loads(response.read().decode("utf-8"))

    print(f"  Downloaded {len(source_data)} sections.")

    converted = convert_law(source_data, law_code)
    sections, chapters_index = converted

    print(f"  Converted {len(sections)} sections across {len(chapters_index)} chapters.")

    with open(info["output"], "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"  Saved to {info['output']}")
    print(f"\n  Chapters:")
    for ch in chapters_index:
        print(f"    Ch {ch['ChapterNo']}: {ch['Name']} ({len(ch['Sections'])} sections)")


def main():
    print("=" * 60)
    print("  Jolly LLB -- Indian Criminal Law Data Download")
    print("=" * 60)

    for law_code, info in SOURCES.items():
        download_and_convert(law_code, info)

    print("\n" + "=" * 60)
    print("[OK] All done! Files created:")
    for law_code, info in SOURCES.items():
        print(f"  - {info['output']} ({law_code})")

    print("\nNext steps:")
    print("  1. Delete existing ChromaDB data: rm -rf chroma_data/")
    print("  2. Re-run Constitution ingestion: python ingest.py")
    print("  3. Run law ingestion: python ingest_laws.py")


if __name__ == "__main__":
    main()
