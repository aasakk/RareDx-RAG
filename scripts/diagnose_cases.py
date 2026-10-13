"""
Diagnostic script: inspect HPO normalization and disease ranking
for the 4 problematic test cases.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from src.phenotype_matcher import PhenotypeMatcher

matcher = PhenotypeMatcher()

CASES = [
    {
        "name": "Dravet syndrome",
        "symptoms": "severe seizures, developmental delay, abnormal gait",
        "onset": "infancy",
        "expected": "Dravet syndrome",
    },
    {
        "name": "Spinal muscular atrophy",
        "symptoms": "severe muscle weakness, hypotonia, absent reflexes",
        "onset": "infancy",
        "expected": "Spinal muscular atrophy",
    },
    {
        "name": "Niemann-Pick disease type C",
        "symptoms": "cherry-red spot, hepatosplenomegaly, no seizures",
        "onset": "infancy",
        "expected": "Niemann-Pick disease type C",
    },
    {
        "name": "Angelman syndrome",
        "symptoms": "severe intellectual disability, absent speech, seizures, happy demeanor",
        "onset": "",
        "expected": "Angelman syndrome",
    },
]


def show_disease_hpos(disease_name, top_n=25):
    """Print the HPO annotations loaded for a disease."""
    df = matcher.diseases
    sub = df[df["display_name"].str.lower() == disease_name.lower()]
    if sub.empty:
        print(f"  [!] '{disease_name}' not found in diseases.csv")
        return
    print(f"\n  === Disease HPOs for '{disease_name}' ({len(sub)} rows) ===")
    for _, row in sub.head(top_n).iterrows():
        print(f"    {row['hpo_id']:15s}  {row.get('phenotype',''):40s}  freq={row.get('frequency','')}")
    if len(sub) > top_n:
        print(f"    ... ({len(sub)-top_n} more)")


def run_case(case):
    print("\n" + "=" * 70)
    print(f"CASE: {case['name']}")
    print(f"  symptoms : {case['symptoms']}")
    print(f"  onset    : {case['onset'] or 'None'}")
    print("=" * 70)

    # 1. Show what the symptoms normalize to
    normalized = matcher.normalize_symptoms(case["symptoms"])
    print("\n[1] Normalized symptoms:")
    for s in normalized:
        neg = " [NEGATIVE]" if s.get("is_negative") else ""
        print(f"    {s['original_symptom']:35s} -> {s['hpo_id']:15s} {s['hpo_name']:40s}  sim={s['similarity']:.3f}{neg}")

    # 2. Show the disease HPO annotations for the expected disease
    show_disease_hpos(case["expected"])

    # 3. Show top-10 rankings
    ranked, _ = matcher.rank_diseases(case["symptoms"], case["onset"] or None)
    print(f"\n[3] Top-10 rankings:")
    for i, r in enumerate(ranked[:10], 1):
        marker = " <-- TARGET" if case["expected"].lower() in r["disease"].lower() else ""
        print(f"    #{i:2d}  score={r['score']:.4f}  {r['disease']}{marker}")

    # 4. Find the target disease and show its match details
    target_result = next(
        (r for r in ranked if case["expected"].lower() in r["disease"].lower()), None
    )
    if target_result:
        actual_rank = ranked.index(target_result) + 1
        print(f"\n[4] Target disease '{case['expected']}' is at rank #{actual_rank}")
        print(f"    Score: {target_result['score']:.4f}")
        print(f"    Matched phenotypes:")
        for line in target_result["matched_phenotypes"]:
            print(f"      {line}")
    else:
        print(f"\n[4] Target disease '{case['expected']}' was NOT found in results!")


for case in CASES:
    run_case(case)
