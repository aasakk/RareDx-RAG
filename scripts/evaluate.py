import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.phenotype_matcher import PhenotypeMatcher


# Prototype benchmark test cases (smoke-test cases for educational evaluation)
TEST_CASES = [
    {
        "case_id": "Case 1",
        "description": "Infantile epileptic encephalopathy with motor delays",
        "symptoms": "severe seizures, developmental delay, abnormal gait",
        "onset": "infancy",
        "expected_disease": "Dravet syndrome"
    },
    {
        "case_id": "Case 2",
        "description": "Neurodevelopmental disorder with severe speech impairment and happy demeanor",
        "symptoms": "severe intellectual disability, absent speech, seizures, happy demeanor",
        "onset": "",
        "expected_disease": "Angelman syndrome"
    },
    {
        "case_id": "Case 3",
        "description": "Infantile motor neuron loss with proximal hypotonia",
        "symptoms": "severe muscle weakness, hypotonia, absent reflexes",
        "onset": "infancy",
        "expected_disease": "Spinal muscular atrophy"
    },
    {
        "case_id": "Case 4",
        "description": "Lysosomal storage presentation with organomegaly, gaze palsy, and absence of early seizures",
        "symptoms": "vertical supranuclear gaze palsy, hepatosplenomegaly, no seizures",
        "onset": "infancy",
        "expected_disease": "Niemann-Pick disease type C"
    },
    {
        "case_id": "Case 5",
        "description": "Developmental regression with characteristic hand stereotypies",
        "symptoms": "regression, hand wringing, seizures",
        "onset": "childhood",
        "expected_disease": "Rett syndrome"
    }
]


def run_evaluation_suite(matcher, test_cases, mode_name="Full Hybrid Matcher", use_onset=True):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {mode_name}")
    print(f"{'='*70}")

    top_1 = 0
    top_3 = 0
    top_5 = 0
    total = len(test_cases)
    results_summary = []

    for i, test in enumerate(test_cases, start=1):
        symptoms = test["symptoms"]
        onset = test["onset"] if use_onset else None
        expected = test["expected_disease"].lower()

        ranked, normalized = matcher.rank_diseases(symptoms, onset)

        found_rank = -1
        for rank, candidate in enumerate(ranked, start=1):
            if expected in candidate["disease"].lower():
                found_rank = rank
                break

        if found_rank > 0:
            if found_rank == 1:
                top_1 += 1
            if found_rank <= 3:
                top_3 += 1
            if found_rank <= 5:
                top_5 += 1
            rank_str = f"Rank #{found_rank}"
        else:
            rank_str = "Not in candidates"

        print(
            f"[{i}/{total}] {test['case_id']}: Expected '{test['expected_disease']}' → {rank_str} "
            f"(Symptoms: '{symptoms[:50]}...', Onset: '{onset or 'None'}')"
        )

        results_summary.append({
            "case": test["case_id"],
            "expected": test["expected_disease"],
            "rank": found_rank,
            "top1": found_rank == 1,
            "top3": 0 < found_rank <= 3,
            "top5": 0 < found_rank <= 5,
        })

    print(f"\n--- Metrics: {mode_name} ---")
    print(f"Total Cases:     {total}")
    print(f"Top-1 Accuracy:  {top_1}/{total} ({(top_1/total)*100:.1f}%)")
    print(f"Top-3 Accuracy:  {top_3}/{total} ({(top_3/total)*100:.1f}%)")
    print(f"Top-5 Accuracy:  {top_5}/{total} ({(top_5/total)*100:.1f}%)")

    return {
        "mode": mode_name,
        "top_1": top_1,
        "top_3": top_3,
        "top_5": top_5,
        "total": total,
        "top_1_pct": (top_1 / total) * 100,
        "top_3_pct": (top_3 / total) * 100,
        "top_5_pct": (top_5 / total) * 100,
        "cases": results_summary
    }


def evaluate():
    print("Initializing PhenotypeMatcher for evaluation benchmark...")
    matcher = PhenotypeMatcher()

    print("\n" + "="*70)
    print("RAREDX-RAG PROTOTYPE EVALUATION BENCHMARK")
    print("NOTE: This is a 5-case prototype validation suite, NOT a clinical trial.")
    print("="*70)

    # 1. Evaluate baseline without onset input
    baseline_res = run_evaluation_suite(
        matcher,
        TEST_CASES,
        mode_name="Baseline (Phenotypes only, No Onset input)",
        use_onset=False
    )

    # 2. Evaluate full pipeline with onset input
    full_res = run_evaluation_suite(
        matcher,
        TEST_CASES,
        mode_name="Full Pipeline (Phenotypes + Onset + Hierarchy + Frequency + Severity)",
        use_onset=True
    )

    print("\n" + "="*70)
    print("COMPARATIVE SUMMARY TABLE")
    print("="*70)
    print(f"{'Configuration':<45} | {'Top-1':<10} | {'Top-3':<10} | {'Top-5':<10}")
    print("-" * 80)
    print(
        f"{baseline_res['mode']:<45} | "
        f"{baseline_res['top_1_pct']:.1f}%     | "
        f"{baseline_res['top_3_pct']:.1f}%     | "
        f"{baseline_res['top_5_pct']:.1f}%"
    )
    print(
        f"{full_res['mode']:<45} | "
        f"{full_res['top_1_pct']:.1f}%     | "
        f"{full_res['top_3_pct']:.1f}%     | "
        f"{full_res['top_5_pct']:.1f}%"
    )
    print("="*70)


if __name__ == "__main__":
    evaluate()
