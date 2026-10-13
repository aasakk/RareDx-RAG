def literature_score(articles):
    if not articles:
        return 0.0

    scores = [
        max(
            0.0,
            min(
                1.0,
                article.get(
                    "similarity",
                    0
                )
            )
        )
        for article in articles
    ]

    return sum(scores) / len(scores)


def calculate_final_score(
    phenotype_score,
    articles,
    clinvar_evidence
):
    pubmed = literature_score(articles)
    genetic = min(len(clinvar_evidence) / 5, 1.0)

    # Phenotype drives ranking almost entirely
    final = (
        0.95 * phenotype_score
        + 0.03 * pubmed
        + 0.02 * genetic
    )

    return {
        "phenotype_score": phenotype_score,
        "literature_score": pubmed,
        "genetic_evidence_score": genetic,
        "final_score": final
    }