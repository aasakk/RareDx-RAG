import requests

from src.config import (
    NCBI_EMAIL,
    NCBI_API_KEY,
    NCBI_TOOL
)


BASE = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils"
)


def base_params():
    params = {
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL
    }

    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    return params


def search_gene_variants(
    gene,
    disease=None,
    max_results=5
):
    query = (
        f'{gene}[gene] AND '
        f'pathogenic[clinical_significance]'
    )

    if disease:
        query += f' AND "{disease}"'

    params = {
        **base_params(),
        "db": "clinvar",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    response = requests.get(
        f"{BASE}/esearch.fcgi",
        params=params,
        timeout=30
    )

    response.raise_for_status()

    ids = (
        response.json()
        .get("esearchresult", {})
        .get("idlist", [])
    )

    # Disease naming varies in ClinVar.
    # Fall back to gene-only pathogenic variants.
    if not ids and disease:
        return search_gene_variants(
            gene,
            disease=None,
            max_results=max_results
        )

    return ids


def fetch_summaries(ids):
    if not ids:
        return []

    params = {
        **base_params(),
        "db": "clinvar",
        "id": ",".join(ids),
        "retmode": "json"
    }

    response = requests.get(
        f"{BASE}/esummary.fcgi",
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json().get(
        "result",
        {}
    )

    variants = []

    for uid in ids:
        record = data.get(
            str(uid),
            {}
        )

        if not record:
            continue

        genes = record.get(
            "genes",
            []
        )

        gene_symbols = []

        for gene in genes:
            if isinstance(gene, dict):
                symbol = gene.get(
                    "symbol"
                )

                if symbol:
                    gene_symbols.append(
                        symbol
                    )

        variation_set = record.get(
            "variation_set",
            []
        )

        variation_name = ""

        if variation_set:
            first = variation_set[0]

            if isinstance(first, dict):
                variation_name = first.get(
                    "variation_name",
                    ""
                )

        variants.append({
            "uid": uid,
            "title": record.get(
                "title",
                ""
            ),
            "variation": variation_name,
            "genes": gene_symbols,
            "clinical_significance": record.get(
                "clinical_significance",
                {}
            )
        })

    return variants


def get_clinvar_evidence(
    genes,
    disease,
    per_gene=3
):
    evidence = []

    for gene in genes[:3]:
        try:
            ids = search_gene_variants(
                gene,
                disease=disease,
                max_results=per_gene
            )

            summaries = fetch_summaries(
                ids
            )

            for item in summaries:
                item["query_gene"] = gene
                evidence.append(item)

        except Exception as exc:
            print(
                f"ClinVar error for {gene}:",
                exc
            )

    return evidence