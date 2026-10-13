import time
import xml.etree.ElementTree as ET

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


def ncbi_params():
    params = {
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL
    }

    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    return params


def search_pubmed(query, max_results=10):
    params = {
        **ncbi_params(),
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance"
    }

    response = requests.get(
        f"{BASE}/esearch.fcgi",
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return (
        response.json()
        .get("esearchresult", {})
        .get("idlist", [])
    )


def extract_abstract(article):
    parts = []

    abstract = article.find(
        ".//Abstract"
    )

    if abstract is None:
        return ""

    for element in abstract.findall(
        "AbstractText"
    ):
        text = "".join(
            element.itertext()
        ).strip()

        label = element.attrib.get(
            "Label",
            ""
        )

        if label:
            parts.append(
                f"{label}: {text}"
            )
        else:
            parts.append(text)

    return " ".join(parts)


def fetch_pubmed(pmids):
    if not pmids:
        return []

    params = {
        **ncbi_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    response = requests.get(
        f"{BASE}/efetch.fcgi",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    articles = []

    for item in root.findall(
        ".//PubmedArticle"
    ):
        pmid = item.findtext(
            ".//PMID",
            default=""
        )

        title_element = item.find(
            ".//ArticleTitle"
        )

        title = ""

        if title_element is not None:
            title = "".join(
                title_element.itertext()
            ).strip()

        abstract = extract_abstract(
            item
        )

        journal = item.findtext(
            ".//Journal/Title",
            default=""
        )

        year = (
            item.findtext(
                ".//PubDate/Year"
            )
            or item.findtext(
                ".//ArticleDate/Year"
            )
            or ""
        )

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year
        })

    time.sleep(0.15)

    return articles


def get_disease_articles(
    disease,
    genes=None,
    max_results=8
):
    genes = genes or []

    query = f'"{disease}"'

    if genes:
        gene_query = " OR ".join(
            genes[:3]
        )

        query += (
            f" AND ({gene_query})"
        )

    pmids = search_pubmed(
        query,
        max_results=max_results
    )

    return fetch_pubmed(pmids)