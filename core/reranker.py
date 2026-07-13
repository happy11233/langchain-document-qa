import math
import re
from collections import Counter


def tokenize(text):
    """适配中英文的轻量分词，用于关键词检索和本地 rerank。"""
    text = (text or "").lower()
    words = re.findall(r"[a-z0-9_]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [
        "".join(chinese_chars[i:i + 2])
        for i in range(max(0, len(chinese_chars) - 1))
    ]
    return words + chinese_chars + chinese_bigrams


class SimpleReranker:
    """轻量本地重排器。

    它不是专用 cross-encoder 模型，但能结合关键词覆盖、短语命中、
    文件名/表头命中和上游检索分数，稳定提升 demo 的排序质量。
    """

    def rerank(self, query, documents, top_k=4):
        query_terms = tokenize(query)
        if not query_terms:
            return documents[:top_k]

        query_counter = Counter(query_terms)
        scored_docs = []
        for doc in documents:
            content = doc.page_content or ""
            metadata_text = " ".join([
                str(doc.metadata.get("source_file", "")),
                str(doc.metadata.get("columns", "")),
                str(doc.metadata.get("sheet_name", "")),
            ])
            doc_terms = tokenize(content + " " + metadata_text)
            doc_counter = Counter(doc_terms)

            overlap = sum(min(count, doc_counter.get(term, 0)) for term, count in query_counter.items())
            coverage = overlap / max(1, sum(query_counter.values()))

            phrase_bonus = 0.0
            lowered_content = (content + " " + metadata_text).lower()
            for term in set(re.findall(r"[a-z0-9_\u4e00-\u9fff]{2,}", query.lower())):
                if term in lowered_content:
                    phrase_bonus += 0.08

            upstream_score = float(doc.metadata.get("hybrid_score", 0) or 0)
            length_penalty = 1 / (1 + math.log(max(1, len(content)) / 800 + 1))
            rerank_score = 0.55 * coverage + 0.30 * upstream_score + phrase_bonus + 0.15 * length_penalty

            doc.metadata["rerank_score"] = round(rerank_score, 4)
            doc.metadata["keyword_coverage"] = round(coverage, 4)
            scored_docs.append((rerank_score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
