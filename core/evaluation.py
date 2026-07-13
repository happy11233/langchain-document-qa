import csv
import io
import json


class RAGEvaluator:
    """简单 RAG 评估器，适合在 demo 阶段做回归测试。"""

    def __init__(self, qa_engine):
        self.qa_engine = qa_engine

    def run(self, cases, filters=None, retrieval_mode="hybrid", top_k=4, use_rerank=True, use_agent=False):
        results = []
        for case in cases:
            question = case.get("question", "").strip()
            if not question:
                continue

            answer_result = self.qa_engine.query(
                question,
                filters=filters,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                use_rerank=use_rerank,
                use_agent=use_agent,
                update_history=False,
            )
            answer = answer_result.get("answer", "")
            source_docs = answer_result.get("source_documents", [])

            expected_keywords = case.get("expected_keywords", [])
            if isinstance(expected_keywords, str):
                expected_keywords = [item.strip() for item in expected_keywords.split("|") if item.strip()]

            keyword_hits = [
                keyword for keyword in expected_keywords
                if keyword and keyword.lower() in answer.lower()
            ]
            keyword_hit_rate = (
                len(keyword_hits) / len(expected_keywords)
                if expected_keywords else None
            )

            expected_source = case.get("expected_source", "")
            source_files = [
                doc.metadata.get("source_file", "")
                for doc in source_docs
            ]
            source_hit = (
                any(expected_source in source for source in source_files)
                if expected_source else None
            )

            results.append({
                "question": question,
                "answer": answer,
                "expected_keywords": expected_keywords,
                "keyword_hits": keyword_hits,
                "keyword_hit_rate": keyword_hit_rate,
                "expected_source": expected_source,
                "source_files": source_files,
                "source_hit": source_hit,
                "retrieved_count": len(source_docs),
            })

        return {
            "summary": self._summarize(results),
            "results": results,
        }

    def _summarize(self, results):
        keyword_rates = [
            item["keyword_hit_rate"] for item in results
            if item["keyword_hit_rate"] is not None
        ]
        source_hits = [
            item["source_hit"] for item in results
            if item["source_hit"] is not None
        ]
        return {
            "case_count": len(results),
            "avg_keyword_hit_rate": round(sum(keyword_rates) / len(keyword_rates), 4) if keyword_rates else None,
            "source_hit_rate": round(sum(1 for hit in source_hits if hit) / len(source_hits), 4) if source_hits else None,
        }


def parse_eval_cases(text):
    """支持 JSON 数组、JSONL 和 CSV 三种评估用例格式。"""
    text = (text or "").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        return [_normalize_case(case) for case in data]
    except json.JSONDecodeError:
        pass

    lines = [line for line in text.splitlines() if line.strip()]
    if lines and all(line.strip().startswith("{") for line in lines):
        return [_normalize_case(json.loads(line)) for line in lines]

    reader = csv.DictReader(io.StringIO(text))
    return [_normalize_case(row) for row in reader]


def _normalize_case(case):
    case = dict(case)
    expected_keywords = case.get("expected_keywords", [])
    if isinstance(expected_keywords, str):
        case["expected_keywords"] = [
            item.strip()
            for item in expected_keywords.split("|")
            if item.strip()
        ]
    return case
