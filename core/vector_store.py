import os

import numpy as np

# Chroma 0.5.x 在部分环境中仍引用 NumPy 1.x 的 np.float_。
# 这里做兼容兜底，避免 Python 3.13 + NumPy 2.x 环境启动时报错。
if not hasattr(np, "float_"):
    np.float_ = np.float64

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document

from core.reranker import tokenize


def metadata_matches(metadata, filters):
    """判断文档 metadata 是否满足过滤条件。"""
    if not filters:
        return True

    for key, expected in filters.items():
        if expected in (None, "", [], ()):
            continue

        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            expected_values = {str(item).lower() for item in expected if item not in (None, "")}
            if expected_values and str(actual).lower() not in expected_values:
                return False
        else:
            if str(actual).lower() != str(expected).lower():
                return False

    return True


class VectorStoreManager:
    """向量数据库管理器"""
    def __init__(self,api_key,persist_directory="./data/chroma_db"):
        self.api_key = api_key
        self.persist_directory = persist_directory

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=api_key,

        )
        self.vectorstore= None
        self.load_or_create()
    def load_or_create(self):
        """加载或创建向量数据库"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    def add_documents(self, documents):
        """添加文档"""
        try:
            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            raise Exception(f"添加文档失败: {str(e)}")

    def get_retriever(self, k=4, filters=None):
        """获取检索器"""
        search_kwargs = {"k": k}
        chroma_filter = self._to_chroma_filter(filters)
        if chroma_filter:
            search_kwargs["filter"] = chroma_filter

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def search(self, query, k=4, filters=None):
        """搜索"""
        kwargs = {}
        chroma_filter = self._to_chroma_filter(filters)
        if chroma_filter:
            kwargs["filter"] = chroma_filter
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_scores(self, query, k=8, filters=None):
        """向量检索，返回带归一化分数的文档。"""
        kwargs = {}
        chroma_filter = self._to_chroma_filter(filters)
        if chroma_filter:
            kwargs["filter"] = chroma_filter

        results = self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)
        scored_docs = []
        for doc, distance in results:
            score = 1 / (1 + float(distance or 0))
            doc.metadata["vector_score"] = round(score, 4)
            scored_docs.append(doc)
        return scored_docs

    def keyword_search(self, query, k=8, filters=None):
        """轻量关键词检索，用于和向量检索做 hybrid fusion。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        results = self._get_collection(include=["documents", "metadatas"])
        documents = results.get("documents", []) if results else []
        metadatas = results.get("metadatas", []) if results else []
        ids = results.get("ids", []) if results else []

        scored = []
        for idx, content in enumerate(documents):
            metadata = metadatas[idx] or {}
            if not metadata_matches(metadata, filters):
                continue

            metadata_text = " ".join([
                str(metadata.get("source_file", "")),
                str(metadata.get("columns", "")),
                str(metadata.get("sheet_name", "")),
                str(metadata.get("department", "")),
                str(metadata.get("category", "")),
            ])
            doc_tokens = tokenize((content or "") + " " + metadata_text)
            if not doc_tokens:
                continue

            doc_set = set(doc_tokens)
            overlap = len(query_set & doc_set)
            if overlap == 0:
                continue

            score = overlap / max(1, len(query_set))
            doc = Document(page_content=content, metadata=dict(metadata))
            doc.metadata["keyword_score"] = round(score, 4)
            doc.metadata["document_id"] = ids[idx] if idx < len(ids) else ""
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def hybrid_search(self, query, k=8, filters=None, vector_weight=0.65, keyword_weight=0.35):
        """融合向量检索和关键词检索结果。"""
        vector_docs = self.similarity_search_with_scores(query, k=k, filters=filters)
        keyword_docs = self.keyword_search(query, k=k, filters=filters)

        merged = {}
        for rank, doc in enumerate(vector_docs, start=1):
            doc_key = self._doc_key(doc)
            vector_score = float(doc.metadata.get("vector_score", 0) or 0)
            merged.setdefault(doc_key, doc)
            merged[doc_key].metadata["vector_score"] = max(
                float(merged[doc_key].metadata.get("vector_score", 0) or 0),
                vector_score,
            )
            merged[doc_key].metadata["vector_rank_score"] = 1 / rank

        for rank, doc in enumerate(keyword_docs, start=1):
            doc_key = self._doc_key(doc)
            keyword_score = float(doc.metadata.get("keyword_score", 0) or 0)
            merged.setdefault(doc_key, doc)
            merged[doc_key].metadata["keyword_score"] = max(
                float(merged[doc_key].metadata.get("keyword_score", 0) or 0),
                keyword_score,
            )
            merged[doc_key].metadata["keyword_rank_score"] = 1 / rank

        docs = list(merged.values())
        for doc in docs:
            vector_score = float(doc.metadata.get("vector_score", 0) or 0)
            keyword_score = float(doc.metadata.get("keyword_score", 0) or 0)
            vector_rank_score = float(doc.metadata.get("vector_rank_score", 0) or 0)
            keyword_rank_score = float(doc.metadata.get("keyword_rank_score", 0) or 0)
            score = (
                vector_weight * max(vector_score, vector_rank_score)
                + keyword_weight * max(keyword_score, keyword_rank_score)
            )
            doc.metadata["hybrid_score"] = round(score, 4)
            doc.metadata["retrieval_mode"] = "hybrid"

        docs.sort(key=lambda doc: doc.metadata.get("hybrid_score", 0), reverse=True)
        return docs[:k]

    def retrieve(self, query, k=8, filters=None, mode="hybrid"):
        """统一检索入口。"""
        if mode == "vector":
            docs = self.similarity_search_with_scores(query, k=k, filters=filters)
            for doc in docs:
                doc.metadata["retrieval_mode"] = "vector"
                doc.metadata["hybrid_score"] = doc.metadata.get("vector_score", 0)
            return docs
        if mode == "keyword":
            docs = self.keyword_search(query, k=k, filters=filters)
            for doc in docs:
                doc.metadata["retrieval_mode"] = "keyword"
                doc.metadata["hybrid_score"] = doc.metadata.get("keyword_score", 0)
            return docs
        return self.hybrid_search(query, k=k, filters=filters)

    def get_document_count(self):
        """获取文档数量"""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0

    def get_all_sources(self):
        """获取所有文档来源"""
        try:
            results = self._get_collection()
            if results and 'metadatas' in results:
                sources = set()
                for metadata in results['metadatas']:
                    if 'source_file' in metadata:
                        sources.add(metadata['source_file'])
                return list(sources)
            return []
        except Exception:
            return []

    def get_source_summaries(self, filters=None):
        """按文件聚合知识库摘要。"""
        try:
            results = self._get_collection(include=["metadatas"])
            summaries = {}
            for metadata in results.get("metadatas", []):
                metadata = metadata or {}
                if not metadata_matches(metadata, filters):
                    continue

                source = metadata.get("source_file", "未知")
                item = summaries.setdefault(source, {
                    "source_file": source,
                    "chunks": 0,
                    "file_type": metadata.get("file_type", "unknown"),
                    "department": metadata.get("department", "未设置"),
                    "category": metadata.get("category", "未设置"),
                    "tags": metadata.get("tags", ""),
                })
                item["chunks"] += 1
            return list(summaries.values())
        except Exception:
            return []

    def get_table_summaries(self, filters=None):
        """获取 CSV/Excel 表格摘要。"""
        try:
            results = self._get_collection(include=["metadatas"])
            tables = []
            seen = set()
            for metadata in results.get("metadatas", []):
                metadata = metadata or {}
                if metadata.get("doc_kind") != "table_summary":
                    continue
                if not metadata_matches(metadata, filters):
                    continue
                key = (metadata.get("source_file"), metadata.get("sheet_name"))
                if key in seen:
                    continue
                seen.add(key)
                tables.append(dict(metadata))
            return tables
        except Exception:
            return []

    def clear_all(self):
        """清空数据库"""
        try:
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.load_or_create()
            return True
        except Exception as e:
            raise Exception(f"清空数据库失败: {str(e)}")

    def _to_chroma_filter(self, filters):
        """将简单 metadata filters 转成 Chroma where 条件。"""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if value in (None, "", [], ()):
                continue
            if isinstance(value, (list, tuple, set)):
                clean_values = [item for item in value if item not in (None, "")]
                if len(clean_values) == 1:
                    conditions.append({key: {"$eq": clean_values[0]}})
                elif clean_values:
                    conditions.append({key: {"$in": clean_values}})
            else:
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _doc_key(self, doc):
        metadata = doc.metadata or {}
        return "|".join([
            str(metadata.get("source_file", "")),
            str(metadata.get("sheet_name", "")),
            str(metadata.get("chunk_id", "")),
            doc.page_content[:80],
        ])

    def _get_collection(self, include=None):
        """兼容不同 Chroma 版本的 get 调用。"""
        if include:
            try:
                return self.vectorstore.get(include=include)
            except TypeError:
                return self.vectorstore.get()
        return self.vectorstore.get()
