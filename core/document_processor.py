from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile
from datetime import datetime

import pandas as pd


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            length_function=len
        )

    def load_document(self, file_path, file_name, extra_metadata=None):
        """加载文档"""
        ext = os.path.splitext(file_path)[1].lower()
        extra_metadata = extra_metadata or {}

        try:
            if ext in ['.csv', '.xlsx', '.xls']:
                documents = self.load_table_document(file_path, file_name, ext)
            elif ext == '.pdf':
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif ext == '.md':
                # Markdown 文件用 TextLoader 加载
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            for doc in documents:
                doc.metadata['source_file'] = file_name
                doc.metadata['file_type'] = ext[1:]
                doc.metadata['uploaded_at'] = extra_metadata.get(
                    'uploaded_at',
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                doc.metadata.update(extra_metadata)

            return documents

        except Exception as e:
            raise Exception(f"加载文档失败: {str(e)}")

    def load_table_document(self, file_path, file_name, ext):
        """将 CSV/Excel 表格转换为可检索的知识片段"""
        if ext == '.csv':
            tables = {"CSV": pd.read_csv(file_path)}
        else:
            excel_file = pd.ExcelFile(file_path)
            tables = {sheet: excel_file.parse(sheet) for sheet in excel_file.sheet_names}

        documents = []
        for sheet_name, df in tables.items():
            df = df.fillna("")
            columns = [str(col) for col in df.columns]
            sample_rows = df.head(5).to_dict(orient="records")

            summary_text = (
                f"表格文件: {file_name}\n"
                f"工作表: {sheet_name}\n"
                f"行数: {len(df)}\n"
                f"列数: {len(columns)}\n"
                f"字段: {', '.join(columns)}\n"
                f"样例数据: {sample_rows}"
            )
            documents.append(Document(
                page_content=summary_text,
                metadata={
                    "source_file": file_name,
                    "file_type": ext[1:],
                    "sheet_name": str(sheet_name),
                    "doc_kind": "table_summary",
                    "row_count": int(len(df)),
                    "column_count": int(len(columns)),
                    "columns": ", ".join(columns),
                }
            ))

            batch_size = 20
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start + batch_size]
                lines = []
                for row_number, (_, row) in enumerate(batch.iterrows(), start=start + 1):
                    values = []
                    for col in columns:
                        value = str(row[col]).strip()
                        if len(value) > 200:
                            value = value[:200] + "..."
                        values.append(f"{col}={value}")
                    lines.append(f"第{row_number}行: " + "; ".join(values))

                documents.append(Document(
                    page_content=(
                        f"表格文件: {file_name}\n"
                        f"工作表: {sheet_name}\n"
                        f"数据范围: 第{start + 1}行-第{start + len(batch)}行\n"
                        + "\n".join(lines)
                    ),
                    metadata={
                        "source_file": file_name,
                        "file_type": ext[1:],
                        "sheet_name": str(sheet_name),
                        "doc_kind": "table_rows",
                        "row_start": int(start + 1),
                        "row_end": int(start + len(batch)),
                    }
                ))

        return documents

    def split_documents(self, documents):
        """切分文档"""
        try:
            table_docs = [
                doc for doc in documents
                if doc.metadata.get('doc_kind', '').startswith('table_')
            ]
            text_docs = [
                doc for doc in documents
                if not doc.metadata.get('doc_kind', '').startswith('table_')
            ]

            for doc in text_docs:
                doc.metadata.setdefault('doc_kind', 'text')

            split_docs = self.text_splitter.split_documents(text_docs) if text_docs else []
            split_docs.extend(table_docs)

            for i, doc in enumerate(split_docs):
                doc.metadata['chunk_id'] = i

            return split_docs

        except Exception as e:
            raise Exception(f"切分文档失败: {str(e)}")

    def process_uploaded_file(self, uploaded_file, extra_metadata=None):
        """处理上传的文件"""
        with tempfile.NamedTemporaryFile(delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            documents = self.load_document(tmp_path, uploaded_file.name, extra_metadata)
            split_docs = self.split_documents(documents)
            return split_docs, len(documents), len(split_docs)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
