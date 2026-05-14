from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile


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

    def load_document(self, file_path, file_name):
        """加载文档"""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext == '.md':
                # Markdown 文件用 TextLoader 加载
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件格式: {ext}")

            documents = loader.load()

            for doc in documents:
                doc.metadata['source_file'] = file_name
                doc.metadata['file_type'] = ext[1:]

            return documents

        except Exception as e:
            raise Exception(f"加载文档失败: {str(e)}")

    def split_documents(self, documents):
        """切分文档"""
        try:
            split_docs = self.text_splitter.split_documents(documents)

            for i, doc in enumerate(split_docs):
                doc.metadata['chunk_id'] = i

            return split_docs

        except Exception as e:
            raise Exception(f"切分文档失败: {str(e)}")

    def process_uploaded_file(self, uploaded_file):
        """处理上传的文件"""
        with tempfile.NamedTemporaryFile(delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            documents = self.load_document(tmp_path, uploaded_file.name)
            split_docs = self.split_documents(documents)
            return split_docs, len(documents), len(split_docs)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)