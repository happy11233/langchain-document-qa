from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os
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

    def get_retriever(self, k=4):
        """获取检索器"""
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def search(self, query, k=4):
        """搜索"""
        return self.vectorstore.similarity_search(query, k=k)

    def get_document_count(self):
        """获取文档数量"""
        try:
            return self.vectorstore._collection.count()
        except:
            return 0

    def get_all_sources(self):
        """获取所有文档来源"""
        try:
            results = self.vectorstore.get()
            if results and 'metadatas' in results:
                sources = set()
                for metadata in results['metadatas']:
                    if 'source_file' in metadata:
                        sources.add(metadata['source_file'])
                return list(sources)
            return []
        except:
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