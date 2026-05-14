from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever


class QAEngine:
    """问答引擎（新版LangChain API）"""

    def __init__(self, api_key, vector_store_manager):
        self.api_key = api_key
        self.vector_store = vector_store_manager

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="qwen-turbo",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7
        )

        # 对话历史
        self.chat_history = []

        # 创建链
        self._create_chains()

    def _create_chains(self):
        """创建检索和问答链"""
        retriever = self.vector_store.get_retriever(k=4)

        # 1. 创建历史感知的检索器
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "根据对话历史和最新问题，重新表述问题使其独立可理解。只输出重新表述的问题。"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            contextualize_q_prompt
        )

        # 2. 创建问答链
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是专业的文档问答助手，根据提供的文档内容回答问题。        

 参考文档：                                                                            
 {context}                                                                             

 回答要求：                                                                         
 1. 基于文档内容回答，不要编造信息                                                     
 2. 如果文档中没有相关信息，明确告知用户                                               
 3. 回答要准确、简洁、易懂                                                             
 4. 可以引用文档中的原文"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )

        # 3. 组合成完整的RAG链
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            question_answer_chain
        )

    def query(self, question):
        """查询"""
        if self.vector_store.get_document_count() == 0:
            return {
                "answer": "知识库为空，请先上传文档。",
                "source_documents": []
            }

        try:
            result = self.rag_chain.invoke({
                "input": question,
                "chat_history": self.chat_history
            })

            # 更新历史
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=result["answer"]))

            return {
                "answer": result["answer"],
                "source_documents": result.get("context", [])
            }

        except Exception as e:
            return {
                "answer": f"查询出错：{str(e)}",
                "source_documents": []
            }

    def clear_memory(self):
        """清空对话记忆"""
        self.chat_history.clear()