from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from core.agent_tools import AgentToolManager
from core.reranker import SimpleReranker


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
        self.reranker = SimpleReranker()
        self.agent_tools = AgentToolManager(vector_store_manager)

        # 创建链
        self._create_chains()

    def _create_chains(self):
        """创建检索和问答链"""
        # 1. 创建历史问题改写链
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "根据对话历史和最新问题，重新表述问题使其独立可理解。只输出重新表述的问题。"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.question_rewrite_chain = contextualize_q_prompt | self.llm

        # 2. 创建问答链
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是专业的文档问答助手，根据提供的文档内容回答问题。        

 参考文档：                                                                            
 {context}                                                                             

 可用工具结果：
 {tool_context}

 回答要求：                                                                         
 1. 基于文档内容回答，不要编造信息                                                     
 2. 如果文档中没有相关信息，明确告知用户                                               
 3. 回答要准确、简洁、易懂                                                             
 4. 可以引用文档中的原文
 5. 如果工具结果和文档都不足以回答，请说明缺少哪些信息"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )

    def query(
        self,
        question,
        filters=None,
        retrieval_mode="hybrid",
        top_k=4,
        use_rerank=True,
        use_agent=True,
        update_history=True,
    ):
        """查询"""
        prepared = self.prepare_query(
            question,
            filters=filters,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_rerank=use_rerank,
            use_agent=use_agent,
        )
        if "answer" in prepared:
            return prepared

        try:
            answer = self.question_answer_chain.invoke({
                "input": question,
                "chat_history": self.chat_history,
                "context": prepared["context_docs"],
                "tool_context": prepared["tool_context"] or "无"
            })

            # 更新历史
            if update_history:
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=answer))

            return {
                "answer": answer,
                "source_documents": prepared["source_documents"],
                "rewritten_question": prepared["rewritten_question"],
                "tool_calls": prepared["tool_calls"]
            }

        except Exception as e:
            return {
                "answer": f"查询出错：{str(e)}",
                "source_documents": [],
                "rewritten_question": question,
                "tool_calls": []
            }

    def prepare_query(
        self,
        question,
        filters=None,
        retrieval_mode="hybrid",
        top_k=4,
        use_rerank=True,
        use_agent=True,
    ):
        """准备检索上下文，供普通输出和流式输出复用。"""
        if self.vector_store.get_document_count() == 0:
            return {
                "answer": "知识库为空，请先上传文档。",
                "source_documents": [],
                "rewritten_question": question,
                "tool_calls": []
            }

        try:
            standalone_question = self._rewrite_question(question)
            candidate_k = max(top_k * 3, 8) if use_rerank else top_k
            source_docs = self.vector_store.retrieve(
                standalone_question,
                k=candidate_k,
                filters=filters,
                mode=retrieval_mode
            )
            if use_rerank:
                source_docs = self.reranker.rerank(
                    standalone_question,
                    source_docs,
                    top_k=top_k
                )
            else:
                source_docs = source_docs[:top_k]

            tool_calls = self.agent_tools.run(question, filters=filters) if use_agent else []
            tool_context = self._format_tool_context(tool_calls)
            context_docs = list(source_docs)
            if tool_context:
                context_docs.append(Document(
                    page_content=tool_context,
                    metadata={"source_file": "agent_tools", "doc_kind": "tool_result"}
                ))

            return {
                "question": question,
                "source_documents": source_docs,
                "rewritten_question": standalone_question,
                "tool_calls": tool_calls,
                "tool_context": tool_context,
                "context_docs": context_docs,
            }

        except Exception as e:
            return {
                "answer": f"查询出错：{str(e)}",
                "source_documents": [],
                "rewritten_question": question,
                "tool_calls": []
            }

    def stream_answer(self, prepared, update_history=True):
        """流式生成答案。"""
        if "answer" in prepared:
            yield prepared["answer"]
            return

        chunks = []
        try:
            stream = self.question_answer_chain.stream({
                "input": prepared["question"],
                "chat_history": self.chat_history,
                "context": prepared["context_docs"],
                "tool_context": prepared["tool_context"] or "无"
            })
            for chunk in stream:
                text = getattr(chunk, "content", chunk)
                if text:
                    text = str(text)
                    chunks.append(text)
                    yield text

            answer = "".join(chunks)
            prepared["answer"] = answer
            if update_history:
                self.chat_history.append(HumanMessage(content=prepared["question"]))
                self.chat_history.append(AIMessage(content=answer))

        except Exception as e:
            yield f"查询出错：{str(e)}"

    def clear_memory(self):
        """清空对话记忆"""
        self.chat_history.clear()

    def _rewrite_question(self, question):
        """结合历史对话改写问题。"""
        if not self.chat_history:
            return question

        rewritten = self.question_rewrite_chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })
        return getattr(rewritten, "content", str(rewritten)).strip() or question

    def _format_tool_context(self, tool_calls):
        if not tool_calls:
            return ""

        lines = []
        for call in tool_calls:
            lines.append(f"工具: {call['tool']}")
            lines.append(f"结果:\n{call['output']}")
        return "\n\n".join(lines)
