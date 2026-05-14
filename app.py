import streamlit as st
import os
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from core.qa_engine import QAEngine
from utils.file_utils import get_file_info, format_file_size, is_supported_file


# 页面配置


st.set_page_config(
    page_title="智能文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 配置
DB_PATH = "./data/chroma_db"

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False

if "show_logout_confirm" not in st.session_state:
    st.session_state.show_logout_confirm = False

# 初始化系统
def init_system(api_key):
    """初始化系统"""
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    vector_store = VectorStoreManager(api_key, DB_PATH)
    qa_engine = QAEngine(api_key, vector_store)
    return processor, vector_store, qa_engine

def validate_api_key(api_key):
    """验证 API Key 是否有效"""
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        # 使用 DashScopeEmbeddings 进行验证
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=api_key
        )
        # 尝试实际调用 embedding 来验证
        test_result = embeddings.embed_query("测试")
        # 如果成功返回结果，说明 API Key 有效
        if test_result and len(test_result) > 0:
            return True, "API Key 验证成功"
        else:
            return False, "API Key 验证失败"
    except Exception as e:
        error_msg = str(e)
        if "Invalid API-key" in error_msg or "Unauthorized" in error_msg or "invalid" in error_msg.lower() or "401" in error_msg:
            return False, "API Key 无效，请检查后重试"
        elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
            return False, "API Key 额度不足"
        else:
            return False, f"API Key 验证失败: {error_msg}"

# 侧边栏：配置和文档管理

with st.sidebar:
    # 如果系统未初始化，显示登录界面
    if not st.session_state.system_initialized:
        st.title("⚙️ 系统配置")
        st.info("欢迎使用智能文档问答系统")

        # API Key 输入
        api_key_input = st.text_input(
            "阿里云 API Key",
            type="password",
            placeholder="sk-xxxxxxxxxxxxxxxx",
            help="请输入你的阿里云通义千问 API Key"
        )

        # 初始化系统按钮
        if api_key_input:
            if st.button("登录系统", type="primary", use_container_width=True):
                try:
                    with st.spinner("正在验证 API Key..."):
                        # 先验证 API Key
                        is_valid, message = validate_api_key(api_key_input)

                        if not is_valid:
                            st.error(message)
                            st.stop()

                        st.success(message)

                    with st.spinner("正在登录系统..."):
                        # 分步初始化，便于定位问题
                        st.info("正在初始化文档处理器...")
                        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

                        st.info("正在初始化向量数据库...")
                        vector_store = VectorStoreManager(api_key_input, DB_PATH)

                        st.info("正在初始化问答引擎...")
                        qa_engine = QAEngine(api_key_input, vector_store)

                        # 只有全部成功后才保存 API Key 和状态
                        st.session_state.api_key = api_key_input
                        st.session_state.processor = processor
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_engine = qa_engine
                        st.session_state.system_initialized = True
                        st.success("✓ 系统登录成功！")
                        st.rerun()
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    st.error(f"登录失败: {str(e)}")
                    with st.expander("查看详细错误信息"):
                        st.code(error_detail)
                    # 确保失败时不保存错误的状态
                    st.session_state.system_initialized = False

        st.divider()

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            **如何获取 API Key？**
            1. 访问 [阿里云通义千问](https://dashscope.aliyun.com/)
            2. 注册/登录账号
            3. 进入控制台创建 API Key
            4. 复制 API Key 并粘贴到上方输入框

            **注意事项：**
            - API Key 仅保存在当前会话
            - 关闭浏览器后需要重新输入
            - 使用 API 会产生费用
            """)
    else:
        # 系统已初始化，所有功能都折叠
        # 系统配置折叠栏（包含退出登录）
        with st.expander("⚙️ 系统配置", expanded=False):
            st.success("✓ 系统已就绪")
            # 显示当前 API Key（部分隐藏）
            masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 12 else "***"
            st.caption(f"当前 API Key: {masked_key}")

            st.divider()

            # 退出登录按钮
            if not st.session_state.show_logout_confirm:
                if st.button("🚪 退出登录", use_container_width=True):
                    st.session_state.show_logout_confirm = True
                    st.rerun()
            else:
                st.warning("退出后将清空所有数据")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("确认退出", type="primary", use_container_width=True):
                        # 清空所有状态
                        st.session_state.api_key = ""
                        st.session_state.system_initialized = False
                        st.session_state.messages = []
                        st.session_state.uploaded_files = []
                        st.session_state.show_logout_confirm = False
                        if 'processor' in st.session_state:
                            del st.session_state.processor
                        if 'vector_store' in st.session_state:
                            del st.session_state.vector_store
                        if 'qa_engine' in st.session_state:
                            del st.session_state.qa_engine
                        st.success("已退出登录")
                        st.rerun()
                with col2:
                    if st.button("取消", use_container_width=True):
                        st.session_state.show_logout_confirm = False
                        st.rerun()


    # 只有系统初始化后才显示文档管理
    if st.session_state.system_initialized:
        processor = st.session_state.processor
        vector_store = st.session_state.vector_store
        qa_engine = st.session_state.qa_engine

        # 文档管理折叠栏
        with st.expander("📚 文档管理", expanded=True):
            # 统计信息
            doc_count = vector_store.get_document_count()
            sources = vector_store.get_all_sources()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("文档段落", doc_count)
            with col2:
                st.metric("文件数量", len(sources))

            st.divider()

            # 文件上传
            st.subheader("📤 上传文档")
            uploaded_files = st.file_uploader(
                "选择文件",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                help="支持 PDF、Word、TXT、Markdown 格式"
            )

            if uploaded_files:
                st.write(f"已选择 {len(uploaded_files)} 个文件")

                if st.button("🚀 开始处理", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    success_count = 0
                    total_chunks = 0

                    for i, file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"处理中: {file.name}")

                            # 处理文档
                            split_docs, page_count, chunk_count = processor.process_uploaded_file(file)

                            # 添加到向量数据库
                            vector_store.add_documents(split_docs)

                            success_count += 1
                            total_chunks += chunk_count

                            # 记录已上传文件
                            file_info = get_file_info(file)
                            file_info['chunks'] = chunk_count
                            st.session_state.uploaded_files.append(file_info)

                        except Exception as e:
                            st.error(f"处理 {file.name} 失败: {str(e)}")

                        progress_bar.progress((i + 1) / len(uploaded_files))

                    status_text.empty()
                    progress_bar.empty()

                    if success_count > 0:
                        st.success(f"✓ 成功处理 {success_count} 个文件，共 {total_chunks}个段落")
                        st.rerun()
                    else:
                        st.error("所有文件处理失败")

            # 已上传文件列表
            if sources:
                st.divider()
                st.subheader("📄 已上传文件")
                for source in sources:
                    st.text(f"• {source}")

            # 操作按钮移到文档管理下面
            st.divider()
            st.subheader("🛠️ 操作")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("清空对话", use_container_width=True):
                    st.session_state.messages = []
                    qa_engine.clear_memory()
                    st.rerun()

            with col2:
                if st.button("清空知识库", use_container_width=True):
                    try:
                        vector_store.clear_all()
                        st.session_state.uploaded_files = []
                        st.session_state.messages = []
                        qa_engine.clear_memory()
                        st.success("知识库已清空")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空失败: {str(e)}")


st.divider()
# 主界面：问答

st.title("智能文档问答系统")
st.caption("上传文档，用自然语言提问，AI 基于文档内容回答")

# 检查系统是否初始化
if not st.session_state.system_initialized:
    st.warning("⚠️ 请先在左侧输入 API Key 并初始化系统")
    st.stop()

# 获取系统组件
vector_store = st.session_state.vector_store
qa_engine = st.session_state.qa_engine

# 显示提示
doc_count = vector_store.get_document_count()
if doc_count == 0:
    st.info("💡 请先在左侧上传文档建立知识库")
else:
    st.success(f"✓ 知识库已就绪，包含 {doc_count} 个文档段落")

    # 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # 显示来源
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 查看来源"):
                for i, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**来源 {i}：** {source['file']}")
                    st.text(source['content'][:200] + "...")
                    st.divider()

                    # 聊天输入
if prompt := st.chat_input("输入你的问题..." if doc_count > 0 else "请先上传文档"):
    if doc_count == 0:
        st.warning("请先上传文档")
    else:
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

            # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = qa_engine.query(prompt)
                answer = result["answer"]
                source_docs = result["source_documents"]

                st.write(answer)

                # 处理来源
                sources = []
                if source_docs:
                    for doc in source_docs:
                        sources.append({
                            "file": doc.metadata.get("source_file", "未知"),
                            "content": doc.page_content
                        })

                    with st.expander("📄 查看来源"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**来源 {i}：** {source['file']}")
                            st.text(source['content'][:200] + "...")
                            st.divider()

                            # 保存消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
# 底部信息
st.divider()
