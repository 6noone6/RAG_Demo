import logging  # 🌟 新增：引入 Python 内置的日志模块
import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
# --- RAG 和 LangChain 核心组件 ---
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever, \
    EnsembleRetriever  # 🌟 新增：多查询重写器
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate  # 🌟 新增：用于自定义重写 Prompt
from langchain_openai import ChatOpenAI

from Embedding_Vector_Store import create_memory_db, get_persistent_db
from chat_storage import init_db, save_session, load_all_sessions  # 🌟 新增：引入数据库模块
from load_and_split_document import load_and_split_document

# ================= 🌟 新增：开启透视日志 =================
# 1. 配置基础的打印格式
logging.basicConfig(level=logging.INFO)

# 2. 专门捕获 MultiQueryRetriever（多查询拆解）的后台输出
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 3. （可选，强烈推荐！）开启 LangChain 的全局 Debug 模式
# 这会在控制台打印出每一个 Chain 输入输出的详细 Token 级内容
# 如果你觉得终端输出太多看花了眼，可以把下面这行注释掉
# langchain.debug = True
# ========================================================

load_dotenv()

# 初始化 SQLite 数据库表
init_db()

st.set_page_config(page_title="企业级 RAG 知识库", page_icon="🏢", layout="wide")


# ================= 1. 性能优化：缓存静态模型组件 =================
@st.cache_resource
def get_base_models():
    llm = ChatOpenAI(
        model="gpt-5-chat-latest",
        temperature=0,
        streaming=True,
        base_url=os.getenv("OPENAI_API_BASE", "https://xiaoai.plus/v1")
    )

    query_len = len(user_input)
    top_n = 5 if query_len < 50 else 8

    compressor = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-multilingual-v3.0",
        top_n=top_n
    )

    return llm, compressor


# ================= 2. 状态初始化：从 SQLite 加载历史 =================
def init_session_state():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    if "db_ready" not in st.session_state:
        st.session_state.db_ready = False
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "chat_sessions" not in st.session_state:
        # 🌟 核心改动：尝试从 SQLite 加载已有会话
        loaded_sessions = load_all_sessions()

        if loaded_sessions:
            st.session_state.chat_sessions = loaded_sessions
            # 默认激活最近一次的对话
            st.session_state.current_session_id = max(loaded_sessions.items(), key=lambda x: x[1]['created_at'])[0]
        else:
            default_session_id = str(uuid.uuid4())
            st.session_state.chat_sessions = {
                default_session_id: {"title": "新对话", "messages": [], "created_at": datetime.now()}
            }
            st.session_state.current_session_id = default_session_id


init_session_state()


def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "title": "新对话",
        "messages": [],
        "created_at": datetime.now()
    }
    st.session_state.current_session_id = new_id
    # 立即存入数据库
    save_session(new_id, "新对话", [], st.session_state.chat_sessions[new_id]["created_at"])


# ================= 3. 侧边栏：主流聊天 UI 布局 =================
with st.sidebar:
    st.markdown(f"👤 **当前用户ID:** `{st.session_state.user_id[:8]}`")
    st.divider()

    st.header("💬 对话历史")
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.write(" ")
    sorted_sessions = sorted(st.session_state.chat_sessions.items(), key=lambda x: x[1]["created_at"], reverse=True)

    for sess_id, sess_data in sorted_sessions:
        is_active = (sess_id == st.session_state.current_session_id)
        btn_label = f"👉 {sess_data['title']}" if is_active else f"📝 {sess_data['title']}"
        if st.button(btn_label, key=sess_id, use_container_width=True):
            st.session_state.current_session_id = sess_id
            st.rerun()

    st.divider()

    st.header("⚙️ 知识库引擎")
    kb_mode = st.radio("选择工作模式：", ["临时文档解析 (纯内存)", "加载本地大底座 (硬盘)"],
                       label_visibility="collapsed")

    if kb_mode == "临时文档解析 (纯内存)":
        uploaded_file = st.file_uploader("请上传知识库文档", type=["pdf", "txt", "docx", "csv"])
        if st.button("构建临时知识库") and uploaded_file is not None:
            with st.spinner("正在内存中极速构建..."):
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    chunks = load_and_split_document(tmp_file_path)
                    st.session_state.bm25 = BM25Retriever.from_documents(chunks)
                    st.session_state.bm25.k = 8
                    st.session_state.vector_db = create_memory_db(chunks)
                    st.session_state.db_ready = True
                    st.success(f"✅ 构建成功！共处理 {len(chunks)} 个文本块。")
                except Exception as e:
                    st.error(f"构建失败: {e}")
                finally:
                    os.remove(tmp_file_path)

    elif kb_mode == "加载本地大底座 (硬盘)":
        if st.button("连接本地数据库"):
            with st.spinner("挂载中..."):
                try:
                    st.session_state.vector_db = get_persistent_db()
                    st.session_state.db_ready = True
                    st.success("✅ 成功连接底座！")
                except Exception as e:
                    st.error(f"连接失败：{e}")

# ================= 4. 主界面：核心问答与子问题拆解链路 =================
st.title("🏢 企业级架构 RAG 知识库系统")

if not st.session_state.db_ready:
    st.warning("👈 请先在左侧边栏配置并就绪知识库。")
else:
    current_session = st.session_state.chat_sessions[st.session_state.current_session_id]
    chat_history = current_session["messages"]

    for msg in chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    user_input = st.chat_input("向知识库提问：")

    if user_input:
        if len(chat_history) == 0:
            current_session["title"] = user_input[:10] + "..." if len(user_input) > 10 else user_input
            # 标题变更，保存一下数据库
            save_session(st.session_state.current_session_id, current_session["title"], chat_history,
                         current_session["created_at"])

        st.chat_message("user").write(user_input)

        # ================= 🌟 核心升级：可视化思考状态栏 =================
        chain_ready = False
        with st.status("🧠 AI 正在深度思考中...", expanded=True) as status:
            try:
                st.write("1️⃣ 正在唤醒大模型与本地向量库...")
                llm, compressor = get_base_models()
                vector_db = st.session_state.vector_db

                # ================= 🌟 核心大升级：多查询与重写 (Query Decomposition) =================
                st.write("2️⃣ 正在分析你的意图，决定检索哪个区域...")

                # 让大模型做一道选择题
                intent_prompt = f"""
                                判断用户的以下问题是想查询论文的哪一部分？
                                - 如果问题关于研究内容、方法、结论、摘要等，请回复: main
                                - 如果问题关于引用的文献、作者、出处、参考资料等，请回复: ref
                                - 如果都有涉及，请回复: both

                                只需回复 main, ref, 或 both 这几个英文单词，不要输出任何其他字符。
                                用户问题：{user_input}
                                """

                # 调用大模型进行判断
                intent = llm.invoke(intent_prompt).content.strip().lower()

                # 根据大模型的决定，动态配置 Chroma 数据库的过滤条件 (Filter)
                search_kwargs = {"k": 8}
                if "main" in intent:
                    st.write("👉 目标锁定：论文【正文】区域！")
                    search_kwargs["filter"] = {"section": "main_body"}
                elif "ref" in intent:
                    st.write("👉 目标锁定：论文【参考文献】区域！")
                    search_kwargs["filter"] = {"section": "references"}
                else:
                    st.write("👉 目标锁定：【全局】扫描！")
                    # both 的情况就不加 filter，全库搜

                st.write("3️⃣ 正在拆解多路子问题并进行过滤检索...")
                # 1. 定义问题拆解的 Prompt
                rewrite_prompt = PromptTemplate(
                    input_variables=["question"],
                    template="""
                你是一个专业的信息检索优化助手。

                请将【原始问题】改写为 3 个高质量检索问题，要求：
                1. 每个问题关注不同语义角度
                2. 不要重复
                3. 更具体、更适合向量检索
                4. 保留原问题核心含义

                只输出 3 行，每行一个问题，不要解释。

                原始问题: {question}
                """
                )

                # 2. 原始的向量库检索器
                base_retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

                # BM25（只在有 chunks 时）
                if "bm25" in st.session_state:
                    bm25_retriever = st.session_state.bm25
                    hybrid_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, base_retriever],
                        weights=[0.4, 0.6]
                    )
                else:
                    hybrid_retriever = base_retriever

                # 3. 包装成 MultiQueryRetriever（自动调用 LLM 进行拆解，并汇集所有子问题的结果）
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=hybrid_retriever,
                    llm=llm,
                    prompt=rewrite_prompt
                )

                st.write("4️⃣ 正在底层数据库海选，并调用 Rerank 模型进行高精度提纯...")
                # 4. 加入精排 (Rerank) 机制压缩去重
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=multi_query_retriever
                )
                # =========================================================================

                # 结合历史上下文的 Prompt
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "给定聊天历史记录和最新问题（该问题可能引用了历史）。请生成一个脱离历史也能被理解的独立问题。不要回答，仅仅重写它。"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                # 这里将包含子问题拆解的检索器传入
                # 手动改写问题（history-aware）
                response = llm.invoke(
                    contextualize_q_prompt.format_messages(
                        chat_history=chat_history,
                        input=user_input
                    )
                )
                standalone_question = response.content.strip()

                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "你是一个严谨的 AI 知识助手。请严格根据以下【上下文内容】来回答。\n\n【上下文内容】：\n{context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                # 当所有准备工作瞬间完成后，将状态栏打个勾，并自动折叠收起！
                status.update(label="✅ 检索提纯完毕，开始生成回答！", state="complete", expanded=False)
                chain_ready = True
            except Exception as e:
                # 如果发生错误，状态栏变成红色并显示错误信息
                status.update(label="❌ 思考链路发生异常", state="error", expanded=True)
                st.error(f"内部构建错误: {e}")
        # ====================================================================

        # ================= 4. 流式输出回答（移出状态栏，让 UI 更清爽） =================
        if chain_ready:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_answer = ""
                context_docs = []
                try:
                    for chunk in rag_chain.stream({
                        "input": standalone_question,
                        "chat_history": chat_history
                    }):
                        if "context" in chunk:
                            context_docs = chunk["context"]
                        if "answer" in chunk:
                            full_answer += chunk["answer"]
                            message_placeholder.markdown(full_answer + "▌")

                    # 去掉结尾的光标
                    message_placeholder.markdown(full_answer)

                    # 🌟 结合了上一步的页码展示功能
                    if context_docs:
                        with st.expander("🔍 查看检索出处与页码"):
                            for i, doc in enumerate(context_docs):
                                # 1. 尝试从元数据中提取页码，如果没有则默认显示 "未知"
                                page_num = doc.metadata.get("page_number", "未知")

                                # 2. 尝试提取原始文件路径，并只保留文件名
                                source_path = doc.metadata.get("source", "未知文件")
                                file_name = os.path.basename(source_path) if source_path != "未知文件" else source_path

                                # 3. 使用更美观的 Markdown 格式渲染标题
                                st.markdown(f"**出处 {i + 1}** (📄 文档: `{file_name}` | 🔖 第 `{page_num}` 页)")

                                # 4. 展示具体的文本内容
                                st.caption(doc.page_content)

                                # 5. 添加一条浅色的分割线，让多个出处之间不会黏在一起
                                st.divider()

                    # 更新当前会话记录
                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=full_answer))

                    # 🌟 核心改动：每次问答结束后，将新记录同步保存到 SQLite 数据库
                    save_session(
                        session_id=st.session_state.current_session_id,
                        title=current_session["title"],
                        messages=chat_history,
                        created_at=current_session["created_at"]
                    )

                except Exception as e:
                    st.error(f"流式生成时发生错误: {e}")
