import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 处理消息历史和占位符的核心模块
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from load_and_split_document import load_and_split_document

# 🌟 改动点 1：导入新的双引擎函数
from Embedding_Vector_Store import create_memory_db, create_persistent_db, get_persistent_db

load_dotenv()

st.set_page_config(page_title="企业级 RAG 知识库", page_icon="🏢", layout="wide")
st.title("🏢 企业级架构 RAG 知识库系统")

# ================= 初始化 Session State =================
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# 🌟 新增：专门用来存当前激活的数据库实例
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ================= 🌟 改动点 2：侧边栏双引擎切换 =================
with st.sidebar:
    st.header("⚙️ 知识库引擎")
    kb_mode = st.radio("选择工作模式：", ["临时文档解析 (纯内存)", "加载本地大底座 (硬盘)"])
    st.divider()

    if kb_mode == "临时文档解析 (纯内存)":
        st.subheader("📁 临时文件上传")
        uploaded_file = st.file_uploader("请上传知识库文档", type=["pdf", "txt", "docx", "csv"])

        if st.button("构建临时知识库") and uploaded_file is not None:
            with st.spinner("正在内存中极速构建..."):
                file_extension = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    chunks = load_and_split_document(tmp_file_path)

                    # 🌟 核心改动：调用内存引擎，并存入 session_state
                    st.session_state.vector_db = create_memory_db(chunks)

                    st.session_state.db_ready = True
                    st.session_state.chat_history = []  # 清空历史，防止串台
                    st.success(f"✅ 构建成功！共处理 {len(chunks)} 个文本块。")
                except Exception as e:
                    st.error(f"构建失败: {e}")
                finally:
                    os.remove(tmp_file_path)

    elif kb_mode == "加载本地大底座 (硬盘)":
        st.subheader("🗄️ 持久化知识库")
        st.info("直接读取本地已有的海量向量数据。")

        if st.button("连接本地数据库"):
            with st.spinner("正在挂载本地硬盘数据..."):
                try:
                    # 🌟 核心改动：直接从硬盘读取大库，瞬间完成
                    st.session_state.vector_db = get_persistent_db()

                    st.session_state.db_ready = True
                    st.session_state.chat_history = []
                    st.success("✅ 成功连接本地知识底座！")
                except Exception as e:
                    st.error(f"连接失败，请确认本地存在 chroma_db_dynamic 文件夹。错误：{e}")

# ================= 主界面：原生聊天 UI 与记忆检索 =================
if st.session_state.db_ready:
    st.info("👇 底层知识库已就绪，请提问！")

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)

    user_input = st.chat_input("向知识库提问：")

    if user_input:
        st.chat_message("user").write(user_input)

        with st.spinner("正在结合上下文检索并生成答案..."):
            try:
                # 🌟 改动点 3：不再调函数重新读取硬盘，直接从口袋里掏出当前激活的数据库
                vector_db = st.session_state.vector_db

                retriever = vector_db.as_retriever(search_kwargs={"k": 3})

                # 注意：我看你上面写了 gpt-5-chat-latest，我原封不动保留了你的配置
                llm = ChatOpenAI(
                    model="gpt-5-chat-latest",
                    temperature=0,
                    streaming=True,
                    base_url=os.getenv("OPENAI_API_BASE", "https://xiaoai.plus/v1")
                )

                contextualize_q_system_prompt = (
                    "给定聊天历史记录和最新的用户问题，该问题可能会引用历史记录中的上下文。"
                    "请生成一个独立的问题，使其在没有聊天历史记录的情况下也能被理解。"
                    "不要回答问题，仅仅重写它，如果不需要重写，就直接返回原问题。"
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )

                qa_system_prompt = (
                    "你是一个严谨的 AI 知识助手。请严格根据以下检索到的【上下文内容】来回答用户的问题。\n"
                    "如果你在上下文中找不到答案，请直接回答“根据提供的文档，我无法回答这个问题”。\n\n"
                    "【上下文内容】：\n{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_answer = ""
                    context_docs = []

                    for chunk in rag_chain.stream({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    }):
                        if "context" in chunk:
                            context_docs = chunk["context"]

                        if "answer" in chunk:
                            full_answer += chunk["answer"]
                            message_placeholder.markdown(full_answer + "▌")

                    message_placeholder.markdown(full_answer)

                    if context_docs:
                        with st.expander("🔍 查看检索来源"):
                            for i, doc in enumerate(context_docs):
                                st.caption(f"**出处 {i + 1}:**\n{doc.page_content}")

                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=full_answer))

            except Exception as e:
                st.error(f"问答链路故障: {e}")
else:
    st.warning("👈 请先在左侧边栏配置并就绪知识库。")