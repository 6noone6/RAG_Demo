import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 新增导入：处理消息历史和占位符的核心模块
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 从你的经典包中导入，新增了 create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from load_and_split_document import load_and_split_document
from Embedding_Vector_Store import create_vector_db, get_vector_db

load_dotenv()

st.set_page_config(page_title="企业级 RAG 知识库", page_icon="🏢", layout="wide")
st.title("🏢 企业级架构 RAG 知识库系统")

# ================= 初始化 Session State =================
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
# 新增：初始化一个列表，用来充当 AI 的“大脑海马体”，存聊天记录
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= 侧边栏：文件上传 =================
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("请上传 PDF 文档", type="pdf")

    if st.button("开始构建知识库") and uploaded_file is not None:
        with st.spinner("正在调度底层模块处理数据..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                chunks = load_and_split_document(tmp_file_path)
                create_vector_db(chunks)
                st.session_state.db_ready = True

                # 关键细节：如果用户传了新文件，要把之前的聊天记录清空，防止串台！
                st.session_state.chat_history = []
                st.success(f"✅ 构建成功！共处理 {len(chunks)} 个文本块。")
            except Exception as e:
                st.error(f"构建失败: {e}")
            finally:
                os.remove(tmp_file_path)

# ================= 主界面：原生聊天 UI 与记忆检索 =================
if st.session_state.db_ready:
    st.info("👇 底层知识库已就绪，请提问！")

    # 1. 渲染历史对话气泡
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)

    # 2. 接收新输入：从 text_input 升级为酷炫的 chat_input
    user_input = st.chat_input("向知识库提问：")

    if user_input:
        # 立即在界面上打印用户的提问
        st.chat_message("user").write(user_input)

        with st.spinner("正在结合上下文检索并生成答案..."):
            try:
                vector_db = get_vector_db()
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI(
                    model="gpt-5-chat-latest",
                    temperature=0,
                    base_url=os.getenv("OPENAI_API_BASE", "https://xiaoai.plus/v1")
                )

                # ================= 核心魔法：两条链 =================

                # 链 1：问题重写（处理代词，比如把“它”还原成具体的名词）
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

                # 链 2：根据重写后的问题和查到的资料进行回答
                qa_system_prompt = (
                    "你是一个严谨的 AI 知识助手。请严格根据以下检索到的【上下文内容】来回答用户的问题。\n"
                    "如果你在上下文中找不到答案，请直接回答“根据提供的文档，我无法回答这个问题”。\n\n"
                    "【上下文内容】：\n{context}"
                )
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),  # 确保模型在回答时也能看到历史
                    ("human", "{input}"),
                ])

                # 组装最终的大链条
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # 3. 传入当前问题和历史记录，执行调用
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })

                # 4. 在界面上展示 AI 回答
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                    with st.expander("🔍 查看检索来源"):
                        for i, doc in enumerate(response["context"]):
                            st.caption(f"**出处 {i + 1}:**\n{doc.page_content}")

                # 5. 把这一轮的问答存入记忆库，供下一次使用
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response["answer"]))

            except Exception as e:
                st.error(f"问答链路故障: {e}")
else:
    st.warning("👈 请先在左侧边栏上传 PDF 文件，并点击构建知识库。")
