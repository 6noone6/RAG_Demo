import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 🔥 核心魔法：导入你自己写的底层模块！
from load_and_split_document import load_and_split_document
from Embedding_Vector_Store import create_vector_db, get_vector_db

load_dotenv()

st.set_page_config(page_title="企业级 RAG 知识库", page_icon="🏢", layout="wide")
st.title("🏢 企业级架构 RAG 知识库系统")

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# ================= 侧边栏：文件上传与底层调度 =================
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("请上传 PDF 文档", type="pdf")

    if st.button("开始构建知识库") and uploaded_file is not None:
        with st.spinner("正在调度底层模块处理数据..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 🚀 优雅调用：只需两行代码，彻底隐藏底层的脏活累活
                chunks = load_and_split_document(tmp_file_path)
                create_vector_db(chunks)

                st.session_state.db_ready = True
                st.success(f"✅ 构建成功！共处理 {len(chunks)} 个文本块。")
            except Exception as e:
                st.error(f"构建失败: {e}")
            finally:
                os.remove(tmp_file_path)

# ================= 主界面：大模型问答调度 =================
if st.session_state.db_ready:
    st.info("👇 底层知识库已就绪，请提问！")
    user_input = st.text_input("向知识库提问：")

    if user_input:
        with st.spinner("正在检索并生成答案..."):
            try:
                # 🚀 优雅调用：一行代码拿到数据库实例
                vector_db = get_vector_db()
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI(
                    model="gpt-5-chat-latest",
                    temperature=0
                )

                system_prompt = (
                    "你是一个严谨的 AI 知识助手。请严格根据以下检索到的【上下文内容】来回答用户的问题。\n"
                    "如果你在上下文中找不到答案，请直接回答“根据提供的文档，我无法回答这个问题”。\n\n"
                    "【上下文内容】：\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                rag_chain = create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, prompt)
                )

                response = rag_chain.invoke({"input": user_input})

                st.markdown("### 💡 AI 回答：")
                st.write(response["answer"])

                with st.expander("🔍 查看检索来源"):
                    for i, doc in enumerate(response["context"]):
                        st.caption(f"**出处 {i + 1}:**\n{doc.page_content}")

            except Exception as e:
                st.error(f"问答链路故障: {e}")
