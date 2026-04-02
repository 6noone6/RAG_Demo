from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()


def get_embeddings_model():
    """统一管理 Embedding 模型配置"""
    return OpenAIEmbeddings(
        model='text-embedding-3-small'
    )


# ================= 引擎 1：纯内存模式 =================
def create_memory_db(chunks):
    """适合临时上传的文件：速度快，随用随弃，绝对不会有文件锁冲突"""
    embeddings = get_embeddings_model()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


# ================= 引擎 2：硬盘持久化模式 =================
def create_persistent_db(chunks, persist_dir="chroma_db_dynamic"):
    """适合构建底层大库：将海量文档向量化并保存到硬盘"""
    embeddings = get_embeddings_model()

    # 【核心修复】：优雅地清空旧数据，代替粗暴的 shutil.rmtree
    try:
        # 先连上现有的库，然后清空里面的集合，完美避开 SQLite 文件锁
        old_db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        old_db.delete_collection()
        print("🧹 已优雅清空旧数据库集合...")
    except Exception:
        pass  # 如果是第一次建库，忽略报错

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vector_db.persist()
    return vector_db


def get_persistent_db(persist_dir="chroma_db_dynamic"):
    """直接读取已经存在的本地硬盘知识库，无需重新向量化"""
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
