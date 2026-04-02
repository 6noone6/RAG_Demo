from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()


def get_embeddings_model(num_chunks=1000):
    if num_chunks < 500:
        batch = 200
    elif num_chunks < 5000:
        batch = 500
    else:
        batch = 1000

    return OpenAIEmbeddings(
        model='text-embedding-3-large',
        chunk_size=batch
    )


from functools import lru_cache


@lru_cache(maxsize=1)
# ================= 引擎 1：纯内存模式 =================
def create_memory_db(chunks):
    """适合临时上传的文件：速度快，随用随弃，绝对不会有文件锁冲突"""
    embeddings = get_embeddings_model(len(chunks))
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


# ================= 引擎 2：硬盘持久化模式 =================
def create_persistent_db(chunks, persist_dir="chroma_db_dynamic"):
    """增量写入向量数据库（不会清空旧数据）"""

    embeddings = get_embeddings_model(len(chunks))

    # 1. 连接已有数据库（如果不存在会自动创建）
    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # 2. 新数据直接追加
    vector_db.add_documents(chunks)

    # 3. 持久化
    # vector_db.persist()

    print(f"✅ 成功新增 {len(chunks)} 条向量数据")

    return vector_db


def get_persistent_db(persist_dir="chroma_db_dynamic"):
    """直接读取已经存在的本地硬盘知识库，无需重新向量化"""
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
