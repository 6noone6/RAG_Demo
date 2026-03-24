from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embeddings_model():
    """统一管理 Embedding 模型配置，方便以后一键切换"""
    return OpenAIEmbeddings(
        model='text-embedding-3-small'
    )


def create_vector_db(chunks, persist_directory="./chroma_db_dynamic"):
    """将文本块向量化并存入数据库"""
    embeddings = get_embeddings_model()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_db


def get_vector_db(persist_directory="./chroma_db_dynamic"):
    """获取已经建好的数据库实例，供检索使用"""
    embeddings = get_embeddings_model()
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_db
