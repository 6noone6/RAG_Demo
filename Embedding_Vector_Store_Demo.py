import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def create_vector_db(chunks, persist_directory="./chroma_db"):
    print("正在加载 Embedding 模型...")

    # 1. 初始化 Embedding 模型
    # 这里使用的是轻量级的开源多语言友好模型，完全本地运行
    # 若你想使用大厂 API，可无缝替换为对应的 Embeddings 类 (如 OpenAIEmbeddings)
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small'
    )

    print("正在将文本向量化并存入 Chroma 数据库...")

    # 2. 创建持久化向量数据库
    # 将我们第一步切分好的 chunks 传入，使用上面的 embeddings 进行向量化
    # persist_directory 指定了数据库保存在本地的文件夹路径
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"✅ 向量数据库已成功构建并保存在: {persist_directory}")
    return vector_db


# --- 真实数据串联执行 ---
if __name__ == "__main__":
    # 导入你写的第一个脚本中的切分函数
    from load_and_split_document_Demo import load_and_split_document

    # 1. 指定真实的 PDF 文件路径
    real_pdf_path = "/Users/noone/PycharmProjects/RAG_Demo/神经网络与深度学习-邱锡鹏.pdf"

    try:
        # 2. 执行第一步：加载并切分文档
        print(">>> 启动步骤 1: 读取并切分真实 PDF 文档...")
        real_chunks = load_and_split_document(real_pdf_path)

        # 3. 执行第二步：存入向量数据库
        if real_chunks:
            print("\n>>> 启动步骤 2: 开始将真实文档写入向量数据库...")
            db = create_vector_db(real_chunks)

            # 4. 拿一个专业问题测试一下最终的检索效果！
            query = "什么是前馈神经网络？"
            print(f"\n--- 🔍 测试真实文档检索: '{query}' ---")
            docs = db.similarity_search(query, k=1)

            if docs:
                print("匹配到的最相关原文片段:")
                print(docs[0].page_content)

    except Exception as e:
        print(f"❌ 发生错误: {e}")
