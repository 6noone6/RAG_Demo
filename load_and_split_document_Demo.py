from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path):
    print(f"正在加载文档: {file_path} ...")

    # 1. 加载文档 (以 PDF 为例)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ 成功加载，共 {len(documents)} 页。")

    # 2. 初始化文本切分器
    # 关键参数解释：
    # chunk_size: 每个文本块的最大字符数
    # chunk_overlap: 相邻文本块之间的重叠字符数（防止一句话被生硬截断，保留上下文）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # 针对中文语境优化切分符优先级：先按段落切，再按句号切，最后是逗号和空格
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )

    # 3. 执行切分
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文档已切分为 {len(chunks)} 个文本块。")

    return chunks


# --- 本地测试 ---
if __name__ == "__main__":
    # 找一个你本地的 PDF 文件测试
    # 强烈建议用你之前写的“51单片机课设报告”或者一篇机器学习方向的论文来测试，
    # 这样后面的问答效果会非常有针对性！
    sample_pdf_path = "/Users/noone/PycharmProjects/RAG_Demo/神经网络与深度学习-邱锡鹏.pdf"  # 请替换为你电脑里实际的文件路径

    try:
        doc_chunks = load_and_split_document(sample_pdf_path)

        # 打印第一个 chunk 看看效果
        if doc_chunks:
            print("\n--- 🔍 第一个文本块内容预览 ---")
            print(doc_chunks[0].page_content)
            print(f"--- 📊 当前块字符长度: {len(doc_chunks[0].page_content)} ---")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("请检查文件路径是否正确，或者是否安装了必要的库。")