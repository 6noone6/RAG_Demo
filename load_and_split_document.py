from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path, chunk_size=500, chunk_overlap=50):
    """
    负责读取 PDF 并切分为文本块的通用工具函数。
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    return chunks