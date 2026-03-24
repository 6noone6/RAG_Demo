import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_document(file_path):
    """
    智能文档加载与切分工厂
    """
    # 1. 获取文件后缀名 (转换为小写，比如 .PDF 变成 .pdf)
    file_extension = os.path.splitext(file_path)[1].lower()

    # 2. 路由分配加载器
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        # txt 文件容易出现编码问题，强制指定 utf-8
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path, encoding='utf-8')
    else:
        # 拦截不支持的格式
        raise ValueError(f"系统暂不支持 {file_extension} 格式的文件解析！")

    # 3. 加载文档内容
    docs = loader.load()

    # 4. 统一进行文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)

    return chunks