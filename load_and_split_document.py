import os
import json
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader


# ================= 1. 各类型文件专属加载函数 =================

def load_pdf(file_path):
    """处理 PDF 文件（调用商业级 AI 视觉解析 API）"""
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=True,
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        url=os.getenv("UNSTRUCTURED_API_URL"),
        mode="single",
        strategy="hi_res"
    )
    return loader.load()


def load_txt(file_path):
    """处理纯文本 TXT 文件"""
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()


def load_docx(file_path):
    """处理 Word 文档"""
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_csv(file_path):
    """处理 CSV 表格文件"""
    loader = CSVLoader(file_path, encoding='utf-8')
    return loader.load()


# ================= 2. 🌟 新增：动态切分策略引擎 =================

def get_split_strategy(file_extension, file_size_bytes):
    """
    根据文件类型和大小动态决定切分参数
    返回: (chunk_size, chunk_overlap)
    """
    size_mb = file_size_bytes / (1024 * 1024)

    # 默认基础配置
    chunk_size = 1000
    chunk_overlap = 100

    if file_extension == '.csv':
        # CSVLoader 默认把每一行转成一个 Document，通常不需要再做细碎的字符级切分
        # 给个大点的 chunk_size 兜底（防止某一行异常长），overlap 设为 0
        chunk_size = 2000
        chunk_overlap = 0

    elif file_extension == '.pdf':
        # PDF 使用了 Unstructured 视觉解析，语义块还原得比较好
        if size_mb < 5:  # 小于 5MB 的 PDF
            chunk_size = 800
            chunk_overlap = 100
        else:  # 大 PDF 文件，稍微调大 chunk_size 减少碎片并保留更多上下文
            chunk_size = 1500
            chunk_overlap = 150

    elif file_extension in ['.txt', '.docx']:
        if size_mb < 1:  # 小于 1MB 的小文本，切细一点方便精准命中
            chunk_size = 500
            chunk_overlap = 50
        else:
            chunk_size = 1000
            chunk_overlap = 100

    return chunk_size, chunk_overlap


# ================= 3. 主处理工厂 =================

def load_and_split_document(file_path):
    """
    智能文档加载与切分工厂（动态策略版）
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(file_path)

    # 字典映射管理文件处理器
    loader_mapping = {
        '.pdf': load_pdf,
        '.txt': load_txt,
        '.docx': load_docx,
        '.csv': load_csv
    }

    if file_extension not in loader_mapping:
        raise ValueError(f"系统暂不支持 {file_extension} 格式的文件解析！")

    # 1. 加载文档
    load_function = loader_mapping[file_extension]
    docs = load_function(file_path)

    # 2. 动态获取切分参数
    chunk_size, chunk_overlap = get_split_strategy(file_extension, file_size_bytes)
    print(
        f"⚙️ 动态切分触发 -> 格式: {file_extension}, 大小: {file_size_bytes / 1024 / 1024:.2f} MB | Chunk: {chunk_size}, Overlap: {chunk_overlap}")

    # 3. 实例化切分器并切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)

    # 4. 过滤掉内容过短的无效块
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    # ================= 🌟 核心升级：动态区域打标与元数据清洗 =================
    is_reference_section = False  # 状态开关：默认一开始都是正文

    for chunk in chunks:
        # 清洗掉列表和字典（防止 Chroma 报错）
        for key, value in list(chunk.metadata.items()):
            if isinstance(value, (list, dict)):
                del chunk.metadata[key]

        # 将文本转小写，方便匹配
        content_lower = chunk.page_content.lower()

        # 启发式规则：如果文本中出现了单独的 "references" 或 "参考文献" 作为小节标题
        # 且通常出现在文档的后半段，我们就把开关打开
        keywords = ["references", "reference", "bibliography", "参考文献"]

        for idx, chunk in enumerate(chunks):
            content_lower = chunk.page_content.lower()

            if any(k in content_lower for k in keywords) and idx > len(chunks) * 0.6:
                is_reference_section = True

            # 根据开关状态，给当前的 chunk 打上 section 标签
            chunk.metadata["section"] = "references" if is_reference_section else "main_body"

    # =======================================================================

    return chunks
