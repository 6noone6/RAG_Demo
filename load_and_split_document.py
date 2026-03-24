import os
import json
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

def load_and_split_document(file_path):
    """
    智能文档加载与切分工厂（巅峰版：接入企业级 AI 视觉识别 API）
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        # ================= 🌟 巅峰改动：接入商业级 AI 视觉解析 API =================
        # 我们用专门针对 RAG 调优的 API，把 PDF 打包传到云端做视觉分析和布局还原。
        loader = UnstructuredLoader(
            file_path=file_path,
            partition_via_api=True,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            url=os.getenv("UNSTRUCTURED_API_URL"),
            # 模式 1: "paged"（适合长文档，按页切）或 "elements"（按语义元素切，最精准）
            mode="single",
            # 策略： "hi_res"（调用顶级视觉模型，精准还原表格和布局，虽然稍微耗Token但效果炸裂）
            strategy="hi_res"
        )
        # =========================================================================

    elif file_extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"系统暂不支持 {file_extension} 格式的文件解析！")

    docs = loader.load()

    # 因为 Unstructured 的 elements 模式已经把文档拆得很干净了，这里的切分参数可以稍微调大一点
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    # ================= 🌟 核心修复逻辑：清洗元数据 =================
    for chunk in chunks:
        # 遍历元数据的每一个键值对
        # 复制一份 key 列表，防止在循环中删除元素导致报错
        for key, value in list(chunk.metadata.items()):
            # 如果值是 列表 或 字典，Chroma 处理不了
            if isinstance(value, (list, dict)):
                # 方案 A：直接删掉（推荐，因为我们检索通常用不到这些链接）
                del chunk.metadata[key]
                # 方案 B：转成字符串（如果你非要保留这些信息）
                # chunk.metadata[key] = json.dumps(value, ensure_ascii=False)
    # ============================================================
    return chunks