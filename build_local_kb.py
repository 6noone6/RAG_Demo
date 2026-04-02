import hashlib
import os
from datetime import datetime

from load_and_split_document import load_and_split_document
from Embedding_Vector_Store import create_persistent_db
from dotenv import load_dotenv

load_dotenv()


def build_offline_knowledge_base(data_folder="data", persist_dir="chroma_db_dynamic"):
    """
    离线知识库构建脚本：一键将文件夹内的所有文档向量化并持久化到硬盘
    """
    # 1. 检查文件夹是否存在
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"📁 已自动创建数据文件夹 '{data_folder}'。")
        print("👇 请把你的 PDF、TXT、DOCX 或 CSV 文件放进这个文件夹，然后重新运行本脚本！")
        return

    allowed_ext = (".pdf", ".txt", ".docx", ".csv")

    valid_files = [
        f for f in os.listdir(data_folder)
        if f.endswith(allowed_ext)
    ]

    if not valid_files:
        print(f"⚠️ '{data_folder}' 文件夹是空的，请放入文档后再试。")
        return

    # all_chunks = []
    print(f"🚀 开始扫描 '{data_folder}' 文件夹，共发现 {len(valid_files)} 个文件...")

    # 2. 遍历文件夹，挨个解析文件
    for file_name in valid_files:
        file_path = os.path.join(data_folder, file_name)
        print(f"📄 正在解析: {file_name} ...")

        try:
            # 调用你之前写好的终极加载与切分工厂
            chunks = load_and_split_document(file_path)

            # ================= 🌟 升级：增强 metadata =================
            for chunk in chunks:
                chunk.metadata.update({
                    "source": file_name,
                    "file_type": os.path.splitext(file_name)[1],
                    "ingest_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "doc_id": hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
                })

            # ========================================================

            # ================= 🌟 新增：分批写入 =================
            print(f"   💾 写入 {len(chunks)} 个 chunks...")
            create_persistent_db(chunks=chunks, persist_dir=persist_dir)
            # ====================================================

            print(f"   ✅ {file_name} 解析成功，切分出 {len(chunks)} 个文本块。")
        except Exception as e:
            print(f"   ❌ {file_name} 解析失败跳过: {e}")


if __name__ == "__main__":
    build_offline_knowledge_base()
