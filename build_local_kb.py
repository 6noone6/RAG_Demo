import os
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

    valid_files = [f for f in os.listdir(data_folder) if not f.startswith('.')]

    if not valid_files:
        print(f"⚠️ '{data_folder}' 文件夹是空的，请放入文档后再试。")
        return

    all_chunks = []
    print(f"🚀 开始扫描 '{data_folder}' 文件夹，共发现 {len(valid_files)} 个文件...")

    # 2. 遍历文件夹，挨个解析文件
    for file_name in valid_files:
        file_path = os.path.join(data_folder, file_name)
        print(f"📄 正在解析: {file_name} ...")

        try:
            # 调用你之前写好的终极加载与切分工厂
            chunks = load_and_split_document(file_path)

            # 🌟 顺手把文件名写进 metadata，方便前端完美展示溯源！
            for chunk in chunks:
                chunk.metadata["source"] = file_name

            all_chunks.extend(chunks)
            print(f"   ✅ {file_name} 解析成功，切分出 {len(chunks)} 个文本块。")
        except Exception as e:
            print(f"   ❌ {file_name} 解析失败跳过: {e}")

    # 3. 统一写入 Chroma 数据库并持久化
    if all_chunks:
        print(f"\n🧠 所有文件解析完毕！总计获取 {len(all_chunks)} 个文本块。")
        print("💾 正在调用 Embedding 模型计算向量，并写入本地 SQLite 硬盘...")

        # 调用你写好的持久化函数
        create_persistent_db(chunks=all_chunks, persist_dir=persist_dir)

        print("\n🎉 灌库大功告成！")
        print("👉 现在你可以去运行 `streamlit run app.py`，并在网页左侧选择【加载本地大底座 (硬盘)】进行对话了！")
    else:
        print("\n⚠️ 未获取到任何有效文本块，请检查文档内容。")


if __name__ == "__main__":
    build_offline_knowledge_base()