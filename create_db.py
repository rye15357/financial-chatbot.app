import os
import re
import sys
import csv
import json
import argparse
import hashlib
import logging
import pdfplumber
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ==== 日誌設定 ====
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/create_db_{now}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==== 參數與目錄 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
VECTORSTORE_DIR = os.path.join(BASE_DIR, 'vectorstore')
PREVIEW_PATH = os.path.join(DATA_PATH, 'preview')
TABLES_PATH = os.path.join(DATA_PATH, 'tables')
CHUNKS_PATH = os.path.join(DATA_PATH, 'chunks')
os.makedirs(PREVIEW_PATH, exist_ok=True)
os.makedirs(TABLES_PATH, exist_ok=True)
os.makedirs(CHUNKS_PATH, exist_ok=True)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
KEYWORDS_PATH = os.path.join(BASE_DIR, 'keywords.txt')
DEFAULT_KEYWORDS = [
    "營業收入", "租賃負債", "現金及約當現金", "公司債", "使用權資產", "存貨",
    "土地", "建築物", "股東權益", "總資產", "總負債", "金融資產", "權益法投資", "資本支出"
]

# ==== 安全資料夾名 (帶hash+timestamp防止重名) ====
def safe_folder_name(company):
    ascii_name = re.sub(r"[^\w]", "_", company)
    ascii_name = re.sub(r"[^a-zA-Z0-9_]", "_", ascii_name)
    hash_part = hashlib.md5(company.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_name.lower()}_{hash_part}"         # 唯一key，永遠一樣


# ==== 同步 company_mapping.json ====
def update_company_mapping(safe_folder, company_name):
    mapping_path = os.path.join(BASE_DIR, "company_mapping.json")
    try:
        mapping = {}
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        mapping[safe_folder] = company_name
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        logging.info(f"✅ company_mapping.json 更新成功: {safe_folder} → {company_name}")
    except Exception as e:
        logging.error(f"❌ company_mapping.json 寫入錯誤: {e}")

def normalize_currency(text):
    text = re.sub(r'(NT\$|新臺幣|台幣|元整|TWD|台幣)', '新台幣', text)
    text = re.sub(r'([0-9,]+)\s*億元', lambda m: str(int(m.group(1).replace(',',''))*1000000) + '仟元', text)
    text = re.sub(r'([0-9,]+)\s*百萬', lambda m: str(int(m.group(1).replace(',',''))*1000) + '仟元', text)
    text = re.sub(r'([0-9,]+)\s*千萬', lambda m: str(int(m.group(1).replace(',',''))*10000) + '仟元', text)
    text = re.sub(r'([0-9,]+)\s*萬元', lambda m: str(int(m.group(1).replace(',',''))*10) + '仟元', text)
    text = re.sub(r'([0-9,]+)\s*元', lambda m: str(int(m.group(1).replace(',',''))//1000) + '仟元', text)
    return text

def load_keywords():
    if os.path.exists(KEYWORDS_PATH):
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            kws = [line.strip() for line in f if line.strip()]
            logging.info(f"載入自定義關鍵字：{kws}")
            return kws
    else:
        logging.info("未偵測到自定義關鍵字，使用預設")
        return DEFAULT_KEYWORDS

def smart_split(text):
    return [s.strip() for s in re.split(r'(?:(?:\n{2,})|(?:\n\s*(?:[一二三四五六七八九十]、|[IVXLCDM]+\.)\s+))', text) if s.strip()]

def extract_text_from_pdf(pdf_path):
    texts = []
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_text = normalize_currency(page_text)
            texts.append({
                "content": page_text.strip(),
                "page": page_num+1,
                "type": "text"
            })
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                table_rows = []
                for row in table:
                    if row and any(row):
                        row_str = " | ".join([cell.strip() if cell else "" for cell in row])
                        row_str = normalize_currency(row_str)
                        table_rows.append(row_str)
                        tables_data.append({"page": page_num+1, "row": row})
                if table_rows:
                    texts.append({
                        "content": "\n".join(table_rows),
                        "page": page_num+1,
                        "type": "table"
                    })
                    csv_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{page_num+1}_tb{table_idx+1}.csv"
                    with open(os.path.join(TABLES_PATH, csv_name), "w", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f)
                        for row in table:
                            writer.writerow([(cell or '').strip() for cell in row])
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    with open(os.path.join(PREVIEW_PATH, f"{basename}.txt"), "w", encoding="utf-8") as f:
        for obj in texts:
            f.write(f"[p{obj['page']}][{obj['type']}]\n{obj['content']}\n\n")
    with open(os.path.join(TABLES_PATH, f"{basename}.tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables_data, f, ensure_ascii=False, indent=2)
    return texts, False

def extract_key_sections(text_objs, keywords):
    results = []
    for idx, obj in enumerate(text_objs):
        if any(kw in obj['content'] for kw in keywords):
            merged = obj['content']
            if obj['type'] == "text":
                for offset in range(1, 4):
                    if idx + offset < len(text_objs) and text_objs[idx + offset]['page'] == obj['page']:
                        nxt = text_objs[idx + offset]['content']
                        if re.search(r"[0-9,\.]+\s*(億|億元|仟元|千元|百萬元|萬元|元)", nxt):
                            merged += "\n" + nxt
                        else:
                            break
            if obj['type'] == "text" and idx > 0 and text_objs[idx-1]['page'] == obj['page']:
                prev = text_objs[idx-1]['content']
                if len(prev) < 30 and re.search(r'(營業|收入|合計|小計|總額|單位|明細)', prev):
                    merged = prev + "\n" + merged
            results.append({
                "content": merged,
                "page": obj['page'],
                "type": obj['type'],
            })
        if obj['type'] == "table":
            merged_table = obj['content']
            if idx > 0 and text_objs[idx-1]['page'] == obj['page']:
                prev = text_objs[idx-1]['content']
                if len(prev) < 30 and re.search(r'(營業收入|收入明細|合計|小計|項目|單位)', prev):
                    merged_table = prev + "\n" + obj['content']
            if any(kw in merged_table for kw in keywords):
                results.append({
                    "content": merged_table,
                    "page": obj['page'],
                    "type": obj['type'],
                })
    return results

def build_vector_db(all_objs, company, safe_folder):
    logging.info(f"公司原始名: {company} | 資料夾安全名: {safe_folder}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for obj in all_objs:
        chunks = splitter.split_text(obj['content'])
        for split in chunks:
            docs.append(Document(
                page_content=split,
                metadata={
                    "company": company,
                    "page": obj['page'],
                    "type": obj['type'],
                }
            ))
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    company_dir = os.path.join(VECTORSTORE_DIR, safe_folder)
    os.makedirs(company_dir, exist_ok=True)
    company_vector_path = os.path.join(company_dir, 'db_faiss')
    os.makedirs(company_vector_path, exist_ok=True)

    if not os.path.exists(company_vector_path):
        logging.error(f"❌ 建立資料夾失敗: {company_vector_path}")
        raise Exception(f"資料夾不存在: {company_vector_path}")

    db = FAISS.from_documents(docs, embedding)
    db.save_local(company_vector_path)

    with open(os.path.join(CHUNKS_PATH, f"{safe_folder}_chunks.json"), "w", encoding="utf-8") as f:
        json.dump([{
            "content": d.page_content,
            **d.metadata
        } for d in docs], f, ensure_ascii=False, indent=2)
    logging.info(f"✅ 向量庫+分段metadata已存：{company_vector_path}")

def process_single_pdf(pdf_path, company):
    safe_folder = safe_folder_name(company)
    logging.info(f"{company}: {pdf_path}")
    keywords = load_keywords()
    texts, _ = extract_text_from_pdf(pdf_path)
    filtered = extract_key_sections(texts, keywords)
    build_vector_db(filtered, company, safe_folder)
    update_company_mapping(safe_folder, company)
    logging.info(f"🏁 {company} 完成向量庫與資料建置")

def batch_process_all(data_path=DATA_PATH):
    pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith(".pdf")]
    logging.info(f"🔍 批次處理 {len(pdf_files)} 份 PDF 檔案")
    for f in pdf_files:
        company = os.path.splitext(os.path.basename(f))[0]
        try:
            process_single_pdf(os.path.join(data_path, f), company)
        except Exception as e:
            logging.error(f"❌ [{company}] 處理錯誤: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, help="PDF 檔案路徑")
    parser.add_argument("--company", type=str, help="公司名稱")
    parser.add_argument("--all", action='store_true', help="批次處理 data/ 內所有 PDF")
    args = parser.parse_args()
    t0 = datetime.now()

    if args.all:
        batch_process_all()
    elif args.pdf and args.company:
        process_single_pdf(args.pdf, args.company)
    else:
        print("請輸入 --pdf 檔案路徑 及 --company 公司名稱，或 --all 進行批次處理")
        sys.exit(0)
    logging.info(f"🏁 全部處理完成，用時 {(datetime.now()-t0).total_seconds():.2f} 秒")
