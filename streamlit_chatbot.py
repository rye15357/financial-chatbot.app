import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from fuzzywuzzy import process
from datetime import datetime
import openai
import torch
import re
import subprocess
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time 
import hashlib
import sys
from lang_pack import LANG_PACK
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shutil

# ====== 中文字型設定（支援本地/雲端）======
FONT_PATH = os.path.join(os.path.dirname(__file__), "NotoSansTC-Regular.ttf")
if os.path.exists(FONT_PATH):
    prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'sans-serif']
else:
    prop = None
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# === .env & Token 初始化 ===
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
newsapi_key = os.getenv("NEWSAPI_KEY")

def login_hf(token):
    try:
        login(token=token)
    except Exception as e:
        st.error(f"Hugging Face Token 登入失敗：{e}")
        st.stop()

if not token:
    st.error("請設定 HUGGINGFACEHUB_API_TOKEN")
    st.stop()
if not openai_api_key:
    st.error("請設定 OPENAI_API_KEY")
    st.stop()

# 只在 Session 第一次 login
if "hf_logged_in" not in st.session_state:
    login_hf(token)
    st.session_state["hf_logged_in"] = True

openai.api_key = openai_api_key

# 路徑初始化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
os.makedirs(VECTOR_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MAPPING_PATH = os.path.join(BASE_DIR, "company_mapping.json")

def load_company_mapping():
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_company_mapping(mapping):
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

company_mapping = load_company_mapping()  # { safe_folder: 中文名 }

# ==== 聊天紀錄相關 ====

def ask_openai(prompt, model="gpt-4o", temperature=0.7, lang="繁體中文", history=None):
    # 動態多段落分組 system prompt
    if lang == "繁體中文":
        system_prompt = (
            "你是一位頂尖的財經產業分析師，專精台灣/全球上市櫃公司、半導體與AI產業。"
            "請根據用戶問題與輸入資料，**自動分段落（每段有明確小標題+條列重點）**，"
            "常見如：財務數據、業務亮點、趨勢、挑戰、展望、風險、投資觀點、產業背景...（依內容自動決定）"
            "每段落標題可加 emoji，內容用條列清楚呈現，無資料時請直接說明。"
            "回答格式請用 markdown，**不要有多餘寒暄或贅詞**。"
        )
    else:
        system_prompt = (
            "You are a top financial/industry analyst. For each user question and data, "
            "dynamically group your answer into multiple sections, each with a clear headline (optionally emoji) and bullet points. "
            "Section topics may include: Financials, Highlights, Trends, Outlook, Risk, Analyst View, Background, etc. (decide based on context)."
            "Output in markdown. No chit-chat. State 'No clear info' if unavailable."
        )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    client = openai.OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1200
    )
    return completion.choices[0].message.content.strip()



def get_cached_summary_path(company):
    return os.path.join(VECTOR_DIR, company, "summary.txt")

def save_summary_to_cache(company, summary):
    summary_path = get_cached_summary_path(company)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

def load_summary_from_cache(company):
    summary_path = get_cached_summary_path(company)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def get_chat_path(user_id, topic):
    user_dir = f"chats/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    return f"{user_dir}/{topic}.json"

def save_chat(messages, user_id, topic):
    path = get_chat_path(user_id, topic)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat(user_id, topic):
    path = get_chat_path(user_id, topic)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def list_topics(user_id):
    user_dir = f"chats/{user_id}"
    if not os.path.exists(user_dir):
        return []
    return [f[:-5] for f in os.listdir(user_dir) if f.endswith('.json')]

def list_users():
    if not os.path.exists("chats"):
        return []
    return [d for d in os.listdir("chats") if os.path.isdir(os.path.join("chats", d))]

# ==== LLM QA & 分析 ====
qa_prompt_template_zh = """
你是一位專業財報 AI 助理。請根據下列財報內容，針對問題用「一句自然、清楚、簡短」的繁體中文回答，不要複製原文，不需要來源與頁碼。

📊 財報內容（參考）：
{context}

🔍 問題：
{question}

請直接說明關鍵數字或結論。如果財報中找不到明確答案，請回覆「查無明確資料」。
"""
qa_prompt_template_en = """
You are a professional financial report AI assistant. Based on the following report content, answer the question with a concise and natural English sentence. Do not copy original text or cite pages.

📊 Financial Content (for reference only):
{context}

🔍 Question:
{question}

Directly state the key figure or conclusion. If no clear answer, reply: "No clear information found."
"""
analysis_prompt_template_zh = """
你是一位財經顧問，根據下列內容進行情緒與前景分析。請用一句話總結，並標示（樂觀／中性／保守）。

📜 財務說明：
{text}
"""
analysis_prompt_template_en = """
You are a financial consultant. Analyze the sentiment and outlook of the following financial statement. Use one sentence and indicate (Optimistic/Neutral/Conservative).

📜 Financial Description:
{text}
"""

@st.cache_resource
def load_llm():
    try:
        model_id = "Qwen/Qwen1.5-1.8B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id or tokenizer.unk_token_id,
            device=0 if torch.cuda.is_available() else -1
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None

@st.cache_resource
def get_qa_bot(DB_FAISS_PATH, lang):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"向量資料庫載入失敗: {e}")
        return None
    llm = load_llm()
    if not llm:
        return None
    if lang == "繁體中文":
        prompt = PromptTemplate(template=qa_prompt_template_zh, input_variables=['context', 'question'])
    else:
        prompt = PromptTemplate(template=qa_prompt_template_en, input_variables=['context', 'question'])
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

@st.cache_resource
def get_analysis_pipeline(lang):
    llm = load_llm()
    if lang == "繁體中文":
        analysis_prompt = PromptTemplate(template=analysis_prompt_template_zh, input_variables=['text'])
    else:
        analysis_prompt = PromptTemplate(template=analysis_prompt_template_en, input_variables=['text'])
    return (analysis_prompt | llm)


def extract_twd_amount(text, query=""):
    keywords = ["營業收入", "收入", "revenue", "sales", "income", "total"]
    try:
        if any(k in query.lower() for k in keywords) or any(k in text for k in keywords):
            match = re.search(r"(營業收入|收入|revenue|sales|income)[^0-9]{0,10}([0-9,\.]+)\s*(億|億元|仟元|千元|百萬元|萬元|元|million|billion)?", text, re.IGNORECASE)
            if match:
                num = match.group(2)
                unit = match.group(3) or ""
                return f"{num}{unit}"
        all_matches = re.findall(r"([0-9,\.]+)\s*(億|億元|仟元|千元|百萬元|萬元|元)", text)
        if all_matches:
            biggest = max(all_matches, key=lambda x: float(x[0].replace(',', '')))
            return f"{biggest[0]}{biggest[1]}"
    except Exception:
        pass
    return None

def extract_eps_amount(text, query=""):
    # 支援EPS問法
    if "eps" in query.lower() or "每股盈餘" in query or "EPS" in text:
        match = re.search(r"(EPS|每股盈餘)[^\d\-\.]{0,10}([\-]?[0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if match:
            return match.group(2)
    return None

def extract_net_income(text, query=""):
    if "淨利" in query or "net income" in query.lower() or "淨利" in text:
        match = re.search(r"(淨利|net income)[^0-9\-]{0,10}([\-]?[0-9,\.]+)\s*(億|百萬|萬元|元)?", text, re.IGNORECASE)
        if match:
            return f"{match.group(2)}{match.group(3) or ''}"
    return None

def extract_gross_margin(text, query=""):
    if "毛利率" in query or "gross margin" in query.lower() or "毛利率" in text:
        match = re.search(r"(毛利率|gross margin)[^\d\-\.]{0,10}([\-]?[0-9\.]+)\s*%", text, re.IGNORECASE)
        if match:
            return f"{match.group(2)}%"
    return None

def extract_amount_by_type(text, query=""):
    # 尋找所有有單位的數字，避免年份
    matches = re.findall(r"([0-9,\.]+)\s*(億|億元|仟元|千元|百萬|百萬元|萬元|元|million|billion|%)", text)
    results = []
    for num, unit in matches:
        try:
            val = float(num.replace(",", ""))
            # 年份範圍通常不會有單位，但還是保守排除1900~2100
            if 1900 <= val <= 2100:
                continue
            results.append(f"{num}{unit}")
        except:
            continue
    if results:
        return results[0]
    return None


def safe_folder_name(name):
    ascii_name = re.sub(r"[^\w]", "_", name)
    ascii_name = re.sub(r"[^a-zA-Z0-9_]", "_", ascii_name)
    hash_part = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_name.lower()}_{hash_part}"



def get_company_year_data(companies, indicator, years):
    """
    查詢多家公司多年度單一指標（如營收/EPS）資料來源：Web > OpenAI > PDF
    參數：
        companies: ["台積電", "鴻海"]
        indicator: "營收" 或 "Revenue"
        years: [2021,2022,2023]
    回傳：DataFrame 欄位 ["公司", "年度", indicator, "來源"]
    """
    rows = []
    search_map = {
        "營收": "營收", "Revenue": "營收",
        "淨利": "淨利", "Net Income": "淨利",
        "EPS": "EPS",
        "毛利率": "毛利率", "Gross Margin": "毛利率"
    }
    for company in companies:
        for year in years:
            # 1. 查 Web
            query = f"{company} {year}年 {search_map[indicator]}"
            web_result = web_search(query)
            match = re.search(r'([0-9,\.]+)\s*(億|仟元|千元|百萬|萬元|元|%)', web_result)
            if match:
                value, unit = match.groups()
                val = float(value.replace(",", ""))
                if "%" in unit:
                    pass
                else:
                    multiplier = {
                        "億": 1e8, "仟元": 1e3, "千元": 1e3,
                        "百萬": 1e6, "萬元": 1e4, "元": 1
                    }.get(unit, 1)
                    val = val * multiplier
                rows.append({"公司": company, "年度": year, indicator: val, "來源": "網路"})
                continue

            # 2. 查 GPT-4o
            prompt = f"請列出{company} {year}年{indicator}（單位：億元或%），只給數字，不要說明，回傳格式：{year},{indicator}=?"
            gpt_result = ask_openai(prompt, model="gpt-4o")
            match2 = re.search(r'([0-9,\.]+)\s*(億|仟元|千元|百萬|萬元|元|%)', gpt_result)
            if match2:
                value, unit = match2.groups()
                val = float(value.replace(",", ""))
                if "%" in unit:
                    pass
                else:
                    multiplier = {
                        "億": 1e8, "仟元": 1e3, "千元": 1e3,
                        "百萬": 1e6, "萬元": 1e4, "元": 1
                    }.get(unit, 1)
                    val = val * multiplier
                rows.append({"公司": company, "年度": year, indicator: val, "來源": "GPT-4o"})
                continue

            # 3. 查 PDF（本地）
            safe_names = [k for k, v in (company_mapping or {}).items() if v == company]
            if safe_names:
                safe_name = safe_names[0]
                DB_FAISS_PATH = os.path.join(VECTOR_DIR, safe_name, "db_faiss")
                if os.path.exists(DB_FAISS_PATH):
                    qa = get_qa_bot(DB_FAISS_PATH, "繁體中文")
                    if qa:
                        pdf_result = qa.invoke({"query": f"{year}年{indicator}多少"})
                        pdf_val = extract_amount_by_type(str(pdf_result), f"{year}年{indicator}")
                        if pdf_val:
                            try:
                                val = float(re.sub(r"[^\d\.]", "", pdf_val))
                                rows.append({"公司": company, "年度": year, indicator: val, "來源": "PDF"})
                            except:
                                pass
    df = pd.DataFrame(rows)
    return df

def extract_company_name(user_input, company_name_list):
    # 直接命中
    for name in sorted(company_name_list, key=len, reverse=True):
        if name in user_input:
            return name
    # 股票代碼
    code_to_name = {code: name for code, name in company_mapping.items()}
    stock_code_match = re.search(r"\d{4}", user_input)
    if stock_code_match and stock_code_match.group(0) in code_to_name:
        return code_to_name[stock_code_match.group(0)]
    # Fuzzy match
    result = process.extractOne(user_input, company_name_list, score_cutoff=80)
    if result:
        match, score = result
        return match
    return None

def goodinfo_web_search(query, max_len=500):
    """
    Goodinfo! 財報數據（僅支持股票代碼，如2330）
    """
    try:
        stock_id = re.search(r"\d{4}", query)
        if not stock_id:
            return "請輸入股票代碼（如2330）"
        code = stock_id.group(0)
        url = f"https://goodinfo.tw/tw/StockFinDetail.asp?RPT_CAT=XX_M_QUAR_ACC&STOCK_ID={code}"
        resp = requests.get(url, timeout=8, headers={"user-agent":"Mozilla/5.0"})
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="b1 p4_2 r0_10 row_mouse_over")
        if not table:
            return "查無財報"
        text = table.get_text(separator="\n", strip=True)
        return f"【Goodinfo!】\n{text[:max_len]}\n{url}"
    except Exception as e:
        return f"Goodinfo! 查詢失敗: {e}"


def cnyes_web_search(query, max_len=600):
    """
    鉅亨網關鍵字爬蟲，只回傳第一筆新聞標題＋摘要＋連結
    """
    try:
        url = f"https://search.cnyes.com/news/newslist?q={query}&t=keyword"
        resp = requests.get(url, timeout=8, headers={"user-agent": "Mozilla/5.0"})
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        item = soup.select_one(".tabList .newsItem")
        if not item:
            return "查無新聞"
        title_tag = item.select_one(".newsItem__title")
        desc_tag = item.select_one(".newsItem__summury")
        link_tag = item.select_one("a")
        title = title_tag.text.strip() if title_tag else ""
        desc = desc_tag.text.strip() if desc_tag else ""
        link = "https://news.cnyes.com" + link_tag['href'] if link_tag else ""
        return f"【鉅亨網】\n{title}\n{desc[:max_len]}\n{link}"
    except Exception as e:
        return f"鉅亨網查詢失敗: {e}"

def twse_api_search(query):
    """
    證交所公開資訊觀測站 OpenAPI
    官方說明：https://openapi.twse.com.tw/
    """
    try:
        stock_id = re.search(r"\d{4}", query)
        if not stock_id:
            return "請輸入股票代碼（如2330）"
        code = stock_id.group(0)
        url = f"https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        for item in data:
            if item.get("公司代號") == code:
                summary = "\n".join([f"{k}: {v}" for k, v in item.items()])
                return f"【證交所 API】\n{summary}"
        return "查無公開資訊"
    except Exception as e:
        return f"證交所 API 查詢失敗: {e}"


def check_company_status_tw(company_name):
    try:
        url = f"https://www.tw-inc.com/company/search?q={company_name}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        result_block = soup.select_one(".company-list .item")
        if not result_block:
            return "查無資料", url
        status_text = result_block.text
        if "解散" in status_text or "撤銷" in status_text or "廢止" in status_text:
            return "解散／已結束營業", url
        elif "核准設立" in status_text or "營業中" in status_text:
            return "營業中", url
        else:
            return "未知狀態", url
    except Exception:
        return "查無資料", ""



def newsapi_search(query, api_key=None, max_len=500):
    """
    NewsAPI 國際新聞聚合查詢（需註冊取得 API KEY）
    """
    if not api_key:
        return "請設定 NewsAPI API KEY"
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "language": "zh", "apiKey": api_key, "pageSize": 1}
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()
        if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
            article = data["articles"][0]
            title = article["title"]
            desc = article["description"] or ""
            link = article["url"]
            return f"【NewsAPI】\n{title}\n{desc[:max_len]}\n{link}"
        return "查無新聞"
    except Exception as e:
        return f"NewsAPI 查詢失敗: {e}"


def synthesize_answers(query, search_results, lang="繁體中文"):
    prompt = (
        "請根據以下兩個來源資料，幫我條列彙整重點並總結結論（不用附網址）：\n\n"
        if lang == "繁體中文" else
        "Based on the following two sources, summarize key points and provide a conclusion (no links):\n\n"
    )
    for src, content in search_results:
        prompt += f"【{src}】\n{content}\n\n"
    prompt += "請強調關鍵數據與不同觀點，最後用你自己的專業語氣總結。"
    return ask_openai(prompt, lang=lang)


def extract_status_from_text(text):
    # 解散關鍵字
    if any(k in text for k in ["解散", "歇業", "撤銷", "結束營業", "廢止", "已不存在", "已下市"]):
        return "已解散"
    # 營業中關鍵字
    if any(k in text for k in ["營業中", "核准設立", "現存", "存續", "上市", "上櫃", "登記設立"]):
        return "營業中"
    # 台灣公司網有公司資訊
    if "統編" in text and ("有限公司" in text or "股份有限公司" in text or "公司" in text):
        return "營業中"
    # 有公司地址、負責人
    if any(k in text for k in ["負責人", "地址", "設立"]):
        return "營業中"
    return "查無"


def get_latest_company_status_from_sources(news_results):
    # 遍歷多個來源回傳的內容，自動判斷公司狀態
    status_list = []
    for src, content in news_results:
        status = extract_status_from_text(content)
        status_list.append(status)
    # 優先已解散，其次營業中
    if "已解散" in status_list:
        return "已解散"
    if "營業中" in status_list:
        return "營業中"
    return "查無"


def integrated_ai_summary(user_input, DB_FAISS_PATH, multi_lang):
    # 1. 多來源搜尋
    news_results = multi_source_search(user_input)
    status_str = get_latest_company_status_from_sources(news_results)
    status_bar = ""
    if status_str == "已解散":
        status_bar = "🔴 **公司目前狀態：已解散／結束營業**\n\n"
    elif status_str == "營業中":
        status_bar = "🟢 **公司目前狀態：營業中**\n\n"


    # ========== 台灣公司網查無資料提醒 ==========
    tw_company_result = ""
    for src, content in news_results:
        if src == "台灣公司網":
            tw_company_result = content
            break
    # 擴充判斷條件
    no_company_info = (
        "查無公司資料" in tw_company_result or
        "查無資料" in tw_company_result or
        "公司不存在" in tw_company_result or
        "找不到該公司" in tw_company_result
    )
    if no_company_info:
        missing_company_msg = (
            "⚠️ 找不到該公司營運資訊，可能已歇業、解散、撤銷、改名或資料已下架。請確認公司名稱正確。\n\n"
            if multi_lang == "繁體中文"
            else "⚠️ Company information not found; it may be dissolved, revoked, renamed, or removed. Please check the company name.\n\n"
        )
    else:
        missing_company_msg = ""

    context_text = ""
    for src, content in news_results:
        context_text += f"[{src}] {content}\n"

    # 2. PDF 本地財報
    pdf_summary = ""
    if DB_FAISS_PATH and os.path.exists(DB_FAISS_PATH):
        qa = get_qa_bot(DB_FAISS_PATH, multi_lang)
        if qa:
            result = qa.invoke({"query": user_input})
            if isinstance(result, dict):
                pdf_summary = result.get("result") or result.get("output_text") or ""
            else:
                pdf_summary = result if isinstance(result, str) else str(result)
    if pdf_summary:
        context_text += f"[PDF 財報] {pdf_summary}\n"

    # 3. 丟給 AI，自動分段 + 小標 + 條列，無資料說明
    prompt = (
        f"{missing_company_msg}" + 
        (
            "請根據所有以下資料，自動分段回答（每段有明確小標題+條列重點），"
            "段落數與標題內容請根據問題動態決定，可包含：數據重點、亮點、趨勢、風險、比較、展望、總結等，"
            "無資料時請說明，全部用 markdown 格式，不要有寒暄。"
            if multi_lang == "繁體中文" else
            "Based on ALL the following sources, auto-group answer into multiple sections (each with a headline & bullet points). "
            "Section count and topics depend on the question, can include: data, highlights, trends, risks, comparison, outlook, summary, etc. "
            "Say 'No clear data' if nothing found. Output markdown only."
        )
    )
    ai_ans = ask_openai(
        f"{prompt}\n\n用戶問題：{user_input}\n\n{context_text}",
        lang=multi_lang
    )
    return status_bar + ai_ans



def multi_source_search(query):
    results = []
    # 1. Google
    results.append(("Google", web_search(query)))
    # 2. Yahoo財經
    results.append(("Yahoo財經", yahoo_finance_web_search(query)))
    # 3. 台灣公司網
    results.append(("台灣公司網", taiwan_company_web_search(query)))
    # 4. 公開資訊觀測站 (MOPS)
    results.append(("公開資訊觀測站", mops_web_search(query)))
    # 5. 鉅亨網
    results.append(("鉅亨網", cnyes_web_search(query)))
    # 6. Goodinfo!
    results.append(("Goodinfo!", goodinfo_web_search(query)))
    # 7. 證交所 API
    results.append(("證交所 API", twse_api_search(query)))
    # 8. NewsAPI 
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        results.append(("NewsAPI", newsapi_search(query, api_key=newsapi_key)))
    else:
        results.append(("NewsAPI", "未設定 API KEY"))
    return results


def yahoo_finance_web_search(query, max_len=600):
    """
    查詢 Yahoo 財經台灣（tw.stock.yahoo.com），抓公司簡易財報（營收、EPS、淨利等）。
    傳入公司名稱或股票代碼皆可。
    """
    try:
        # 嘗試從 query 抓出公司股票代碼（如2330）
        stock_id = re.search(r"\d{4}", query)
        if stock_id:
            stock_code = stock_id.group(0)
        else:
            # 如果只給公司名要轉股票代碼，可用字典 mapping 或用公開資訊觀測站/台灣公司網補查
            return "請輸入股票代碼或公司名稱"

        url = f"https://tw.stock.yahoo.com/quote/{stock_code}/financial"
        resp = requests.get(url, timeout=8, headers={"user-agent":"Mozilla/5.0"})
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        # 抓財報表格
        table = soup.find("table")
        if not table:
            return "查無財報資料"
        text = table.get_text(separator="\n", strip=True)
        return f"【Yahoo財經】\n{text[:max_len]}\n{url}"
    except Exception as e:
        return f"Yahoo財經查詢失敗: {e}"


def mops_web_search(query, max_len=600):
    """
    用公開資訊觀測站（MOPS）查公司基本資料、重大訊息
    """
    try:
        # 這裡舉例用公開資訊觀測站公司查詢頁面
        search_url = f"https://mops.twse.com.tw/mops/web/t05st01"
        params = {
            "TYPEK": "all",
            "firstin": "true",
            "co_id": "",      
            "keyword": query, 
        }
        # 直接查關鍵字其實有限制，建議可用台灣證券公司代碼對照表輔助
        resp = requests.get(search_url, params=params, timeout=10)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "hasBorder"})
        if not table:
            return "查無公司資料"
        text = table.get_text(separator="\n", strip=True)
        return f"【公開資訊觀測站】\n{text[:max_len]}\n{search_url}"
    except Exception as e:
        return f"公開資訊觀測站查詢失敗: {e}"


def taiwan_company_web_search(query, max_len=500):
    """爬取台灣公司網（twincn.com）關鍵公司資訊"""
    try:
        url = f"https://www.twincn.com/search?q={query}"
        resp = requests.get(url, timeout=8)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        link_tag = soup.select_one("div.r-list > a")
        if not link_tag:
            return "查無公司資料"
        company_url = "https://www.twincn.com" + link_tag['href']
        company_resp = requests.get(company_url, timeout=8)
        company_resp.encoding = "utf-8"
        company_soup = BeautifulSoup(company_resp.text, "html.parser")
        summary_div = company_soup.select_one("div.r-info")
        summary_text = summary_div.get_text(separator="\n", strip=True) if summary_div else ""
        return f"【台灣公司網】\n{summary_text[:max_len]}\n{company_url}"
    except Exception as e:
        return f"台灣公司網查詢失敗: {e}"


def web_search(query, retry=2):
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    for i in range(retry+1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=8)
            data = resp.json()
            results = []
            all_snippets = ""
            for r in data.get("organic", [])[:3]:
                title = r.get("title")
                link = r.get("link")
                snippet = r.get("snippet", "")
                results.append(f"【{title}】\n{snippet}\n{link}")
                all_snippets += snippet + " "
            match = re.search(r'([0-9,\.]+)\s*(億|億元|仟元|千萬元|百萬元|萬元|元)', all_snippets)
            if match:
                amount = match.group(0)
                return f"\U0001f310 來源：網路\n\U0001f4b0 金額擷取：**{amount}**\n\n" + "\n\n".join(results)
            return "\U0001f310 來源：網路\n" + "\n\n".join(results) if results else "❌ 無法從網路查得資料"
        except Exception as e:
            if i < retry:
                continue
            return f"❌ 網路查詢錯誤：{e}"

def parse_web_search_result(news_result, query=""):
    amount = extract_amount_by_type(news_result, query)
    news_list = []
    source_snippets = []
    news_pattern = r"【(.+?)】\n(.+?)\n(https?://[^\s]+)"
    for m in re.finditer(news_pattern, news_result):
        news_list.append({
            "title": m.group(1),
            "desc": m.group(2),
            "link": m.group(3)
        })
        source_snippets.append(m.group(2))
    return amount or "（查無）", news_list, source_snippets


def build_ai_reply(company, user_input, answer, amount, growth, news_list, source_snippets, T):
    bullets = []
    if answer:
        bullets.append(f"{company} {T['revenue']} {amount}, {T['yoy_growth']} {growth}.")
        bullets.append(T["ai_bullet1"])
        bullets.append(T["ai_bullet2"])
    else:
        bullets.append(T["no_data"])
    news_md = "\n".join([f"- [{n['title']}]({n['link']})：{n['desc']}" for n in news_list]) if news_list else T["no_news"]
    sources_md = "\n".join([f"> {s}" for s in source_snippets]) if source_snippets else ""
    reply = f"""

{T['ai_reply_title']}

---
{T['key_data_card']}
- **{T['company']}**：{company}
- **{T['query']}**：{user_input}
- **{T['revenue']}**：<span style="color:orange;font-weight:bold;">{amount}</span>
- **{T['yoy_growth']}**：{growth}

---

{T['ai_bullets']}
{chr(10).join([f"{i+1}. {b}" for i, b in enumerate(bullets)])}

---

{T['news_sources']}
{news_md}

---

{sources_md}
{T['also_ask']}

---
{T['ai_footer']}
"""
    return reply


# ==== 公司/向量/語言 ====
def get_companies_list():
    if not os.path.exists(VECTOR_DIR):
        return []
    return [d for d in os.listdir(VECTOR_DIR) if os.path.isdir(os.path.join(VECTOR_DIR, d))]

# ==== Streamlit 介面 ====
st.set_page_config(page_title="財報助理AI", page_icon=":robot_face:", layout="wide")
multi_lang = st.sidebar.selectbox(
    LANG_PACK["繁體中文"]["language_select"],
    list(LANG_PACK.keys()),
    key="lang_select"
)
if "lang_last" not in st.session_state or st.session_state["lang_last"] != multi_lang:
    st.session_state["lang_last"] = multi_lang
    st.rerun()
T = LANG_PACK[multi_lang]

# --- Sidebar: 用戶/主題 ---
st.sidebar.markdown("## ⚙️ " + T["model_setting"])
model_options = ["Qwen1.8", "OpenAI GPT-4o"]
selected_model = st.sidebar.selectbox(T["model_select"], model_options, key="model_select")

st.sidebar.header(T["user_and_topic"])
users = list_users()
if not users:
    user_id = st.sidebar.text_input(T["user_input"], value="guest", key="user_id_new")
    if st.sidebar.button(T["add_user"]):
        if user_id:
            os.makedirs(f"chats/{user_id}", exist_ok=True)
            st.success(f"{T['add_user']}：{user_id}")
            st.rerun()
else:
    user_options = users + [T["add_user"]]
    user_id = st.sidebar.selectbox("👤 " + T["user_label"], user_options, index=0, key="user_select")

    if user_id == T["add_user"]:
        new_user = st.sidebar.text_input(T["user_input"], key="new_user_id")
        if st.sidebar.button("✅ " + T["add_user_btn"], key="add_user_btn"):
            if new_user and new_user not in users:
                os.makedirs(f"chats/{new_user}", exist_ok=True)
                st.success(f"✅ {T['add_user']}：{new_user}")
                st.rerun()
            else:
                st.warning(T["user_input"])
    else:
        st.session_state["user_id"] = user_id

# ==== 📁 主題選擇 ====
all_add_topic_names = [v["add_topic"] for v in LANG_PACK.values()] + ["+ 新增主題", "+ Add Topic"]
topics = list_topics(user_id)
topic_options = [t for t in topics if t not in all_add_topic_names]
topic_options.append(T["add_topic"])
topic = st.sidebar.selectbox(
    "📁 " + T["topic_label"],
    topic_options,
    index=0,
    key=f"topic_select_{user_id}"
)


if topic == T["add_topic"]:
    topic_new = st.sidebar.text_input(T["topic_input"], key="new_topic_name")
    if st.sidebar.button("✅ " + T["confirm_add_topic"], key="add_topic_btn"):
        path = f"chats/{user_id}/{topic_new}.json"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.success(f"✅ 已新增主題：{topic_new}")
            st.rerun()
        else:
            st.warning("⚠️ 主題已存在" if multi_lang == "繁體中文" else "⚠️ Topic already exists")
else:
    st.session_state["topic"] = topic

# 刪除用戶（兩階段確認）
delete_user_key = f"delete_user_confirm_{user_id}"
if user_id and user_id != T["add_user"]:
    if st.sidebar.button(f"{T['delete_user']} [{user_id}]", key=f"delete_user_btn_{user_id}"):
        st.session_state[delete_user_key] = True
    if st.session_state.get(delete_user_key, False):
        st.sidebar.warning("⚠️ 此操作不可復原，請再次點擊下方按鈕確認！" if multi_lang=="繁體中文" else "⚠️ This cannot be undone, click again to confirm!")
        if st.sidebar.button("⚡️ 確認永久刪除" if multi_lang=="繁體中文" else "⚡️ Confirm Permanent Delete", key=f"delete_user_really_{user_id}"):
            user_dir = os.path.join("chats", user_id)
            try:
                shutil.rmtree(user_dir)
                st.success(f"{T['delete_user']}：{user_id}")
                st.session_state[delete_user_key] = False
                st.rerun()
            except Exception as e:
                st.error(f"刪除用戶失敗：{e}" if multi_lang=="繁體中文" else f"Delete user failed: {e}")


else:
    if st.sidebar.button("切換主題" if multi_lang=="繁體中文" else "Switch Topic"):
        st.session_state["messages"] = load_chat(user_id, topic)
        st.success(f"切換到用戶 [{user_id}] 主題 [{topic}]" if multi_lang=="繁體中文" else f"Switched to user [{user_id}], topic [{topic}]")
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_chat(user_id, topic)

# 刪除主題（兩階段）

delete_topic_key = f"delete_topic_confirm_{topic}"
if topic != T["add_topic"]:  # 排除新增主題選項
    if st.sidebar.button(f"{T['delete_topic']} [{topic}]", key=f"delete_topic_btn_{topic}"):
        st.session_state[delete_topic_key] = True
    if st.session_state.get(delete_topic_key, False):
        st.sidebar.warning("⚠️ 此操作不可復原，請再次點擊下方按鈕確認！" if multi_lang=="繁體中文" else "⚠️ This cannot be undone, click again to confirm!")
        if st.sidebar.button("⚡️ 確認永久刪除" if multi_lang=="繁體中文" else "⚡️ Confirm Permanent Delete", key=f"delete_topic_really_{topic}"):
            chat_file = get_chat_path(user_id, topic)
            try:
                if os.path.exists(chat_file):
                    os.remove(chat_file)
                    st.success(f"{T['delete_topic']}：{topic}")
                    st.session_state[delete_topic_key] = False
                    st.rerun()
                else:
                    st.warning("主題檔案不存在" if multi_lang=="繁體中文" else "Topic file not found")
            except Exception as e:
                st.error(f"刪除主題失敗：{e}" if multi_lang=="繁體中文" else f"Delete topic failed: {e}")


# Sidebar: 公司+PDF
if st.session_state.get("after_build_db"):
    st.session_state["sidebar_company_input"] = ""
    st.session_state["sidebar_pdf_uploader"] = None
    st.session_state["after_build_db"] = False

# ==== 公司PDF上傳 ====
st.sidebar.header(T["upload_pdf"])
company_name = st.sidebar.text_input(T["company_name"], value="", key="sidebar_company_input")
if not company_name.strip():
    st.sidebar.warning(T["pdf_upload_tip"])
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader(T["upload_pdf"], type="pdf", key="sidebar_pdf_uploader")
    if uploaded_file is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = safe_folder_name(company_name)
        pdf_filename = f"{safe_name}_{now}.pdf"
        pdf_path = os.path.join("data", pdf_filename)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("建構向量資料庫中...（請勿重新整理或操作，預計 10-30 秒）" if multi_lang=="繁體中文" else "Building vector DB... Please wait."):
            try:
                result = subprocess.run(
                    [sys.executable, "create_db.py", "--pdf", pdf_path, "--company", company_name],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    company_mapping[safe_name] = company_name
                    save_company_mapping(company_mapping)
                    waited = 0
                    while waited < 5:
                        new_map = load_company_mapping()
                        new_list = get_companies_list()
                        if safe_name in new_map and safe_name in new_list:
                            break
                        time.sleep(0.1)
                        waited += 0.1
                    if not (safe_name in new_map and safe_name in new_list):
                        st.warning("⚠️ 建庫完成但同步延遲，請手動刷新頁面" if multi_lang=="繁體中文" else "⚠️ Sync delay, please refresh the page manually.")
                    st.toast(f"✅ {company_name} 資料庫建立完成！" if multi_lang=="繁體中文" else f"✅ {company_name} database created!", icon="✅")
                    st.session_state["company_selected"] = safe_name
                    # 只設 flag，清空操作到下次 rerun
                    st.session_state["after_build_db"] = True
                    st.rerun()
                else:
                    st.error(f"❌ 資料庫建立失敗：\n{result.stderr}" if multi_lang=="繁體中文" else f"❌ DB build failed:\n{result.stderr}")
            except Exception as e:
                st.error(f"❌ 建庫執行失敗：{e}" if multi_lang=="繁體中文" else f"❌ Build process failed: {e}")


companies = get_companies_list()  
company_mapping = load_company_mapping() 

# 用 mapping 取得所有顯示用中文名（若無就用 safe_name）
display_names = [company_mapping.get(code, code) for code in companies]
# 反查表（顯示名 → safe_name，確保顯示名唯一即可）
display_to_code = {display: code for display, code in zip(display_names, companies)}

print("companies =", companies)
print("company_mapping =", company_mapping)
print("display_names =", display_names)

# 根據 session_state 設定預設 index
selected_safe_name = st.session_state.get("company_selected", None)
if selected_safe_name and selected_safe_name in companies:
    selected_display_name = company_mapping.get(selected_safe_name, selected_safe_name)
else:
    selected_display_name = None

if selected_display_name and selected_display_name in display_names:
    selected_index = display_names.index(selected_display_name) + 1
else:
    selected_index = 0

# 這裡 sidebar 下拉，顯示的只有公司中文名
company_display = st.sidebar.selectbox(
    T["select_company"],
    [T["select_company"]] + display_names,
    index=selected_index,
    key="company_selected_display"
)
if company_display == T["select_company"]:
    DB_FAISS_PATH = None
    company_selected = None
else:
    # 選到的公司 display name → safe name
    company_selected = display_to_code[company_display]
    st.session_state["company_selected"] = company_selected
    DB_FAISS_PATH = os.path.join(VECTOR_DIR, company_selected, "db_faiss")

# 多語切換（顯示在最上面已經有，不需再重複）

# ====== 進階功能/聊天紀錄搜尋/使用建議 ======
with st.sidebar.expander("📌 進階功能" if multi_lang=="繁體中文" else "📌 Advanced"):
    st.markdown("🔎 **聊天紀錄搜尋**" if multi_lang=="繁體中文" else "🔎 **Chat Log Search**")
    search_key = st.text_input(
        "輸入關鍵字搜尋聊天紀錄" if multi_lang=="繁體中文" else "Search keyword in chat logs", 
        key="search_history"
    )
    # 搜尋範圍下拉選單（中英支援）
    if multi_lang == "繁體中文":
        scope_options = ["目前主題", "所有主題", "所有用戶"]
        scope_label = "搜尋範圍"
    else:
        scope_options = ["Current topic", "All topics", "All users"]
        scope_label = "Scope"
    search_scope = st.selectbox(scope_label, scope_options, key="search_scope")

    # 中英 mapping，搜尋時統一用中文做邏輯
    search_scope_mapping = {
        "目前主題": "目前主題", "Current topic": "目前主題",
        "所有主題": "所有主題", "All topics": "所有主題",
        "所有用戶": "所有用戶", "All users": "所有用戶"
    }
    search_results = []

    if search_key:
        scope_internal = search_scope_mapping.get(search_scope, "目前主題")
        if scope_internal == "目前主題":
            history = load_chat(user_id, topic)
            for m in history:
                if search_key in m["content"]:
                    search_results.append({
                        "user": user_id,
                        "topic": topic,
                        "role": m["role"],
                        "content": m["content"]
                    })
        elif scope_internal == "所有主題":
            topics = list_topics(user_id)
            for t in topics:
                history = load_chat(user_id, t)
                for m in history:
                    if search_key in m["content"]:
                        search_results.append({
                            "user": user_id,
                            "topic": t,
                            "role": m["role"],
                            "content": m["content"]
                        })
        elif scope_internal == "所有用戶":
            users = list_users()
            for u in users:
                topics = list_topics(u)
                for t in topics:
                    history = load_chat(u, t)
                    for m in history:
                        if search_key in m["content"]:
                            search_results.append({
                                "user": u,
                                "topic": t,
                                "role": m["role"],
                                "content": m["content"]
                            })

        st.markdown(
            f"共找到 <span style='color:orange;font-weight:bold'>{len(search_results)}</span> 筆：" 
            if multi_lang=="繁體中文" 
            else f"Found <span style='color:orange;font-weight:bold'>{len(search_results)}</span> record(s):", 
            unsafe_allow_html=True
        )
        for i, m in enumerate(search_results):
            st.markdown(
                f"<b>{m['user']}/{m['topic']}</b> | {m['role']}：{m['content'][:100]}{'...' if len(m['content'])>100 else ''}",
                unsafe_allow_html=True
            )
    st.caption(
        "可選搜尋「目前主題」、「所有主題」、「所有用戶」" if multi_lang=="繁體中文" 
        else "You can search in current topic / all topics / all users"
    )

st.sidebar.info(
    """  
💡 使用建議：
- 先上傳公司財報 PDF，自動建立資料庫
- 可切換公司/語言，自由查詢
- 可分主題管理對話與歷史查詢
- 提問後可用 /分析 進行情緒預測
- 回覆內容更貼近 AI 口吻，自然簡明
""" if multi_lang=="繁體中文" else
    """  
💡 Suggestions:
- Upload company financial PDF first to build the database
- Switch company/language freely for queries
- Manage topics and chat history
- Use /analyze for sentiment analysis
- Answers are now more conversational and AI-like
""")

tab1, tab2, tab3 = st.tabs([T["tab1"], T["tab2"], T["tab3"]])

# <<<< 初始化 session_state["messages"]
if "messages" not in st.session_state:
    try:
        st.session_state["messages"] = load_chat(user_id, topic)
    except Exception:
        st.session_state["messages"] = []

# ========== 分頁1：AI問答 ==========

# 初始化分頁1的聊天紀錄
if "messages_tab1" not in st.session_state:
    try:
        st.session_state["messages_tab1"] = load_chat(user_id, topic)
    except Exception:
        st.session_state["messages_tab1"] = []

with tab1:
    st.title(T["chat_title"])
    st.markdown(
        """請輸入您的問題（中英文皆可），例如：\n- 台積電 2024 年第一季的營業收入是多少？\n- What is TSMC's Q1 2024 revenue?\n- /分析 預期未來折舊費用將上升"""
        if multi_lang=="繁體中文" else
        """Ask your question (either language), e.g.:\n- What is TSMC's Q1 2024 revenue?\n- 台積電 2024 年第一季的營業收入是多少？\n- /analyze Depreciation expense is expected to rise"""
    )

    if st.button(T["clear_chat"], key="clear_chat"):
        st.session_state["messages_tab1"] = []
        save_chat(st.session_state["messages_tab1"], user_id, topic)
        st.rerun()

    for chat in st.session_state["messages_tab1"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"], unsafe_allow_html=True)

user_input = st.chat_input(T["chat_input"], key="ai_chat")
if user_input:
    st.session_state["messages_tab1"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("AI 正在查詢 ..." if multi_lang == "繁體中文" else "AI is searching ..."):
        try:
            reply = integrated_ai_summary(user_input, DB_FAISS_PATH, multi_lang)
            # ==這裡原本有公司查核警語區塊，直接移除==
        except Exception as e:
            reply = f"❌ 查詢失敗：{e}" if multi_lang == "繁體中文" else f"❌ Error: {e}"

    st.session_state["messages_tab1"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply, unsafe_allow_html=True)
    save_chat(st.session_state["messages_tab1"], user_id, topic)
    st.toast("已完成多來源查詢" if multi_lang == "繁體中文" else "Multi-source search complete", icon="🤖")


# ========== 分頁2：財報摘要 ==========
with tab2:
    st.title(T["tab2_title"])
    st.markdown(T["tab2_markdown"])

    input_text = st.text_area(
        T["tab2_input_placeholder"], ""
    )

    if st.button(T["tab2_analyze_btn2"]):
        if not input_text.strip():
            st.warning(T["tab2_enter_content"])
        else:
            with st.spinner(T["tab2_spinner_analyze"]):
                # 條列三個重點
                summary_prompt = (
                    "請閱讀下列內容，條列三個經營重點（每點15字內，不要抄原文）：\n\n"
                    if multi_lang == "繁體中文"
                    else "Read the following and list three business highlights (no more than 15 words each, do not copy the original):\n\n"
                ) + input_text
                summary = ask_openai(summary_prompt, lang=multi_lang)

                # 判斷情緒
                sentiment_prompt = (
                    "請閱讀下列內容，判斷經營情緒並只回覆一個詞（樂觀/中性/保守）：\n\n"
                    if multi_lang == "繁體中文"
                    else "Read the following, judge the management sentiment, and only reply with one word (Optimistic/Neutral/Conservative):\n\n"
                ) + input_text
                sentiment = ask_openai(sentiment_prompt, lang=multi_lang)

                # 顯示
                st.markdown(f"#### {T['tab2_key_summary']}")
                st.markdown(summary)
                st.markdown(
                    f"#### {T['tab2_sentiment']}<span style='color:orange;font-weight:bold'>{sentiment}</span>",
                    unsafe_allow_html=True
                )

with tab3:
    st.title(T["tab3"])
    st.markdown(T["tab3_title"])
    indicator_options = ["營收", "淨利", "EPS", "毛利率"] if multi_lang == "繁體中文" else ["Revenue", "Net Income", "EPS", "Gross Margin"]

    st.subheader(T["tab3_multi_title"])
    selected_companies_display = st.multiselect(
        T["tab3_multi_company"],
        display_names,
        default=[display_names[0]] if display_names else [],
        key="tab3_multi_company"
    )
    selected_companies = [display_to_code[d] for d in selected_companies_display]
    compare_indicator = st.selectbox(
        T["tab3_multi_indicator"],
        indicator_options,
        index=0,
        key="compare_indicator"
    )
    year_range = st.slider(
        T["tab3_year"], 
        min_value=2015, 
        max_value=datetime.now().year, 
        value=(2020, datetime.now().year), 
        key="tab3_year_multi"
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        compare_btn_clicked = st.button(T["tab3_multi_btn"], key="compare_btn", use_container_width=True)
    with btn_col2:
        delete_btn_clicked = st.button("🗑️", key="delete_chart_btn", use_container_width=True, help="刪除圖表")

    if "multi_chart_df" not in st.session_state:
        st.session_state["multi_chart_df"] = None

    if compare_btn_clicked:
        if not selected_companies:
            st.warning(T["tab3_multi_no_company"])
            st.session_state["multi_chart_df"] = None
        else:
            with st.spinner(T["tab3_multi_spinner"]):
                selected_display_names = [company_mapping.get(code, code) for code in selected_companies]
                years = list(range(year_range[0], year_range[1]+1))
                # 這裡用中文欄位查詢
                indicator_map = {
                    "Revenue": "營收", "Net Income": "淨利", "EPS": "EPS", "Gross Margin": "毛利率",
                    "營收": "營收", "淨利": "淨利", "毛利率": "毛利率"
                }
                indicator_key = indicator_map.get(compare_indicator, compare_indicator)
                df = get_company_year_data(selected_display_names, indicator_key, years)
                # 如果不是百分比再除以 1e8
                if not df.empty and "%" not in indicator_key:
                    df[indicator_key] = df[indicator_key] / 1e8
                st.session_state["multi_chart_df"] = df

    if delete_btn_clicked:
        st.session_state["multi_chart_df"] = None

    df = st.session_state.get("multi_chart_df", None)
    if df is not None and not df.empty:
        # 這裡 indicator_key 也要用
        indicator_map = {
            "Revenue": "營收", "Net Income": "淨利", "EPS": "EPS", "Gross Margin": "毛利率",
            "營收": "營收", "淨利": "淨利", "毛利率": "毛利率"
        }
        indicator_key = indicator_map.get(compare_indicator, compare_indicator)

        chart_col, slider_col = st.columns([5, 1])
        with slider_col:
            chart_width = st.slider("寬度", min_value=2.0, max_value=8.0, value=3.2, step=0.1, key="chart_width_multi")
            chart_height = st.slider("高度", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="chart_height_multi")
        with chart_col:
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            for company in df["公司"].unique():
                df_c = df[df["公司"] == company].sort_values("年度")
                label = f"{company} ({df_c['來源'].iloc[0]})"
                ax.plot(df_c["年度"], df_c[indicator_key], marker="o", label=label)
            # 標題/軸/圖例都用 prop
            if prop:
                if "%" in indicator_key or indicator_key in ["毛利率", "Gross Margin"]:
                    ax.set_ylabel(indicator_key + ("（%）" if multi_lang == "繁體中文" else " (%)"), fontproperties=prop)
                else:
                    ax.set_ylabel(indicator_key + ("（億元）" if multi_lang == "繁體中文" else " (100M)"), fontproperties=prop)
                ax.set_xlabel("年度" if multi_lang == "繁體中文" else "Year", fontproperties=prop)
                ax.set_title(T["tab3_chart_title"].format(indicator=compare_indicator), fontsize=14, fontproperties=prop)
                ax.legend(
                    fontsize=8,
                    bbox_to_anchor=(1.01, 0.5),
                    loc='center left',
                    borderaxespad=0.,
                    prop=prop
                )
            else:
                if "%" in indicator_key or indicator_key in ["毛利率", "Gross Margin"]:
                    ax.set_ylabel(indicator_key + ("（%）" if multi_lang == "繁體中文" else " (%)"))
                else:
                    ax.set_ylabel(indicator_key + ("（億元）" if multi_lang == "繁體中文" else " (100M)"))
                ax.set_xlabel("年度" if multi_lang == "繁體中文" else "Year")
                ax.set_title(T["tab3_chart_title"].format(indicator=compare_indicator), fontsize=14)
                ax.legend(
                    fontsize=8,
                    bbox_to_anchor=(1.01, 0.5),
                    loc='center left',
                    borderaxespad=0.
                )
            ax.grid(True)
            st.pyplot(fig, use_container_width=False)
            st.write(df)
    elif df is not None and df.empty:
        st.warning(T["tab3_multi_no_data"])