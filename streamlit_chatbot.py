import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from datetime import datetime
import openai
import torch
import re
import subprocess
import requests
import pandas as pd
import time 
import hashlib
from lang_pack import LANG_PACK
import json
import matplotlib.pyplot as plt
import shutil

plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei',           # Windows
    'Noto Sans CJK TC',             # Linux/Google Cloud, 很常見
    'AR PL UMing TW',               # Ubuntu 也有
    'sans-serif'                    # 其他預設
]
plt.rcParams['axes.unicode_minus'] = False

# === .env & Token 初始化 ===
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

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
    # 強化 AI 分析師風格
    if lang == "繁體中文":
        system_prompt = (
            "你是一位頂尖的財經產業分析師，專精台灣/全球上市櫃公司、半導體與AI產業。"
            "針對用戶問題，請用以下結構回答：\n"
            "1. 條列【關鍵數據】或【產業重點】\n"
            "2. 條列【影響因素】或【趨勢/挑戰】\n"
            "3. 最後給一段【專業觀點/投資建議】，務必明確、有深度。\n"
            "如果資料不足，請明說『資訊有限』，並根據產業趨勢合理推估。"
            "回覆務必嚴謹、專業，不要杜撰來源，不要用太口語的語氣。"
        )
    else:
        system_prompt = (
            "You are a top financial and industry analyst, specializing in global and Taiwanese listed companies, semiconductors, and AI. "
            "For any question, structure your answer as follows:\n"
            "1. List [Key Figures] or [Industry Highlights]\n"
            "2. List [Factors/Trends/Challenges]\n"
            "3. Conclude with a clear [Analyst View/Investment Suggestion].\n"
            "If data is limited, explicitly say so, and provide reasoned industry-based estimates. "
            "Your tone should be precise and professional, avoid making up data, and do not be overly casual."
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
        max_tokens=1000
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
    # 如果有多個，通常第一個就是
    if results:
        return results[0]
    # 如果沒找到有單位的數字，可以保留你原本的萬用條件
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
            safe_names = [k for k, v in company_mapping.items() if v == company]
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
if topic != T["add_topic"]:  # ✅ 排除新增主題選項
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
# 放在所有 sidebar 輸入元件之前
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
                    ["python", "create_db.py", "--pdf", pdf_path, "--company", company_name],
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



companies = get_companies_list()  # ['___23bc3472', '___0b8f314d']
company_mapping = load_company_mapping()  # {'___23bc3472': '鴻海', ...}

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
    # 先取得中文名
    selected_display_name = company_mapping.get(selected_safe_name, selected_safe_name)
    selected_index = display_names.index(selected_display_name) + 1
else:
    selected_display_name = None  
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
    with st.spinner("AI 正在查詢資料..." if multi_lang == "繁體中文" else "AI is searching..."):
        try:
            if selected_model == "Qwen1.8":
                answer = ""
                amount = "（查無）" if multi_lang == "繁體中文" else "(No data)"
                growth = "（查無）" if multi_lang == "繁體中文" else "(No data)"
                source_snippets = []
                news_list = []
                used_pdf = False
                if DB_FAISS_PATH and os.path.exists(DB_FAISS_PATH):
                    qa = get_qa_bot(DB_FAISS_PATH, multi_lang)
                    if qa:
                        result = qa.invoke({"query": user_input})
                        if isinstance(result, dict):
                            answer = result.get("result") or result.get("output_text") or ""
                            source_documents = result.get("source_documents", [])
                        else:
                            answer = result if isinstance(result, str) else str(result)
                            source_documents = []
                        amount = extract_amount_by_type(answer, user_input)
                        if not amount and source_documents:
                            for doc in source_documents:
                                amount = extract_amount_by_type(doc.page_content, user_input)
                                if amount:
                                    break
                        def extract_growth(text):
                            match = re.search(r"(年增率|年成長率)[\s:：]*([\-0-9\.]+%)", text)
                            return match.group(2) if match else None
                        growth = extract_growth(answer)
                        if not growth and source_documents:
                            for doc in source_documents:
                                growth = extract_growth(doc.page_content)
                                if growth:
                                    break
                        if amount:
                            used_pdf = True
                            source_snippets = []
                            for doc in source_documents[:3]:
                                snippet = doc.page_content.strip()
                                if len(snippet) > 80:
                                    snippet = snippet[:80] + "..."
                                source_snippets.append(snippet)

                # 沒查到 PDF 金額 → 自動補網路
                if not used_pdf:
                    st.info("❗️未在財報PDF中擷取到關鍵金額，已自動補上網路資料" if multi_lang == "繁體中文" else "❗️Not found in PDF, using web data")
                    news_result = web_search(user_input)
                    amount, news_list, source_snippets = parse_web_search_result(news_result, user_input)
                    answer = "（來自網路）" if multi_lang == "繁體中文" else "(From web)"
                    growth = "（查無）" if multi_lang == "繁體中文" else "(No data)"

                def extract_company_from_question(question):
                    known_companies = ["台積電", "鴻海", "聯發科", "聯電", "大立光", "日月光"]
                    for c in known_companies:
                        if c in question:
                            return c
                    return company_selected or ("（自動判斷公司）" if multi_lang == "繁體中文" else "(Auto company detect)")
                company_name = extract_company_from_question(user_input)

                reply = build_ai_reply(
                    company_mapping.get(company_name, company_name),  # 顯示用中文名
                    user_input,
                    answer,
                    amount or ("（查無）" if multi_lang == "繁體中文" else "(No data)"),
                    growth or ("（查無）" if multi_lang == "繁體中文" else "(No data)"),
                    news_list,
                    source_snippets,
                    T
                )

            elif selected_model == "OpenAI GPT-4o":
                # ==== 改：永遠優先網路抓數字，抓到就直接用 ====
                news_result = web_search(user_input)
                amount, news_list, source_snippets = parse_web_search_result(news_result, user_input)

                if amount and amount not in ["（查無）", "(No data)"]:
                    # 如果有數字 → 直接顯示網路數字與來源（可選GPT輔助摘要）
                    gpt_summary = ask_openai(
                        f"根據以下最新公開資訊，請條列本季財報重點、趨勢與專業觀點：\n\n{news_result}",
                        lang=multi_lang,
                        history=[]
                    )
                    reply = f"""🌐 **網路最新數字：**  
- **{amount}**

{news_result}

---

**AI 分析補充：**
{gpt_summary}
"""
                else:
                    # 沒抓到網路數字才用GPT
                    history_msgs = []
                    for m in st.session_state["messages_tab1"]:
                        if m["role"] in ("user", "assistant"):
                            content = m["content"]
                            if len(content) > 1500:
                                content = content[:1500] + "..."
                            history_msgs.append({"role": m["role"], "content": content})
                    answer = ask_openai(user_input, lang=multi_lang, history=history_msgs)
                    reply = f"🌐 **OpenAI GPT 回覆：**\n\n{answer}" if multi_lang == "繁體中文" else f"🌐 **OpenAI GPT Answer:**\n\n{answer}"

            else:
                reply = "請選擇有效的模型。" if multi_lang == "繁體中文" else "Please select a valid model."
        except Exception as e:
            reply = f"❌ 問答過程發生錯誤：{e}" if multi_lang == "繁體中文" else f"❌ Error occurred: {e}"
    st.session_state["messages_tab1"].append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply, unsafe_allow_html=True)
    save_chat(st.session_state["messages_tab1"], user_id, topic)
    st.toast("已完成回答" if multi_lang == "繁體中文" else "Answer complete", icon="🤖")


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

# ========== 分頁3：財報圖表 ==========
with tab3:
    st.title(T["tab3"])
    st.markdown(T["tab3_title"])
    indicator_options = ["營收", "淨利", "EPS", "毛利率"] if multi_lang == "繁體中文" else ["Revenue", "Net Income", "EPS", "Gross Margin"]

    # ...選公司略...

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

    # --- 比較/刪除兩顆按鈕同一行 ---
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        compare_btn_clicked = st.button(T["tab3_multi_btn"], key="compare_btn", use_container_width=True)
    with btn_col2:
        delete_btn_clicked = st.button("🗑️", key="delete_chart_btn", use_container_width=True, help="刪除圖表")

    # 存放圖表資料
    if "multi_chart_df" not in st.session_state:
        st.session_state["multi_chart_df"] = None

    # 查詢與刪除動作
    if compare_btn_clicked:
        if not selected_companies:
            st.warning(T["tab3_multi_no_company"])
            st.session_state["multi_chart_df"] = None
        else:
            with st.spinner(T["tab3_multi_spinner"]):
                selected_display_names = [company_mapping.get(code, code) for code in selected_companies]
                years = list(range(year_range[0], year_range[1]+1))
                df = get_company_year_data(selected_display_names, compare_indicator, years)
                if not df.empty and "%" not in compare_indicator:
                    df[compare_indicator] = df[compare_indicator] / 1e8
                st.session_state["multi_chart_df"] = df

    if delete_btn_clicked:
        st.session_state["multi_chart_df"] = None

    df = st.session_state.get("multi_chart_df", None)
    # 圖表與寬高滑條同一行（只有查詢過才出現）
    if df is not None and not df.empty:
        chart_col, slider_col = st.columns([5, 1])
        # 滑條只要有圖就一直在旁邊、即時更新
        with slider_col:
            chart_width = st.slider("寬度", min_value=2.0, max_value=8.0, value=3.2, step=0.1, key="chart_width_multi")
            chart_height = st.slider("高度", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="chart_height_multi")
        with chart_col:
            fig, ax = plt.subplots(figsize=(chart_width, chart_height))
            for company in df["公司"].unique():
                df_c = df[df["公司"] == company].sort_values("年度")
                label = f"{company} ({df_c['來源'].iloc[0]})"
                ax.plot(df_c["年度"], df_c[compare_indicator], marker="o", label=label)
            if "%" in compare_indicator or compare_indicator in ["毛利率", "Gross Margin"]:
                ax.set_ylabel(compare_indicator + ("（%）" if multi_lang == "繁體中文" else " (%)"))
            else:
                ax.set_ylabel(compare_indicator + ("（億元）" if multi_lang == "繁體中文" else " (100M)"))
            ax.set_xlabel("年度" if multi_lang == "繁體中文" else "Year")
            ax.set_title(T["tab3_chart_title"].format(indicator=compare_indicator), fontsize=14)
            # ==== legend 放圖外右側、字體縮小 ====
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

