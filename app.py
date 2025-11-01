from dotenv import load_dotenv

load_dotenv()

# app.py
# -------------------------------------------------------
# 使い方：
# 1) 「専門家モード」を A / B から選ぶ
# 2) テキストを入力
# 3) 送信で、選んだ専門家の観点で回答が表示される
# -------------------------------------------------------

# --- 指示どおりの dotenv 呼び出し。ただし未導入でも落ちないように no-op フォールバック ---
try:
    from dotenv import load_dotenv  # 外部依存
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
load_dotenv()

import os               # 標準ライブラリのみ
import importlib        # 標準
import traceback        # 標準
import streamlit as st  # Cloud 既定で利用可能

# Streamlit の最初の呼び出しは page_config（他の st.* より前）
try:
    st.set_page_config(page_title="Expert Mode LLM App", page_icon="🧠", layout="centered")
except Exception:
    pass  # 古いStreamlitでも落とさない

# -------------------------
# 画面ヘッダ（ここまでで UI は必ず表示される）
# -------------------------
st.title("🧠 Expert Mode LLM App")
st.caption("A/B から専門家モードを選び、テキストを送信すると、選んだ専門家の観点で回答します。")

with st.expander("ℹ️ このアプリの概要と操作方法", expanded=False):
    st.markdown(
        "- **概要**: 単一入力フォーム＋専門家モード切替で、LangChain を通して LLM に質問し、結果を表示。\n"
        "- **操作**: ラジオでモード選択 → テキスト入力 → **送信** ボタン。\n"
        "- **注意**: 生成結果は一般的情報であり、最終判断は自己責任で。"
    )

# -------------------------
# 専門家モード定義（要件：選択でシステムメッセージを切替）
# -------------------------
EXPERTS = {
    "A": {
        "label": "A｜生成AI実装コンサルタント",
        "system": (
            "あなたは生成AI/LLM実装に強いコンサルタントである。"
            "要件定義→設計→実装→評価の順で、短く具体的に助言する。"
            "箇条書きを好み、前提・制約・リスクも簡潔に触れる。"
            "冗長な比喩は避け、日本語で明快に述べる。"
        ),
    },
    "B": {
        "label": "B｜空港アクセス・モビリティアナリスト",
        "system": (
            "あなたは空港アクセス/モビリティのアナリストである。"
            "交通モード比較、需要見立て、運用・収益の観点から助言する。"
            "推定値は根拠と不確実性を明示し、日本語で簡潔に述べる。"
        ),
    },
}

# -------------------------
# APIキー取得（Cloudでも落ちない。Secrets優先→環境変数）
# -------------------------
def resolve_openai_api_key():
    try:
        if hasattr(st, "secrets"):
            key = st.secrets.get("OPENAI_API_KEY", None)
            if not key and "openai" in st.secrets:
                blk = st.secrets.get("openai", {})
                if isinstance(blk, dict):
                    key = blk.get("api_key")
            if key:
                return key, "secrets"
    except Exception:
        pass
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key, "env"
    return None, "missing"

# -------------------------
# LangChain 遅延 import（新旧APIを探索）。失敗しても UI は維持。
# -------------------------
def lazy_import_langchain():
    errors = []

    ChatOpenAI = None
    for path, name in [
        ("langchain_openai", "ChatOpenAI"),                 # 新API
        ("langchain.chat_models", "ChatOpenAI"),            # 旧API（分割前）
        ("langchain_community.chat_models", "ChatOpenAI"),  # 旧API（分割期）
    ]:
        try:
            mod = importlib.import_module(path)
            ChatOpenAI = getattr(mod, name)
            break
        except Exception as e:
            errors.append(f"{path}.{name}: {e}")

    SystemMessage = HumanMessage = None
    for path, sys_name, hum_name in [
        ("langchain_core.messages", "SystemMessage", "HumanMessage"),
        ("langchain.schema", "SystemMessage", "HumanMessage"),
    ]:
        try:
            mod = importlib.import_module(path)
            SystemMessage = getattr(mod, sys_name)
            HumanMessage = getattr(mod, hum_name)
            break
        except Exception as e:
            errors.append(f"{path}: {e}")

    if ChatOpenAI and SystemMessage and HumanMessage:
        return ChatOpenAI, SystemMessage, HumanMessage, None
    return None, None, None, "\n".join(errors) if errors else "unknown import error"

# -------------------------
# 要件の関数：入力テキスト＋選択値 → 文字列で回答
# -------------------------
def run_llm(user_text: str, expert_key: str) -> str:
    # 1) APIキー確認（未設定でも UI は保つ）
    api_key, key_source = resolve_openai_api_key()
    if not api_key:
        return "OpenAI APIキーが見つかりませんでした。Cloudでは App → Settings → Secrets に `OPENAI_API_KEY` を設定してください。"
    os.environ["OPENAI_API_KEY"] = api_key  # LangChain/SDKが参照

    # 2) LangChain を遅延読み込み
    ChatOpenAI, SystemMessage, HumanMessage, import_err = lazy_import_langchain()
    if import_err:
        return (
            "LangChain の読み込みに失敗しました。requirements.txt を次の例で用意してください：\n"
            "streamlit>=1.36\n"
            "langchain>=0.3.0\n"
            "langchain-openai>=0.1.21\n"
            "openai>=1.51.0\n"
            "python-dotenv>=1.0.1\n\n"
            f"詳細: {import_err}"
        )

    # 3) プロンプト（システム＋ユーザ）
    system_msg = EXPERTS.get(expert_key, EXPERTS["A"])["system"]
    messages = [SystemMessage(content=system_msg), HumanMessage(content=user_text)]

    # 4) モデル初期化（引数差を吸収）
    llm = None
    last_err = None
    for kwargs in (
        {"model": "gpt-4o-mini", "temperature": 0.3},
        {"model_name": "gpt-4o-mini", "temperature": 0.3},
        {"model": "gpt-4o-mini", "temperature": 0.3, "openai_api_key": api_key},
        {"model_name": "gpt-4o-mini", "temperature": 0.3, "openai_api_key": api_key},
    ):
        try:
            llm = ChatOpenAI(**kwargs)
            break
        except Exception as e:
            last_err = e
    if llm is None:
        return f"モデル初期化に失敗しました（gpt-4o-mini）。詳細: {last_err}"

    # 5) 呼び出し（invoke / __call__ / predict_messages / generate の順にフォールバック）
    try:
        try:
            res = llm.invoke(messages)  # 新API
        except Exception:
            try:
                res = llm(messages)      # 旧API
            except Exception:
                if hasattr(llm, "predict_messages"):
                    res = llm.predict_messages(messages)
                elif hasattr(llm, "generate"):
                    res = llm.generate([messages])  # LLMResult
                else:
                    raise RuntimeError("互換呼び出しメソッドが見つかりません。")
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return f"LLM呼び出しに失敗しました。詳細: {e}\n{tb}"

    # 6) 出力抽出
    try:
        if hasattr(res, "content"):
            answer = res.content
        elif isinstance(res, dict) and "content" in res:
            answer = res["content"]
        elif hasattr(res, "generations"):  # LLMResult
            gens = getattr(res, "generations", None)
            if gens and gens[0]:
                g0 = gens[0][0]
                if hasattr(g0, "message") and hasattr(g0.message, "content"):
                    answer = g0.message.content
                elif hasattr(g0, "text"):
                    answer = g0.text
                else:
                    answer = str(res)
            else:
                answer = str(res)
        else:
            answer = str(res)
    except Exception:
        answer = str(res)

    suffix = f"\n\n---\n（使用モード: {EXPERTS[expert_key]['label']}｜モデル: gpt-4o-mini｜キー取得元: {key_source}）"
    return answer + suffix

# -------------------------
# フォーム（ここまでで UI は確実に表示される）
# -------------------------
with st.form("main_form", clear_on_submit=False):
    labels = [v["label"] for v in EXPERTS.values()]
    key_by_label = {v["label"]: k for k, v in EXPERTS.items()}
    # horizontal=True は古いStreamlitで落ちることがあるため未使用
    selected_label = st.radio("専門家モードを選択", labels, index=0)
    selected_key = key_by_label[selected_label]

    user_text = st.text_area(
        "入力テキスト",
        placeholder="ここに質問や課題、要件などを記入してください。",
        height=180,
    )
    submitted = st.form_submit_button("送信")

# -------------------------
# 実行と表示（例外は画面内テキストとして返す＝クラッシュしない）
# -------------------------
if submitted:
    if not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLMが回答を生成中…"):
            answer = run_llm(user_text=user_text, expert_key=selected_key)
            st.markdown("### ✅ 回答")
            st.write(answer)

# -------------------------
# 起動時ステータス表示（画面下部に情報表示のみ）
# -------------------------
with st.expander("🔧 設定ステータス（表示のみ／安全）", expanded=False):
    key, src = resolve_openai_api_key()
    st.write(f"- OPENAI_API_KEY 検出: {'はい' if key else 'いいえ'}（ソース: {src}）")
    st.write("- LangChain / OpenAI は送信時に自動読み込みします。未導入でもUIは表示されます。")
