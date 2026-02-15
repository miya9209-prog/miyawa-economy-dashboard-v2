import re
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import feedparser
import yfinance as yf


# =========================
# Page Config + CSS (타이틀 잘림 방지)
# =========================
st.set_page_config(page_title="재테크 대시보드 v4.2", layout="wide")

st.markdown(
    """
<style>
/* 상단 여백 크게: 타이틀 잘림 방지 */
.block-container { padding-top: 2.6rem !important; padding-bottom: 2rem !important; }

/* 제목/텍스트 가독성 */
html, body, [class*="css"]  { font-size: 16px !important; }
h1 { font-size: 2.1rem !important; margin: 0.2rem 0 0.4rem 0 !important; line-height: 1.2 !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.15rem !important; }

/* 탭 크게 */
div[data-baseweb="tab"] button {
  font-size: 18px !important;
  padding: 10px 16px !important;
}

/* metric 값 크게 */
div[data-testid="stMetricValue"] { font-size: 24px !important; }

/* 버튼/폼 약간 촘촘히 */
div.stButton > button { padding: 0.45rem 0.9rem !important; }
</style>
""",
    unsafe_allow_html=True
)


# =========================
# Helpers
# =========================
def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.{decimals}f}"

def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):+.2f}%"

def safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "euc-kr", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="ignore")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)

    # exact match
    for c in candidates:
        if c in cols:
            return c

    # normalize (remove spaces)
    norm = {re.sub(r"\s+", "", str(x)).lower(): x for x in cols}
    for c in candidates:
        k = re.sub(r"\s+", "", str(c)).lower()
        if k in norm:
            return norm[k]

    # contains
    cols_s = [str(c) for c in cols]
    for c in candidates:
        for real in cols_s:
            if str(c).lower() in real.lower():
                return real

    return None

def is_month_col(name: str) -> bool:
    return re.fullmatch(r"\d{4}[.\-/]\d{2}", name.strip()) is not None


# =========================
# Local data loading
# =========================
@st.cache_data
def load_local_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers_raw = safe_read_csv("data/kr_tickers.csv")
    etfs_raw = safe_read_csv("data/kr_etfs.csv")
    realestate_raw = pd.read_excel("data/realestate.xlsx")
    return tickers_raw, etfs_raw, realestate_raw


# =========================
# Normalize tickers (주식 CSV가 어떤 컬럼명이든 표준화)
# =========================
def normalize_tickers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()

    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "Code", "code"])
    name_col = pick_col(df, ["종목명", "한글종목명", "한글 종목명", "종목약명", "Name", "name"])
    market_col = pick_col(df, ["시장구분", "시장", "Market", "market"])
    industry_col = pick_col(df, ["업종", "업종명", "Sector", "sector"])

    out = pd.DataFrame()

    if code_col is not None:
        out["code"] = (
            df[code_col].astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
            .str.zfill(6)
        )
    else:
        out["code"] = ""

    if name_col is not None:
        out["name"] = df[name_col].astype(str).str.strip()
    else:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        out["name"] = df[text_cols[0]].astype(str).str.strip() if text_cols else ""

    out["market"] = df[market_col].astype(str).str.strip() if market_col is not None else ""
    out["industry"] = df[industry_col].astype(str).str.strip() if industry_col is not None else ""

    meta = {
        "code_col": code_col,
        "name_col": name_col,
        "market_col": market_col,
        "industry_col": industry_col,
    }
    return out, meta


# =========================
# Market snapshot with fallback (금값 복원)
# =========================
@st.cache_data(ttl=60 * 15)
def fetch_snapshot(symbol: str) -> Dict[str, Optional[float]]:
    out = {"last": None, "chg_pct": None}
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="10d", interval="1d")
        if hist is None or hist.empty:
            return out
        closes = hist["Close"].dropna()
        if len(closes) < 1:
            return out

        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
        chg_pct = ((last - prev) / prev * 100.0) if (prev is not None and prev != 0) else None

        out["last"] = last
        out["chg_pct"] = chg_pct
        return out
    except Exception:
        return out

def fetch_first_available(symbols: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    symbols 중에서 데이터가 정상인 첫 번째를 사용
    return: (last, chg_pct, used_symbol)
    """
    for sym in symbols:
        snap = fetch_snapshot(sym)
        if snap.get("last") is not None:
            return snap.get("last"), snap.get("chg_pct"), sym
    return None, None, None


# 주요지표 (금은 3단계 fallback)
MARKET_ITEMS: List[Tuple[str, List[str], int]] = [
    ("KOSPI", ["^KS11"], 2),
    ("KOSDAQ", ["^KQ11"], 2),
    ("USD/KRW", ["KRW=X"], 4),

    # GOLD fallback: XAUUSD -> GC -> GLD(ETF)
    ("GOLD", ["XAUUSD=X", "GC=F", "GLD"], 2),

    ("WTI", ["CL=F"], 2),
    ("DXY", ["DX-Y.NYB"], 2),
    ("US10Y", ["^TNX"], 2),
    ("BTC", ["BTC-USD"], 0),
]


# =========================
# News RSS
# =========================
NEWS_FEEDS: List[Tuple[str, str]] = [
    ("연합뉴스 경제", "https://www.yna.co.kr/rss/economy.xml"),
    ("한국경제", "https://rss.hankyung.com/economy.xml"),
    ("매일경제", "https://file.mk.co.kr/news/rss/rss_30000001.xml"),
    ("조선비즈", "https://biz.chosun.com/arc/outboundfeeds/rss/category/economy/?outputType=xml"),
]

@st.cache_data(ttl=60 * 20)
def fetch_news(max_items_total: int = 25) -> List[dict]:
    items: List[dict] = []
    per_feed = max(5, max_items_total // max(1, len(NEWS_FEEDS)))
    for name, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:per_feed]:
                items.append({
                    "source": name,
                    "title": getattr(e, "title", "").strip(),
                    "link": getattr(e, "link", "").strip(),
                    "published": getattr(e, "published", "") or getattr(e, "updated", ""),
                    "summary": getattr(e, "summary", "") or getattr(e, "description", ""),
                })
        except Exception:
            continue
    return items[:max_items_total]


# =========================
# Session init (관심종목)
# =========================
if "watchlist" not in st.session_state:
    # dict: code -> name
    st.session_state["watchlist"] = {}

def add_to_watchlist(code: str, name: str):
    if code and name:
        st.session_state["watchlist"][code] = name

def remove_from_watchlist(code: str):
    if code in st.session_state["watchlist"]:
        del st.session_state["watchlist"][code]


# =========================
# Header
# =========================
st.markdown("# 📊 재테크 대시보드 v4.2")
st.caption("주요지표(금 포함) · 주식/ETF/부동산 · 뉴스 · 관심종목")

top1, top2 = st.columns([1.0, 2.2])
with top1:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
with top2:
    st.info("로컬 데이터는 GitHub `data/`에서 읽습니다. 지표/뉴스는 인터넷이 필요합니다.", icon="ℹ️")

st.markdown("---")


# =========================
# Load & normalize
# =========================
try:
    tickers_raw, etfs_raw, realestate_raw = load_local_data()
    tickers, tickers_meta = normalize_tickers(tickers_raw)
except Exception as e:
    st.error("data 폴더의 파일을 읽지 못했습니다. 파일명/경로/인코딩을 확인해주세요.")
    st.exception(e)
    st.stop()


# =========================
# Tabs
# =========================
tab_dash, tab_stock, tab_etf, tab_re, tab_news, tab_watch, tab_debug = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산", "📰 뉴스", "⭐ 관심종목", "🛠️ 진단"]
)

# =========================
# TAB: Dashboard
# =========================
with tab_dash:
    st.subheader("오늘의 주요지표")

    # 2줄로 예쁘게(카드가 너무 많아도 깨지지 않게)
    per_row = 4
    rows = [MARKET_ITEMS[i:i+per_row] for i in range(0, len(MARKET_ITEMS), per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for i, (label, symbols, decimals) in enumerate(row):
            last, chg_pct, used = fetch_first_available(symbols)
            delta = fmt_pct(chg_pct) if chg_pct is not None else None
            cols[i].metric(label, fmt_num(last, decimals), delta=delta)

            # 금처럼 fallback이 걸린 경우 어떤 심볼로 떴는지 아주 작게 힌트
            if used and len(symbols) > 1:
                cols[i].caption(f"source: {used}")

    st.markdown("---")
    st.markdown("### 오늘의 뉴스 Top 8")
    news = fetch_news(8)
    if not news:
        st.warning("뉴스를 불러오지 못했습니다. (RSS/네트워크 제한 가능)")
    else:
        for x in news:
            title = x.get("title") or "(제목 없음)"
            link = x.get("link") or ""
            source = x.get("source") or ""
            published = x.get("published") or ""
            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")
            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)
            st.divider()


# =========================
# TAB: Stocks (검색 + 관심종목 담기 완성)
# =========================
with tab_stock:
    st.subheader("국내 주식 종목 탐색")

    # 검색폼: 버튼 기반 + Enter도 submit으로 동작
    with st.form("stock_search_form", clear_on_submit=False):
        kw = st.text_input("종목명 또는 6자리 코드", value=st.session_state.get("stock_kw", "삼성전자"))
        submit = st.form_submit_button("🔎 검색")

    if submit:
        st.session_state["stock_kw"] = kw

    kw2 = st.session_state.get("stock_kw", "").strip()

    df = tickers.copy()

    # 실제 필터 적용(이게 핵심)
    if kw2:
        df = df[
            df["name"].astype(str).str.contains(kw2, case=False, na=False)
            | df["code"].astype(str).str.contains(kw2, case=False, na=False)
        ]

    left, right = st.columns([1.35, 1.0])
    with right:
        st.markdown("### 결과 요약")
        st.metric("검색 결과", f"{len(df):,}개")

    with left:
        if len(df) == 0:
            st.warning("검색 결과가 없습니다. 키워드를 바꿔보세요.")
        else:
            show_cols = ["code", "name"]
            if df["market"].astype(str).str.strip().any():
                show_cols.append("market")
            if df["industry"].astype(str).str.strip().any():
                show_cols.append("industry")
            st.dataframe(df[show_cols], use_container_width=True, height=520)

    st.markdown("---")
    st.markdown("### 종목 상세 + 관심종목 담기")

    if len(df) > 0:
        options = (df["name"].astype(str) + " (" + df["code"].astype(str) + ")").tolist()[:5000]
        sel = st.selectbox("종목 선택", options=options)

        m = re.search(r"\((\d{6})\)$", sel)
        sel_code = m.group(1) if m else ""
        row = df[df["code"].astype(str) == sel_code].head(1)

        if not row.empty:
            r = row.iloc[0].to_dict()
            sel_name = r.get("name", "")
            st.write({
                "종목코드": r.get("code"),
                "종목명": sel_name,
                "시장": r.get("market"),
                "업종": r.get("industry"),
            })

            b1, b2, b3 = st.columns([1.1, 1.1, 2.0])
            with b1:
                if st.button("⭐ 관심종목에 담기", key=f"add_{sel_code}"):
                    add_to_watchlist(sel_code, sel_name)
                    st.success("관심종목에 추가했습니다.")
            with b2:
                if st.button("🗑️ 관심종목에서 제거", key=f"rm_{sel_code}"):
                    remove_from_watchlist(sel_code)
                    st.info("관심종목에서 제거했습니다.")
            with b3:
                st.caption("관심종목은 ‘⭐ 관심종목’ 탭에서 관리/다운로드할 수 있습니다.")


# =========================
# TAB: ETF (원본 그대로 보여주되 검색만 안정적으로)
# =========================
with tab_etf:
    st.subheader("국내 ETF 탐색")

    kw = st.text_input("ETF 이름/코드 검색", value="")
    df = etfs_raw.copy()

    name_col = pick_col(df, ["종목명", "ETF명", "한글종목명", "한글 종목명", "name", "Name"])
    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "code", "Code"])

    if kw.strip():
        cond = pd.Series([False] * len(df))
        if name_col:
            cond = cond | df[name_col].astype(str).str.contains(kw.strip(), case=False, na=False)
        if code_col:
            cond = cond | df[code_col].astype(str).str.contains(kw.strip(), case=False, na=False)
        df = df[cond]

    st.metric("표시 ETF", f"{len(df):,}개")
    st.dataframe(df, use_container_width=True, height=650)


# =========================
# TAB: Real Estate (깨져도 표는 무조건 보여주기)
# =========================
with tab_re:
    st.subheader("부동산 데이터")

    df = realestate_raw.copy()

    region_cols = [c for c in df.columns if str(c).startswith("지역")]
    type_cols = [c for c in df.columns if ("유형" in str(c)) or str(c).startswith("주택")]
    month_cols = [c for c in df.columns if is_month_col(str(c))]

    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])

    with f1:
        sel_regions = []
        if region_cols:
            region_key = region_cols[0]
            regions = sorted(df[region_key].dropna().astype(str).unique().tolist())
            sel_regions = st.multiselect(f"지역 선택 ({region_key})", options=regions, default=[])
        else:
            st.caption("지역 컬럼 자동감지 실패(표는 표시됩니다)")

    with f2:
        sel_months = []
        if month_cols:
            default_months = month_cols[-6:] if len(month_cols) >= 6 else month_cols
            sel_months = st.multiselect("표시할 월 선택", options=month_cols, default=default_months)
        else:
            st.caption("월(YYYY.MM) 컬럼 자동감지 실패(표는 표시됩니다)")

    with f3:
        st.metric("행 수", f"{len(df):,}건")

    if region_cols and sel_regions:
        df = df[df[region_cols[0]].astype(str).isin(sel_regions)]

    base_cols = []
    base_cols += type_cols[:1]
    base_cols += region_cols[:4]
    show_cols = [c for c in base_cols if c in df.columns]
    if sel_months:
        show_cols += [c for c in sel_months if c in df.columns]

    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=650)


# =========================
# TAB: News
# =========================
with tab_news:
    st.subheader("경제 뉴스 게시판")

    n1, n2 = st.columns([1.0, 2.0])
    with n1:
        max_items = st.slider("뉴스 개수", 10, 60, 25, 5)
    with n2:
        keyword = st.text_input("키워드 필터(선택)", value="")

    items = fetch_news(max_items)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS/네트워크 제한 가능)")
    else:
        if keyword.strip():
            kw = keyword.strip().lower()
            items = [x for x in items if kw in (x.get("title","") + " " + x.get("summary","")).lower()]

        st.metric("표시 뉴스", f"{len(items):,}건")
        for x in items:
            title = x.get("title") or "(제목 없음)"
            link = x.get("link") or ""
            source = x.get("source") or ""
            published = x.get("published") or ""

            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)
            st.divider()


# =========================
# TAB: Watchlist (관심종목 관리/다운로드)
# =========================
with tab_watch:
    st.subheader("⭐ 관심종목")

    wl = st.session_state.get("watchlist", {})
    if not wl:
        st.info("아직 담긴 관심종목이 없습니다. 주식 탭에서 ⭐ 버튼으로 담아주세요.")
    else:
        wdf = pd.DataFrame([{"code": c, "name": n} for c, n in wl.items()]).sort_values("name")
        st.metric("관심종목 수", f"{len(wdf):,}개")
        st.dataframe(wdf, use_container_width=True, height=420)

        c1, c2 = st.columns([1.2, 2.0])
        with c1:
            # CSV 다운로드
            csv_bytes = wdf.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 관심종목 CSV 다운로드", data=csv_bytes, file_name="watchlist.csv", mime="text/csv")
        with c2:
            # 개별 삭제 UI
            opts = (wdf["name"] + " (" + wdf["code"] + ")").tolist()
            sel = st.selectbox("삭제할 종목 선택", options=opts)
            m = re.search(r"\((\d{6})\)$", sel)
            code = m.group(1) if m else ""
            if st.button("🗑️ 선택 종목 삭제"):
                remove_from_watchlist(code)
                st.success("삭제했습니다.")
                st.rerun()


# =========================
# TAB: Debug
# =========================
with tab_debug:
    st.subheader("🛠️ 진단 패널")

    st.markdown("### 1) 종목 CSV 컬럼/매핑")
    st.write("원본 컬럼:", list(tickers_raw.columns))
    st.write("매핑 결과:", tickers_meta)
    st.write("표준 컬럼:", list(tickers.columns))
    st.dataframe(tickers.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2) 금값이 안 뜰 때 확인")
    st.write("GOLD는 아래 순서로 시도합니다: XAUUSD=X → GC=F → GLD")
    for sym in ["XAUUSD=X", "GC=F", "GLD"]:
        snap = fetch_snapshot(sym)
        st.write(sym, "last=", snap.get("last"), "chg_pct=", snap.get("chg_pct"))

st.markdown("---")
st.caption("© 재테크 대시보드 v4.2 | 핵심: 타이틀/검색/관심종목/금값 안정화")
