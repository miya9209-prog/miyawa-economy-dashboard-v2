import re
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd

# (뉴스/지표는 인터넷 필요)
import feedparser
import yfinance as yf


# =========================
# Page Config + CSS
# =========================
st.set_page_config(page_title="재테크 대시보드 v4", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { font-size: 16px !important; }
h1 { font-size: 2.0rem !important; margin-bottom: 0.15rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }
div[data-baseweb="tab"] button { font-size: 18px !important; padding: 10px 16px !important; }
div[data-testid="stMetricValue"] { font-size: 24px !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
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
    # KRX/ETF 파일 인코딩 혼재 방어
    for enc in ("utf-8", "utf-8-sig", "euc-kr", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 최후 수단
    return pd.read_csv(path, encoding="utf-8", errors="ignore")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    # 1) 정확 일치
    for c in candidates:
        if c in cols:
            return c
    # 2) 공백 제거한 비교
    norm = {re.sub(r"\s+", "", str(x)).lower(): x for x in cols}
    for c in candidates:
        k = re.sub(r"\s+", "", str(c)).lower()
        if k in norm:
            return norm[k]
    # 3) 포함(contains) 기반
    cols_s = [str(c) for c in cols]
    for c in candidates:
        for real in cols_s:
            if str(c).lower() in real.lower():
                return real
    return None

def is_month_col(name: str) -> bool:
    # 2024.10 / 2024-10 / 2024/10
    return re.fullmatch(r"\d{4}[.\-/]\d{2}", name.strip()) is not None


# =========================
# Load local files
# =========================
@st.cache_data
def load_local_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tickers_raw = safe_read_csv("data/kr_tickers.csv")
    etfs_df = safe_read_csv("data/kr_etfs.csv")
    realestate_df = pd.read_excel("data/realestate.xlsx")
    return tickers_raw, etfs_df, realestate_df


# =========================
# Normalize tickers (주식)
# =========================
def normalize_tickers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()

    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "code", "Code"])
    name_col = pick_col(df, ["종목명", "한글종목명", "한글 종목명", "종목약명", "name", "Name"])
    market_col = pick_col(df, ["시장구분", "시장", "Market", "market"])
    industry_col = pick_col(df, ["업종", "업종명", "Sector", "sector"])

    out = pd.DataFrame()

    # code
    if code_col is not None:
        out["code"] = (
            df[code_col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
            .str.zfill(6)
        )
    else:
        out["code"] = ""

    # name
    if name_col is not None:
        out["name"] = df[name_col].astype(str).str.strip()
    else:
        # 텍스트 컬럼 하나라도 있으면 그걸 name으로
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        out["name"] = df[text_cols[0]].astype(str).str.strip() if text_cols else ""

    # market / industry (없어도 괜찮게)
    out["market"] = df[market_col].astype(str).str.strip() if market_col is not None else ""
    out["industry"] = df[industry_col].astype(str).str.strip() if industry_col is not None else ""

    # 원본 인덱스 보존(디버깅용)
    out["_raw_index"] = df.index

    meta = {
        "code_col": code_col,
        "name_col": name_col,
        "market_col": market_col,
        "industry_col": industry_col,
    }
    return out, meta


# =========================
# Normalize ETF (ETF)
# =========================
def normalize_etfs(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()

    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "code", "Code"])
    name_col = pick_col(df, ["종목명", "ETF명", "한글종목명", "한글 종목명", "name", "Name"])
    nav_col = pick_col(df, ["순자산가치(NAV)", "NAV", "nav"])
    price_col = pick_col(df, ["종가", "현재가", "가격", "price", "Price"])

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6) if code_col else ""
    out["name"] = df[name_col].astype(str).str.strip() if name_col else (df[df.columns[0]].astype(str) if len(df.columns) else "")
    out["price"] = df[price_col] if price_col else None
    out["nav"] = df[nav_col] if nav_col else None

    meta = {"code_col": code_col, "name_col": name_col, "price_col": price_col, "nav_col": nav_col}
    return out, meta


# =========================
# Realestate helpers
# =========================
def normalize_realestate(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df = raw.copy()

    # "지역별(1..)" 같은 계층 컬럼
    region_cols = [c for c in df.columns if str(c).startswith("지역")]
    # "주택유형별(1)" 같은 유형 컬럼
    type_cols = [c for c in df.columns if ("유형" in str(c)) or str(c).startswith("주택")]
    # 월 컬럼(2024.10 형태)
    month_cols = [c for c in df.columns if is_month_col(str(c))]

    meta = {
        "region_cols": [str(c) for c in region_cols],
        "type_cols": [str(c) for c in type_cols],
        "month_cols": [str(c) for c in month_cols],
    }
    return df, meta


# =========================
# Market Snapshot (지표) - 복원/확장
# =========================
# 금값: XAUUSD=X (현물) or GC=F (선물) 둘 중 하나라도 뜨면 보여줌
MARKET_TICKERS: List[Tuple[str, str, int]] = [
    ("KOSPI", "^KS11", 2),
    ("KOSDAQ", "^KQ11", 2),
    ("USD/KRW", "KRW=X", 4),
    ("GOLD (XAU/USD)", "XAUUSD=X", 2),
    ("GOLD FUT (GC=F)", "GC=F", 2),
    ("WTI", "CL=F", 2),
    ("DXY", "DX-Y.NYB", 2),        # 달러 인덱스
    ("US10Y", "^TNX", 2),          # 10년물(지수값, %*10 형태일 수 있음)
    ("BTC", "BTC-USD", 0),
]

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
# UI Header
# =========================
st.title("📊 재테크 대시보드 v4 (완성 안정판)")
st.caption("주요지표(금 포함) · 주식/ETF/부동산 탐색 · 뉴스 게시판")

h1, h2 = st.columns([1.0, 2.0])
with h1:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
with h2:
    st.info("로컬 데이터는 GitHub `data/`에서 읽습니다. 지표/뉴스는 인터넷이 필요합니다.", icon="ℹ️")

st.markdown("---")


# =========================
# Load & normalize all
# =========================
try:
    tickers_raw, etfs_raw, realestate_raw = load_local_data()
    tickers, tickers_meta = normalize_tickers(tickers_raw)
    etfs, etfs_meta = normalize_etfs(etfs_raw)
    realestate, realestate_meta = normalize_realestate(realestate_raw)
except Exception as e:
    st.error("data 폴더의 파일을 읽지 못했습니다. 파일명/경로/인코딩을 확인해주세요.")
    st.exception(e)
    st.stop()


# =========================
# Tabs
# =========================
tab_dash, tab_stock, tab_etf, tab_re, tab_news, tab_debug = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산", "📰 뉴스", "🛠️ 진단"]
)

# =========================
# Dashboard
# =========================
with tab_dash:
    st.subheader("오늘의 주요지표 (금 포함)")

    # 금은 XAUUSD가 안 뜨는 환경도 있어 GC=F까지 같이 두었고,
    # 둘 다 뜨면 둘 다 보이게(사용자가 선호 선택 가능)
    cards_per_row = 5
    rows = [MARKET_TICKERS[i:i+cards_per_row] for i in range(0, len(MARKET_TICKERS), cards_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for i, (label, symbol, decimals) in enumerate(row):
            snap = fetch_snapshot(symbol)
            delta = fmt_pct(snap["chg_pct"]) if snap["chg_pct"] is not None else None
            cols[i].metric(label, fmt_num(snap["last"], decimals), delta=delta)

    st.markdown("### 데이터 한눈에")
    c1, c2, c3 = st.columns(3)
    c1.metric("KRX 종목 수", f"{len(tickers):,}")
    c2.metric("ETF 행 수", f"{len(etfs_raw):,}")
    c3.metric("부동산 행 수", f"{len(realestate):,}")

    st.markdown("---")
    st.markdown("### 오늘의 뉴스 Top 10")
    items = fetch_news(10)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS/네트워크 제한 가능)")
    else:
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
# Stocks
# =========================
with tab_stock:
    st.subheader("국내 주식 종목 탐색")

    # 검색폼 (버튼 트리거)
    with st.form("stock_search_form", clear_on_submit=False):
        kw = st.text_input("종목명 또는 6자리 코드 검색", value=st.session_state.get("stock_kw", ""))
        markets = sorted([m for m in tickers["market"].dropna().unique().tolist() if str(m).strip()])
        industries = sorted([m for m in tickers["industry"].dropna().unique().tolist() if str(m).strip()])

        sel_markets = st.multiselect(
            "시장 선택(선택)",
            options=markets,
            default=st.session_state.get("stock_markets", markets)
        ) if markets else []

        sel_industries = st.multiselect(
            "업종 선택(선택)",
            options=industries,
            default=st.session_state.get("stock_industries", [])
        ) if industries else []

        submit = st.form_submit_button("🔎 검색")

    if submit:
        st.session_state["stock_kw"] = kw
        st.session_state["stock_markets"] = sel_markets
        st.session_state["stock_industries"] = sel_industries

    # 적용
    kw2 = st.session_state.get("stock_kw", "").strip()
    sel_markets2 = st.session_state.get("stock_markets", markets if markets else [])
    sel_industries2 = st.session_state.get("stock_industries", [])

    df = tickers.copy()
    if sel_markets2:
        df = df[df["market"].astype(str).isin([str(x) for x in sel_markets2])]
    if sel_industries2:
        df = df[df["industry"].astype(str).isin([str(x) for x in sel_industries2])]
    if kw2:
        df = df[
            df["name"].astype(str).str.contains(kw2, case=False, na=False)
            | df["code"].astype(str).str.contains(kw2, case=False, na=False)
        ]

    left, right = st.columns([1.35, 1.0])
    with right:
        st.markdown("### 결과 요약")
        st.metric("검색 결과", f"{len(df):,}개")
        st.caption("표는 최소 code/name을 항상 표시합니다.")

    with left:
        if len(df) == 0:
            st.warning("검색 결과가 없습니다. 키워드를 바꿔보세요.")
        else:
            show_cols = ["code", "name"]
            if df["market"].astype(str).str.strip().any():
                show_cols.append("market")
            if df["industry"].astype(str).str.strip().any():
                show_cols.append("industry")
            st.dataframe(df[show_cols], use_container_width=True, height=560)

    st.markdown("---")
    st.markdown("### 종목 상세(기본)")
    if len(df) > 0:
        options = (df["name"].astype(str) + " (" + df["code"].astype(str) + ")").tolist()[:5000]
        sel = st.selectbox("종목 선택", options=options)
        m = re.search(r"\((\d{6})\)$", sel)
        sel_code = m.group(1) if m else ""
        row = df[df["code"].astype(str) == sel_code].head(1)
        if not row.empty:
            r = row.iloc[0].to_dict()
            st.write({
                "종목코드": r.get("code"),
                "종목명": r.get("name"),
                "시장": r.get("market"),
                "업종": r.get("industry"),
            })

# =========================
# ETF
# =========================
with tab_etf:
    st.subheader("국내 ETF 탐색")

    kw = st.text_input("ETF 이름/코드 검색", value="")
    df = etfs_raw.copy()

    # ETF 이름 컬럼을 최대한 찾아서 검색이 먹게
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
# Real Estate
# =========================
with tab_re:
    st.subheader("부동산 데이터")

    df = realestate.copy()

    region_cols = [c for c in df.columns if str(c).startswith("지역")]
    type_cols = [c for c in df.columns if ("유형" in str(c)) or str(c).startswith("주택")]
    month_cols = [c for c in df.columns if is_month_col(str(c))]

    # 필터 UI
    f1, f2, f3 = st.columns([1.2, 1.2, 1.6])

    with f1:
        if region_cols:
            region_key = region_cols[0]  # 지역별(1) 우선
            regions = sorted(df[region_key].dropna().astype(str).unique().tolist())
            sel_regions = st.multiselect(f"지역 선택 ({region_key})", options=regions, default=[])
        else:
            sel_regions = []
            st.warning("지역 컬럼을 찾지 못했습니다. (하지만 데이터는 표로 표시됩니다)")

    with f2:
        if month_cols:
            default_months = month_cols[-6:] if len(month_cols) >= 6 else month_cols
            sel_months = st.multiselect("표시할 월 선택", options=month_cols, default=default_months)
        else:
            sel_months = []
            st.warning("월(YYYY.MM) 컬럼을 찾지 못했습니다. (하지만 데이터는 표로 표시됩니다)")

    with f3:
        st.metric("행 수", f"{len(df):,}건")

    # 적용
    if region_cols and sel_regions:
        df = df[df[region_cols[0]].astype(str).isin(sel_regions)]

    # 표시 컬럼: 유형/지역 계층 + 선택 월
    base_cols = []
    base_cols += type_cols[:1]
    base_cols += region_cols[:4]  # 지역별(1~4)
    show_cols = [c for c in base_cols if c in df.columns]
    if sel_months:
        show_cols += [c for c in sel_months if c in df.columns]

    if show_cols:
        st.dataframe(df[show_cols], use_container_width=True, height=650)
    else:
        st.dataframe(df, use_container_width=True, height=650)

# =========================
# News
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
# Debug tab (필수)
# =========================
with tab_debug:
    st.subheader("🛠️ 진단 패널 (문제 원인 1분 내 확인)")

    st.markdown("### 1) 종목 CSV 컬럼/매핑")
    st.write("원본 컬럼:", list(tickers_raw.columns))
    st.write("매핑 결과:", tickers_meta)
    st.write("표준화된 컬럼:", list(tickers.columns))
    st.dataframe(tickers.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2) ETF CSV 컬럼/매핑")
    st.write("원본 컬럼:", list(etfs_raw.columns))
    st.write("ETF 매핑:", etfs_meta)
    st.dataframe(etfs_raw.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3) 부동산 XLSX 컬럼 감지")
    st.write("감지 결과:", realestate_meta)
    st.write("원본 컬럼:", list(realestate_raw.columns))
    st.dataframe(realestate_raw.head(20), use_container_width=True)

st.markdown("---")
st.caption("© 재테크 대시보드 v4 | 안정판: ‘새로고침 버튼’ 중심 · 지표/뉴스는 인터넷 필요")
