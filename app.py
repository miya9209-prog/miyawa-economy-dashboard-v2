import re
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import feedparser
import yfinance as yf


# =========================
# Page Config
# =========================
st.set_page_config(page_title="재테크 대시보드 v4.1", layout="wide")

# =========================
# Global CSS
# =========================
st.markdown(
    """
<style>
html, body, [class*="css"]  { font-size: 16px !important; }
h1 { font-size: 2.1rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.25rem !important; }

div[data-baseweb="tab"] button {
  font-size: 18px !important;
  padding: 10px 16px !important;
}

div[data-testid="stMetric"] label { font-size: 15px !important; }
div[data-testid="stMetricValue"] { font-size: 24px !important; }

.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
""",
    unsafe_allow_html=True
)

# =========================
# Helpers
# =========================
def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:+.2f}%"

def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:,.{decimals}f}"

def safe_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "euc-kr", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 마지막 수단
    return pd.read_csv(path, encoding="utf-8", errors="ignore")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # 부분 매칭(공백/언더바 차이 대응)
    norm = {re.sub(r"\s+", "", str(x)).lower(): x for x in cols}
    for c in candidates:
        k = re.sub(r"\s+", "", str(c)).lower()
        if k in norm:
            return norm[k]
    return None

def first_match_contains(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    for p in patterns:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None

# =========================
# Normalize Tickers (핵심)
# =========================
def normalize_tickers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    다양한 KRX 컬럼명을 표준 컬럼으로 매핑:
    - code, name, market, industry (가능한 범위에서)
    """
    df = raw.copy()

    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "Code", "code"])
    name_col = pick_col(df, ["종목명", "한글종목명", "한글 종목명", "종목약명", "Name", "name"])
    market_col = pick_col(df, ["시장구분", "시장", "Market", "market"])
    industry_col = pick_col(df, ["업종", "업종명", "Sector", "sector"])

    # KRX 파일에서 흔히 보이는 컬럼명 패턴도 추가 탐색
    if code_col is None:
        code_col = first_match_contains(df, ["코드"])
    if name_col is None:
        name_col = first_match_contains(df, ["종목", "회사"])
    if market_col is None:
        market_col = first_match_contains(df, ["시장"])
    if industry_col is None:
        industry_col = first_match_contains(df, ["업종", "섹터"])

    # 표준 컬럼 만들기
    out = pd.DataFrame()

    if code_col is not None:
        out["code"] = df[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    else:
        out["code"] = ""

    if name_col is not None:
        out["name"] = df[name_col].astype(str)
    else:
        # 이름 컬럼을 못 찾으면 첫 번째 텍스트 컬럼을 사용
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        out["name"] = df[text_cols[0]].astype(str) if text_cols else ""

    out["market"] = df[market_col].astype(str) if market_col is not None else ""
    out["industry"] = df[industry_col].astype(str) if industry_col is not None else ""

    # 원본도 붙여서 확인 가능하게
    out["_raw_index"] = df.index

    meta = {
        "code_col": code_col,
        "name_col": name_col,
        "market_col": market_col,
        "industry_col": industry_col
    }
    return out, meta

# =========================
# Realestate helpers
# =========================
def detect_region_cols(df: pd.DataFrame) -> List[str]:
    # 예: 지역별(1), 지역별(2) ...
    return [c for c in df.columns if str(c).startswith("지역")]

def detect_month_cols(df: pd.DataFrame) -> List[str]:
    # 예: 2024.10, 2025.03, 2024-10, 2025/03 등
    cols = []
    for c in df.columns:
        s = str(c)
        if re.fullmatch(r"\d{4}[.\-/]\d{2}", s):
            cols.append(c)
    return cols

def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")

# =========================
# Data Loaders
# =========================
@st.cache_data
def load_local_data():
    tickers_raw = safe_read_csv("data/kr_tickers.csv")
    etfs_df = safe_read_csv("data/kr_etfs.csv")
    realestate_df = pd.read_excel("data/realestate.xlsx")
    return tickers_raw, etfs_df, realestate_df

# =========================
# Indices
# =========================
INDEX_TICKERS = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "USD/KRW": "KRW=X",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
}

@st.cache_data(ttl=60 * 15)
def fetch_index_snapshot(symbol: str) -> Dict[str, Optional[float]]:
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
# News
# =========================
NEWS_FEEDS: List[Tuple[str, str]] = [
    ("연합뉴스 경제", "https://www.yna.co.kr/rss/economy.xml"),
    ("한국경제", "https://rss.hankyung.com/economy.xml"),
    ("매일경제", "https://file.mk.co.kr/news/rss/rss_30000001.xml"),
    ("조선비즈", "https://biz.chosun.com/arc/outboundfeeds/rss/category/economy/?outputType=xml"),
]

@st.cache_data(ttl=60 * 20)
def fetch_news_items(max_items_total: int = 20) -> List[dict]:
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
# Header
# =========================
st.title("📊 재테크 대시보드 v4.1")
st.caption("주요지수 요약 · 주식/ETF/부동산 탐색 · 뉴스 게시판")

c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
with c1:
    auto_refresh = st.toggle("자동 새로고침(5분)", value=False)
with c2:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
with c3:
    st.info("로컬 데이터는 GitHub `data/` 폴더에서 읽습니다. 지수/뉴스는 인터넷이 필요합니다.", icon="ℹ️")

if auto_refresh:
    st.autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5min")

st.markdown("---")

# =========================
# Load & Normalize
# =========================
try:
    tickers_raw, etfs_df, realestate_df = load_local_data()
    tickers_df, tickers_meta = normalize_tickers(tickers_raw)
except Exception as e:
    st.error("data 폴더의 파일을 읽지 못했습니다. 파일명/경로/인코딩을 확인해주세요.")
    st.exception(e)
    st.stop()

# =========================
# Tabs
# =========================
tab_dashboard, tab_stocks, tab_etf, tab_realestate, tab_news = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산", "📰 뉴스"]
)

# =========================
# TAB: Dashboard
# =========================
with tab_dashboard:
    st.subheader("오늘의 주요지수 요약")

    cols = st.columns(len(INDEX_TICKERS))
    for i, (label, symbol) in enumerate(INDEX_TICKERS.items()):
        snap = fetch_index_snapshot(symbol)
        delta = fmt_pct(snap["chg_pct"]) if snap["chg_pct"] is not None else None
        cols[i].metric(label, fmt_num(snap["last"], 2), delta=delta)

    st.markdown("### 데이터 한눈에")
    d1, d2, d3 = st.columns(3)
    d1.metric("KRX 종목 수", f"{len(tickers_df):,}")
    d2.metric("ETF 데이터 행", f"{len(etfs_df):,}")
    d3.metric("부동산 데이터 행", f"{len(realestate_df):,}")

    with st.expander("디버그: 종목 CSV 컬럼 인식 결과", expanded=False):
        st.write("원본 컬럼 목록:", list(tickers_raw.columns))
        st.write("매핑 결과:", tickers_meta)
        st.write("표준 컬럼:", list(tickers_df.columns))

    st.markdown("---")
    st.markdown("### 오늘의 경제 뉴스 Top")
    items = fetch_news_items(max_items_total=10)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS/네트워크 이슈 가능)")
    else:
        for x in items[:10]:
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
# TAB: Stocks
# =========================
with tab_stocks:
    st.subheader("📈 국내 주식 종목 탐색")

    left_filters, right_info = st.columns([1.6, 1.0])

    # 검색 조건(버튼 트리거)
    with left_filters:
        with st.form("stock_search_form", clear_on_submit=False):
            st.markdown("#### 검색 조건")

            keyword_input = st.text_input(
                "종목명/코드 검색",
                value=st.session_state.get("stock_kw", "")
            )

            market_opts = sorted([m for m in tickers_df["market"].dropna().unique().tolist() if str(m).strip()]) if "market" in tickers_df.columns else []
            industry_opts = sorted([m for m in tickers_df["industry"].dropna().unique().tolist() if str(m).strip()]) if "industry" in tickers_df.columns else []

            markets = st.multiselect(
                "시장 선택(선택)",
                options=market_opts,
                default=st.session_state.get("stock_markets", market_opts)
            ) if market_opts else []

            industries = st.multiselect(
                "업종 선택(선택)",
                options=industry_opts,
                default=st.session_state.get("stock_industries", [])
            ) if industry_opts else []

            submit = st.form_submit_button("🔎 검색")

        if submit:
            st.session_state["stock_kw"] = keyword_input
            st.session_state["stock_markets"] = markets
            st.session_state["stock_industries"] = industries

    # 조건 적용(세션 기반)
    kw = st.session_state.get("stock_kw", "").strip()
    selected_markets = st.session_state.get("stock_markets", [])
    selected_industries = st.session_state.get("stock_industries", [])

    filtered = tickers_df.copy()

    if selected_markets:
        filtered = filtered[filtered["market"].astype(str).isin([str(x) for x in selected_markets])]
    if selected_industries:
        filtered = filtered[filtered["industry"].astype(str).isin([str(x) for x in selected_industries])]

    if kw:
        # 코드 검색도 같이 지원
        filtered = filtered[
            filtered["name"].astype(str).str.contains(kw, case=False, na=False)
            | filtered["code"].astype(str).str.contains(kw, case=False, na=False)
        ]

    with right_info:
        st.markdown("#### 결과 요약")
        st.metric("검색 결과", f"{len(filtered):,}개")
        st.caption("검색 조건은 ‘검색’ 버튼을 눌렀을 때만 적용됩니다.")

    st.markdown("---")

    table_col, detail_col = st.columns([1.35, 1.0])

    with table_col:
        # 표는 항상 최소 code/name은 보여주기
        show_cols = ["code", "name"]
        if "market" in filtered.columns and filtered["market"].astype(str).str.strip().any():
            show_cols.append("market")
        if "industry" in filtered.columns and filtered["industry"].astype(str).str.strip().any():
            show_cols.append("industry")

        if len(filtered) == 0:
            st.warning("검색 결과가 없습니다. 키워드를 바꿔보세요.")
        else:
            st.dataframe(filtered[show_cols], use_container_width=True, height=600)

    with detail_col:
        st.markdown("#### 종목 상세(기본)")
        if len(filtered) > 0:
            options = (filtered["name"].astype(str) + " (" + filtered["code"].astype(str) + ")").tolist()
            options = options[:5000]
            selected = st.selectbox("종목 선택", options=options)

            # 선택된 code 파싱
            m = re.search(r"\((\d{6})\)$", selected)
            sel_code = m.group(1) if m else ""
            row = filtered[filtered["code"].astype(str) == sel_code].head(1)

            if not row.empty:
                r = row.iloc[0].to_dict()
                st.write({
                    "종목코드": r.get("code"),
                    "종목명": r.get("name"),
                    "시장구분": r.get("market"),
                    "업종": r.get("industry"),
                })

        st.markdown("---")
        st.caption("다음 확장: 선택 종목 주가 차트/변동률/관심종목 저장")

    with st.expander("디버그: 종목 CSV 매핑 확인", expanded=False):
        st.write("원본 컬럼:", list(tickers_raw.columns))
        st.write("매핑:", tickers_meta)

# =========================
# TAB: ETF
# =========================
with tab_etf:
    st.subheader("📊 국내 ETF 탐색")

    kw = st.text_input("ETF 이름 검색", value="")

    df = etfs_df.copy()

    # 이름 컬럼 추정(ETF 파일은 다양)
    name_col = pick_col(df, ["종목명", "ETF명", "한글종목명", "한글 종목명", "name", "Name"])
    if name_col is None:
        # 첫 번째 텍스트 컬럼 우선
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        name_col = text_cols[0] if text_cols else df.columns[0]

    if kw.strip():
        df = df[df[name_col].astype(str).str.contains(kw.strip(), case=False, na=False)]

    st.metric("표시 ETF", f"{len(df):,}개")
    st.dataframe(df, use_container_width=True, height=650)

# =========================
# TAB: Real Estate
# =========================
with tab_realestate:
    st.subheader("🏠 부동산 데이터")

    df = realestate_df.copy()

    region_cols = detect_region_cols(df)
    month_cols = detect_month_cols(df)

    f1, f2, f3 = st.columns([1.1, 1.1, 2.0])

    with f1:
        # 지역 필터(가능하면 지역별(1) 사용)
        if region_cols:
            primary_region_col = region_cols[0]  # 보통 지역별(1)이 핵심
            regions = sorted(df[primary_region_col].dropna().astype(str).unique().tolist())
            sel_regions = st.multiselect(f"지역 선택 ({primary_region_col})", options=regions, default=[])
        else:
            sel_regions = []
            st.caption("지역 컬럼을 찾지 못했습니다.")

    with f2:
        # 월 컬럼 필터
        if month_cols:
            sel_months = st.multiselect("월 컬럼 선택(표시)", options=month_cols, default=month_cols[-6:] if len(month_cols) >= 6 else month_cols)
        else:
            sel_months = []
            st.caption("월(YYYY.MM) 컬럼을 찾지 못했습니다.")

    with f3:
        st.metric("표시 행", f"{len(df):,}건")

    # 필터 적용
    if region_cols and sel_regions:
        df = df[df[region_cols[0]].astype(str).isin(sel_regions)]

    # 표시 컬럼 구성: 앞쪽 범주 컬럼 + 선택 월 컬럼
    base_cols = [c for c in df.columns if c in region_cols or str(c).startswith("주택") or "유형" in str(c)]
    base_cols = base_cols[:6]  # 너무 길어지지 않게

    show_cols = base_cols + sel_months if sel_months else base_cols
    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, height=650)

    with st.expander("디버그: 부동산 컬럼 감지", expanded=False):
        st.write("지역 컬럼 후보:", region_cols)
        st.write("월 컬럼 후보(YYYY.MM):", month_cols)
        st.write("전체 컬럼:", list(realestate_df.columns))

# =========================
# TAB: News
# =========================
with tab_news:
    st.subheader("📰 경제 뉴스 게시판")

    n1, n2, n3 = st.columns([1.0, 1.0, 2.0])
    with n1:
        max_items = st.slider("뉴스 개수", min_value=10, max_value=60, value=25, step=5)
    with n2:
        show_summary = st.toggle("요약 표시", value=False)
    with n3:
        keyword = st.text_input("키워드 필터(선택)", value="")

    items = fetch_news_items(max_items_total=max_items)

    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS/네트워크 이슈 가능)")
    else:
        if keyword.strip():
            kw = keyword.strip().lower()
            items = [x for x in items if kw in (x.get("title", "") + " " + x.get("summary", "")).lower()]

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

            if show_summary and x.get("summary"):
                st.write(x.get("summary"))

            st.divider()

st.markdown("---")
st.caption("© 재테크 대시보드 v4.1 | Local data: /data · Indices/News: internet")
