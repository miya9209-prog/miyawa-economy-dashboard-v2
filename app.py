import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Tuple
import requests
import feedparser
import yfinance as yf


# =========================
# Page Config
# =========================
st.set_page_config(page_title="재테크 대시보드 v4", layout="wide")

# =========================
# Global CSS (폰트/탭/버튼/레이아웃)
# =========================
st.markdown(
    """
<style>
/* 전체 기본 글씨 약간 키우기 */
html, body, [class*="css"]  {
  font-size: 16px !important;
}

/* 타이틀/섹션 제목 더 또렷하게 */
h1 { font-size: 2.1rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.25rem !important; }

/* 탭 버튼 글씨 키우기 */
div[data-baseweb="tab"] button {
  font-size: 18px !important;
  padding: 10px 16px !important;
}

/* metric 글씨 조금 키우기 */
div[data-testid="stMetric"] label {
  font-size: 15px !important;
}
div[data-testid="stMetricValue"] {
  font-size: 24px !important;
}

/* 데이터프레임 위 아래 여백 */
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
    # csv 인코딩 혼재 대비
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, encoding="euc-kr")

# =========================
# Local Data Loaders
# =========================
@st.cache_data
def load_tickers() -> pd.DataFrame:
    df = safe_read_csv("data/kr_tickers.csv")
    # 종목코드 6자리 정규화
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df

@st.cache_data
def load_etfs() -> pd.DataFrame:
    return safe_read_csv("data/kr_etfs.csv")

@st.cache_data
def load_realestate() -> pd.DataFrame:
    return pd.read_excel("data/realestate.xlsx")

# =========================
# Market Snapshot (Indices)
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
    out = {"last": None, "prev": None, "chg": None, "chg_pct": None}
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
        chg = (last - prev) if prev is not None else None
        chg_pct = (chg / prev * 100.0) if (prev is not None and prev != 0) else None

        out.update({"last": last, "prev": prev, "chg": chg, "chg_pct": chg_pct})
        return out
    except Exception:
        return out

# =========================
# News (RSS)
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
st.title("📊 재테크 대시보드 v4")
st.caption("주요지수 요약 · 주식/ETF/부동산 탐색 · 뉴스 게시판")

top_c1, top_c2, top_c3 = st.columns([1.2, 1.2, 2.0])
with top_c1:
    auto_refresh = st.toggle("자동 새로고침(5분)", value=False)
with top_c2:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
with top_c3:
    st.info("로컬 데이터는 GitHub `data/` 폴더에서 읽습니다. 지수/뉴스는 인터넷이 필요합니다.", icon="ℹ️")

if auto_refresh:
    st.autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5min")

st.markdown("---")

# =========================
# Load Local Data (fail fast)
# =========================
try:
    tickers_df = load_tickers()
    etfs_df = load_etfs()
    realestate_df = load_realestate()
except Exception as e:
    st.error("data 폴더의 파일을 읽지 못했습니다. 파일명/경로/인코딩을 확인해주세요.")
    st.exception(e)
    st.stop()

# =========================
# Top Tabs (클릭 가능한 실제 탭)
# =========================
tab_dashboard, tab_stocks, tab_etf, tab_realestate, tab_news = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산", "📰 뉴스"]
)

# =========================
# TAB 1: Dashboard
# =========================
with tab_dashboard:
    st.subheader("오늘의 주요지수 요약")

    cols = st.columns(len(INDEX_TICKERS))
    for i, (label, symbol) in enumerate(INDEX_TICKERS.items()):
        snap = fetch_index_snapshot(symbol)
        last = snap["last"]
        delta = fmt_pct(snap["chg_pct"]) if snap["chg_pct"] is not None else None
        cols[i].metric(label, fmt_num(last, 2), delta=delta)

    st.markdown("### 오늘의 데이터 한눈에")
    c1, c2, c3 = st.columns(3)
    c1.metric("KRX 종목 수", f"{len(tickers_df):,}")
    c2.metric("ETF 데이터 행 수", f"{len(etfs_df):,}")
    c3.metric("부동산 데이터 행 수", f"{len(realestate_df):,}")

    st.markdown("---")
    st.markdown("### 오늘의 경제 뉴스 Top")
    items = fetch_news_items(max_items_total=10)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS 접근/네트워크 이슈 가능)")
    else:
        for x in items[:10]:
            title = x.get("title", "") or "(제목 없음)"
            link = x.get("link", "")
            source = x.get("source", "")
            published = x.get("published", "")

            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")
            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)
            st.divider()

# =========================
# TAB 2: Stocks (검색 버튼 있는 폼)
# =========================
with tab_stocks:
    st.subheader("📈 국내 주식 종목 탐색")

    # 검색/필터 컨트롤 영역
    left_filters, right_info = st.columns([1.6, 1.0])

    # --- 검색 폼: '검색' 버튼으로 트리거
    with left_filters:
        with st.form("stock_search_form", clear_on_submit=False):
            st.markdown("#### 검색 조건")
            keyword_input = st.text_input("종목명 검색", value=st.session_state.get("stock_kw", ""))
            market_col = "시장구분" if "시장구분" in tickers_df.columns else None
            industry_col = "업종" if "업종" in tickers_df.columns else None

            markets = None
            industries = None

            if market_col:
                market_opts = sorted(tickers_df[market_col].dropna().unique().tolist())
                markets = st.multiselect("시장 선택", options=market_opts, default=st.session_state.get("stock_markets", market_opts))

            if industry_col:
                ind_opts = sorted(tickers_df[industry_col].dropna().astype(str).unique().tolist())
                industries = st.multiselect("업종 선택(선택)", options=ind_opts, default=st.session_state.get("stock_industries", []))

            submit = st.form_submit_button("🔎 검색")

        # 검색 버튼을 눌렀을 때만 조건 적용(=“검색 버튼이 없어서 안된다” 해결)
        if submit:
            st.session_state["stock_kw"] = keyword_input
            st.session_state["stock_markets"] = markets if markets is not None else []
            st.session_state["stock_industries"] = industries if industries is not None else []

    # --- 조건 적용 (세션값 기준)
    kw = st.session_state.get("stock_kw", "")
    selected_markets = st.session_state.get("stock_markets", None)
    selected_industries = st.session_state.get("stock_industries", [])

    filtered = tickers_df.copy()

    if "시장구분" in filtered.columns and selected_markets:
        filtered = filtered[filtered["시장구분"].isin(selected_markets)]

    if "업종" in filtered.columns and selected_industries:
        filtered = filtered[filtered["업종"].astype(str).isin([str(x) for x in selected_industries])]

    if kw.strip() and "종목명" in filtered.columns:
        filtered = filtered[filtered["종목명"].astype(str).str.contains(kw.strip(), case=False, na=False)]

    with right_info:
        st.markdown("#### 결과 요약")
        st.metric("검색 결과", f"{len(filtered):,}개")
        st.caption("검색 조건은 ‘검색’ 버튼을 눌렀을 때만 적용됩니다.")

    st.markdown("---")

    # 표 + 상세 패널
    table_col, detail_col = st.columns([1.35, 1.0])

    with table_col:
        show_cols = [c for c in ["종목코드", "종목명", "시장구분", "업종"] if c in filtered.columns]
        st.dataframe(filtered[show_cols], use_container_width=True, height=600)

    with detail_col:
        st.markdown("#### 종목 상세(기본)")
        if "종목명" in filtered.columns and len(filtered) > 0:
            # 너무 많은 옵션 방지: 상위 5000개까지만 (필요시 늘리기)
            names = filtered["종목명"].astype(str).tolist()[:5000]
            selected_name = st.selectbox("종목 선택", options=names)

            row = filtered[filtered["종목명"].astype(str) == selected_name].head(1)
            if not row.empty:
                r = row.iloc[0].to_dict()
                st.write({
                    "종목코드": r.get("종목코드"),
                    "종목명": r.get("종목명"),
                    "시장구분": r.get("시장구분"),
                    "업종": r.get("업종"),
                })

        st.markdown("---")
        st.caption("다음 확장(원하시면): 선택 종목 주가 차트/변동률/관심종목 저장")

# =========================
# TAB 3: ETF
# =========================
with tab_etf:
    st.subheader("📊 국내 ETF 탐색")

    # ETF 검색은 즉시 반영해도 UX가 좋아서 버튼 없이 진행
    kw = st.text_input("ETF 이름 검색", value="")

    df = etfs_df.copy()

    # 이름 컬럼 추정
    name_col = None
    for cand in ["종목명", "ETF명", "한글종목명", "name", "Name"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        name_col = df.columns[0]

    if kw.strip():
        df = df[df[name_col].astype(str).str.contains(kw.strip(), case=False, na=False)]

    st.metric("표시 ETF", f"{len(df):,}개")
    st.dataframe(df, use_container_width=True, height=650)

# =========================
# TAB 4: Real Estate
# =========================
with tab_realestate:
    st.subheader("🏠 부동산 데이터")

    df = realestate_df.copy()

    # 컬럼 추정
    region_col = None
    for cand in ["지역", "시도", "구", "동", "Region", "지역명"]:
        if cand in df.columns:
            region_col = cand
            break

    year_col = None
    for cand in ["연도", "년도", "year", "Year"]:
        if cand in df.columns:
            year_col = cand
            break

    f1, f2, f3 = st.columns([1.2, 1.2, 2.0])
    with f1:
        if region_col:
            regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
            sel_regions = st.multiselect("지역 선택(선택)", options=regions, default=[])
        else:
            sel_regions = []
            st.caption("지역 컬럼을 찾지 못했습니다.")
    with f2:
        if year_col:
            years = sorted(df[year_col].dropna().unique().tolist())
            sel_years = st.multiselect("연도 선택(선택)", options=years, default=[])
        else:
            sel_years = []
            st.caption("연도 컬럼을 찾지 못했습니다.")
    with f3:
        st.metric("표시 행", f"{len(df):,}건")

    if region_col and sel_regions:
        df = df[df[region_col].astype(str).isin(sel_regions)]
    if year_col and sel_years:
        df = df[df[year_col].isin(sel_years)]

    st.dataframe(df, use_container_width=True, height=650)

# =========================
# TAB 5: News Board
# =========================
with tab_news:
    st.subheader("📰 경제 뉴스 게시판")

    nc1, nc2, nc3 = st.columns([1.0, 1.0, 2.0])
    with nc1:
        max_items = st.slider("뉴스 개수", min_value=10, max_value=60, value=25, step=5)
    with nc2:
        show_summary = st.toggle("요약 표시", value=False)
    with nc3:
        keyword = st.text_input("키워드 필터(선택)", value="")

    items = fetch_news_items(max_items_total=max_items)

    if not items:
        st.warning("뉴스를 불러오지 못했습니다. (RSS 접근/네트워크 이슈 가능)")
    else:
        if keyword.strip():
            kw = keyword.strip().lower()
            items = [x for x in items if kw in (x.get("title","") + " " + x.get("summary","")).lower()]

        st.metric("표시 뉴스", f"{len(items):,}건")

        for x in items:
            title = x.get("title", "") or "(제목 없음)"
            link = x.get("link", "")
            source = x.get("source", "")
            published = x.get("published", "")

            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)

            if show_summary:
                summary = x.get("summary", "")
                if summary:
                    st.write(summary)

            st.divider()

st.markdown("---")
st.caption("© 재테크 대시보드 v4 | Local data: /data  · Indices/News: internet")
