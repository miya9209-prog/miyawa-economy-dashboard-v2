import streamlit as st
import pandas as pd
import datetime as dt
from typing import Dict, List, Tuple, Optional

import requests
import feedparser
import yfinance as yf


# =========================
# Page Config
# =========================
st.set_page_config(page_title="미야와 재테크 대시보드 v4", layout="wide")

# -------------------------
# UI Helpers
# -------------------------
def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:+.2f}%"

def _fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:,.{decimals}f}"

def _safe_request_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _safe_request_text(url: str, timeout: int = 10) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return None


# =========================
# Data Loaders (Local files)
# =========================
@st.cache_data
def load_tickers() -> pd.DataFrame:
    # KRX CSV는 보통 euc-kr
    df = pd.read_csv("data/kr_tickers.csv", encoding="euc-kr")
    # 종목코드 6자리 정규화
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df

@st.cache_data
def load_etfs() -> pd.DataFrame:
    # 업로드된 파일이 utf-8 또는 euc-kr 혼재할 수 있어 안전 처리
    try:
        return pd.read_csv("data/kr_etfs.csv", encoding="utf-8")
    except Exception:
        return pd.read_csv("data/kr_etfs.csv", encoding="euc-kr")

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

@st.cache_data(ttl=60 * 15)  # 15 minutes
def fetch_index_snapshot(symbol: str) -> Dict[str, Optional[float]]:
    """
    Returns dict: {"last": ..., "prev": ..., "chg": ..., "chg_pct": ...}
    Uses yfinance (internet required).
    """
    out = {"last": None, "prev": None, "chg": None, "chg_pct": None}

    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="7d", interval="1d")
        if hist is None or hist.empty:
            return out

        # 마지막 2개 영업일 종가 기준
        closes = hist["Close"].dropna()
        if len(closes) == 0:
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
# News Board (RSS)
# =========================
NEWS_FEEDS: List[Tuple[str, str]] = [
    # URLs are inside code (OK)
    ("연합뉴스 경제", "https://www.yna.co.kr/rss/economy.xml"),
    ("한국경제", "https://rss.hankyung.com/economy.xml"),
    ("매일경제", "https://file.mk.co.kr/news/rss/rss_30000001.xml"),
    ("조선비즈", "https://biz.chosun.com/arc/outboundfeeds/rss/category/economy/?outputType=xml"),
]

@st.cache_data(ttl=60 * 20)  # 20 minutes
def fetch_news_items(max_items_total: int = 20) -> List[dict]:
    items: List[dict] = []
    for name, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[: max(5, max_items_total // 4)]:
                title = getattr(e, "title", "").strip()
                link = getattr(e, "link", "").strip()
                published = getattr(e, "published", "") or getattr(e, "updated", "")
                summary = getattr(e, "summary", "") or getattr(e, "description", "")
                items.append(
                    {
                        "source": name,
                        "title": title,
                        "link": link,
                        "published": published,
                        "summary": summary,
                    }
                )
        except Exception:
            continue

    # 간단 정렬: published 문자열이 제각각이라, 일단 최신 느낌으로 앞쪽 우선(수집 순서+부분 정렬)
    # 필요하면 추후 날짜 파싱 로직 추가
    return items[:max_items_total]


# =========================
# App Header
# =========================
st.title("📊 미야와 재테크 대시보드 v4")
st.caption("주요지수 요약 · 국내종목 · ETF · 부동산 · 뉴스 게시판")

# Top controls
top_col1, top_col2, top_col3 = st.columns([1.2, 1.2, 1.6])
with top_col1:
    auto_refresh = st.toggle("자동 새로고침(5분)", value=False)
with top_col2:
    if st.button("🔄 지금 새로고침"):
        st.cache_data.clear()
        st.rerun()
with top_col3:
    st.write("")
    st.info("데이터 파일은 GitHub의 `data/` 폴더에서 읽습니다. 지수/뉴스는 인터넷 연결이 필요합니다.", icon="ℹ️")

# Auto refresh
if auto_refresh:
    st.write("")
    st.write("자동 새로고침이 켜져 있습니다. (5분)")
    st.autorefresh(interval=5 * 60 * 1000, key="auto_refresh_5min")


# =========================
# Market Snapshot Section
# =========================
st.markdown("## 오늘의 주요지수 요약")

snap_cols = st.columns(len(INDEX_TICKERS))
for i, (label, symbol) in enumerate(INDEX_TICKERS.items()):
    snap = fetch_index_snapshot(symbol)
    last = snap["last"]
    chg_pct = snap["chg_pct"]
    delta = _fmt_pct(chg_pct) if chg_pct is not None else None
    snap_cols[i].metric(label, _fmt_num(last, 2), delta=delta)

with st.expander("지수 데이터가 비어있다면? (해결 체크)", expanded=False):
    st.write(
        "- Streamlit Cloud가 외부 인터넷 요청을 차단한 상태인지 확인\n"
        "- `requirements.txt`에 `yfinance`가 들어있는지 확인\n"
        "- 앱 우측 하단/Manage app에서 Reboot 후 재확인"
    )


st.markdown("---")

# =========================
# Load local datasets
# =========================
try:
    tickers_df = load_tickers()
    etfs_df = load_etfs()
    realestate_df = load_realestate()
except Exception as e:
    st.error("로컬 데이터 파일을 불러오는 중 오류가 발생했습니다. `data/` 폴더와 파일명을 확인해주세요.")
    st.exception(e)
    st.stop()


# =========================
# Sidebar Navigation
# =========================
st.sidebar.header("📌 메뉴")
menu = st.sidebar.radio(
    "이동",
    ["국내 주식 종목", "국내 ETF", "부동산 데이터", "뉴스 게시판"],
    index=0
)

# =========================
# Page: 국내 주식 종목
# =========================
if menu == "국내 주식 종목":
    st.subheader("📈 국내 주식 종목 탐색")

    # Filters
    st.sidebar.subheader("종목 필터")
    keyword = st.sidebar.text_input("종목명 검색", value="")
    market_col = "시장구분" if "시장구분" in tickers_df.columns else None
    industry_col = "업종" if "업종" in tickers_df.columns else None

    markets = []
    if market_col:
        markets = st.sidebar.multiselect(
            "시장 선택",
            options=sorted(tickers_df[market_col].dropna().unique()),
            default=sorted(tickers_df[market_col].dropna().unique()),
        )

    industries = []
    if industry_col:
        industries = st.sidebar.multiselect(
            "업종 선택",
            options=sorted(tickers_df[industry_col].dropna().unique()),
            default=[],
        )

    filtered = tickers_df.copy()

    if markets and market_col:
        filtered = filtered[filtered[market_col].isin(markets)]

    if industries and industry_col:
        filtered = filtered[filtered[industry_col].isin(industries)]

    if keyword.strip():
        if "종목명" in filtered.columns:
            filtered = filtered[filtered["종목명"].astype(str).str.contains(keyword.strip(), case=False, na=False)]

    left, right = st.columns([1.25, 1])
    with left:
        st.write(f"총 **{len(filtered):,}개** 종목")
        show_cols = [c for c in ["종목코드", "종목명", "시장구분", "업종"] if c in filtered.columns]
        st.dataframe(filtered[show_cols], use_container_width=True, height=580)

    with right:
        st.markdown("### 종목 상세(선택)")
        st.caption("v4에서는 기본 틀만 제공합니다. 다음 단계에서 종목 클릭 → 차트/재무를 붙입니다.")
        # 간단 선택 UI
        if "종목명" in filtered.columns:
            selected_name = st.selectbox("종목 선택", options=filtered["종목명"].astype(str).tolist()[:5000])
            row = filtered[filtered["종목명"].astype(str) == selected_name].head(1)
            if not row.empty:
                r = row.iloc[0].to_dict()
                st.write({
                    "종목코드": r.get("종목코드"),
                    "종목명": r.get("종목명"),
                    "시장구분": r.get("시장구분"),
                    "업종": r.get("업종"),
                })

        st.markdown("### 다음 확장(v5) 후보")
        st.write("- 선택 종목 주가 차트 / 변동률\n- 거래대금/거래량\n- 재무지표(PER/PBR/ROE)\n- 관심종목 저장")

# =========================
# Page: 국내 ETF
# =========================
elif menu == "국내 ETF":
    st.subheader("📊 국내 ETF 탐색")

    st.sidebar.subheader("ETF 필터")
    keyword = st.sidebar.text_input("ETF 이름 검색", value="")

    df = etfs_df.copy()

    # 이름 컬럼 추정
    name_col = None
    for cand in ["종목명", "ETF명", "한글종목명", "name", "Name"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        name_col = df.columns[0]

    if keyword.strip():
        df = df[df[name_col].astype(str).str.contains(keyword.strip(), case=False, na=False)]

    st.write(f"총 **{len(df):,}개** ETF")
    st.dataframe(df, use_container_width=True, height=650)

    st.markdown("### 다음 확장(v5) 후보")
    st.write("- ETF 유형 자동 분류(국내/해외/채권/원자재)\n- 배당/보수/추적지수 요약\n- 관심 ETF 저장")

# =========================
# Page: 부동산 데이터
# =========================
elif menu == "부동산 데이터":
    st.subheader("🏠 부동산 데이터")

    st.sidebar.subheader("부동산 필터")
    # 컬럼 기반 간단 필터: 지역/연도 컬럼이 있다면 제공
    df = realestate_df.copy()

    # 지역 추정 컬럼
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

    if region_col:
        regions = sorted(df[region_col].dropna().astype(str).unique().tolist())
        sel_regions = st.sidebar.multiselect("지역 선택", options=regions, default=[])
        if sel_regions:
            df = df[df[region_col].astype(str).isin(sel_regions)]

    if year_col:
        years = sorted(df[year_col].dropna().unique().tolist())
        sel_years = st.sidebar.multiselect("연도 선택", options=years, default=[])
        if sel_years:
            df = df[df[year_col].isin(sel_years)]

    st.write(f"총 **{len(df):,}건**")
    st.dataframe(df, use_container_width=True, height=650)

    st.markdown("### 다음 확장(v5) 후보")
    st.write("- 지역별 평균/중위값 요약 카드\n- 연도별 추이 차트\n- 관심 지역 저장")

# =========================
# Page: 뉴스 게시판
# =========================
elif menu == "뉴스 게시판":
    st.subheader("📰 오늘의 경제 뉴스 게시판")

    st.sidebar.subheader("뉴스 설정")
    max_items = st.sidebar.slider("가져올 뉴스 개수", min_value=10, max_value=60, value=20, step=5)
    show_summary = st.sidebar.toggle("요약(원문 요약 텍스트) 펼치기", value=False)

    items = fetch_news_items(max_items_total=max_items)

    if not items:
        st.warning("뉴스를 불러오지 못했습니다. 네트워크/피드 URL을 확인해주세요.")
    else:
        # 간단 키워드 필터
        kw = st.text_input("키워드 필터(선택)", value="")
        if kw.strip():
            items = [
                x for x in items
                if kw.lower() in (x.get("title","") + " " + x.get("summary","")).lower()
            ]

        st.write(f"표시: **{len(items):,}건**")

        for x in items:
            title = x.get("title", "").strip() or "(제목 없음)"
            source = x.get("source", "")
            link = x.get("link", "")
            published = x.get("published", "")

            # 제목을 링크로
            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")

            meta_line = " · ".join([p for p in [source, published] if p])
            if meta_line:
                st.caption(meta_line)

            if show_summary:
                summary = x.get("summary", "")
                if summary:
                    # RSS summary가 HTML일 수 있어 안전하게 텍스트로 표시
                    st.write(summary)

            st.divider()

    with st.expander("피드가 깨지면 이렇게 처리합니다", expanded=False):
        st.write(
            "- 일부 언론사는 RSS를 가끔 막거나 형식을 바꿉니다.\n"
            "- 그럴 땐 `NEWS_FEEDS` 목록에서 해당 항목을 빼고 다른 RSS를 추가하면 됩니다.\n"
            "- 원하시면 ‘형준님 취향(재테크/부동산/ETF/금리)’으로 RSS 세트를 더 안정적으로 구성해드릴게요."
        )


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("© miya9209-prog | 재테크 대시보드 v4 | Local data: /data  · Market/News: internet")
