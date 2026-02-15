import re
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import feedparser
import yfinance as yf


# =========================
# Page + CSS
# =========================
st.set_page_config(page_title="재테크 대시보드 v5.1.1", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 2.4rem !important; padding-bottom: 2rem !important; }
html, body, [class*="css"]  { font-size: 16px !important; }
h1 { font-size: 2.1rem !important; margin: 0.25rem 0 0.5rem 0 !important; line-height: 1.2 !important; }
div[data-baseweb="tab"] button { font-size: 18px !important; padding: 10px 16px !important; }
div[data-testid="stMetricValue"] { font-size: 24px !important; }
</style>
""",
    unsafe_allow_html=True
)


# =========================
# Utils
# =========================
def fmt_num(x: Optional[float], d: int = 2) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):,.{d}f}"

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
    for c in candidates:
        if c in cols:
            return c
    norm = {re.sub(r"\s+", "", str(x)).lower(): x for x in cols}
    for c in candidates:
        k = re.sub(r"\s+", "", str(c)).lower()
        if k in norm:
            return norm[k]
    cols_s = [str(c) for c in cols]
    for c in candidates:
        for real in cols_s:
            if str(c).lower() in real.lower():
                return real
    return None

def is_month_col(s: str) -> bool:
    return re.fullmatch(r"\d{4}[.\-/]\d{2}", str(s).strip()) is not None

def month_to_timestamp(s: str) -> pd.Timestamp:
    parts = re.split(r"[.\-/]", str(s).strip())
    y, m = int(parts[0]), int(parts[1])
    return pd.Timestamp(year=y, month=m, day=1)


# =========================
# Load local data
# =========================
@st.cache_data
def load_local_data():
    tickers_raw = safe_read_csv("data/kr_tickers.csv")
    etfs_raw = safe_read_csv("data/kr_etfs.csv")
    realestate_raw = pd.read_excel("data/realestate.xlsx")
    return tickers_raw, etfs_raw, realestate_raw


# =========================
# Normalize: Stocks / ETFs
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

    for c in ["code", "name", "market", "industry"]:
        out[c] = out[c].fillna("").astype(str).str.strip()
    out = out[out["name"] != ""].reset_index(drop=True)

    meta = {"code_col": code_col, "name_col": name_col, "market_col": market_col, "industry_col": industry_col}
    return out, meta

def normalize_etfs(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()
    code_col = pick_col(df, ["종목코드", "단축코드", "표준코드", "Code", "code"])
    name_col = pick_col(df, ["종목명", "ETF명", "한글종목명", "한글 종목명", "Name", "name"])

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

    for c in ["code", "name"]:
        out[c] = out[c].fillna("").astype(str).str.strip()
    out = out[out["name"] != ""].reset_index(drop=True)

    meta = {"code_col": code_col, "name_col": name_col}
    return out, meta


# =========================
# Real estate: Major city trend (SAFE)
# =========================
MAJOR_CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]

def detect_region_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("지역")]

def detect_type_col(df: pd.DataFrame) -> Optional[str]:
    return pick_col(df, ["주택유형별(1)", "주택유형", "유형", "주택유형별"])

def detect_month_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_month_col(str(c))]

def find_city_level(df: pd.DataFrame, region_cols: List[str]) -> Optional[str]:
    if not region_cols:
        return None
    for c in region_cols:
        vals = df[c].astype(str).fillna("").unique().tolist()
        hit = sum(1 for v in vals if v in MAJOR_CITIES)
        if hit >= 3:
            return c
    for c in region_cols:
        vals = df[c].astype(str).fillna("")
        if vals.str.contains("서울").any():
            return c
    return region_cols[-1]

def build_city_trend_long(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = raw.copy()
    region_cols = detect_region_cols(df)
    type_col = detect_type_col(df)
    month_cols = detect_month_cols(df)

    meta = {
        "region_cols": region_cols,
        "type_col": type_col,
        "month_cols": month_cols,
        "city_level_col": None,
    }

    if not month_cols:
        return pd.DataFrame(), meta

    city_level = find_city_level(df, region_cols)
    meta["city_level_col"] = city_level

    id_cols = []
    if type_col and type_col in df.columns:
        id_cols.append(type_col)
    if city_level and city_level in df.columns:
        id_cols.append(city_level)
    if not id_cols and region_cols:
        id_cols = [region_cols[0]]

    long = df[id_cols + month_cols].copy()

    for m in month_cols:
        long[m] = pd.to_numeric(long[m].astype(str).str.replace(",", ""), errors="coerce")

    long = long.melt(id_vars=id_cols, value_vars=month_cols, var_name="month", value_name="value")
    long["date"] = long["month"].astype(str).apply(month_to_timestamp)

    if type_col and type_col in id_cols:
        long = long.rename(columns={type_col: "type"})
    else:
        long["type"] = "전체"

    if city_level and city_level in id_cols:
        long = long.rename(columns={city_level: "city"})
    else:
        long["city"] = "전체"

    long["city"] = long["city"].astype(str).str.strip()
    long["type"] = long["type"].astype(str).str.strip()

    long = long.dropna(subset=["value"]).sort_values("date")
    return long, meta


# =========================
# Market snapshot (gold fallback)
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
    for sym in symbols:
        snap = fetch_snapshot(sym)
        if snap.get("last") is not None:
            return snap.get("last"), snap.get("chg_pct"), sym
    return None, None, None

MARKET_ITEMS: List[Tuple[str, List[str], int]] = [
    ("KOSPI", ["^KS11"], 2),
    ("KOSDAQ", ["^KQ11"], 2),
    ("USD/KRW", ["KRW=X"], 4),
    ("GOLD", ["XAUUSD=X", "GC=F", "GLD"], 2),
    ("WTI", ["CL=F"], 2),
    ("DXY", ["DX-Y.NYB"], 2),
    ("US10Y", ["^TNX"], 2),
    ("BTC", ["BTC-USD"], 0),
]


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
                })
        except Exception:
            continue
    return items[:max_items_total]


# =========================
# Session: watchlists
# =========================
if "watch_stocks" not in st.session_state:
    st.session_state["watch_stocks"] = {}
if "watch_etfs" not in st.session_state:
    st.session_state["watch_etfs"] = {}

def add_watch(kind: str, code: str, name: str):
    if not code or not name:
        return
    if kind == "stock":
        st.session_state["watch_stocks"][code] = name
    elif kind == "etf":
        st.session_state["watch_etfs"][code] = name


# =========================
# Header
# =========================
st.markdown("# 📊 재테크 대시보드 v5.1.1 (슬라이더 완전 안전판)")
st.caption("부동산 월 데이터가 6개여도 앱이 절대 죽지 않게 수정됨")

b1, b2 = st.columns([1.0, 2.2])
with b1:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()
with b2:
    st.info("로컬 데이터는 GitHub `data/`에서 읽습니다. 지표/뉴스는 인터넷이 필요합니다.", icon="ℹ️")

st.markdown("---")


# =========================
# Load
# =========================
try:
    tickers_raw, etfs_raw, realestate_raw = load_local_data()
    tickers, tickers_meta = normalize_tickers(tickers_raw)
    etfs, etfs_meta = normalize_etfs(etfs_raw)
    re_long, re_meta = build_city_trend_long(realestate_raw)
except Exception as e:
    st.error("data 폴더 파일을 읽지 못했습니다. 파일명/경로/인코딩을 확인해주세요.")
    st.exception(e)
    st.stop()


# =========================
# Tabs
# =========================
tab_dash, tab_stock, tab_etf, tab_re, tab_news, tab_debug = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산(대도시 추이)", "📰 뉴스", "🛠️ 진단"]
)


# =========================
# Dashboard
# =========================
with tab_dash:
    st.subheader("오늘의 주요지표 (금 포함)")
    per_row = 4
    rows = [MARKET_ITEMS[i:i+per_row] for i in range(0, len(MARKET_ITEMS), per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for i, (label, symbols, decimals) in enumerate(row):
            last, chg_pct, used = fetch_first_available(symbols)
            cols[i].metric(label, fmt_num(last, decimals), delta=fmt_pct(chg_pct) if chg_pct is not None else None)
            if used and len(symbols) > 1:
                cols[i].caption(f"source: {used}")

    st.markdown("---")
    st.markdown("### 뉴스 Top 8")
    items = fetch_news(8)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다.")
    else:
        for x in items:
            title = x.get("title") or "(제목 없음)"
            link = x.get("link") or ""
            source = x.get("source") or ""
            published = x.get("published") or ""
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)
            st.divider()


# =========================
# Stocks
# =========================
with tab_stock:
    st.subheader("주식 검색 & 담기")

    kw = st.text_input("종목명 또는 6자리 코드", value="삼성전자")
    df = tickers.copy()
    if kw.strip():
        df = df[df["name"].str.contains(kw, case=False, na=False) | df["code"].str.contains(kw, case=False, na=False)]

    st.metric("검색 결과", f"{len(df):,}개")

    if len(df) == 0:
        st.warning("검색 결과가 없습니다.")
    else:
        st.dataframe(df[["code", "name", "market", "industry"]].head(200), use_container_width=True, height=520)
        options = (df["name"] + " (" + df["code"] + ")").tolist()[:5000]
        sel = st.selectbox("종목 선택", options=options)
        m = re.search(r"\((\d{6})\)$", sel)
        code = m.group(1) if m else ""
        row = df[df["code"] == code].head(1)
        if not row.empty:
            name = row.iloc[0]["name"]
            if st.button("⭐ 관심주식 담기"):
                add_watch("stock", code, name)
                st.success("담았습니다(세션 저장).")


# =========================
# ETF
# =========================
with tab_etf:
    st.subheader("ETF 검색 & 담기")

    kw = st.text_input("ETF명 또는 코드", value="")
    df = etfs.copy()
    if kw.strip():
        df = df[df["name"].str.contains(kw, case=False, na=False) | df["code"].str.contains(kw, case=False, na=False)]

    st.metric("검색 결과", f"{len(df):,}개")

    if len(df) == 0:
        st.warning("검색 결과가 없습니다.")
    else:
        st.dataframe(df[["code", "name"]].head(200), use_container_width=True, height=520)
        options = (df["name"] + " (" + df["code"] + ")").tolist()[:5000]
        sel = st.selectbox("ETF 선택", options=options)
        m = re.search(r"\((\d{6})\)$", sel)
        code = m.group(1) if m else ""
        row = df[df["code"] == code].head(1)
        if not row.empty:
            name = row.iloc[0]["name"]
            if st.button("⭐ 관심ETF 담기"):
                add_watch("etf", code, name)
                st.success("담았습니다(세션 저장).")


# =========================
# Real Estate (슬라이더 완전 안전)
# =========================
with tab_re:
    st.subheader("주요 대도시 매매가격 추이 (슬라이더 안전판)")

    if re_long.empty:
        st.error("부동산 데이터에서 월 컬럼(예: 2024.10)을 찾지 못했습니다. 진단 탭에서 month_cols를 확인해주세요.")
    else:
        all_types = sorted(re_long["type"].dropna().unique().tolist())
        sel_type = st.selectbox("주택유형 선택", options=all_types, index=0)

        cities_all = sorted(re_long["city"].dropna().unique().tolist())
        major_candidates = []
        for c in cities_all:
            for mc in MAJOR_CITIES:
                if c == mc or mc in c:
                    major_candidates.append(c)
                    break
        major_candidates = sorted(list(dict.fromkeys(major_candidates))) or cities_all

        default_sel = [c for c in major_candidates if "서울" in c][:1] + major_candidates[:4]
        default_sel = list(dict.fromkeys(default_sel))[:6]
        sel_cities = st.multiselect("대도시 선택", options=major_candidates, default=default_sel)

        # ---- 핵심 수정: total_months <= 6 이면 슬라이더를 만들지 않는다 (6~6 방지)
        all_dates = sorted(pd.to_datetime(re_long["date"]).unique())
        total_months = len(all_dates)

        if total_months == 0:
            st.warning("월 데이터(date)가 비어 있습니다. 엑셀 월 컬럼 인식 실패 가능성이 큽니다.")
            n_months = 0
        elif total_months <= 6:
            st.info(f"월 데이터가 {total_months}개라 슬라이더 없이 전체 기간을 표시합니다.")
            n_months = total_months
        else:
            default_n = 12 if total_months >= 12 else total_months
            n_months = st.slider("최근 몇 개월 표시", min_value=6, max_value=total_months, value=default_n, step=1)

        df = re_long.copy()
        df = df[df["type"] == sel_type]
        if sel_cities:
            df = df[df["city"].isin(sel_cities)]

        if n_months > 0:
            df = df[df["date"].isin(all_dates[-n_months:])]

        pv = df.pivot_table(index="date", columns="city", values="value", aggfunc="mean").sort_index()
        st.line_chart(pv, use_container_width=True)

        st.caption(f"도시 레벨 자동감지 컬럼: {re_meta.get('city_level_col')}")


# =========================
# News
# =========================
with tab_news:
    st.subheader("경제 뉴스")
    max_items = st.slider("뉴스 개수", 10, 60, 25, 5)
    items = fetch_news(max_items)
    if not items:
        st.warning("뉴스를 불러오지 못했습니다.")
    else:
        for x in items:
            title = x.get("title") or "(제목 없음)"
            link = x.get("link") or ""
            source = x.get("source") or ""
            published = x.get("published") or ""
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            meta = " · ".join([p for p in [source, published] if p])
            if meta:
                st.caption(meta)
            st.divider()


# =========================
# Debug
# =========================
with tab_debug:
    st.subheader("🛠️ 진단")

    st.markdown("### 주식 CSV 매핑")
    st.write("원본 컬럼:", list(tickers_raw.columns))
    st.write("매핑:", tickers_meta)
    st.dataframe(tickers.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("### ETF CSV 매핑")
    st.write("원본 컬럼:", list(etfs_raw.columns))
    st.write("매핑:", etfs_meta)
    st.dataframe(etfs.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("### 부동산 감지")
    st.write("region_cols:", re_meta.get("region_cols"))
    st.write("type_col:", re_meta.get("type_col"))
    st.write("month_cols count:", len(re_meta.get("month_cols", [])))
    st.write("month_cols sample:", (re_meta.get("month_cols", [])[:15]))
    st.write("city_level_col:", re_meta.get("city_level_col"))
    if re_long.empty:
        st.warning("re_long이 비었습니다(월 컬럼 감지 실패 가능).")
    else:
        st.dataframe(re_long.head(50), use_container_width=True)

st.caption("© v5.1.1 | 슬라이더(6~6) 예외 완전 차단")
