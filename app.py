import re
from typing import Optional, Dict, List, Tuple

import streamlit as st
import pandas as pd
import feedparser
import yfinance as yf


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(page_title="재테크 대시보드 v6.2", layout="wide")

st.markdown(
    """
<style>
/* 상단 타이틀 잘림 방지 */
.block-container { padding-top: 3.4rem !important; padding-bottom: 2rem !important; }

/* 탭/폰트 크게 */
html, body, [class*="css"]  { font-size: 16px !important; }
h1 { font-size: 2.1rem !important; margin: 0.25rem 0 0.6rem 0 !important; line-height: 1.2 !important; }
div[data-baseweb="tab"] button { font-size: 18px !important; padding: 10px 16px !important; }
div[data-testid="stMetricValue"] { font-size: 26px !important; }

/* 버튼/입력 박스 */
button[kind="secondary"], button[kind="primary"] { padding: 0.55rem 0.9rem !important; }
</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# 유틸
# =========================================================
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

def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    s = s.strip()
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s.lower()

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    if not cols:
        return None
    norm_map = {norm_key(c): c for c in cols}

    for cand in candidates:
        k = norm_key(cand)
        if k in norm_map:
            return norm_map[k]

    cols_norm = [(c, norm_key(c)) for c in cols]
    for cand in candidates:
        ck = norm_key(cand)
        for real, rk in cols_norm:
            if ck and ck in rk:
                return real
    return None

def to_6digit(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.str.split().str[0]
    s = s.str.replace(r"[^0-9]", "", regex=True)
    return s.str.zfill(6)

def normalize_text_for_search(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", "", regex=True).str.replace(r"[^0-9A-Za-z가-힣]", "", regex=True)

def is_month_col(s: str) -> bool:
    return re.fullmatch(r"\d{4}[.\-/]\d{2}", str(s).strip()) is not None

def month_to_timestamp(s: str) -> pd.Timestamp:
    parts = re.split(r"[.\-/]", str(s).strip())
    y, m = int(parts[0]), int(parts[1])
    return pd.Timestamp(year=y, month=m, day=1)


# =========================================================
# 로컬 데이터 로드
# =========================================================
@st.cache_data
def load_local_data():
    tickers_raw = safe_read_csv("data/kr_tickers.csv")
    etfs_raw = safe_read_csv("data/kr_etfs.csv")
    realestate_raw = pd.read_excel("data/realestate.xlsx")
    return tickers_raw, etfs_raw, realestate_raw


# =========================================================
# 주식/ETF 표준화
# =========================================================
def normalize_tickers(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()
    code_col = pick_col(df, ["6자리코드", "6자리 코드", "단축코드", "종목코드", "code"])
    name_col = pick_col(df, ["종목명", "한글종목명", "종목약명", "name"])
    market_col = pick_col(df, ["시장구분", "시장", "market"])
    shares_col = pick_col(df, ["상장주식수", "상장주식수량", "shares"])

    out = pd.DataFrame()
    out["code"] = to_6digit(df[code_col]) if code_col is not None else ""
    out["name"] = df[name_col].astype(str).str.strip() if name_col is not None else ""
    out["market"] = df[market_col].astype(str).str.strip() if market_col is not None else ""
    out["shares"] = df[shares_col] if shares_col is not None else None

    out["name_key"] = normalize_text_for_search(out["name"])
    out["code_key"] = normalize_text_for_search(out["code"])
    out = out[out["name"].notna() & (out["name"].astype(str).str.strip() != "")]
    out = out.drop_duplicates(subset=["code", "name"]).reset_index(drop=True)

    meta = {"code_col": code_col, "name_col": name_col, "market_col": market_col, "shares_col": shares_col}
    return out, meta

def normalize_etfs(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = raw.copy()
    code_col = pick_col(df, ["종목코드", "6자리코드", "단축코드", "code"])
    name_col = pick_col(df, ["종목명", "ETF명", "name"])

    out = pd.DataFrame()
    out["code"] = to_6digit(df[code_col]) if code_col is not None else ""
    out["name"] = df[name_col].astype(str).str.strip() if name_col is not None else ""

    out["name_key"] = normalize_text_for_search(out["name"])
    out["code_key"] = normalize_text_for_search(out["code"])
    out = out[out["name"].notna() & (out["name"].astype(str).str.strip() != "")]
    out = out.drop_duplicates(subset=["code", "name"]).reset_index(drop=True)

    meta = {"code_col": code_col, "name_col": name_col}
    return out, meta


# =========================================================
# 부동산(대도시 매매가격 추이)
# =========================================================
MAJOR_CITIES = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종"]

def detect_region_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).strip().startswith("지역")]

def detect_type_col(df: pd.DataFrame) -> Optional[str]:
    return pick_col(df, ["주택유형별(1)", "주택유형", "유형", "주택유형별"])

def detect_month_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_month_col(str(c).strip())]

def best_city_level(df: pd.DataFrame, region_cols: List[str]) -> Optional[str]:
    if not region_cols:
        return None
    best, best_score = None, -1
    for c in region_cols:
        vals = df[c].astype(str).fillna("").unique().tolist()
        score = 0
        for v in vals:
            for mc in MAJOR_CITIES:
                if v == mc or mc in v:
                    score += 1
                    break
        if score > best_score:
            best_score, best = score, c
    return best if best_score > 0 else (region_cols[-1] if region_cols else None)

def build_city_trend_long(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = raw.copy()
    region_cols = detect_region_cols(df)
    type_col = detect_type_col(df)
    month_cols = detect_month_cols(df)

    meta = {"region_cols": region_cols, "type_col": type_col, "month_cols": month_cols, "city_level_col": None}
    if not month_cols:
        return pd.DataFrame(), meta

    city_col = best_city_level(df, region_cols)
    meta["city_level_col"] = city_col

    id_cols = []
    if type_col and type_col in df.columns:
        id_cols.append(type_col)
    if city_col and city_col in df.columns:
        id_cols.append(city_col)
    if not id_cols and region_cols:
        id_cols = [region_cols[0]]

    work = df[id_cols + month_cols].copy()

    # ✅ nan 제거: 엑셀 병합/빈 셀 구간을 위 값으로 채움
    for c in id_cols:
        work[c] = work[c].ffill()

    for m in month_cols:
        work[m] = pd.to_numeric(work[m].astype(str).str.replace(",", ""), errors="coerce")

    long = work.melt(id_vars=id_cols, value_vars=month_cols, var_name="month", value_name="value")
    long["date"] = long["month"].astype(str).apply(month_to_timestamp)

    if type_col and type_col in id_cols:
        long = long.rename(columns={type_col: "type"})
    else:
        long["type"] = "종합"

    if city_col and city_col in id_cols:
        long = long.rename(columns={city_col: "city"})
    else:
        long["city"] = "전체"

    long["type"] = long["type"].astype(str).str.strip()
    long["city"] = long["city"].astype(str).str.strip()

    long = long.dropna(subset=["value"])
    long = long[(long["city"] != "") & (long["city"].str.lower() != "nan")]
    long = long[(long["type"] != "") & (long["type"].str.lower() != "nan")]
    long = long.sort_values("date").reset_index(drop=True)
    return long, meta


# =========================================================
# 주요지표 + 금(한돈) + (대시보드 핵심) 10대 대표주 그래프
# =========================================================
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

@st.cache_data(ttl=60 * 15)
def calc_gold_don_krw() -> Dict[str, Optional[float]]:
    gold_usd_oz, gold_chg, gold_sym = fetch_first_available(["XAUUSD=X", "GC=F"])
    fx_krw, fx_chg, fx_sym = fetch_first_available(["KRW=X"])
    if gold_usd_oz is None or fx_krw is None:
        return {"don_krw": None, "chg_pct": None, "gold_sym": gold_sym, "fx_sym": fx_sym}
    usd_per_g = gold_usd_oz / 31.1034768
    krw_per_g = usd_per_g * fx_krw
    don_krw = krw_per_g * 3.75
    chg_pct = (float(gold_chg) + float(fx_chg)) if (gold_chg is not None and fx_chg is not None) else None
    return {"don_krw": float(don_krw), "chg_pct": chg_pct, "gold_sym": gold_sym, "fx_sym": fx_sym}

MARKET_ITEMS = [
    ("KOSPI", ["^KS11"], 2, "pt"),
    ("KOSDAQ", ["^KQ11"], 2, "pt"),
    ("USD/KRW", ["KRW=X"], 2, "원"),
    ("WTI", ["CL=F"], 2, "USD/bbl"),
    ("DXY", ["DX-Y.NYB"], 2, "index"),
    ("US10Y", ["^TNX"], 2, "%"),
    ("BTC", ["BTC-USD"], 0, "USD"),
]

# ✅ “최초 대시보드 핵심” 복구: 10대 대표주 그래프 (기본 하드코딩 + 추후 자동화 가능)
TOP10_STOCKS = [
    ("삼성전자", "005930.KS"),
    ("SK하이닉스", "000660.KS"),
    ("LG에너지솔루션", "373220.KS"),
    ("삼성바이오로직스", "207940.KS"),
    ("현대차", "005380.KS"),
    ("기아", "000270.KS"),
    ("NAVER", "035420.KS"),
    ("POSCO홀딩스", "005490.KS"),
    ("KB금융", "105560.KS"),
    ("삼성물산", "028260.KS"),
]

@st.cache_data(ttl=60 * 30)
def fetch_top10_prices(period: str = "6mo") -> pd.DataFrame:
    data = {}
    for name, sym in TOP10_STOCKS:
        try:
            hist = yf.Ticker(sym).history(period=period, interval="1d")
            if hist is None or hist.empty:
                continue
            s = hist["Close"].dropna()
            if s.empty:
                continue
            data[name] = s
        except Exception:
            continue
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).dropna(how="all")
    return df

def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    # 그래프를 “비교”하기 위해 시작값=100으로 정규화
    out = df.copy()
    for c in out.columns:
        s = out[c].dropna()
        if len(s) == 0:
            continue
        base = float(s.iloc[0])
        if base != 0:
            out[c] = out[c] / base * 100.0
    return out


# =========================================================
# 뉴스
# =========================================================
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


# =========================================================
# 세션 상태
# =========================================================
if "stock_query" not in st.session_state:
    st.session_state["stock_query"] = "삼성전자"
if "etf_query" not in st.session_state:
    st.session_state["etf_query"] = "코덱스 200"
if "watch_stocks" not in st.session_state:
    st.session_state["watch_stocks"] = {}  # code -> name
if "watch_etfs" not in st.session_state:
    st.session_state["watch_etfs"] = {}    # code -> name

def add_watch(kind: str, code: str, name: str):
    if not code or not name:
        return
    if kind == "stock":
        st.session_state["watch_stocks"][code] = name
    elif kind == "etf":
        st.session_state["watch_etfs"][code] = name

def remove_watch(kind: str, code: str):
    if kind == "stock":
        st.session_state["watch_stocks"].pop(code, None)
    elif kind == "etf":
        st.session_state["watch_etfs"].pop(code, None)


# =========================================================
# 헤더/가이드
# =========================================================
st.markdown("# 📊 재테크 대시보드 v6.2")
st.caption("‘최초 대시보드 핵심(10대 대표주 그래프)’을 기준으로 고정 + 검색/담기/부동산/뉴스/진단 통합")

c1, c2 = st.columns([1.0, 2.4])
with c1:
    if st.button("🔄 새로고침(캐시 초기화)"):
        st.cache_data.clear()
        st.rerun()
with c2:
    st.info("로컬 데이터는 GitHub `data/`에서 읽습니다. 주요지표/뉴스/10대주 그래프는 인터넷이 필요합니다.", icon="ℹ️")

with st.expander("📌 활용 가이드(복구)", expanded=True):
    st.markdown(
        """
**대시보드**
- 주요지표 7종 + **금(한돈) 원화** + **10대 대표주 비교 그래프(정규화 100 기준)**

**주식**
- 종목명/코드 입력 → **[검색]** → 결과 선택 → **[관심주식 담기]**

**ETF**
- ETF명/코드 입력(띄어쓰기 무시: ‘코덱스 200’ OK) → **[검색]** → 선택 → **[관심ETF 담기]**

**부동산**
- 주택유형/대도시 선택 → 추이 그래프
- 월 데이터가 6개 이하일 땐 슬라이더 없이 전체 표시(에러 방지)

**관심목록**
- 담긴 주식/ETF 확인, 삭제, CSV 다운로드
        """
    )

st.markdown("---")


# =========================================================
# 데이터 로드
# =========================================================
try:
    tickers_raw, etfs_raw, realestate_raw = load_local_data()
    tickers, tickers_meta = normalize_tickers(tickers_raw)
    etfs, etfs_meta = normalize_etfs(etfs_raw)
    re_long, re_meta = build_city_trend_long(realestate_raw)
except Exception as e:
    st.error("data 폴더 파일을 읽지 못했습니다. (data/kr_tickers.csv, data/kr_etfs.csv, data/realestate.xlsx)")
    st.exception(e)
    st.stop()


# =========================================================
# 탭
# =========================================================
tab_dash, tab_stock, tab_etf, tab_re, tab_news, tab_watch, tab_debug = st.tabs(
    ["📌 대시보드", "📈 주식", "📊 ETF", "🏠 부동산(대도시 추이)", "📰 뉴스", "⭐ 관심목록", "🛠️ 진단"]
)


# =========================================================
# 대시보드 (핵심: 10대 대표주 그래프 복구)
# =========================================================
with tab_dash:
    st.subheader("오늘의 주요지표 (단위 표기)")

    per_row = 4
    rows = [MARKET_ITEMS[i:i + per_row] for i in range(0, len(MARKET_ITEMS), per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for i, (label, symbols, decimals, unit) in enumerate(row):
            last, chg_pct, used = fetch_first_available(symbols)
            cols[i].metric(f"{label} ({unit})", fmt_num(last, decimals), delta=fmt_pct(chg_pct) if chg_pct is not None else None)
            if used and len(symbols) > 1:
                cols[i].caption(f"source: {used}")

    st.markdown("### 🟡 금(한돈) 가격 (원/3.75g)")
    g = calc_gold_don_krw()
    gcols = st.columns([1.2, 1.2, 3.0])
    gcols[0].metric("금 한돈", fmt_num(g.get("don_krw"), 0), delta=fmt_pct(g.get("chg_pct")) if g.get("chg_pct") is not None else None)
    gcols[1].metric("금 국제시세", fmt_num(fetch_first_available(["XAUUSD=X", "GC=F"])[0], 2), delta=None)
    gcols[2].caption(f"금 소스: {g.get('gold_sym')}, 환율 소스: {g.get('fx_sym')}  (변동률은 금%+환율% 근사)")

    st.markdown("---")
    st.subheader("📈 국내 10대 대표주 비교 그래프 (정규화: 시작=100)")
    p = st.radio("기간 선택", options=["3mo", "6mo", "1y"], horizontal=True, index=1)
    top_df = fetch_top10_prices(period=p)

    if top_df.empty:
        st.warning("10대 대표주 데이터를 불러오지 못했습니다. (인터넷/야후 파이낸스 차단 가능)")
    else:
        norm_df = normalize_index(top_df)
        st.caption("서로 가격 단위가 달라 비교가 어려워서, **시작값=100**으로 맞춰서 추이를 비교합니다.")
        st.line_chart(norm_df, use_container_width=True)
        with st.expander("원본(종가) 표 보기", expanded=False):
            st.dataframe(top_df.tail(30), use_container_width=True)

    st.markdown("---")
    st.subheader("📰 뉴스 Top 10")
    items = fetch_news(10)
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


# =========================================================
# 주식 (검색 버튼 + 담기 버튼)
# =========================================================
with tab_stock:
    st.subheader("주식 검색 & 담기")

    with st.form("stock_search_form", clear_on_submit=False):
        q = st.text_input("종목명 또는 6자리 코드", value=st.session_state["stock_query"])
        submitted = st.form_submit_button("🔎 검색")
    if submitted:
        st.session_state["stock_query"] = q

    q_use = st.session_state["stock_query"].strip()
    q_key = re.sub(r"\s+", "", q_use)
    key = normalize_text_for_search(pd.Series([q_key])).iloc[0] if q_use else ""

    df = tickers.copy()
    if q_use:
        df = df[
            df["name_key"].str.contains(key, na=False)
            | df["code_key"].str.contains(key, na=False)
            | df["name"].str.contains(q_use, na=False)
            | df["code"].str.contains(q_use, na=False)
        ]

    st.metric("검색 결과", f"{len(df):,}개")

    if df.empty:
        st.warning("검색 결과가 없습니다.")
    else:
        st.dataframe(df[["code", "name", "market"]].head(200), use_container_width=True, height=420)
        options = (df["name"] + " (" + df["code"] + ")").tolist()[:5000]
        sel = st.selectbox("검색 결과에서 종목 선택", options=options, key="stock_pick")
        m = re.search(r"\((\d{6})\)$", sel)
        code = m.group(1) if m else ""
        row = df[df["code"] == code].head(1)
        if not row.empty:
            name = row.iloc[0]["name"]
            b1, b2 = st.columns([1.1, 2.9])
            with b1:
                if st.button("⭐ 관심주식 담기", key=f"btn_add_stock_{code}"):
                    add_watch("stock", code, name)
                    st.success(f"담기 완료: {name} ({code})")
            with b2:
                st.caption("관심목록 탭에서 확인/삭제/다운로드 할 수 있습니다.")

    st.divider()
    st.markdown("#### 현재 관심주식")
    wl = st.session_state["watch_stocks"]
    if not wl:
        st.info("아직 담긴 관심주식이 없습니다.")
    else:
        sdf = pd.DataFrame([{"code": k, "name": v} for k, v in wl.items()]).sort_values("name")
        st.dataframe(sdf, use_container_width=True, height=220)


# =========================================================
# ETF (검색 버튼 + 담기 버튼 + 띄어쓰기 무시)
# =========================================================
with tab_etf:
    st.subheader("ETF 검색 & 담기")

    with st.form("etf_search_form", clear_on_submit=False):
        q = st.text_input("ETF명 또는 코드", value=st.session_state["etf_query"])
        submitted = st.form_submit_button("🔎 검색")
    if submitted:
        st.session_state["etf_query"] = q

    q_use = st.session_state["etf_query"].strip()
    q_key = re.sub(r"\s+", "", q_use)
    key = normalize_text_for_search(pd.Series([q_key])).iloc[0] if q_use else ""

    df = etfs.copy()
    if q_use:
        df = df[
            df["name_key"].str.contains(key, na=False)
            | df["code_key"].str.contains(key, na=False)
            | df["name"].str.contains(q_use, na=False)
            | df["code"].str.contains(q_use, na=False)
        ]

    st.metric("검색 결과", f"{len(df):,}개")

    if df.empty:
        st.warning("검색 결과가 없습니다. 예) '코덱스200' / '코덱스 200' 모두 가능")
    else:
        st.dataframe(df[["code", "name"]].head(200), use_container_width=True, height=420)
        options = (df["name"] + " (" + df["code"] + ")").tolist()[:5000]
        sel = st.selectbox("검색 결과에서 ETF 선택", options=options, key="etf_pick")
        m = re.search(r"\((\d{6})\)$", sel)
        code = m.group(1) if m else ""
        row = df[df["code"] == code].head(1)
        if not row.empty:
            name = row.iloc[0]["name"]
            b1, b2 = st.columns([1.1, 2.9])
            with b1:
                if st.button("⭐ 관심ETF 담기", key=f"btn_add_etf_{code}"):
                    add_watch("etf", code, name)
                    st.success(f"담기 완료: {name} ({code})")
            with b2:
                st.caption("관심목록 탭에서 확인/삭제/다운로드 할 수 있습니다.")

    st.divider()
    st.markdown("#### 현재 관심ETF")
    wl = st.session_state["watch_etfs"]
    if not wl:
        st.info("아직 담긴 관심ETF가 없습니다.")
    else:
        edf = pd.DataFrame([{"code": k, "name": v} for k, v in wl.items()]).sort_values("name")
        st.dataframe(edf, use_container_width=True, height=220)


# =========================================================
# 부동산 (대도시 추이 + nan 제거 + 슬라이더 에러 방지)
# =========================================================
with tab_re:
    st.subheader("주요 대도시 매매가격 추이")

    if re_long.empty:
        st.error("부동산 데이터에서 월 컬럼(예: 2024.10)을 찾지 못했습니다. 진단 탭에서 month_cols를 확인해주세요.")
    else:
        types = sorted(re_long["type"].dropna().unique().tolist()) or ["종합"]
        sel_type = st.selectbox("주택유형 선택", options=types, index=0)

        cities_all = sorted(re_long["city"].dropna().unique().tolist())
        major_candidates = []
        for c in cities_all:
            for mc in MAJOR_CITIES:
                if c == mc or mc in c:
                    major_candidates.append(c)
                    break
        major_candidates = sorted(list(dict.fromkeys(major_candidates))) or cities_all

        default_sel = [c for c in major_candidates if "서울" in c][:1] + major_candidates[:5]
        default_sel = list(dict.fromkeys(default_sel))[:6]
        sel_cities = st.multiselect("대도시 선택", options=major_candidates, default=default_sel)

        all_dates = sorted(pd.to_datetime(re_long["date"]).unique())
        total_months = len(all_dates)

        if total_months == 0:
            st.warning("월 데이터(date)가 비어 있습니다.")
            n_months = 0
        elif total_months <= 6:
            st.info(f"월 데이터가 {total_months}개라 슬라이더 없이 전체 기간을 표시합니다.")
            n_months = total_months
        else:
            default_n = 12 if total_months >= 12 else total_months
            n_months = st.slider("최근 몇 개월 표시", min_value=6, max_value=total_months, value=default_n, step=1)

        df = re_long[re_long["type"] == sel_type].copy()
        if sel_cities:
            df = df[df["city"].isin(sel_cities)]
        if n_months > 0:
            df = df[df["date"].isin(all_dates[-n_months:])]

        pv = df.pivot_table(index="date", columns="city", values="value", aggfunc="mean").sort_index()

        st.markdown("### 📈 대도시 매매가격 추이(지수)")
        st.line_chart(pv, use_container_width=True)

        with st.expander("표로 보기", expanded=False):
            st.dataframe(pv.reset_index(), use_container_width=True, height=320)

        st.caption(f"도시 레벨 자동감지 컬럼: {re_meta.get('city_level_col')} | month_cols: {len(re_meta.get('month_cols', []))}개")


# =========================================================
# 뉴스
# =========================================================
with tab_news:
    st.subheader("경제 뉴스 게시판")
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


# =========================================================
# 관심목록 (삭제 + 다운로드)
# =========================================================
with tab_watch:
    st.subheader("⭐ 관심목록 (주식/ETF)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 관심주식")
        wl = st.session_state["watch_stocks"]
        if not wl:
            st.info("관심주식이 없습니다.")
        else:
            sdf = pd.DataFrame([{"code": k, "name": v} for k, v in wl.items()]).sort_values("name")
            st.dataframe(sdf, use_container_width=True, height=320)

            rm = st.selectbox("삭제할 종목 선택", options=["(선택안함)"] + (sdf["name"] + " (" + sdf["code"] + ")").tolist(), key="rm_stock")
            if rm != "(선택안함)":
                m = re.search(r"\((\d{6})\)$", rm)
                code = m.group(1) if m else None
                if code and st.button("🗑️ 선택 종목 삭제", key="btn_rm_stock"):
                    remove_watch("stock", code)
                    st.rerun()

            st.download_button(
                "⬇️ 관심주식 CSV 다운로드",
                data=sdf.to_csv(index=False).encode("utf-8-sig"),
                file_name="watch_stocks.csv",
                mime="text/csv",
            )

    with col2:
        st.markdown("### 관심ETF")
        wl = st.session_state["watch_etfs"]
        if not wl:
            st.info("관심ETF가 없습니다.")
        else:
            edf = pd.DataFrame([{"code": k, "name": v} for k, v in wl.items()]).sort_values("name")
            st.dataframe(edf, use_container_width=True, height=320)

            rm = st.selectbox("삭제할 ETF 선택", options=["(선택안함)"] + (edf["name"] + " (" + edf["code"] + ")").tolist(), key="rm_etf")
            if rm != "(선택안함)":
                m = re.search(r"\((\d{6})\)$", rm)
                code = m.group(1) if m else None
                if code and st.button("🗑️ 선택 ETF 삭제", key="btn_rm_etf"):
                    remove_watch("etf", code)
                    st.rerun()

            st.download_button(
                "⬇️ 관심ETF CSV 다운로드",
                data=edf.to_csv(index=False).encode("utf-8-sig"),
                file_name="watch_etfs.csv",
                mime="text/csv",
            )


# =========================================================
# 진단 (컬럼 매핑/부동산 감지/10대주 데이터 확인)
# =========================================================
with tab_debug:
    st.subheader("🛠️ 진단")

    st.markdown("### 1) 주식 CSV 매핑")
    st.write("원본 컬럼:", list(tickers_raw.columns))
    st.write("매핑:", tickers_meta)
    st.dataframe(tickers.head(30)[["code", "name", "market"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 2) ETF CSV 매핑")
    st.write("원본 컬럼:", list(etfs_raw.columns))
    st.write("매핑:", etfs_meta)
    st.dataframe(etfs.head(30)[["code", "name"]], use_container_width=True)

    st.markdown("---")
    st.markdown("### 3) 부동산 감지")
    st.write("region_cols:", re_meta.get("region_cols"))
    st.write("type_col:", re_meta.get("type_col"))
    st.write("month_cols count:", len(re_meta.get("month_cols", [])))
    st.write("month_cols sample:", re_meta.get("month_cols", [])[:15])
    st.write("city_level_col:", re_meta.get("city_level_col"))
    st.dataframe(re_long.head(50), use_container_width=True)

    st.markdown("---")
    st.markdown("### 4) 10대 대표주 그래프 데이터 확인")
    tdf = fetch_top10_prices(period="3mo")
    st.write("가져온 컬럼(종목 수):", list(tdf.columns), f" / {len(tdf.columns)}개")
    st.dataframe(tdf.tail(5), use_container_width=True)

    st.markdown("---")
    st.markdown("### 5) 금(한돈) 계산 진단")
    st.write(calc_gold_don_krw())

st.caption("© 재테크 대시보드 v6.2 | ‘최초 대시보드 핵심’(10대 주가 그래프) 고정 + 기능 통합")
