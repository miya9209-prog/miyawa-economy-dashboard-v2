import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import requests
import feedparser
import yfinance as yf
import plotly.graph_objects as go
import FinanceDataReader as fdr

# =========================
# Page / CSS
# =========================
st.set_page_config(page_title="재테크 핵심지표 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

header[data-testid="stHeader"] { height: 0rem; }
.block-container { max-width: 1240px; padding-top: 2.8rem; padding-bottom: 2rem; } /* 타이틀 잘림 방지 */

h1 { font-size:1.75rem !important; font-weight:800; line-height:1.25 !important; margin:0 !important; }
.small { color:#666; font-size:.92rem; }

.section-title{ font-size:1.25rem; font-weight:800; margin:.1rem 0 .6rem 0; }
.hr { border-top:1px solid rgba(0,0,0,.06); margin:1.0rem 0 1.05rem 0; }

.card{
 border:1px solid rgba(0,0,0,.06);
 border-radius:16px;
 padding:12px 14px;
 background:#fff;
 box-shadow:0 8px 20px rgba(0,0,0,.05)
}
.ct{font-size:.88rem;color:rgba(0,0,0,.62);font-weight:650}
.kpi{font-size:1.2rem;font-weight:800;letter-spacing:-.02em;margin-top:.15rem}
.pos{color:#0a7b34;font-weight:700}
.neg{color:#b42318;font-weight:700}

.footer{
  margin-top:42px;
  text-align:center;
  font-size:.85rem;
  color:rgba(0,0,0,.55);
  padding:18px 0 6px 0;
}

.smallbtn .stButton>button{ border-radius:14px; padding:.55rem .75rem; }
</style>
""", unsafe_allow_html=True)

# =========================
# Utils
# =========================
def now(): return datetime.now()

def is_korean_text(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def clean_ticker_6(s: str) -> str | None:
    m = re.search(r"(\d{6})", (s or "").strip())
    return m.group(1) if m else None

def safe_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            x = x.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")
    try:
        s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
        return s
    except Exception:
        return pd.Series(dtype="float64")

def normalize_100(s: pd.Series) -> pd.Series:
    s = safe_series(s)
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

def resample_close(s: pd.Series, freq: str) -> pd.Series:
    s = safe_series(s)
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    if freq == "D":
        return s
    if freq == "W":
        return s.resample("W-FRI").last().dropna()
    if freq == "M":
        return s.resample("M").last().dropna()
    return s

@st.cache_data(ttl=60*10)
def yf_close(symbol: str, start: str) -> pd.Series:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # MultiIndex 대응
    if isinstance(df.columns, pd.MultiIndex):
        try:
            c = df["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            return safe_series(c)
        except Exception:
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
            if "Close" in flat.columns:
                return safe_series(flat["Close"])
            return pd.Series(dtype="float64")

    if "Close" in df.columns:
        return safe_series(df["Close"])
    return pd.Series(dtype="float64")

def kpi_delta(s: pd.Series):
    s = safe_series(s)
    if s.empty or len(s) < 2:
        return None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    d = last - prev
    p = (d / prev * 100) if prev != 0 else None
    return last, d, p

def kpi_card(title, value, unit, delta_tuple=None, precision=2):
    delta_html = ""
    if delta_tuple is not None and delta_tuple[2] is not None:
        _, d, p = delta_tuple
        cls = "pos" if d > 0 else "neg" if d < 0 else ""
        delta_html = f'<div class="{cls}">{d:+.{precision}f} ({p:+.2f}%)</div>'
    st.markdown(f"""
    <div class="card">
      <div class="ct">{title}</div>
      <div class="kpi">{value} <span style="font-size:.9rem; font-weight:650; color:rgba(0,0,0,.55)">{unit}</span></div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)

def plot_df(df: pd.DataFrame, title: str, height: int = 360):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=64, b=92),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(size=12))
    )
    st.plotly_chart(fig, use_container_width=True)

def get_start(freq: str) -> str:
    day = {"D": 220, "W": 1200, "M": 4200}.get(freq, 365)
    return (now() - timedelta(days=day)).strftime("%Y-%m-%d")

# =========================
# Sidebar (간단/정돈)
# =========================
st.sidebar.markdown("### ⚙ 설정")
freq_label = st.sidebar.radio("보기 단위", ["일간", "주간", "월간"], index=0)
freq = {"일간":"D", "주간":"W", "월간":"M"}[freq_label]

news_n = st.sidebar.slider("뉴스 표시 개수", 5, 50, 20, step=5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ 관심종목 (한글 검색 지원)")

# =========================
# KRX Listing for Korean Search
# =========================
@st.cache_data(ttl=60*60*24)
def krx_listings():
    frames = []
    try:
        frames.append(fdr.StockListing("KRX")[["Code", "Name", "Market"]])
    except Exception:
        pass
    try:
        etf = fdr.StockListing("ETF/KR")
        if "Symbol" in etf.columns:
            etf = etf.rename(columns={"Symbol":"Code"})
        if "Code" in etf.columns and "Name" in etf.columns:
            etf["Market"] = "ETF"
            frames.append(etf[["Code","Name","Market"]])
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=["Code","Name","Market"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df

KRX = krx_listings()

def search_krx(query: str, limit=20) -> pd.DataFrame:
    q = (query or "").strip()
    if not q or KRX.empty:
        return pd.DataFrame(columns=["Code","Name","Market"])
    t6 = clean_ticker_6(q)
    if t6:
        hit = KRX[KRX["Code"] == t6]
        return hit.head(limit)
    hit = KRX[KRX["Name"].str.contains(q, case=False, na=False)]
    return hit.head(limit)

# =========================
# Watchlist state
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

def add_watch(name, code6):
    code6 = (code6 or "").strip()
    name = (name or code6).strip()
    if not code6 or len(code6) != 6:
        return "bad"
    if len(st.session_state.watchlist) >= 10:
        return "full"
    if any(it["code"] == code6 for it in st.session_state.watchlist):
        return "dup"
    st.session_state.watchlist.append({"name": name, "code": code6})
    return "ok"

def remove_watch(i):
    if 0 <= i < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(i)

with st.sidebar.form("kr_search_form"):
    q = st.text_input("회사명/ETF명/6자리 티커", value="")
    do = st.form_submit_button("검색")

if do:
    st.session_state.kr_results = search_krx(q, limit=20)

results = st.session_state.get("kr_results", pd.DataFrame())
if isinstance(results, pd.DataFrame) and not results.empty:
    st.sidebar.markdown("**검색 결과**")
    for _, row in results.iterrows():
        code = row["Code"]; nm = row["Name"]; mk = row.get("Market","")
        cols = st.sidebar.columns([4,1])
        with cols[0]:
            st.sidebar.write(f"{nm} ({code})")
            st.sidebar.caption(mk)
        with cols[1]:
            if st.sidebar.button("＋", key=f"add_{code}"):
                res = add_watch(nm, code)
                if res == "ok": st.sidebar.success("추가 완료")
                elif res == "dup": st.sidebar.info("이미 등록됨")
                elif res == "full": st.sidebar.warning("최대 10개")
                st.rerun()
elif do:
    st.sidebar.info("검색 결과가 없습니다. 정확한 회사명으로 다시 시도해보세요.")

st.sidebar.markdown("---")
st.sidebar.markdown("**현재 관심종목**")
for i, it in enumerate(st.session_state.watchlist):
    cols = st.sidebar.columns([4,1])
    with cols[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['code']}")
    with cols[1]:
        if st.sidebar.button("－", key=f"rm_{it['code']}"):
            remove_watch(i); st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("지금 새로고침"):
    st.rerun()

# =========================
# Header + Guide
# =========================
l, r = st.columns([5, 1.4])
with l:
    st.title("재테크 핵심지표 대시보드")
    st.markdown('<div class="small">시장 한눈에 → 국내 Top10/ETF10 → 내 관심종목 → 부동산(주간) → 경제뉴스</div>', unsafe_allow_html=True)
with r:
    if st.button("활용법 가이드", use_container_width=True):
        st.session_state.show_guide = True

if st.session_state.get("show_guide", False):
    if hasattr(st, "dialog"):
        @st.dialog("활용법 가이드")
        def guide():
            st.markdown("""
- **보기 단위(좌측)**: 일간(변동) / 주간(추세) / 월간(사이클)  
- **정규화 100**: 시작점을 100으로 맞춰 “상대 강도” 비교  
- **USD/KRW, DXY**: 달러 강세/원화 약세 압력 체크  
- **Gold, WTI**: 인플레/원자재 사이클 확인  
- **부동산(주간)**: 주식 대비 후행일 수 있어, 흐름 확인용으로 활용  
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        guide()
    else:
        st.info("가이드 팝업 미지원 환경이라 상단에 표시됩니다.")
        if st.button("가이드 닫기"):
            st.session_state.show_guide = False
            st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Data fetch wrappers
# =========================
def kr_yf_series(code6: str, start: str) -> pd.Series:
    s = yf_close(f"{code6}.KS", start)
    if s.empty:
        s = yf_close(f"{code6}.KQ", start)
    return s

KR_TOP10_DEFAULT = [
    ("삼성전자", "005930"), ("SK하이닉스", "000660"), ("LG에너지솔루션", "373220"),
    ("삼성바이오로직스", "207940"), ("현대차", "005380"), ("기아", "000270"),
    ("삼성전자우", "005935"), ("셀트리온", "068270"), ("NAVER", "035420"), ("KB금융", "105560"),
]
KR_ETF10_DEFAULT = [
    ("KODEX 200", "069500"), ("KODEX 코스닥150", "229200"), ("KODEX 레버리지", "122630"),
    ("KODEX 인버스", "114800"), ("KODEX 200선물인버스2X", "252670"), ("KODEX 반도체", "091160"),
    ("KODEX 은행", "091170"), ("KODEX 자동차", "091180"), ("KODEX 미국S&P500TR", "379800"),
    ("KODEX 2차전지산업", "305720"),
]

# =========================
# Tabs (정돈된 흐름)
# =========================
tabs = st.tabs(["시장 한눈에", "국내 Top10/ETF10", "내 관심종목", "부동산(주간)"])

# --------- TAB 1: Market Overview
with tabs[0]:
    start = get_start(freq)

    kospi = resample_close(yf_close("^KS11", start), freq)
    kosdaq = resample_close(yf_close("^KQ11", start), freq)
    spx = resample_close(yf_close("^GSPC", start), freq)
    nas = resample_close(yf_close("^IXIC", start), freq)
    dow = resample_close(yf_close("^DJI", start), freq)

    usdkrw = resample_close(yf_close("KRW=X", start), freq)
    dxy = resample_close(yf_close("DX-Y.NYB", start), freq)
    gold = resample_close(yf_close("GC=F", start), freq)   # USD/oz
    wti = resample_close(yf_close("CL=F", start), freq)    # USD/barrel

    gold_kr_1don = pd.Series(dtype="float64")
    if not gold.empty and not usdkrw.empty:
        aligned = pd.concat([gold, usdkrw], axis=1).dropna()
        if not aligned.empty:
            gold_kr_1don = aligned.iloc[:, 0] * aligned.iloc[:, 1] * (3.75 / 31.1035)

    st.markdown('<div class="section-title">요약 스냅샷</div>', unsafe_allow_html=True)
    # ✅ 3열 카드 배치(보기 좋게)
    row1 = st.columns(3)
    with row1[0]: kpi_card("KOSPI", f"{kospi.iloc[-1]:,.1f}" if not kospi.empty else "-", "pt", kpi_delta(kospi), precision=1)
    with row1[1]: kpi_card("KOSDAQ", f"{kosdaq.iloc[-1]:,.1f}" if not kosdaq.empty else "-", "pt", kpi_delta(kosdaq), precision=1)
    with row1[2]: kpi_card("USD/KRW", f"{usdkrw.iloc[-1]:,.1f}" if not usdkrw.empty else "-", "원", kpi_delta(usdkrw), precision=1)

    row2 = st.columns(3)
    with row2[0]: kpi_card("S&P 500", f"{spx.iloc[-1]:,.1f}" if not spx.empty else "-", "pt", kpi_delta(spx), precision=1)
    with row2[1]: kpi_card("NASDAQ", f"{nas.iloc[-1]:,.1f}" if not nas.empty else "-", "pt", kpi_delta(nas), precision=1)
    with row2[2]: kpi_card("DOW", f"{dow.iloc[-1]:,.1f}" if not dow.empty else "-", "pt", kpi_delta(dow), precision=1)

    row3 = st.columns(3)
    with row3[0]: kpi_card("DXY", f"{dxy.iloc[-1]:.2f}" if not dxy.empty else "-", "index", kpi_delta(dxy), precision=2)
    with row3[1]: kpi_card("WTI", f"{wti.iloc[-1]:.2f}" if not wti.empty else "-", "USD/배럴", kpi_delta(wti), precision=2)
    with row3[2]: kpi_card("금 1돈", f"{gold_kr_1don.iloc[-1]:,.0f}" if not gold_kr_1don.empty else "-", "원/돈")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">주요 지수 추세 (정규화 100)</div>', unsafe_allow_html=True)
    df_k = pd.DataFrame({
        "KOSPI": normalize_100(kospi),
        "KOSDAQ": normalize_100(kosdaq)
    }).dropna(how="all")
    plot_df(df_k, "KOSPI vs KOSDAQ (정규화 100)", height=340)

    df_u = pd.DataFrame({
        "S&P500": normalize_100(spx),
        "NASDAQ": normalize_100(nas),
        "DOW": normalize_100(dow)
    }).dropna(how="all")
    plot_df(df_u, "US Indices (정규화 100)", height=340)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">환율/원자재 (정규화 100)</div>', unsafe_allow_html=True)
    df_fx = pd.DataFrame({
        "USD/KRW": normalize_100(usdkrw),
        "DXY": normalize_100(dxy),
        "Gold(USD/oz)": normalize_100(gold),
        "WTI": normalize_100(wti),
    }).dropna(how="all")
    plot_df(df_fx, "FX & Commodities (정규화 100)", height=360)

# --------- TAB 2: Top10 / ETF10
with tabs[1]:
    start = get_start(freq)
    st.markdown('<div class="section-title">국내 10대 기업 (정규화 100)</div>', unsafe_allow_html=True)
    series = {}
    for nm, code in KR_TOP10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty:
            series[nm] = normalize_100(s)
    plot_df(pd.DataFrame(series).dropna(how="all"), "KR Top10 (정규화 100)", height=380)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">한국 대표 ETF 10 (정규화 100)</div>', unsafe_allow_html=True)
    series2 = {}
    for nm, code in KR_ETF10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty:
            series2[nm] = normalize_100(s)
    plot_df(pd.DataFrame(series2).dropna(how="all"), "KR ETF10 (정규화 100)", height=380)

# --------- TAB 3: Watchlist
with tabs[2]:
    start = get_start(freq)
    st.markdown('<div class="section-title">내 관심종목 (정규화 100)</div>', unsafe_allow_html=True)
    if not st.session_state.watchlist:
        st.info("좌측에서 관심종목을 검색/추가해주세요. (한글 회사명 검색 지원)")
    else:
        series = {}
        for it in st.session_state.watchlist[:10]:
            s = resample_close(kr_yf_series(it["code"], start), freq)
            if not s.empty:
                series[it["name"]] = normalize_100(s)
        plot_df(pd.DataFrame(series).dropna(how="all"), "내 관심종목 (정규화 100)", height=420)

# --------- TAB 4: Real Estate (KOSIS)
KOSIS_ORG_ID_DEFAULT = st.secrets.get("KOSIS_ORG_ID", "408")
KOSIS_TBL_ID_DEFAULT = st.secrets.get("KOSIS_TBL_ID", "DT_304004_WEEK_002_A")
KOSIS_ITM_ID_DEFAULT = st.secrets.get("KOSIS_ITM_ID", "T1")
REGION_CODES_DEFAULT = {"서울": "11", "부산": "21", "대구": "22", "경기": "31"}

@st.cache_data(ttl=60*60*6)
def kosis_weekly_apt_sale_index(api_key: str, weeks: int = 260):
    base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    out = {}
    for name, code in REGION_CODES_DEFAULT.items():
        params = {
            "method": "getList",
            "apiKey": api_key,
            "orgId": KOSIS_ORG_ID_DEFAULT,
            "tblId": KOSIS_TBL_ID_DEFAULT,
            "itmId": KOSIS_ITM_ID_DEFAULT,
            "objL1": code,
            "format": "json",
            "jsonVD": "Y",
            "prdSe": "W",
            "loadGubun": "2",
            "newEstPrdCnt": str(weeks),
        }
        r = requests.get(base_url, params=params, timeout=20)
        try:
            data = r.json()
        except Exception:
            out[name] = pd.Series(dtype="float64")
            continue
        if not isinstance(data, list):  # 에러 dict 방지
            out[name] = pd.Series(dtype="float64")
            continue
        df = pd.DataFrame(data)
        if df.empty or "PRD_DE" not in df.columns:
            out[name] = pd.Series(dtype="float64")
            continue
        val_col = "DT" if "DT" in df.columns else ("DATA_VALUE" if "DATA_VALUE" in df.columns else None)
        if val_col is None:
            out[name] = pd.Series(dtype="float64")
            continue
        idx = pd.to_datetime(df["PRD_DE"].astype(str), errors="coerce", format="%Y%m%d")
        s = pd.to_numeric(df[val_col], errors="coerce")
        s.index = idx
        out[name] = s.dropna().sort_index()
    return pd.DataFrame(out).dropna(how="all")

with tabs[3]:
    st.markdown('<div class="section-title">주간 아파트 매매가격지수 (서울 · 부산 · 대구 · 경기)</div>', unsafe_allow_html=True)
    st.caption("출처: KOSIS OpenAPI (선택1)")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    if not api_key:
        st.info("KOSIS_API_KEY가 설정되지 않아 부동산 지표는 표시되지 않습니다. (다른 탭은 정상)")
    else:
        df = kosis_weekly_apt_sale_index(api_key=api_key)
        if df.empty:
            st.warning("부동산 데이터 호출은 실패했습니다. (다른 탭/기능은 정상)")
            st.info("KOSIS URL보기에서 tblId/itmId를 확인 후 Secrets에 KOSIS_TBL_ID / KOSIS_ITM_ID를 넣어주세요.")
        else:
            norm = df.apply(normalize_100)
            plot_df(norm, "주간 아파트 매매가격지수 (정규화 100)", height=420)
            with st.expander("원자료(최근 30개)"):
                st.dataframe(df.tail(30), use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Bottom: Shortcuts + News (요청대로 하단 배치)
# =========================
st.markdown("### 🔎 국내 주요경제지표 바로가기")
btn_cols = st.columns(5)
links = [
    ("한국은행 ECOS", "https://ecos.bok.or.kr/"),
    ("KOSIS", "https://kosis.kr/"),
    ("통계청", "https://kostat.go.kr/"),
    ("금융통계정보시스템", "https://fisis.fss.or.kr/"),
    ("한국부동산원 R-ONE", "https://www.reb.or.kr/r-one/"),
    ("국토교통부 통계누리", "https://stat.molit.go.kr/"),
    ("관세청 수출입통계", "https://unipass.customs.go.kr/ets/"),
    ("무역협회(KITA) 통계", "https://stat.kita.net/"),
    ("FRED(미국지표)", "https://fred.stlouisfed.org/"),
    ("Investing(참고)", "https://www.investing.com/"),
]
for i, (nm, url) in enumerate(links):
    with btn_cols[i % 5]:
        st.link_button(nm, url)

@st.cache_data(ttl=60*5)
def fetch_news_rss(limit=20, keyword=""):
    feeds = [
        "https://news.google.com/rss/search?q=%EA%B2%BD%EC%A0%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%EC%A6%9D%EC%8B%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%ED%99%98%EC%9C%A8+when:1d&hl=ko&gl=KR&ceid=KR:ko",
    ]
    items = []
    for u in feeds:
        try:
            d = feedparser.parse(u)
            for e in d.entries:
                title = getattr(e, "title", "")
                link = getattr(e, "link", "")
                published = getattr(e, "published", "")
                items.append((title, link, published))
        except Exception:
            continue
    df = pd.DataFrame(items, columns=["title", "link", "published"]).drop_duplicates(subset=["title","link"])
    if keyword:
        df = df[df["title"].str.contains(keyword, case=False, na=False)]
    return df.head(limit)

st.markdown("### 📰 실시간 경제뉴스")
news_df = fetch_news_rss(limit=news_n, keyword=keyword.strip())
if news_df.empty:
    st.info("뉴스를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
else:
    for _, row in news_df.iterrows():
        st.markdown(f"- [{row['title']}]({row['link']})  \n  <span class='small'>{row['published']}</span>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
