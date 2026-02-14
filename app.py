import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
import requests
import feedparser
import yfinance as yf
import plotly.graph_objects as go

# FinanceDataReader는 "있으면 쓰고, 없으면/막히면 우회"로 처리
try:
    import FinanceDataReader as fdr
    HAS_FDR = True
except Exception:
    HAS_FDR = False


# =========================================================
# Page / CSS
# =========================================================
st.set_page_config(
    page_title="재테크 핵심지표 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
header[data-testid="stHeader"] { height: 0rem; }

.block-container {
  max-width: 1240px;
  padding-top: 3.2rem;
  padding-bottom: 2rem;
}
h1 {
  font-size: 1.8rem !important;
  font-weight: 800;
  line-height: 1.35 !important;
  margin: 0 !important;
}
.small { color:#666; font-size:.92rem; }

.section-title { font-size:1.25rem; font-weight:800; margin:.1rem 0 .6rem 0; }
.hr { border-top:1px solid rgba(0,0,0,.06); margin:1.0rem 0 1.05rem 0; }

.card {
  border:1px solid rgba(0,0,0,.06);
  border-radius:16px;
  padding:12px 14px;
  background:#fff;
  box-shadow:0 8px 20px rgba(0,0,0,.05)
}
.ct { font-size:.88rem; color:rgba(0,0,0,.62); font-weight:650; }
.kpi { font-size:1.2rem; font-weight:800; letter-spacing:-.02em; margin-top:.15rem; }
.pos { color:#0a7b34; font-weight:700; }
.neg { color:#b42318; font-weight:700; }

.footer {
  margin-top: 42px;
  text-align: center;
  font-size: .85rem;
  color: rgba(0,0,0,.55);
  padding: 18px 0 6px 0;
}

@media (max-width: 768px) {
  .block-container { padding-top: 2.4rem; padding-left: 1rem; padding-right: 1rem; }
  h1 { font-size: 1.45rem !important; }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# Utils
# =========================================================
def now():
    return datetime.now()

def safe_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            x = x.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")
    if isinstance(x, (int, float, np.number)):
        return pd.Series([float(x)])
    try:
        s = pd.Series(x)
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    except Exception:
        return pd.Series(dtype="float64")

def last_value(x):
    try:
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            if x.empty:
                return None
            x = x.iloc[:, 0]
        if isinstance(x, pd.Series):
            if x.empty:
                return None
            return float(pd.to_numeric(x, errors="coerce").dropna().iloc[-1])
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = safe_series(x)
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None

def normalize_100(s: pd.Series) -> pd.Series:
    s = safe_series(s)
    if s.empty:
        return s
    base = s.iloc[0]
    if base == 0:
        return s
    return (s / base) * 100

def resample_close(s: pd.Series, freq: str) -> pd.Series:
    s = safe_series(s)
    if s.empty:
        return s
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
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

    if isinstance(df.columns, pd.MultiIndex):
        try:
            c = df["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            return safe_series(c)
        except Exception:
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
            return safe_series(flat["Close"]) if "Close" in flat.columns else pd.Series(dtype="float64")

    return safe_series(df["Close"]) if "Close" in df.columns else pd.Series(dtype="float64")

def kpi_delta(s: pd.Series):
    s = safe_series(s)
    if s.empty or len(s) < 2:
        return None
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
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

def plot_df(df: pd.DataFrame, title: str, height: int = 380):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=70, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0, font=dict(size=12)),
    )
    st.plotly_chart(fig, use_container_width=True)

def get_start(freq: str) -> str:
    day = {"D": 220, "W": 1200, "M": 4200}.get(freq, 365)
    return (now() - timedelta(days=day)).strftime("%Y-%m-%d")

def clean_ticker_6(s: str) -> str | None:
    m = re.search(r"(\d{6})", (s or "").strip())
    return m.group(1) if m else None

def is_hangul(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))


# =========================================================
# Sidebar (설정)
# =========================================================
st.sidebar.markdown("### ⚙ 설정")
freq_label = st.sidebar.radio("보기 단위", ["일간", "주간", "월간"], index=0)
freq = {"일간": "D", "주간": "W", "월간": "M"}[freq_label]

news_n = st.sidebar.slider("뉴스 표시 개수", 5, 50, 20, step=5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="")


# =========================================================
# ✅ 한글 종목검색: 네이버 금융 검색(HTML) 우회 (FDR 불안정 대비)
# =========================================================
NAVER_SEARCH_URL = "https://finance.naver.com/search/searchList.naver"

def naver_search(query: str, limit=15):
    """네이버 금융 검색: (이름, 6자리코드, 타입추정) 리스트 반환"""
    q = (query or "").strip()
    if not q:
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    }
    try:
        r = requests.get(NAVER_SEARCH_URL, params={"query": q}, headers=headers, timeout=(6, 12))
        if r.status_code != 200:
            return []
        html = r.text
    except Exception:
        return []

    # 코드 패턴: /item/main.naver?code=005930
    items = []
    for m in re.finditer(r"/item/main\.naver\?code=(\d{6})", html):
        code = m.group(1)
        items.append(code)

    # 중복 제거
    codes = []
    seen = set()
    for c in items:
        if c not in seen:
            seen.add(c)
            codes.append(c)

    # 이름 추출(간단 파싱): 코드 주변 링크 텍스트를 찾는다
    results = []
    for code in codes[:limit*2]:
        # <a href="/item/main.naver?code=005930">삼성전자</a> 형태를 찾음
        mm = re.search(rf'/item/main\.naver\?code={code}[^>]*>\s*([^<]+)\s*<', html)
        name = mm.group(1).strip() if mm else code

        # ETF 추정: 이름에 KODEX/TIGER/KBSTAR/ARIRANG/RIZE 등
        typ = "ETF" if re.search(r"(KODEX|TIGER|KBSTAR|ARIRANG|RISE|HANARO|SOL)", name, re.IGNORECASE) else "STOCK"
        results.append({"Name": name, "Code": code, "Type": typ})

    # 정렬: (정확일치 > 시작 > 포함) + STOCK 우선
    q0 = q
    def score(x):
        nm = x["Name"]
        if nm == q0: ms = 0
        elif nm.startswith(q0): ms = 1
        elif q0 in nm: ms = 2
        else: ms = 9
        ts = 0 if x["Type"] == "STOCK" else 1
        return (ms, ts, nm)

    results = sorted(results, key=score)
    # 상위 limit
    return results[:limit]


# =========================================================
# 관심종목 상태
# =========================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "kr_results" not in st.session_state:
    st.session_state.kr_results = []

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

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ 관심종목 검색/추가")
st.sidebar.caption("한글 회사명도 검색됩니다(네이버 금융). 6자리 티커로 추가하는 게 가장 안정적입니다.")

with st.sidebar.form("kr_search_form"):
    q = st.text_input("회사명/ETF명/6자리 티커", value="")
    do = st.form_submit_button("검색")

if do:
    q = (q or "").strip()
    t6 = clean_ticker_6(q)
    if t6:
        # 6자리면 바로
        st.session_state.kr_results = [{"Name": t6, "Code": t6, "Type": "STOCK"}]
    else:
        # ✅ 한글은 네이버 우선
        if is_hangul(q):
            st.session_state.kr_results = naver_search(q, limit=15)
        else:
            # 영문은 네이버로도 되고, 그냥 네이버로 통일
            st.session_state.kr_results = naver_search(q, limit=15)

results = st.session_state.kr_results
if results:
    st.sidebar.markdown("**검색 결과**")
    for row in results:
        code = row["Code"]
        nm = row["Name"]
        typ = row.get("Type", "")
        cols = st.sidebar.columns([4, 1])
        with cols[0]:
            st.sidebar.write(f"{nm} ({code})")
            st.sidebar.caption(typ)
        with cols[1]:
            if st.sidebar.button("＋", key=f"add_{code}"):
                res = add_watch(nm, code)
                if res == "ok":
                    st.sidebar.success("추가 완료")
                elif res == "dup":
                    st.sidebar.info("이미 등록됨")
                elif res == "full":
                    st.sidebar.warning("최대 10개")
                else:
                    st.sidebar.warning("추가 실패")
                st.rerun()
elif do:
    st.sidebar.info("검색 결과가 없습니다. (회사명 일부만 넣어도 됩니다)")

st.sidebar.markdown("**현재 관심종목**")
for i, it in enumerate(st.session_state.watchlist):
    cols = st.sidebar.columns([4, 1])
    with cols[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['code']}")
    with cols[1]:
        if st.sidebar.button("－", key=f"rm_{it['code']}"):
            remove_watch(i)
            st.rerun()


# =========================================================
# Header + Guide
# =========================================================
l, r = st.columns([5, 1.4])
with l:
    st.title("재테크 핵심지표 대시보드")
    st.markdown('<div class="small">시장 한눈에 → 국내 Top10/ETF10 → 내 관심종목 → 부동산(KOSIS) → 경제뉴스</div>', unsafe_allow_html=True)
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
- **DXY(달러인덱스)**: 달러 강세/약세 흐름  
- **WTI(서부텍사스유)**: 국제 유가(원유) 흐름  
- **부동산(KOSIS)**: 느릴 수 있어 “수동 불러오기” + 실패 시 쿨다운으로 무한로딩 방지  
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        guide()
    else:
        with st.expander("활용법 가이드", expanded=True):
            st.markdown("""
- **보기 단위(좌측)**: 일간(변동) / 주간(추세) / 월간(사이클)  
- **정규화 100**: 시작점을 100으로 맞춰 “상대 강도” 비교  
- **DXY(달러인덱스)**: 달러 강세/약세 흐름  
- **WTI(서부텍사스유)**: 국제 유가(원유) 흐름  
- **부동산(KOSIS)**: 느릴 수 있어 “수동 불러오기” + 실패 시 쿨다운으로 무한로딩 방지  
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================================================
# Tabs
# =========================================================
tabs = st.tabs(["시장 한눈에", "국내 Top10/ETF10", "내 관심종목", "부동산(KOSIS)", "경제뉴스"])


# =========================================================
# Tab 0: 시장 한눈에
# =========================================================
with tabs[0]:
    start = get_start(freq)

    kospi  = resample_close(yf_close("^KS11", start), freq)
    kosdaq = resample_close(yf_close("^KQ11", start), freq)
    spx    = resample_close(yf_close("^GSPC", start), freq)
    nas    = resample_close(yf_close("^IXIC", start), freq)
    dow    = resample_close(yf_close("^DJI", start), freq)

    usdkrw = resample_close(yf_close("KRW=X", start), freq)
    dxy    = resample_close(yf_close("DX-Y.NYB", start), freq)
    gold   = resample_close(yf_close("GC=F", start), freq)
    wti    = resample_close(yf_close("CL=F", start), freq)

    # 금 1돈(원/돈) 근사: 금선물(USD/oz)*환율*(3.75g/oz)
    gold_kr_1don = pd.Series(dtype="float64")
    if not gold.empty and not usdkrw.empty:
        aligned = pd.concat([gold, usdkrw], axis=1).dropna()
        if not aligned.empty:
            gold_kr_1don = aligned.iloc[:, 0] * aligned.iloc[:, 1] * (3.75 / 31.1035)

    st.markdown('<div class="section-title">요약 스냅샷</div>', unsafe_allow_html=True)

    row1 = st.columns(3)
    with row1[0]:
        v = last_value(kospi)
        kpi_card("KOSPI", f"{v:,.1f}" if v is not None else "-", "pt", kpi_delta(kospi), precision=1)
    with row1[1]:
        v = last_value(kosdaq)
        kpi_card("KOSDAQ", f"{v:,.1f}" if v is not None else "-", "pt", kpi_delta(kosdaq), precision=1)
    with row1[2]:
        v = last_value(usdkrw)
        kpi_card("USD/KRW", f"{v:,.1f}" if v is not None else "-", "원", kpi_delta(usdkrw), precision=1)

    row2 = st.columns(3)
    with row2[0]:
        v = last_value(spx)
        kpi_card("S&P 500", f"{v:,.1f}" if v is not None else "-", "pt", kpi_delta(spx), precision=1)
    with row2[1]:
        v = last_value(nas)
        kpi_card("NASDAQ", f"{v:,.1f}" if v is not None else "-", "pt", kpi_delta(nas), precision=1)
    with row2[2]:
        v = last_value(dow)
        kpi_card("DOW", f"{v:,.1f}" if v is not None else "-", "pt", kpi_delta(dow), precision=1)

    row3 = st.columns(3)
    with row3[0]:
        v = last_value(dxy)
        kpi_card("DXY", f"{v:.2f}" if v is not None else "-", "index", kpi_delta(dxy), precision=2)
    with row3[1]:
        v = last_value(wti)
        kpi_card("WTI", f"{v:.2f}" if v is not None else "-", "USD/배럴", kpi_delta(wti), precision=2)
    with row3[2]:
        v = last_value(gold_kr_1don)
        kpi_card("금 1돈", f"{v:,.0f}" if v is not None else "-", "원/돈")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    df_kr = pd.DataFrame({"KOSPI": normalize_100(kospi), "KOSDAQ": normalize_100(kosdaq)}).dropna(how="all")
    df_us = pd.DataFrame({"S&P500": normalize_100(spx), "NASDAQ": normalize_100(nas), "DOW": normalize_100(dow)}).dropna(how="all")
    plot_df(df_kr, "주요 주가지수 (정규화=100)", height=380)
    plot_df(df_us, "미국 증시 (정규화=100)", height=380)


# =========================================================
# Tab 1: 국내 Top10 / ETF10
# =========================================================
with tabs[1]:
    start = get_start(freq)

    KR_TOP10 = [
        ("삼성전자", "005930"),
        ("SK하이닉스", "000660"),
        ("LG에너지솔루션", "373220"),
        ("삼성바이오로직스", "207940"),
        ("현대차", "005380"),
        ("기아", "000270"),
        ("셀트리온", "068270"),
        ("NAVER", "035420"),
        ("KB금융", "105560"),
        ("삼성SDI", "006400"),
    ]

    KR_ETF10 = [
        ("KODEX 200", "069500"),
        ("KODEX 코스닥150", "229200"),
        ("KODEX 레버리지", "122630"),
        ("KODEX 인버스", "114800"),
        ("KODEX 반도체", "091160"),
        ("KODEX 은행", "091170"),
        ("KODEX 자동차", "091180"),
        ("KODEX 미국S&P500TR", "379800"),
        ("KODEX 2차전지산업", "305720"),
        ("KODEX 200선물인버스2X", "252670"),
    ]

    st.markdown('<div class="section-title">국내 10대 기업 (정규화 100 비교)</div>', unsafe_allow_html=True)
    series = {}
    for name, code in KR_TOP10:
        s = yf_close(f"{code}.KS", start)
        if s.empty:
            s = yf_close(f"{code}.KQ", start)
        s = resample_close(s, freq)
        if not s.empty:
            series[name] = normalize_100(s)
    plot_df(pd.DataFrame(series).dropna(how="all"), "KR Top10 Companies (Normalized=100)", height=420)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">한국 대표 ETF 10 (정규화 100 비교)</div>', unsafe_allow_html=True)
    series = {}
    for name, code in KR_ETF10:
        s = yf_close(f"{code}.KS", start)
        if s.empty:
            s = yf_close(f"{code}.KQ", start)
        s = resample_close(s, freq)
        if not s.empty:
            series[name] = normalize_100(s)
    plot_df(pd.DataFrame(series).dropna(how="all"), "KR ETF Top10 (Normalized=100)", height=420)


# =========================================================
# Tab 2: 내 관심종목
# =========================================================
with tabs[2]:
    start = get_start(freq)
    st.markdown('<div class="section-title">내 관심종목 (정규화 100)</div>', unsafe_allow_html=True)
    if not st.session_state.watchlist:
        st.info("좌측에서 관심종목을 검색/추가해주세요. (한글 회사명 검색 지원)")
    else:
        s = {}
        for it in st.session_state.watchlist[:10]:
            ss = yf_close(f"{it['code']}.KS", start)
            if ss.empty:
                ss = yf_close(f"{it['code']}.KQ", start)
            ss = resample_close(ss, freq)
            if not ss.empty:
                s[it["name"]] = normalize_100(ss)
        plot_df(pd.DataFrame(s).dropna(how="all"), "내 관심종목 (정규화 100)", height=420)


# =========================================================
# Tab 3: 부동산(KOSIS) — ✅ 무한 로딩 방지 버전
# =========================================================
KOSIS_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

def kosis_request(params, timeout=10, retries=1):
    """실패하면 즉시 반환(무한로딩 방지). User-Agent 필수."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(KOSIS_URL, params=params, headers=headers, timeout=(5, timeout))
            if r.status_code != 200:
                last_err = {"err": "http", "errMsg": f"HTTP {r.status_code}"}
                continue
            return r.json(), None
        except Exception as e:
            last_err = {"err": "timeout", "errMsg": str(e)}
    return None, last_err

def parse_dt(x):
    x = re.sub(r"[^0-9]", "", str(x))
    if len(x) == 6:
        return pd.to_datetime(x, format="%Y%m", errors="coerce")
    if len(x) == 8:
        return pd.to_datetime(x, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def pick_cols(df):
    date_col = "PRD_DE" if "PRD_DE" in df.columns else ("TIME" if "TIME" in df.columns else None)
    val_col = None
    for c in ["DT", "DATA_VALUE", "VALUE", "VAL"]:
        if c in df.columns:
            val_col = c
            break
    nm_cols = [c for c in df.columns if c.endswith("_NM") or c in ["OBJ_NM", "C1_NM", "C2_NM", "C3_NM"]]
    return date_col, val_col, nm_cols

# 캐시/쿨다운(중요)
if "kosis_cache_df" not in st.session_state:
    st.session_state.kosis_cache_df = None
if "kosis_cache_time" not in st.session_state:
    st.session_state.kosis_cache_time = None
if "kosis_last_fail_time" not in st.session_state:
    st.session_state.kosis_last_fail_time = None
if "kosis_last_fail_msg" not in st.session_state:
    st.session_state.kosis_last_fail_msg = None

with tabs[3]:
    st.markdown('<div class="section-title">부동산 지표 (KOSIS)</div>', unsafe_allow_html=True)
    st.caption("✅ ‘불러오기’ 버튼을 눌렀을 때만 호출합니다. 실패하면 3분 쿨다운으로 재호출을 막습니다.")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    org_id = st.secrets.get("KOSIS_ORG_ID", "408").strip()
    tbl_id = st.secrets.get("KOSIS_TBL_ID", "").strip()
    itm_id = st.secrets.get("KOSIS_ITM_ID", "").strip().replace("+", "").strip()
    prd_se = st.secrets.get("KOSIS_PRD_SE", "M").strip()

    c1, c2, c3 = st.columns([1.1, 2.6, 1.3])
    with c1:
        desired = st.selectbox("가져올 기간(최근 개수)", [3, 6, 12], index=1)
    with c2:
        st.info("표가 크면 제한/타임아웃이 나기 쉬워서, 실패 시 자동으로 기간을 줄여 재시도합니다.", icon="ℹ️")
    with c3:
        fetch = st.button("불러오기/새로고침", use_container_width=True)

    status = st.empty()
    chart_box = st.empty()

    # Secrets 미설정
    if not api_key or not tbl_id or not itm_id:
        status.warning("Secrets에 KOSIS_API_KEY / KOSIS_TBL_ID / KOSIS_ITM_ID 가 필요합니다.")
    else:
        # 쿨다운 체크
        if fetch:
            if st.session_state.kosis_last_fail_time and (now() - st.session_state.kosis_last_fail_time).total_seconds() < 180:
                status.warning("방금 실패해서 잠시(약 3분) 후 다시 시도해주세요. (무한로딩 방지)")
                if st.session_state.kosis_last_fail_msg:
                    st.code(st.session_state.kosis_last_fail_msg)
            else:
                candidates = [desired] + [c for c in [12, 6, 3] if c != desired]
                ok_df = None
                last_msg = None

                with st.spinner("KOSIS 호출 중… (최대 10초 타임아웃)"):
                    for cnt in candidates:
                        params = {
                            "method": "getList",
                            "apiKey": api_key,
                            "orgId": org_id,
                            "tblId": tbl_id,
                            "itmId": itm_id,
                            "objL1": "ALL",
                            "objL2": "ALL",
                            "format": "json",
                            "jsonVD": "Y",
                            "prdSe": prd_se,
                            "newEstPrdCnt": str(cnt),
                            "loadGubun": "2",
                        }
                        data, err = kosis_request(params, timeout=10, retries=1)
                        if err is not None:
                            last_msg = err
                            continue
                        if isinstance(data, dict) and "err" in data:
                            last_msg = data
                            continue
                        if isinstance(data, list) and len(data) > 0:
                            ok_df = pd.DataFrame(data)
                            break

                if ok_df is None or ok_df.empty:
                    st.session_state.kosis_last_fail_time = now()
                    st.session_state.kosis_last_fail_msg = last_msg
                    status.warning("부동산 데이터를 가져오지 못했습니다. (다른 탭/기능은 정상)")
                    st.info(f"확인: orgId={org_id}, tblId={tbl_id}, itmId={itm_id}, prdSe={prd_se}")
                    if last_msg:
                        st.code(last_msg, language="json")
                else:
                    st.session_state.kosis_cache_df = ok_df
                    st.session_state.kosis_cache_time = now()
                    st.session_state.kosis_last_fail_time = None
                    st.session_state.kosis_last_fail_msg = None
                    status.success("호출 성공 ✅ (마지막 성공 데이터를 캐시해 계속 표시합니다.)")

        # ✅ 캐시가 있으면 항상 보여주기
        df0 = st.session_state.kosis_cache_df
        if st.session_state.kosis_cache_time:
            st.caption(f"마지막 성공 업데이트: {st.session_state.kosis_cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if df0 is None:
            status.info("현재 캐시된 부동산 데이터가 없습니다. ‘불러오기/새로고침’을 눌러 주세요.")
        else:
            date_col, val_col, nm_cols = pick_cols(df0)
            if date_col is None or val_col is None:
                chart_box.warning("시계열 변환에 필요한 컬럼을 찾지 못했습니다. (표 구조가 다를 수 있어요)")
                with st.expander("raw 상위 50행"):
                    st.dataframe(df0.head(50), use_container_width=True)
            else:
                region_col = None
                for cand in ["C1_NM", "OBJ_NM", "C2_NM"]:
                    if cand in df0.columns:
                        region_col = cand
                        break
                if region_col is None and nm_cols:
                    region_col = nm_cols[0]

                tmp = pd.DataFrame({
                    "date": df0[date_col].apply(parse_dt),
                    "region": df0[region_col].astype(str) if region_col else "ALL",
                    "value": pd.to_numeric(df0[val_col], errors="coerce")
                }).dropna()

                piv = tmp.pivot_table(index="date", columns="region", values="value", aggfunc="last").sort_index()
                auto = [c for c in piv.columns if any(k in c for k in ["서울", "부산", "대구", "경기"])]
                default = auto[:8] if auto else list(piv.columns)[:8]
                pick = st.multiselect("표시할 지역 선택", options=list(piv.columns), default=default)

                view = piv[pick] if pick else piv
                plot_df(view.apply(normalize_100), f"KOSIS 부동산 지표 (정규화100) · prdSe={prd_se}", height=420)


# =========================================================
# Tab 4: 경제뉴스 + 바로가기 버튼
# =========================================================
with tabs[4]:
    st.markdown('<div class="section-title">📰 실시간 경제뉴스</div>', unsafe_allow_html=True)

    links = [
        ("한국은행", "https://www.bok.or.kr/portal/main/main.do"),
        ("KOSIS", "https://kosis.kr/"),
        ("금융위원회", "https://www.fsc.go.kr/"),
        ("금감원", "https://www.fss.or.kr/"),
        ("KRX", "https://www.krx.co.kr/"),
        ("통계청", "https://kostat.go.kr/"),
        ("기재부", "https://www.moef.go.kr/"),
        ("네이버 금융", "https://finance.naver.com/"),
        ("다음 금융", "https://finance.daum.net/"),
        ("구글 경제뉴스", "https://news.google.com/search?q=%EA%B2%BD%EC%A0%9C&hl=ko&gl=KR&ceid=KR:ko"),
    ]
    cols = st.columns(5)
    for i, (t, u) in enumerate(links):
        with cols[i % 5]:
            st.link_button(t, u, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    @st.cache_data(ttl=60*5)
    def fetch_news_rss(limit=20, keyword=""):
        feeds = [
            "https://news.google.com/rss/search?q=%EA%B2%BD%EC%A0%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=%EC%A6%9D%EC%8B%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=%ED%99%98%EC%9C%A8+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=%EB%B6%80%EB%8F%99%EC%82%B0+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        ]
        items = []
        for u in feeds:
            try:
                d = feedparser.parse(u)
                for e in d.entries:
                    items.append((getattr(e, "title", ""), getattr(e, "link", ""), getattr(e, "published", "")))
            except Exception:
                continue

        df = pd.DataFrame(items, columns=["title", "link", "published"]).drop_duplicates(subset=["title", "link"])
        if keyword:
            df = df[df["title"].str.contains(keyword, case=False, na=False)]
        return df.head(limit)

    news_df = fetch_news_rss(limit=news_n, keyword=keyword.strip())
    if news_df.empty:
        st.info("뉴스를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        for _, row in news_df.iterrows():
            st.markdown(
                f"- [{row['title']}]({row['link']})  \n  <span class='small'>{row['published']}</span>",
                unsafe_allow_html=True
            )


# =========================================================
# Footer
# =========================================================
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
