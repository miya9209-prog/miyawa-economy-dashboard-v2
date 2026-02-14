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
.block-container { max-width: 1240px; padding-top: 2.7rem; padding-bottom: 2rem; } /* ✅ 타이틀 잘림 방지 */

h1 { font-size:1.7rem !important; font-weight:800; line-height:1.25 !important; margin-top:.1rem !important; }
.small { color:#666; font-size:.92rem; }

.card{
 border:1px solid rgba(0,0,0,.06);
 border-radius:16px;
 padding:12px;
 background:#fff;
 box-shadow:0 8px 20px rgba(0,0,0,.05)
}
.ct{font-size:.9rem;color:rgba(0,0,0,.62);font-weight:650}
.kpi{font-size:1.2rem;font-weight:800;letter-spacing:-.02em}
.pos{color:#0a7b34;font-weight:650}
.neg{color:#b42318;font-weight:650}

.hr { border-top:1px solid rgba(0,0,0,.06); margin:.9rem 0 1.05rem 0; }

.footer{
  margin-top:42px;
  text-align:center;
  font-size:.85rem;
  color:rgba(0,0,0,.55);
  padding:18px 0 6px 0;
}
.smallbtn .stButton>button{ border-radius:12px; padding:.45rem .7rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# Utils
# =========================
def now():
    return datetime.now()

def is_korean_text(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def clean_ticker_6(s: str) -> str | None:
    m = re.search(r"(\d{6})", (s or "").strip())
    return m.group(1) if m else None

def safe_series(x) -> pd.Series:
    """어떤 형태든 숫자 Series로."""
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            x = x.iloc[:, 0]
        else:
            return pd.Series(dtype="float64")
    try:
        s = pd.to_numeric(pd.Series(x), errors="coerce")
        s = s.dropna()
        return s
    except Exception:
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

def plot_df(df: pd.DataFrame, title: str, height: int = 340):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=str(c)))
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=64, b=90),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, font=dict(size=12))
    )
    st.plotly_chart(fig, use_container_width=True)

def normalize_100(s: pd.Series) -> pd.Series:
    s = safe_series(s)
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

def resample_close(s: pd.Series, freq: str) -> pd.Series:
    """일간 Series를 주간/월간 종가로 리샘플."""
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
    """yfinance Close를 항상 Series로."""
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
            # fallback
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
            if "Close" in flat.columns:
                return safe_series(flat["Close"])
            return pd.Series(dtype="float64")

    if "Close" in df.columns:
        return safe_series(df["Close"])
    return pd.Series(dtype="float64")


# =========================
# Sidebar - Settings
# =========================
st.sidebar.markdown("### ⚙ 설정")
mobile_opt = st.sidebar.toggle("모바일 보기 최적화", value=True)
news_n = st.sidebar.slider("뉴스 표시 개수", 5, 50, 25, step=5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="")

st.sidebar.markdown("---")

# =========================
# Korean Search (KRX listing)
# =========================
@st.cache_data(ttl=60*60*24)
def krx_listings():
    # KRX 종목 + ETF (가능한 범위에서 합치기)
    frames = []
    try:
        frames.append(fdr.StockListing("KRX")[["Code", "Name", "Market"]])
    except Exception:
        pass
    try:
        etf = fdr.StockListing("ETF/KR")
        # 컬럼 통일
        if "Symbol" in etf.columns:
            etf = etf.rename(columns={"Symbol":"Code", "Name":"Name"})
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
    if not q:
        return pd.DataFrame(columns=["Code","Name","Market"])
    if KRX.empty:
        return pd.DataFrame(columns=["Code","Name","Market"])

    # 6자리면 정확 매칭 우선
    t6 = clean_ticker_6(q)
    if t6:
        hit = KRX[KRX["Code"] == t6]
        if not hit.empty:
            return hit.head(limit)

    # 한글/영문 부분검색
    hit = KRX[KRX["Name"].str.contains(q, case=False, na=False)]
    if hit.empty:
        # 영문 ticker 입력일 수도 → Code에는 6자리라 제외
        return hit
    return hit.head(limit)

# =========================
# Watchlist (관심종목)
# =========================
if "watchlist" not in st.session_state:
    # 기본 1개 넣어둔 상태면 유지, 아니면 빈 리스트
    st.session_state.watchlist = st.session_state.watchlist if "watchlist" in st.session_state else []

def add_watch(name, code6):
    code6 = (code6 or "").strip()
    name = (name or code6).strip()
    if not code6 or len(code6) != 6:
        return "bad"
    if len(st.session_state.watchlist) >= 10:
        return "full"
    for it in st.session_state.watchlist:
        if it["code"] == code6:
            return "dup"
    st.session_state.watchlist.append({"name": name, "code": code6})
    return "ok"

def remove_watch(i):
    if 0 <= i < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(i)

st.sidebar.markdown("### ⭐ 관심종목 검색/추가")
with st.sidebar.form("kr_search_form"):
    q = st.text_input("검색(한글 회사명/ETF명/6자리 티커)", value="")
    do = st.form_submit_button("검색")

if do:
    st.session_state.kr_results = search_krx(q, limit=20)

results = st.session_state.get("kr_results", pd.DataFrame())
if isinstance(results, pd.DataFrame) and not results.empty:
    st.sidebar.markdown("**검색 결과(최대 20개)**")
    for idx, row in results.iterrows():
        c = row["Code"]
        nm = row["Name"]
        mk = row.get("Market", "")
        cols = st.sidebar.columns([4, 1])
        with cols[0]:
            st.sidebar.write(f"{nm} ({c})")
            st.sidebar.caption(mk)
        with cols[1]:
            if st.sidebar.button("＋", key=f"add_{c}"):
                res = add_watch(nm, c)
                if res == "ok":
                    st.sidebar.success("추가 완료")
                elif res == "dup":
                    st.sidebar.info("이미 등록됨")
                elif res == "full":
                    st.sidebar.warning("최대 10개")
                st.rerun()
elif do and (isinstance(results, pd.DataFrame) and results.empty):
    st.sidebar.info("검색 결과가 없습니다. (띄어쓰기/정확한 회사명으로 다시 시도해보세요)")

st.sidebar.markdown("---")
st.sidebar.markdown("**현재 관심종목 (최대 10개)**")
for i, it in enumerate(st.session_state.watchlist):
    cols = st.sidebar.columns([4, 1])
    with cols[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['code']}")
    with cols[1]:
        if st.sidebar.button("－", key=f"rm_{it['code']}"):
            remove_watch(i)
            st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("지금 새로고침"):
    st.rerun()


# =========================
# Header + Guide popup
# =========================
l, r = st.columns([5, 1.3])
with l:
    st.title("재테크 핵심지표 대시보드")
    st.markdown('<div class="small">국내/미국 지수 · 국내 Top10 · 대표 ETF10 · 환율/금/유가/DXY · 관심종목 · 실시간 경제뉴스 · 부동산(주간)</div>', unsafe_allow_html=True)
with r:
    if st.button("활용법 가이드", use_container_width=True):
        st.session_state.show_guide = True

if st.session_state.get("show_guide", False):
    if hasattr(st, "dialog"):
        @st.dialog("활용법 가이드")
        def guide():
            st.markdown("""
- **일간/주간/월간 탭**: 단기 변동/추세/사이클을 나눠서 보세요.
- **정규화 100 그래프**: 시작점을 100으로 맞춰 “상대강도”를 비교합니다.
- **USD/KRW**: 원화 약세면 수입물가/외국인 수급에 영향.
- **DXY**: 달러 강세 압력이 큰지 확인.
- **Gold / WTI**: 인플레이션/리스크오프/원자재 사이클 체크.
- **부동산 주간 지수**: 주식보다 후행인 경우가 많아, 흐름 확인용으로 추천.
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        guide()
    else:
        st.info("가이드 팝업이 미지원이라 상단에 표시됩니다.")
        if st.button("가이드 닫기"):
            st.session_state.show_guide = False
            st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================
# Shortcuts (경제지표 바로가기 10개)
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

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================
# News (RSS)
# =========================
@st.cache_data(ttl=60*5)
def fetch_news_rss(limit=25, keyword=""):
    feeds = [
        # Google News RSS (KR)
        "https://news.google.com/rss/search?q=%EA%B2%BD%EC%A0%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%EC%A6%9D%EC%8B%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%ED%99%98%EC%9C%A8+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        # 네이버 경제 RSS(일부)
        "https://rss.hankyung.com/feed/economy.xml",  # 한국경제(예시)
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

    df = pd.DataFrame(items, columns=["title", "link", "published"]).drop_duplicates(subset=["title", "link"])
    if keyword:
        df = df[df["title"].str.contains(keyword, case=False, na=False)]
    return df.head(limit)

st.markdown("### 📰 실시간 경제뉴스")
news_df = fetch_news_rss(limit=news_n, keyword=keyword.strip())
if news_df.empty:
    st.info("뉴스를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
else:
    for _, row in news_df.iterrows():
        t, link, pub = row["title"], row["link"], row["published"]
        st.markdown(f"- [{t}]({link})  \n  <span class='small'>{pub}</span>", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# =========================
# Main Tabs: D / W / M
# =========================
tab_titles = ["일간", "주간", "월간"]
freqs = ["D", "W", "M"]
tabs = st.tabs(tab_titles)

def get_start(freq: str) -> str:
    # 더 넉넉하게 잡아야 월간/주간 리샘플에서 빈값이 안 나옴
    day = {"D": 200, "W": 1100, "M": 4000}.get(freq, 365)
    return (now() - timedelta(days=day)).strftime("%Y-%m-%d")

# 기본 Top10/ETF10 (형준님이 원하시면 나중에 사이드에서 편집 UI도 추가 가능)
KR_TOP10_DEFAULT = [
    ("삼성전자", "005930"),
    ("SK하이닉스", "000660"),
    ("LG에너지솔루션", "373220"),
    ("삼성바이오로직스", "207940"),
    ("현대차", "005380"),
    ("기아", "000270"),
    ("삼성전자우", "005935"),
    ("셀트리온", "068270"),
    ("NAVER", "035420"),
    ("KB금융", "105560"),
]
KR_ETF10_DEFAULT = [
    ("KODEX 200", "069500"),
    ("KODEX 코스닥150", "229200"),
    ("KODEX 레버리지", "122630"),
    ("KODEX 인버스", "114800"),
    ("KODEX 200선물인버스2X", "252670"),
    ("KODEX 반도체", "091160"),
    ("KODEX 은행", "091170"),
    ("KODEX 자동차", "091180"),
    ("KODEX 미국S&P500TR", "379800"),
    ("KODEX 2차전지산업", "305720"),
]

def kr_yf_series(code6: str, start: str) -> pd.Series:
    # KS가 대부분, 안 되면 KQ 시도
    s = yf_close(f"{code6}.KS", start)
    if s.empty:
        s = yf_close(f"{code6}.KQ", start)
    return s

def snapshot_block(freq: str):
    start = get_start(freq)

    # 주요 지표
    kospi = resample_close(yf_close("^KS11", start), freq)
    kosdaq = resample_close(yf_close("^KQ11", start), freq)
    spx = resample_close(yf_close("^GSPC", start), freq)
    nas = resample_close(yf_close("^IXIC", start), freq)
    dow = resample_close(yf_close("^DJI", start), freq)

    usdkrw = resample_close(yf_close("KRW=X", start), freq)  # 원/달러
    dxy = resample_close(yf_close("DX-Y.NYB", start), freq)   # 달러인덱스
    gold = resample_close(yf_close("GC=F", start), freq)      # 금(USD/oz)
    wti = resample_close(yf_close("CL=F", start), freq)       # WTI(USD/배럴)

    # 금 1돈(원/돈) = Gold(USD/oz)*환율*(3.75/31.1035)
    gold_kr_1don = pd.Series(dtype="float64")
    if not gold.empty and not usdkrw.empty:
        aligned = pd.concat([gold, usdkrw], axis=1).dropna()
        if not aligned.empty:
            gold_kr_1don = aligned.iloc[:, 0] * aligned.iloc[:, 1] * (3.75 / 31.1035)

    st.subheader("요약 스냅샷")
    c1 = st.columns(4)
    kpi_card("KOSPI", f"{kospi.iloc[-1]:,.1f}" if not kospi.empty else "-", "pt", kpi_delta(kospi), precision=1)
    kpi_card("KOSDAQ", f"{kosdaq.iloc[-1]:,.1f}" if not kosdaq.empty else "-", "pt", kpi_delta(kosdaq), precision=1)
    kpi_card("S&P 500", f"{spx.iloc[-1]:,.1f}" if not spx.empty else "-", "pt", kpi_delta(spx), precision=1)
    kpi_card("NASDAQ", f"{nas.iloc[-1]:,.1f}" if not nas.empty else "-", "pt", kpi_delta(nas), precision=1)

    c2 = st.columns(4)
    with c2[0]:
        kpi_card("DOW", f"{dow.iloc[-1]:,.1f}" if not dow.empty else "-", "pt", kpi_delta(dow), precision=1)
    with c2[1]:
        kpi_card("USD/KRW", f"{usdkrw.iloc[-1]:,.1f}" if not usdkrw.empty else "-", "원", kpi_delta(usdkrw), precision=1)
    with c2[2]:
        kpi_card("DXY", f"{dxy.iloc[-1]:.2f}" if not dxy.empty else "-", "index", kpi_delta(dxy), precision=2)
    with c2[3]:
        kpi_card("WTI", f"{wti.iloc[-1]:.2f}" if not wti.empty else "-", "USD/배럴", kpi_delta(wti), precision=2)

    c3 = st.columns(4)
    with c3[0]:
        kpi_card("Gold", f"{gold.iloc[-1]:.2f}" if not gold.empty else "-", "USD/oz", kpi_delta(gold), precision=2)
    with c3[1]:
        kpi_card("금 1돈", f"{gold_kr_1don.iloc[-1]:,.0f}" if not gold_kr_1don.empty else "-", "원/돈")
    with c3[2]:
        st.markdown('<div class="card"><div class="ct">단위</div><div style="color:rgba(0,0,0,.55);font-weight:650">pt / 원 / index / USD</div></div>', unsafe_allow_html=True)
    with c3[3]:
        st.markdown('<div class="card"><div class="ct">팁</div><div style="color:rgba(0,0,0,.55);font-weight:650">정규화100으로 상대강도 비교</div></div>', unsafe_allow_html=True)

def indices_charts(freq: str):
    start = get_start(freq)
    kos = normalize_100(resample_close(yf_close("^KS11", start), freq))
    koq = normalize_100(resample_close(yf_close("^KQ11", start), freq))
    df1 = pd.DataFrame({"KOSPI": kos, "KOSDAQ": koq}).dropna(how="all")
    plot_df(df1, "주요 주가지수: KOSPI vs KOSDAQ (정규화 100)", height=330)

    spx = normalize_100(resample_close(yf_close("^GSPC", start), freq))
    nas = normalize_100(resample_close(yf_close("^IXIC", start), freq))
    dow = normalize_100(resample_close(yf_close("^DJI", start), freq))
    df2 = pd.DataFrame({"S&P500": spx, "NASDAQ": nas, "DOW": dow}).dropna(how="all")
    plot_df(df2, "미국 증시 지수 (정규화 100)", height=330)

def top10_charts(freq: str):
    start = get_start(freq)

    # 국내 Top10
    series = {}
    for nm, code in KR_TOP10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty:
            series[nm] = normalize_100(s)
    df = pd.DataFrame(series).dropna(how="all")
    plot_df(df, "국내 10대 기업 (정규화 100 비교)", height=360)

    # 대표 ETF10
    series2 = {}
    for nm, code in KR_ETF10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty:
            series2[nm] = normalize_100(s)
    df2 = pd.DataFrame(series2).dropna(how="all")
    plot_df(df2, "한국 대표 ETF 10 (정규화 100 비교)", height=360)

def watchlist_chart(freq: str):
    start = get_start(freq)
    st.subheader("내 관심종목 (정규화 100)")
    if not st.session_state.watchlist:
        st.info("좌측에서 관심종목을 검색/추가해주세요. (한글 회사명 검색 지원)")
        return
    series = {}
    for it in st.session_state.watchlist[:10]:
        code = it["code"]
        nm = it["name"]
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty:
            series[nm] = normalize_100(s)
    df = pd.DataFrame(series).dropna(how="all")
    plot_df(df, "관심종목 (정규화 100)", height=360)

# =========================
# KOSIS Real Estate (부분 실패 처리)
# =========================
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
            "objL2": "",
            "objL3": "",
            "objL4": "",
            "objL5": "",
            "objL6": "",
            "objL7": "",
            "objL8": "",
            "format": "json",
            "jsonVD": "Y",
            "prdSe": "W",
            "loadGubun": "2",
            "newEstPrdCnt": str(weeks),
        }
        r = requests.get(base_url, params=params, timeout=20)

        # ✅ KOSIS가 에러를 dict로 주는 경우가 많음 → pd.DataFrame() 만들면 “scalar index” 에러
        try:
            data = r.json()
        except Exception:
            out[name] = pd.Series(dtype="float64")
            continue

        # list가 아니면(=에러 dict 가능성) 스킵
        if not isinstance(data, list):
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
        s = s.dropna().sort_index()
        out[name] = s

    return pd.DataFrame(out).dropna(how="all")

def realestate_block():
    st.subheader("🏠 주간 아파트 매매가격지수 (서울 · 부산 · 대구 · 경기)")
    st.caption("출처: KOSIS OpenAPI (선택1)")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    if not api_key:
        st.info("KOSIS_API_KEY가 설정되지 않아 부동산 지표는 표시되지 않습니다. (나머지 지표는 정상)")
        return

    df = kosis_weekly_apt_sale_index(api_key=api_key)
    if df.empty:
        st.warning("부동산 데이터 호출은 실패했지만, 대시보드 나머지 기능은 정상입니다.")
        st.info("KOSIS URL보기에서 tblId/itmId가 맞는지 확인 후 Secrets에 KOSIS_TBL_ID / KOSIS_ITM_ID를 넣어주세요.")
        return

    norm = df.apply(normalize_100)
    plot_df(norm, "주간 아파트 매매가격지수 (정규화 100)", height=360)
    with st.expander("원자료(최근 30개)"):
        st.dataframe(df.tail(30), use_container_width=True)


# =========================
# Render Tabs
# =========================
for i, tab in enumerate(tabs):
    with tab:
        freq = freqs[i]

        snapshot_block(freq)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown("### 📊 주요 지수")
        indices_charts(freq)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        st.markdown("### 🏢 국내 Top10 / ETF10")
        top10_charts(freq)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        watchlist_chart(freq)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        realestate_block()

# =========================
# Footer (최하단 중앙)
# =========================
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
