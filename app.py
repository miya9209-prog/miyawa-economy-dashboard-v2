import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import feedparser
import yfinance as yf
import plotly.graph_objects as go
import re

# =========================
# Page / CSS
# =========================
st.set_page_config(page_title="재테크 핵심지표 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

/* ✅ 타이틀 잘림 방지 */
header[data-testid="stHeader"] { height: 0.0rem; }
.block-container { max-width: 1240px; padding-top: 2.0rem; padding-bottom: 2rem; }

h1 {
  font-size: 1.65rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
  line-height: 1.25 !important;
  margin-top: .25rem !important;
  padding-top: .15rem !important;
}

/* ✅ 섹션 제목 간격 */
h2, h3 { letter-spacing: -0.02em; }
.small { color: rgba(0,0,0,.55); font-size: .92rem; }

/* ✅ 카드 */
.card { border:1px solid rgba(0,0,0,.06); border-radius:16px; padding:12px 12px 10px 12px;
        background:#fff; box-shadow:0 8px 24px rgba(0,0,0,.05); }
.ct { color: rgba(0,0,0,.62); font-weight:650; font-size:.92rem; margin-bottom:6px; }
.kpi { font-size: 1.20rem; font-weight: 800; letter-spacing: -0.02em; }
.dpos { color:#0a7b34; font-weight:650; font-size:.90rem; }
.dneg { color:#b42318; font-weight:650; font-size:.90rem; }
.dflat{ color:rgba(0,0,0,.55); font-weight:650; font-size:.90rem; }
.hr { border-top:1px solid rgba(0,0,0,.06); margin:.9rem 0 1.05rem 0; }

/* ✅ 탭 */
.stTabs [data-baseweb="tab-list"]{ gap: 8px; }
.stTabs [data-baseweb="tab"]{
  height: 40px;
  border-radius: 12px;
  padding: 0 14px;
  border: 1px solid rgba(0,0,0,0.06);
}
.stTabs [aria-selected="true"]{
  border: 1px solid rgba(0,0,0,0.12) !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

a { text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def clean_ticker(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"(\d{6})", raw)
    if m:
        return m.group(1)
    return raw.replace(" ", "")

def now_local() -> datetime:
    return datetime.now()

def days_for_freq(freq: str) -> int:
    return {"D": 180, "W": 365 * 3, "M": 365 * 10}.get(freq, 365)

def metric(df_close: pd.DataFrame):
    if df_close is None or df_close.empty:
        return None, None, None
    s = df_close["Close"].dropna()
    if len(s) < 2:
        return float(s.iloc[-1]) if len(s) else None, None, None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    delta = last - prev
    pct = (delta / prev) * 100 if prev != 0 else None
    return last, delta, pct

def delta_cls(delta):
    if delta is None:
        return "dflat"
    if abs(delta) < 1e-12:
        return "dflat"
    return "dpos" if delta > 0 else "dneg"

def kpi_card(title, last, delta, pct, precision=2, suffix=""):
    if last is None:
        v = "-"
        d = ""
        cls = "dflat"
    else:
        v = f"{last:,.{precision}f}{suffix}"
        if pct is None or delta is None:
            d = ""
            cls = "dflat"
        else:
            sign = "+" if delta > 0 else ""
            d = f"{sign}{delta:,.{precision}f} ({pct:+.2f}%)"
            cls = delta_cls(delta)

    st.markdown(f"""
    <div class="card">
      <div class="ct">{title}</div>
      <div class="kpi">{v}</div>
      <div class="{cls}">{d}</div>
    </div>
    """, unsafe_allow_html=True)

def _close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    if isinstance(out.columns, pd.MultiIndex):
        if "Close" in out.columns.get_level_values(0):
            c = out.xs("Close", axis=1, level=0)
            if isinstance(c, pd.DataFrame):
                return c.iloc[:, 0].dropna()
            return c.dropna()

    if "Close" in out.columns:
        c = out["Close"]
        if isinstance(c, pd.DataFrame):
            return c.iloc[:, 0].dropna()
        return c.dropna()

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    if not num_cols:
        return pd.Series(dtype="float64")
    c = out[num_cols[0]]
    if isinstance(c, pd.DataFrame):
        return c.iloc[:, 0].dropna()
    return c.dropna()

def resample_close(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    s = _close_series(df)
    if s.empty:
        return pd.DataFrame()
    if freq == "D":
        return s.to_frame("Close")
    rule = "W-FRI" if freq == "W" else "M"
    return s.resample(rule).last().dropna().to_frame("Close")

# ✅ Plotly: 그래프 위 텍스트 겹침 해결(legend를 그래프 아래로 내림)
def plot_lines(df: pd.DataFrame, title: str, height: int = 320, normalized: bool = False):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return

    d = df.copy().dropna(how="all")
    if d.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return

    if normalized:
        for c in d.columns:
            s = d[c].dropna()
            if len(s):
                d[c] = (d[c] / s.iloc[0]) * 100

    fig = go.Figure()
    for c in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d[c], mode="lines", name=str(c)))

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=64, b=90),  # ✅ 아래 여백 크게
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,        # ✅ 그래프 아래로 내림
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Data (yfinance only)
# =========================
@st.cache_data(ttl=60 * 30)
def yf_close(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)

    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        return pd.DataFrame()

    return df[["Close"]].dropna()

@st.cache_data(ttl=60 * 30)
def yf_close_kr(ticker6: str, start: str) -> pd.DataFrame:
    t = clean_ticker(ticker6)
    if re.fullmatch(r"\d{6}", t):
        for suf in [".KS", ".KQ"]:
            sym = f"{t}{suf}"
            df = yf_close(sym, start)
            if not df.empty:
                return df
        return pd.DataFrame()
    return yf_close(t, start)

@st.cache_data(ttl=60 * 10)
def rss_items(url: str, limit: int = 25):
    d = feedparser.parse(url)
    out = []
    for e in d.entries[:limit]:
        out.append({
            "title": getattr(e, "title", "").strip(),
            "link": getattr(e, "link", "").strip(),
            "published": getattr(e, "published", "") or getattr(e, "updated", "")
        })
    return out

# =========================
# Sidebar
# =========================
st.sidebar.markdown("### ⚙️ 설정")
mobile = st.sidebar.toggle("모바일 보기 최적화", value=True)
refresh_on = st.sidebar.toggle("자동 새로고침", value=False)
refresh_min = st.sidebar.select_slider("갱신 주기(분)", options=[2, 3, 5, 10, 15], value=5)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 60, 25, 5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="").strip()

st.sidebar.markdown("---")
if st.sidebar.button("지금 새로고침"):
    st.cache_data.clear()
    st.rerun()

if refresh_on:
    ms = int(refresh_min * 60 * 1000)
    st.components.v1.html(f"<script>setTimeout(()=>window.location.reload(), {ms});</script>", height=0)

st.sidebar.markdown("---")
st.sidebar.caption("✍️ 10대 기업/ETF 목록은 언제든 수정 가능")

DEFAULT_TOP10_COMP = [
    ("삼성전자", "005930"),
    ("SK하이닉스", "000660"),
    ("LG에너지솔루션", "373220"),
    ("삼성바이오로직스", "207940"),
    ("현대차", "005380"),
    ("삼성전자우", "005935"),
    ("기아", "000270"),
    ("셀트리온", "068270"),
    ("NAVER", "035420"),
    ("KB금융", "105560"),
]

DEFAULT_ETF10 = [
    ("KODEX 200", "069500"),
    ("KODEX 코스닥150", "229200"),
    ("KODEX 레버리지", "122630"),
    ("KODEX 인버스", "114800"),
    ("KODEX 200선물인버스2X", "252670"),
    ("KODEX 2차전지산업", "305720"),
    ("KODEX 반도체", "091160"),
    ("KODEX 은행", "091170"),
    ("KODEX 자동차", "091180"),
    ("KODEX 미국S&P500TR", "379800"),
]

top10_text = st.sidebar.text_area(
    "10대 기업 (형식: 이름,티커 / 한 줄에 하나)",
    value="\n".join([f"{n},{t}" for n, t in DEFAULT_TOP10_COMP]),
    height=170
)
etf10_text = st.sidebar.text_area(
    "대표 ETF 10 (형식: 이름,티커 / 한 줄에 하나)",
    value="\n".join([f"{n},{t}" for n, t in DEFAULT_ETF10]),
    height=170
)

def parse_list(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        name, tick = line.split(",", 1)
        name = name.strip()
        tick = tick.strip()
        if name and tick:
            out.append((name, tick))
    return out

TOP10_COMP = parse_list(top10_text)
ETF10 = parse_list(etf10_text)

KPI_COLS = 2 if mobile else 4
CHART_H = 260 if mobile else 340  # ✅ 차트 높이 약간 늘려 레전드 공간 여유
NEWS_COLS = 1 if mobile else 2

# =========================
# Header
# =========================
st.title("재테크 핵심지표 대시보드")
st.markdown(
    '<div class="small">국내/미국 지수 · 국내 10대 기업 · 대표 ETF 10 · 환율/금/유가 · 실시간 경제뉴스</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Render Sections
# =========================
def render_overview(freq: str, start: str):
    st.subheader("요약 스냅샷")
    cols = st.columns(KPI_COLS)

    kpi_defs = [
        ("KOSPI", "^KS11", "", 2),
        ("KOSDAQ", "^KQ11", "", 2),
        ("S&P 500", "^GSPC", "", 2),
        ("NASDAQ", "^IXIC", "", 2),
        ("USD/KRW", "KRW=X", "", 4),
        ("Gold", "GC=F", "", 2),
        ("WTI", "CL=F", "", 2),
        ("DXY (달러인덱스)", "DX-Y.NYB", "", 2),
    ]

    for i, (title, sym, suffix, prec) in enumerate(kpi_defs):
        with cols[i % KPI_COLS]:
            base = yf_close(sym, start)
            df = resample_close(base, freq)
            last, delta, pct = metric(df)
            kpi_card(title, last, delta, pct, precision=prec, suffix=suffix)

def render_indices(freq: str, start: str):
    st.subheader("주요 주가지수")

    kr = {}
    for name, sym in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11")]:
        d = resample_close(yf_close(sym, start), freq)
        if not d.empty:
            kr[name] = d["Close"]

    us = {}
    for name, sym in [("S&P500", "^GSPC"), ("NASDAQ", "^IXIC"), ("DOW", "^DJI")]:
        d = resample_close(yf_close(sym, start), freq)
        if not d.empty:
            us[name] = d["Close"]

    if kr:
        plot_lines(pd.DataFrame(kr), "KOSPI vs KOSDAQ (정규화=100)", height=CHART_H, normalized=True)
    else:
        st.info("국내 지수 데이터를 가져오지 못했습니다.")

    if us:
        plot_lines(pd.DataFrame(us), "US Indices (정규화=100)", height=CHART_H, normalized=True)
    else:
        st.info("미국 지수 데이터를 가져오지 못했습니다.")

def render_top10_companies(freq: str, start: str):
    st.subheader("국내 10대 기업 (정규화 100 비교)")
    prices = {}
    for name, ticker in TOP10_COMP:
        d = resample_close(yf_close_kr(ticker, start), freq)
        if not d.empty:
            prices[name] = d["Close"]

    if prices:
        plot_lines(pd.DataFrame(prices), "KR Top10 Companies (정규화=100)", height=CHART_H + 40, normalized=True)
    else:
        st.info("기업 주가 데이터를 가져오지 못했습니다. (티커 확인)")

def render_etf10(freq: str, start: str):
    st.subheader("한국 대표 ETF 10 (정규화 100 비교)")
    prices = {}
    for name, ticker in ETF10:
        d = resample_close(yf_close_kr(ticker, start), freq)
        if not d.empty:
            prices[name] = d["Close"]

    if prices:
        plot_lines(pd.DataFrame(prices), "KR ETF10 (정규화=100)", height=CHART_H + 40, normalized=True)
    else:
        st.info("ETF 데이터를 가져오지 못했습니다. (티커 확인)")

def render_fx_gold_oil(freq: str, start: str):
    st.subheader("환율 · 금 · 유가")

    series = [
        ("USD/KRW", "KRW=X", 4),
        ("Gold", "GC=F", 2),
        ("WTI", "CL=F", 2),
    ]

    cols = st.columns(KPI_COLS)
    for i, (title, sym, prec) in enumerate(series):
        with cols[i % KPI_COLS]:
            base = yf_close(sym, start)
            df = resample_close(base, freq)
            last, delta, pct = metric(df)
            kpi_card(title, last, delta, pct, precision=prec)

def render_news():
    st.subheader("실시간 경제 뉴스")

    daum_rss = "http://media.daum.net/rss/part/primary/economic/rss2.xml"
    google_rss = "http://news.google.co.kr/news?pz=1&cf=all&ned=kr&hl=ko&topic=b&output=rss"

    def render_list(items):
        if keyword:
            k = keyword.lower()
            items = [it for it in items if k in (it["title"] or "").lower()]
        if not items:
            st.caption("표시할 뉴스가 없습니다.")
            return
        for it in items[:news_limit]:
            title = it.get("title", "").strip()
            link = it.get("link", "").strip()
            pub = it.get("published", "").strip()
            pub_txt = f" · {pub}" if pub else ""
            st.markdown(f"- [{title}]({link}){pub_txt}")

    cols = st.columns(NEWS_COLS)
    with cols[0]:
        st.markdown("**DAUM (RSS)**")
        try:
            render_list(rss_items(daum_rss, limit=max(news_limit, 35)))
        except Exception as e:
            st.warning(f"다음 RSS 실패: {e}")

    if NEWS_COLS == 2:
        with cols[1]:
            st.markdown("**GOOGLE NEWS (RSS)**")
            try:
                render_list(rss_items(google_rss, limit=max(news_limit, 35)))
            except Exception as e:
                st.warning(f"구글 RSS 실패: {e}")
    else:
        st.markdown("**GOOGLE NEWS (RSS)**")
        try:
            render_list(rss_items(google_rss, limit=max(news_limit, 35)))
        except Exception as e:
            st.warning(f"구글 RSS 실패: {e}")

# =========================
# Main
# =========================
def render_tab(freq: str):
    start_dt = now_local() - timedelta(days=days_for_freq(freq))
    start = start_dt.strftime("%Y-%m-%d")

    render_overview(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_indices(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_top10_companies(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_etf10(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_fx_gold_oil(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_news()
    st.caption("※ 무료 데이터 소스 특성상 간헐적 누락이 있을 수 있어요. 그럴 땐 ‘지금 새로고침’을 눌러주세요.")

tabs = st.tabs(["일간", "주간", "월간"])
with tabs[0]:
    render_tab("D")
with tabs[1]:
    render_tab("W")
with tabs[2]:
    render_tab("M")
