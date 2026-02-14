import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import yfinance as yf
import FinanceDataReader as fdr
import plotly.graph_objects as go


# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="재테크 핵심지표 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.block-container { max-width: 1240px; padding-top: 1.1rem; padding-bottom: 2rem; }
h1 { font-size: 1.65rem !important; font-weight: 800 !important; letter-spacing: -0.02em; }
.small { color: rgba(0,0,0,.55); font-size: .92rem; }
.card { border:1px solid rgba(0,0,0,.06); border-radius:16px; padding:12px 12px 10px 12px;
        background:#fff; box-shadow:0 8px 24px rgba(0,0,0,.05); }
.ct { color: rgba(0,0,0,.62); font-weight:650; font-size:.92rem; margin-bottom:6px; }
.kpi { font-size: 1.20rem; font-weight: 800; letter-spacing: -0.02em; }
.dpos { color:#0a7b34; font-weight:650; font-size:.90rem; }
.dneg { color:#b42318; font-weight:650; font-size:.90rem; }
.dflat{ color:rgba(0,0,0,.55); font-weight:650; font-size:.90rem; }
.hr { border-top:1px solid rgba(0,0,0,.06); margin:.9rem 0 1.05rem 0; }
</style>
""", unsafe_allow_html=True)


# -------------------------
# Helpers
# -------------------------
def resample_close(df, freq):
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    s = df["Close"].dropna()
    if s.empty:
        return pd.DataFrame()
    if freq == "D":
        return s.to_frame("Close")
    rule = "W-FRI" if freq == "W" else "M"
    return s.resample(rule).last().dropna().to_frame("Close")

def metric(df):
    if df is None or df.empty:
        return None, None, None
    s = df["Close"].dropna()
    if len(s) < 2:
        return float(s.iloc[-1]) if len(s) else None, None, None
    last = float(s.iloc[-1]); prev = float(s.iloc[-2])
    delta = last - prev
    pct = (delta / prev) * 100 if prev != 0 else None
    return last, delta, pct

def delta_cls(delta):
    if delta is None:
        return "dflat"
    if abs(delta) < 1e-12:
        return "dflat"
    return "dpos" if delta > 0 else "dneg"

def kpi_card(title, last, delta, pct, precision=2):
    if last is None:
        v = "-"
        d = ""
        cls = "dflat"
    else:
        v = f"{last:,.{precision}f}"
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

def plot_lines(df, title, height=260, normalized=False):
    if df is None or df.empty:
        st.info(f"{title}: 데이터가 없습니다.")
        return
    d = df.copy().dropna(how="all")
    if normalized:
        for c in d.columns:
            s = d[c].dropna()
            if len(s):
                d[c] = (d[c] / s.iloc[0]) * 100

    fig = go.Figure()
    for c in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d[c], mode="lines", name=str(c)))
    fig.update_layout(title=title, height=height, margin=dict(l=8, r=8, t=42, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=60*30)
def fdr_close(symbol, start):
    df = fdr.DataReader(symbol, start)
    if df is None or df.empty:
        return pd.DataFrame()
    if "Close" not in df.columns:
        # 일부 지표는 컬럼명이 다를 수 있어 첫 numeric 열 사용
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num:
            return pd.DataFrame()
        df = df[[num[0]]].rename(columns={num[0]:"Close"})
    return df[["Close"]]

@st.cache_data(ttl=60*30)
def yf_close(symbol, start):
    df = yf.download(symbol, start=start, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df[["Close"]].dropna()

@st.cache_data(ttl=60*10)
def rss_items(url, limit=25):
    d = feedparser.parse(url)
    out = []
    for e in d.entries[:limit]:
        out.append({
            "title": getattr(e, "title", "").strip(),
            "link": getattr(e, "link", "").strip(),
            "published": getattr(e, "published", "") or getattr(e, "updated", "")
        })
    return out


# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("### ⚙️ 설정")
mobile = st.sidebar.toggle("모바일 보기 최적화", value=True)
news_limit = st.sidebar.slider("뉴스 표시 개수", 10, 60, 25, 5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="").strip()

DEFAULT_TOP10 = [
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

st.sidebar.caption("✍️ 10대 기업/ETF 목록은 언제든 수정 가능")
top10_text = st.sidebar.text_area("10대 기업 (이름,티커)", value="\n".join([f"{n},{t}" for n,t in DEFAULT_TOP10]), height=160)
etf10_text = st.sidebar.text_area("ETF 10 (이름,티커)", value="\n".join([f"{n},{t}" for n,t in DEFAULT_ETF10]), height=160)

def parse_list(text):
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or "," not in line:
            continue
        n,t=line.split(",",1)
        n=n.strip(); t=t.strip()
        if n and t:
            out.append((n,t))
    return out

TOP10 = parse_list(top10_text)
ETF10 = parse_list(etf10_text)

KPI_COLS = 2 if mobile else 4
CHART_H = 240 if mobile else 300

# -------------------------
# Header
# -------------------------
st.title("재테크 핵심지표 대시보드")
st.markdown('<div class="small">국내/미국 지수 · 국내 10대 기업 · 대표 ETF 10 · 환율/금/유가 · 실시간 경제뉴스</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)


# -------------------------
# Tabs
# -------------------------
def render(freq):
    start = (datetime.now() - timedelta(days={"D":180,"W":365*3,"M":365*10}[freq])).strftime("%Y-%m-%d")

    # Snapshot
    cols = st.columns(KPI_COLS)
    kpis = [
        ("KOSPI", ("FDR","KS11")),
        ("KOSDAQ", ("FDR","KQ11")),
        ("S&P 500", ("YF","^GSPC")),
        ("NASDAQ", ("YF","^IXIC")),
        ("USD/KRW", ("FDR","USD/KRW")),
        ("Gold", ("YF","GC=F")),
        ("WTI", ("YF","CL=F")),
    ]
    for i,(name,(src,sym)) in enumerate(kpis):
        with cols[i % KPI_COLS]:
            base = fdr_close(sym,start) if src=="FDR" else yf_close(sym,start)
            df = resample_close(base, freq)
            last,delta,pct = metric(df)
            kpi_card(name,last,delta,pct,2)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Indices charts
    st.subheader("주요 주가지수")
    kr = {}
    for name,sym in [("KOSPI","KS11"),("KOSDAQ","KQ11")]:
        d = resample_close(fdr_close(sym,start), freq)
        if not d.empty: kr[name]=d["Close"]
    us = {}
    for name,sym in [("S&P500","^GSPC"),("NASDAQ","^IXIC"),("DOW","^DJI")]:
        d = resample_close(yf_close(sym,start), freq)
        if not d.empty: us[name]=d["Close"]

    if kr:
        plot_lines(pd.DataFrame(kr), "KOSPI vs KOSDAQ (Normalized=100)", height=CHART_H, normalized=True)
    if us:
        plot_lines(pd.DataFrame(us), "US Indices (Normalized=100)", height=CHART_H, normalized=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Top10 companies
    st.subheader("국내 10대 기업 (정규화 100 비교)")
    prices={}
    for n,t in TOP10:
        d = resample_close(fdr_close(t,start), freq)
        if not d.empty: prices[n]=d["Close"]
    if prices:
        plot_lines(pd.DataFrame(prices), "KR Top10 Companies (Normalized=100)", height=CHART_H+60, normalized=True)
    else:
        st.info("기업 데이터가 없습니다. 티커를 확인해주세요.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ETF 10
    st.subheader("한국 대표 ETF 10 (정규화 100 비교)")
    prices={}
    for n,t in ETF10:
        d = resample_close(fdr_close(t,start), freq)
        if not d.empty: prices[n]=d["Close"]
    if prices:
        plot_lines(pd.DataFrame(prices), "KR ETF 10 (Normalized=100)", height=CHART_H+60, normalized=True)
    else:
        st.info("ETF 데이터가 없습니다. 티커를 확인해주세요.")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # News (RSS stable)
    st.subheader("실시간 경제 뉴스")
    feeds = [
        ("Google 경제", "http://news.google.co.kr/news?pz=1&cf=all&ned=kr&hl=ko&topic=b&output=rss"),
        ("Daum 경제", "http://media.daum.net/rss/part/primary/economic/rss2.xml"),
    ]
    for label,url in feeds:
        st.markdown(f"**{label}**")
        items = rss_items(url, limit=max(news_limit, 35))
        if keyword:
            k = keyword.lower()
            items = [it for it in items if k in (it["title"] or "").lower()]
        for it in items[:news_limit]:
            pub = f" · {it['published']}" if it.get("published") else ""
            st.markdown(f"- [{it['title']}]({it['link']}){pub}")

tabs = st.tabs(["일간", "주간", "월간"])
with tabs[0]: render("D")
with tabs[1]: render("W")
with tabs[2]: render("M")
