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
.block-container { max-width: 1240px; padding-top: 2.6rem; padding-bottom: 2rem; }

h1 {
  font-size: 1.65rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.02em;
  line-height: 1.25 !important;
  margin-top: .25rem !important;
  padding-top: .15rem !important;
}

.small { color: rgba(0,0,0,.55); font-size: .92rem; }

/* ✅ 카드 */
.card {
  border:1px solid rgba(0,0,0,.06);
  border-radius:16px;
  padding:12px 12px 10px 12px;
  background:#fff;
  box-shadow:0 8px 24px rgba(0,0,0,.05);
}
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

/* ✅ footer 중앙정렬 */
.footer {
  text-align:center;
  color: rgba(0,0,0,.45);
  font-size: 0.86rem;
  padding: 22px 0 6px 0;
}
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
        margin=dict(l=10, r=10, t=64, b=90),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

def link_button(label: str, url: str):
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True)
    else:
        st.markdown(f"- [{label}]({url})")

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

def has_yf_search():
    return getattr(yf, "Search", None) is not None

def try_yf_search(query: str, max_results: int = 10):
    """가능하면 yfinance.Search 사용, 아니면 빈 리스트."""
    results = []
    query = (query or "").strip()
    if not query:
        return results
    Search = getattr(yf, "Search", None)
    if Search is None:
        return results

    try:
        s = Search(query)
        quotes = getattr(s, "quotes", None) or []
        for q in quotes[:max_results]:
            sym = q.get("symbol") or ""
            name = q.get("shortname") or q.get("longname") or q.get("name") or ""
            exch = q.get("exchange") or q.get("exchDisp") or ""
            if sym:
                results.append({"symbol": sym, "name": name, "exchange": exch})
        return results
    except Exception:
        return results

def resolve_kr_by_6digit(ticker6: str):
    """6자리 입력이면 .KS/.KQ를 시도해서 이름 추정(가능한 범위)."""
    t = clean_ticker(ticker6)
    if not re.fullmatch(r"\d{6}", t):
        return None

    # yfinance info는 느리거나 제한될 수 있음. 실패해도 안전하게 None 반환.
    for suf in [".KS", ".KQ"]:
        sym = f"{t}{suf}"
        try:
            info = yf.Ticker(sym).info
            nm = info.get("longName") or info.get("shortName") or info.get("displayName")
            if nm:
                return {"symbol": sym, "name": nm, "exchange": "KRX"}
        except Exception:
            pass
    return None

# =========================
# Session State: 관심종목(내 주식)
# =========================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = [
        {"name": "삼성전자", "ticker": "005930"},
        {"name": "SK하이닉스", "ticker": "000660"},
        {"name": "현대차", "ticker": "005380"},
        {"name": "기아", "ticker": "000270"},
        {"name": "NAVER", "ticker": "035420"},
        {"name": "카카오", "ticker": "035720"},
        {"name": "셀트리온", "ticker": "068270"},
        {"name": "삼성바이오로직스", "ticker": "207940"},
        {"name": "LG화학", "ticker": "051910"},
        {"name": "KB금융", "ticker": "105560"},
    ]

def watchlist_add(name: str, ticker: str):
    name = (name or "").strip()
    ticker = (ticker or "").strip()
    if not name or not ticker:
        return "empty"
    for it in st.session_state.watchlist:
        if clean_ticker(it.get("ticker", "")) == clean_ticker(ticker):
            return "dup"
    if len(st.session_state.watchlist) >= 10:
        return "full"
    st.session_state.watchlist.append({"name": name, "ticker": ticker})
    return "ok"

def watchlist_remove(idx: int):
    if 0 <= idx < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(idx)

# =========================
# Guide Content
# =========================
GUIDE_MD = """
### 이 대시보드, 이렇게 보시면 됩니다

#### 1) ‘일간/주간/월간’ 탭 차이
- **일간(D)**: 최근 흐름(단기 변동, 뉴스 영향)을 빠르게 확인
- **주간(W)**: 단기 노이즈를 줄이고 추세(스윙) 확인
- **월간(M)**: 큰 사이클(중장기 방향) 확인

#### 2) 정규화 100 그래프(가장 중요한 핵심)
- 시작점을 **100으로 맞춰**서 “누가 더 강한지(상대수익률)”를 비교하는 방식입니다.
- 120이면 시작 대비 **+20%**, 80이면 **-20%** 같은 의미예요.
- 10대 기업/ETF 비교 그래프가 바로 이 방식입니다.

#### 3) 주요 지표 뜻 (초간단)
- **KOSPI/KOSDAQ**: 국내 시장 체력(대형/성장·중소형)
- **S&P500/NASDAQ/DOW**: 미국 시장(빅테크/대형/전통)
- **USD/KRW**: 환율. 원화 강/약 → 수입물가·외국인 수급에 영향
- **Gold(금)**: 불안/인플레 헤지 성격
- **WTI(유가)**: 경기·물가·금리 방향에 영향을 주는 원자재 핵심
- **DXY(달러인덱스)**: 달러 힘. 강달러면 신흥국·위험자산에 부담이 되는 경우가 많음

#### 4) 빠른 해석 팁(자주 쓰는 조합)
- **DXY↑ + USD/KRW↑**: 강달러/원화약세 → 국내 주식에 부담일 때가 많음
- **WTI↑**: 물가 압력/금리 부담이 커질 수 있음(업종별 영향 다름)
- **Gold↑ + 주가지수↓**: 위험회피 분위기 가능성 체크

#### 5) ‘내 관심종목’ 활용법
- 관심종목을 **최대 10개**까지 등록해두고,
- **KOSPI/KOSDAQ/미국 지수**와 ‘상대적인 강도’를 같이 보세요.
- 강한 종목은 약세장에서도 덜 빠지거나 빨리 회복합니다.

> 참고: 데이터는 무료 소스 특성상 간헐적으로 누락될 수 있어요.  
> 그럴 땐 좌측 **‘지금 새로고침’**을 눌러주세요.
"""

# =========================
# Sidebar: 메뉴 + 설정 + 관심종목 관리(+/-)
# =========================
st.sidebar.markdown("### 📌 메뉴")
section = st.sidebar.radio("이동", ["대시보드", "관심종목 관리"], label_visibility="collapsed")

st.sidebar.markdown("---")
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

# ✅ 좌측: 관심종목 +/− (검색 동작 확실히: form 사용)
st.sidebar.markdown("### ⭐ 내 관심종목 (+ / -)")
st.sidebar.caption("최대 10개. 검색/추가/삭제를 여기서 직관적으로 관리합니다.")

with st.sidebar.expander("➕ 종목 검색/추가", expanded=True):
    with st.form("watchlist_search_form", clear_on_submit=False):
        q = st.text_input("검색(회사명/티커)", value="", key="wl_q")
        manual_name = st.text_input("이름(수동)", value="", key="wl_manual_name")
        manual_ticker = st.text_input("티커(수동, 6자리/심볼)", value="", key="wl_manual_ticker")

        c1, c2 = st.columns(2)
        search_submit = c1.form_submit_button("검색", use_container_width=True)
        manual_submit = c2.form_submit_button("수동 추가", use_container_width=True)

    # 검색 버튼 눌렀을 때
    if search_submit:
        st.session_state.wl_results = []
        qq = (q or "").strip()

        # 1) yfinance.Search 가능하면 사용
        if has_yf_search() and qq:
            st.session_state.wl_results = try_yf_search(qq, max_results=10)

        # 2) Search가 없거나 결과가 없으면: 6자리 입력이면 .KS/.KQ 시도
        if (not st.session_state.wl_results) and re.search(r"\d{6}", qq):
            resolved = resolve_kr_by_6digit(qq)
            if resolved:
                st.session_state.wl_results = [resolved]

    # 수동 추가 버튼 눌렀을 때
    if manual_submit:
        if len(st.session_state.watchlist) >= 10:
            st.warning("관심종목은 최대 10개까지예요. 먼저 하나 삭제해 주세요.")
        else:
            nm = manual_name or q
            tk = manual_ticker.strip()
            # 6자리면 6자리로 저장(내부에서 KR 처리), 심볼이면 그대로
            tk = clean_ticker(tk) if re.search(r"\d{6}", tk) else tk
            res = watchlist_add(nm, tk)
            if res == "dup":
                st.info("이미 등록된 티커입니다.")
            elif res == "empty":
                st.warning("이름/티커를 입력해주세요.")
            elif res == "full":
                st.warning("관심종목은 최대 10개까지입니다.")
            else:
                st.success("추가 완료!")
            st.rerun()

    # 검색 결과 표시 (+)
    results = st.session_state.get("wl_results", [])
    if search_submit and not results:
        st.caption("검색 결과가 없습니다. (환경에 따라 검색이 제한될 수 있어요) 아래 ‘수동 추가’를 사용해주세요.")

    if results:
        st.markdown("**검색 결과 (＋로 추가)**")
        for r in results:
            sym = r.get("symbol", "")
            name = r.get("name", "") or "(이름없음)"
            label = f"{name} ({sym})"
            add_cols = st.columns([4, 1])
            with add_cols[0]:
                st.caption(label)
            with add_cols[1]:
                if st.button("＋", key=f"add_{sym}", use_container_width=True):
                    if len(st.session_state.watchlist) >= 10:
                        st.warning("관심종목은 최대 10개까지예요. 먼저 하나 삭제해 주세요.")
                    else:
                        # KR 6자리면 6자리로 저장, 아니면 심볼 그대로
                        m = re.search(r"(\d{6})", sym)
                        tk = m.group(1) if m else sym
                        watchlist_add(name, tk)
                        st.rerun()

st.sidebar.markdown("**현재 관심종목 (최대 10개)**")
for i, it in enumerate(st.session_state.watchlist):
    row = st.sidebar.columns([4, 1])
    with row[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['ticker']}")
    with row[1]:
        if st.sidebar.button("－", key=f"rm_{i}", use_container_width=True):
            watchlist_remove(i)
            st.rerun()

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
CHART_H = 260 if mobile else 340
NEWS_COLS = 1 if mobile else 2

# =========================
# Top header + Guide button (popup)
# =========================
left, right = st.columns([5, 1.3])
with left:
    st.title("재테크 핵심지표 대시보드")
    st.markdown(
        '<div class="small">국내/미국 지수 · 국내 10대 기업 · 대표 ETF 10 · 환율/금/유가 · 내 관심종목 · 실시간 경제뉴스</div>',
        unsafe_allow_html=True
    )
with right:
    st.write("")
    st.write("")
    if st.button("활용법 가이드", use_container_width=True):
        st.session_state.show_guide = True

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ✅ dialog가 있으면 진짜 팝업으로, 없으면 페이지 상단에 안내 박스로 표시
if st.session_state.get("show_guide", False):
    if hasattr(st, "dialog"):
        @st.dialog("활용법 가이드")
        def _guide_dialog():
            st.markdown(GUIDE_MD)
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        _guide_dialog()
    else:
        with st.container():
            st.info("이 Streamlit 버전에서는 팝업 대신 상단에 표시됩니다.")
            st.markdown(GUIDE_MD)
            if st.button("가이드 닫기"):
                st.session_state.show_guide = False
                st.rerun()

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
        ("DXY (달러인덱스)", "DX-Y.NYB", 2),
    ]
    cols = st.columns(KPI_COLS)
    for i, (title, sym, prec) in enumerate(series):
        with cols[i % KPI_COLS]:
            base = yf_close(sym, start)
            df = resample_close(base, freq)
            last, delta, pct = metric(df)
            kpi_card(title, last, delta, pct, precision=prec)

def render_watchlist(freq: str, start: str):
    st.subheader("내 주식 관심종목")
    wl = st.session_state.watchlist[:10]
    if not wl:
        st.info("관심종목이 비어있어요. 좌측에서 +로 추가해 주세요.")
        return

    cols = st.columns(KPI_COLS)
    series = {}
    for i, it in enumerate(wl):
        name = it["name"]
        tick = it["ticker"]
        if re.fullmatch(r"\d{6}", clean_ticker(tick)):
            base = yf_close_kr(tick, start)
        else:
            base = yf_close(tick, start)

        df = resample_close(base, freq)
        last, delta, pct = metric(df)

        with cols[i % KPI_COLS]:
            kpi_card(f"{name}", last, delta, pct, precision=2)

        if not df.empty:
            series[name] = df["Close"]

    if series:
        plot_lines(pd.DataFrame(series), "관심종목 (정규화=100)", height=CHART_H + 40, normalized=True)

def render_econ_shortcuts():
    st.markdown("**국내 주요 경제지표 바로가기**")
    links = [
        ("한국은행 ECOS", "https://ecos.bok.or.kr/"),
        ("통계청 KOSIS", "https://kosis.kr/"),
        ("기획재정부", "https://www.moef.go.kr/"),
        ("금융위원회", "https://www.fsc.go.kr/"),
        ("금융감독원", "https://www.fss.or.kr/"),
        ("한국거래소(KRX)", "https://www.krx.co.kr/"),
        ("e-나라지표", "https://www.index.go.kr/"),
        ("네이버 금융", "https://finance.naver.com/"),
        ("연합인포맥스", "https://news.einfomax.co.kr/"),
        ("FRED(미국 지표)", "https://fred.stlouisfed.org/"),
    ]
    cols_n = 2 if mobile else 5
    cols = st.columns(cols_n)
    for i, (label, url) in enumerate(links[:10]):
        with cols[i % cols_n]:
            link_button(label, url)

def render_news():
    st.subheader("실시간 경제 뉴스")
    render_econ_shortcuts()
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

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

    render_watchlist(freq, start)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    render_news()
    st.caption("※ 무료 데이터 소스 특성상 간헐적 누락이 있을 수 있어요. 그럴 땐 좌측 ‘지금 새로고침’을 눌러주세요.")

def render_watchlist_manager_page():
    st.subheader("관심종목 관리")
    st.write("좌측 사이드바에서도 +/−로 관리할 수 있고, 여기서 한 번에 정리할 수도 있어요.")

    wl = st.session_state.watchlist[:10]
    df = pd.DataFrame(wl) if wl else pd.DataFrame(columns=["name", "ticker"])
    if not df.empty:
        df.index = np.arange(1, len(df) + 1)
        st.dataframe(df, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("좌측 ‘➕ 종목 검색/추가’에서 검색/추가/삭제를 진행해주세요. (여기서는 보기만 제공합니다.)")

# =========================
# Footer
# =========================
def render_footer():
    st.markdown("""
<div class="footer">
  © 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br/>
  © MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)

# =========================
# Run
# =========================
if section == "관심종목 관리":
    render_watchlist_manager_page()
    render_footer()
else:
    tabs = st.tabs(["일간", "주간", "월간"])
    with tabs[0]:
        render_tab("D")
    with tabs[1]:
        render_tab("W")
    with tabs[2]:
        render_tab("M")
    render_footer()
