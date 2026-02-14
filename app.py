import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import re
import requests

# =========================
# Page / CSS
# =========================
st.set_page_config(page_title="재테크 핵심지표 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

header[data-testid="stHeader"] { height: 0rem; }
.block-container { max-width: 1240px; padding-top: 2.6rem; padding-bottom: 2rem; }

h1 { font-size:1.7rem !important; font-weight:800; line-height:1.25 !important; }
.small { color:#666; font-size:.92rem; }

.card {
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
.footer{margin-top:40px;text-align:center;font-size:.85rem;color:rgba(0,0,0,.55); padding:18px 0 6px 0;}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def now(): return datetime.now()

def days(freq): return {"D":180, "W":900, "M":3650}.get(freq, 365)

def is_korean_text(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def clean_ticker(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"(\d{6})", raw)
    return m.group(1) if m else raw.replace(" ", "")

def _to_series_close(df: pd.DataFrame) -> pd.Series:
    """
    yfinance 결과가
    - MultiIndex 컬럼
    - DataFrame/Series 혼재
    어떤 형태든 'Close' 단일 Series로 정리.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    # MultiIndex면 Close 레벨을 먼저 시도
    if isinstance(df.columns, pd.MultiIndex):
        # 가장 흔한 형태: ('Close', 'TICKER') 같은 구조
        try:
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return pd.to_numeric(close, errors="coerce").dropna()
        except Exception:
            # 다른 MultiIndex 구조도 대비: 2레벨에서 'Close' 찾기
            cols = df.columns
            close_cols = [c for c in cols if (isinstance(c, tuple) and "Close" in c)]
            if close_cols:
                close = df[close_cols[0]]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return pd.to_numeric(close, errors="coerce").dropna()
            return pd.Series(dtype="float64")

    # 일반 컬럼
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.to_numeric(close, errors="coerce").dropna()

    # 혹시 Close가 없으면 마지막 컬럼을 시도(예외 케이스)
    close = df.iloc[:, -1]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()

@st.cache_data(ttl=60*10)
def yf_close(sym: str, start: str) -> pd.Series:
    df = yf.download(sym, start=start, progress=False, auto_adjust=False)
    return _to_series_close(df)

def safe_last_float(s: pd.Series):
    """무조건 숫자 float로 마지막 값 반환 (실패 시 None)"""
    if s is None or len(s) == 0:
        return None
    try:
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None

def kpi_delta(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s is None or s.empty or len(s) < 2:
        return None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    d = last - prev
    p = (d / prev * 100) if prev != 0 else None
    return last, d, p

def kpi_card(title, value, unit, delta_tuple=None, precision=2):
    delta_html = ""
    cls = ""
    if delta_tuple is not None:
        _, d, p = delta_tuple
        if p is not None:
            cls = "pos" if d > 0 else "neg" if d < 0 else ""
            delta_html = f'<div class="{cls}">{d:+.{precision}f} ({p:+.2f}%)</div>'
    st.markdown(f"""
    <div class="card">
      <div class="ct">{title}</div>
      <div class="kpi">{value} <span style="font-size:.9rem; font-weight:650; color:rgba(0,0,0,.55)">{unit}</span></div>
      {delta_html}
    </div>
    """, unsafe_allow_html=True)

def plot_df(df: pd.DataFrame, title: str, height: int = 330):
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
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

# =========================
# KOSIS (선택1: 주간 아파트 매매가격지수)
# =========================
KOSIS_ORG_ID_DEFAULT = st.secrets.get("KOSIS_ORG_ID", "408")
KOSIS_TBL_ID_DEFAULT = st.secrets.get("KOSIS_TBL_ID", "DT_304004_WEEK_002_A")
KOSIS_ITM_ID_DEFAULT = st.secrets.get("KOSIS_ITM_ID", "T1")

REGION_CODES_DEFAULT = {
    "서울": "11",
    "부산": "21",
    "대구": "22",
    "경기": "31",
}

def _kosis_params(api_key, org_id, tbl_id, itm_id, objL1, weeks=260):
    return {
        "method": "getList",
        "apiKey": api_key,
        "orgId": org_id,
        "tblId": tbl_id,
        "itmId": itm_id,
        "objL1": objL1,
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

@st.cache_data(ttl=60*60*6)
def kosis_weekly_apt_sale_index(api_key: str,
                               org_id: str = KOSIS_ORG_ID_DEFAULT,
                               tbl_id: str = KOSIS_TBL_ID_DEFAULT,
                               itm_id: str = KOSIS_ITM_ID_DEFAULT,
                               region_codes: dict = REGION_CODES_DEFAULT,
                               weeks: int = 260):
    base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    out = {}

    for name, code in region_codes.items():
        params = _kosis_params(api_key, org_id, tbl_id, itm_id, code, weeks)
        r = requests.get(base_url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

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

def render_realestate():
    st.subheader("🏠 주간 아파트 매매가격지수 (서울 · 부산 · 대구 · 경기)")
    st.caption("출처: KOSIS OpenAPI (선택1)")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    if not api_key:
        st.error("KOSIS_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 KOSIS_API_KEY를 추가해주세요.")
        return

    try:
        df = kosis_weekly_apt_sale_index(api_key=api_key)
    except Exception as e:
        st.error("KOSIS 호출 실패")
        st.code(str(e))
        st.info("KOSIS URL보기에서 tblId/itmId가 맞는지 확인 후 Secrets(KOSIS_TBL_ID, KOSIS_ITM_ID)로 넣어주세요.")
        return

    if df.empty:
        st.warning("부동산 데이터가 비어있습니다. KOSIS_TBL_ID / KOSIS_ITM_ID가 표와 다를 가능성이 큽니다.")
        st.info("KOSIS URL보기에서 tblId, itmId를 확인해 Secrets에 넣고 다시 실행해주세요.")
        return

    norm = df.apply(normalize_100)
    plot_df(norm, "주간 아파트 매매가격지수 (정규화 100)", height=340)
    with st.expander("원자료(최근 30개)"):
        st.dataframe(df.tail(30), use_container_width=True)

# =========================
# Sidebar (관심종목)
# =========================
st.sidebar.markdown("### ⭐ 관심종목 검색/추가")
st.sidebar.caption("한글 회사명 검색은 제한이 있어요. **6자리 티커**로 추가하는 방식이 가장 안정적입니다.")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

def watchlist_add(name, ticker):
    if len(st.session_state.watchlist) >= 10:
        return "full"
    ticker = (ticker or "").strip()
    name = (name or "").strip()
    if not ticker or not name:
        return "empty"
    for it in st.session_state.watchlist:
        if clean_ticker(it["ticker"]) == clean_ticker(ticker):
            return "dup"
    st.session_state.watchlist.append({"name": name, "ticker": ticker})
    return "ok"

def watchlist_remove(i):
    if 0 <= i < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(i)

with st.sidebar.form("search_form"):
    q = st.text_input("검색(영문 심볼 / 6자리 티커)", value="")
    submitted = st.form_submit_button("검색")

if submitted:
    st.session_state.search_msg = ""
    st.session_state.search_results = []

    qq = (q or "").strip()
    if not qq:
        st.session_state.search_msg = "검색어를 입력해주세요."
    elif is_korean_text(qq):
        st.session_state.search_msg = "한글 회사명 검색은 제한됩니다. 예: 삼성전자 → **005930**"
    else:
        m = re.fullmatch(r"\d{6}", clean_ticker(qq))
        if m:
            t6 = m.group(0)
            st.session_state.search_results = [{"name": f"{t6}", "ticker": t6, "hint": f"{t6}.KS / {t6}.KQ 자동 시도"}]
        else:
            st.session_state.search_results = [{"name": "심볼(영문)", "ticker": qq, "hint": "예: AAPL, SPY, ^GSPC 등"}]

msg = st.session_state.get("search_msg", "")
if msg:
    st.sidebar.info(msg)

results = st.session_state.get("search_results", [])
if results:
    st.sidebar.markdown("**검색 결과**")
    for i, r in enumerate(results):
        cols = st.sidebar.columns([4,1])
        with cols[0]:
            st.sidebar.write(f"{r['name']} · {r['ticker']}")
            st.sidebar.caption(r.get("hint",""))
        with cols[1]:
            if st.sidebar.button("＋", key=f"addres_{i}"):
                res = watchlist_add(r["name"], r["ticker"])
                if res == "full":
                    st.sidebar.warning("최대 10개까지입니다.")
                elif res == "dup":
                    st.sidebar.info("이미 등록됨")
                elif res == "ok":
                    st.sidebar.success("추가 완료!")
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**현재 관심종목 (최대 10개)**")
for i, it in enumerate(st.session_state.watchlist):
    row = st.sidebar.columns([4,1])
    with row[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['ticker']}")
    with row[1]:
        if st.sidebar.button("－", key=f"rm_{i}"):
            watchlist_remove(i)
            st.rerun()

# =========================
# Header + Guide
# =========================
l, r = st.columns([5, 1.3])
with l:
    st.title("재테크 핵심지표 대시보드")
    st.markdown('<div class="small">주식 · 환율 · 금 · 유가 · 부동산(주간) · 관심종목</div>', unsafe_allow_html=True)
with r:
    if st.button("활용법 가이드", use_container_width=True):
        st.session_state.show_guide = True

if st.session_state.get("show_guide", False):
    st.info("""
- **정규화 100**: 도시/종목 상대강도 비교  
- **DXY/환율↑**: 위험자산에 부담일 때가 많음  
- **WTI↑**: 물가/금리 압력 체크  
- **부동산 지수**: 보통 주식 대비 후행, 흐름 확인용
""")
    if st.button("가이드 닫기"):
        st.session_state.show_guide = False
        st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Main Tabs
# =========================
tab_titles = ["일간", "주간", "월간"]
freqs = ["D", "W", "M"]
tabs = st.tabs(tab_titles)

for idx, tab in enumerate(tabs):
    with tab:
        freq = freqs[idx]
        start = (now() - timedelta(days=days(freq))).strftime("%Y-%m-%d")

        st.subheader("요약 스냅샷")
        cols = st.columns(4)

        kospi = yf_close("^KS11", start)
        kosdaq = yf_close("^KQ11", start)
        usdkrw = yf_close("KRW=X", start)
        gold_usd_oz = yf_close("GC=F", start)  # USD/oz
        wti = yf_close("CL=F", start)          # USD/배럴
        dxy = yf_close("DX-Y.NYB", start)      # index

        # 금 1돈(원/돈) = 금(USD/oz) * 환율(KRW/USD) * (3.75/31.1035)
        gold_kr_1don = pd.Series(dtype="float64")
        if not gold_usd_oz.empty and not usdkrw.empty:
            aligned = pd.concat([gold_usd_oz, usdkrw], axis=1).dropna()
            if not aligned.empty:
                gold_kr_1don = aligned.iloc[:, 0] * aligned.iloc[:, 1] * (3.75 / 31.1035)

        # ✅ 여기서부터는 무조건 float로 안전 변환
        kospi_last = safe_last_float(kospi)
        usd_last = safe_last_float(usdkrw)
        golddon_last = safe_last_float(gold_kr_1don)
        wti_last = safe_last_float(wti)

        with cols[0]:
            kpi_card("KOSPI", f"{kospi_last:.1f}" if kospi_last is not None else "-", "pt", kpi_delta(kospi), precision=1)
        with cols[1]:
            kpi_card("USD/KRW", f"{usd_last:.1f}" if usd_last is not None else "-", "원", kpi_delta(usdkrw), precision=1)
        with cols[2]:
            kpi_card("금 1돈", f"{golddon_last:,.0f}" if golddon_last is not None else "-", "원/돈")
        with cols[3]:
            kpi_card("WTI", f"{wti_last:.2f}" if wti_last is not None else "-", "USD/배럴", kpi_delta(wti), precision=2)

        cols2 = st.columns(4)
        kosdaq_last = safe_last_float(kosdaq)
        dxy_last = safe_last_float(dxy)
        gold_last = safe_last_float(gold_usd_oz)

        with cols2[0]:
            kpi_card("KOSDAQ", f"{kosdaq_last:.1f}" if kosdaq_last is not None else "-", "pt", kpi_delta(kosdaq), precision=1)
        with cols2[1]:
            kpi_card("DXY", f"{dxy_last:.2f}" if dxy_last is not None else "-", "index", kpi_delta(dxy), precision=2)
        with cols2[2]:
            kpi_card("Gold", f"{gold_last:.2f}" if gold_last is not None else "-", "USD/oz", kpi_delta(gold_usd_oz), precision=2)
        with cols2[3]:
            st.markdown('<div class="card"><div class="ct">단위</div><div style="color:rgba(0,0,0,.55);font-weight:650">pt / 원 / index / USD</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # 관심종목 그래프
        st.subheader("내 관심종목 (정규화 100)")
        if not st.session_state.watchlist:
            st.info("좌측에서 관심종목을 추가해주세요.")
        else:
            series = {}
            for it in st.session_state.watchlist[:10]:
                tick = it["ticker"]
                name = it["name"]
                if re.fullmatch(r"\d{6}", clean_ticker(tick)):
                    s = pd.Series(dtype="float64")
                    for suf in [".KS", ".KQ"]:
                        s = yf_close(f"{tick}{suf}", start)
                        if not s.empty:
                            break
                else:
                    s = yf_close(tick, start)

                if not s.empty:
                    series[name] = normalize_100(s)

            if series:
                df = pd.DataFrame(series).dropna(how="all")
                plot_df(df, "관심종목 (정규화 100)", height=340)
            else:
                st.warning("관심종목 데이터를 가져오지 못했습니다. 티커 확인 필요.")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # 부동산(선택1)
        render_realestate()

# Footer
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
