import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import feedparser
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
def now():
    return datetime.now()

def days(freq):
    return {"D":180, "W":900, "M":3650}.get(freq, 365)

def is_korean_text(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def clean_ticker(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"(\d{6})", raw)
    return m.group(1) if m else raw.replace(" ", "")

def yf_close(sym: str, start: str) -> pd.Series:
    df = yf.download(sym, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if "Close" not in df.columns:
        return pd.Series(dtype="float64")
    return df["Close"].dropna()

def kpi_delta(s: pd.Series):
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
    s = s.dropna()
    if s.empty:
        return s
    return (s / s.iloc[0]) * 100

# =========================
# KOSIS OpenAPI (주간 아파트 매매가격지수)
# =========================
KOSIS_ORG_ID_DEFAULT = "408"                      # 한국부동산원 계열에서 자주 사용
KOSIS_TBL_ID_DEFAULT = "DT_304004_WEEK_002_A"     # "주간 아파트 매매가격지수" (KOSIS 통계표)
KOSIS_ITM_ID_DEFAULT = "T1"                       # 많은 KOSIS 표에서 지수 항목이 T1인 경우가 많음(표마다 다를 수 있어 자동 안내 포함)
REGION_CODES_DEFAULT = {                          # 일반적인 시도 코드(표마다 동일하지 않을 수 있어 실패 시 안내)
    "서울": "11",
    "부산": "21",
    "대구": "22",
    "경기": "31",
}

def _kosis_request_params(api_key, org_id, tbl_id, itm_id, objL1, prdSe, startPrdDe=None, endPrdDe=None, newEstPrdCnt=None):
    params = {
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
        "prdSe": prdSe,
        "loadGubun": "2",
    }
    if startPrdDe and endPrdDe:
        params["startPrdDe"] = startPrdDe
        params["endPrdDe"] = endPrdDe
    elif newEstPrdCnt:
        params["newEstPrdCnt"] = str(newEstPrdCnt)
    return params

@st.cache_data(ttl=60*60*6)
def kosis_weekly_apt_sale_index(api_key: str,
                               org_id: str = KOSIS_ORG_ID_DEFAULT,
                               tbl_id: str = KOSIS_TBL_ID_DEFAULT,
                               itm_id: str = KOSIS_ITM_ID_DEFAULT,
                               region_codes: dict = REGION_CODES_DEFAULT,
                               weeks: int = 260):
    """
    주간 아파트 매매가격지수(서울/부산/대구/경기) 조회.
    - prdSe=W (주간)
    - newEstPrdCnt로 최근 N개 시점 조회(여기서는 260주 ≈ 5년)
    """
    base_url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    out = {}

    for name, code in region_codes.items():
        params = _kosis_request_params(
            api_key=api_key,
            org_id=org_id,
            tbl_id=tbl_id,
            itm_id=itm_id,
            objL1=code,
            prdSe="W",
            newEstPrdCnt=weeks
        )
        r = requests.get(base_url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        # KOSIS 응답이 에러 텍스트일 때 대비
        if isinstance(data, dict) and data.get("err") is not None:
            raise ValueError(str(data))

        df = pd.DataFrame(data)
        if df.empty:
            out[name] = pd.Series(dtype="float64")
            continue

        # 대표 컬럼들: PRD_DE(시점), DT(값) / 일부 표는 DATA_VALUE 등
        val_col = "DT" if "DT" in df.columns else ("DATA_VALUE" if "DATA_VALUE" in df.columns else None)
        if val_col is None or "PRD_DE" not in df.columns:
            out[name] = pd.Series(dtype="float64")
            continue

        s = pd.to_numeric(df[val_col], errors="coerce")
        x = df["PRD_DE"].astype(str)

        # 주간은 YYYYMMDD 형태가 일반적
        idx = pd.to_datetime(x, errors="coerce", format="%Y%m%d")
        s.index = idx
        s = s.dropna()
        s = s.sort_index()
        out[name] = s

    result = pd.DataFrame(out).dropna(how="all")
    return result

def render_realestate_kosis(freq: str):
    """
    선택1: 주간 아파트 매매가격지수
    - 탭이 주간(W)/월간(M)일 때도 보여주되, 데이터 자체는 주간으로 고정.
    """
    st.subheader("🏠 주간 아파트 매매가격지수 (서울 · 부산 · 대구 · 경기)")
    st.caption("출처: KOSIS OpenAPI (한국부동산원 전국주택가격동향조사)")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    if not api_key:
        st.error("KOSIS_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 KOSIS_API_KEY를 추가해주세요.")
        return

    # (선택) 값이 다를 경우 secrets로 오버라이드 가능
    org_id = st.secrets.get("KOSIS_ORG_ID", KOSIS_ORG_ID_DEFAULT)
    tbl_id = st.secrets.get("KOSIS_TBL_ID", KOSIS_TBL_ID_DEFAULT)
    itm_id = st.secrets.get("KOSIS_ITM_ID", KOSIS_ITM_ID_DEFAULT)

    try:
        df = kosis_weekly_apt_sale_index(api_key=api_key, org_id=org_id, tbl_id=tbl_id, itm_id=itm_id, weeks=260)
    except Exception as e:
        st.error("KOSIS 부동산 지표 호출에 실패했습니다.")
        st.code(str(e))
        st.info("""
✅ 해결 방법(딱 한 번만 하면 끝)

1) KOSIS 공유서비스(OpenAPI) → **통계자료** → **통계표선택**에서  
   '주간 아파트 매매가격지수' 표를 열고  
   - 서울/부산/대구/경기 선택
   - 'URL보기' 버튼 클릭
2) URL 안에서 아래 값을 확인한 뒤 Secrets에 넣어주세요.
   - KOSIS_ORG_ID
   - KOSIS_TBL_ID
   - KOSIS_ITM_ID

Secrets 예시:
KOSIS_API_KEY="..."
KOSIS_ORG_ID="408"
KOSIS_TBL_ID="DT_304004_WEEK_002_A"
KOSIS_ITM_ID="T1"   ← 여기가 표마다 달라서 실패하는 경우가 가장 많습니다.
""")
        return

    if df.empty or df.dropna(how="all").empty:
        st.warning("데이터가 비어있습니다. (항목코드 itmId 또는 지역코드 objL1이 이 표와 다를 가능성이 큽니다)")
        st.info("""
✅ 가장 확실한 방법:
KOSIS에서 같은 표를 열고 'URL보기'로 생성된 URL의 itmId/objL1 값을 그대로 Secrets에 반영해 주세요.
""")
        return

    # 탭이 일간/월간이어도 부동산은 주간 데이터 그대로 보여줌
    # 정규화 100 비교 (도시별 상대 흐름)
    norm = df.apply(normalize_100)
    plot_df(norm, "주간 아파트 매매가격지수 (정규화 100)", height=340)

    with st.expander("원자료(지수값) 보기"):
        st.dataframe(df.tail(30), use_container_width=True)

# =========================
# Sidebar (관심종목)
# =========================
st.sidebar.markdown("### ⭐ 관심종목 검색/추가")
st.sidebar.caption("한글 회사명 검색은 yfinance에서 제한이 있어요. **6자리 티커**로 추가하는 방식이 가장 안정적입니다.")

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
        st.session_state.search_msg = "한글 회사명 검색은 제한됩니다. 예: 삼성전자 → **005930** 처럼 6자리 티커로 입력해주세요."
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
                    st.sidebar.warning("최대 10개까지입니다. 먼저 삭제해주세요.")
                elif res == "dup":
                    st.sidebar.info("이미 등록된 종목입니다.")
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
    if hasattr(st, "dialog"):
        @st.dialog("활용법 가이드")
        def guide():
            st.markdown("""
- **일간/주간/월간**: 단기/추세/사이클 확인  
- **정규화 100**: “누가 더 강한가(상대강도)” 비교  
- **DXY/환율↑**: 위험자산에 부담일 때가 많음  
- **WTI↑**: 물가/금리 압력 체크  
- **부동산 지수**: 보통 주식 대비 후행, 흐름 확인용
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        guide()
    else:
        st.info("가이드 팝업이 미지원이라 상단에 표시됩니다.")
        st.markdown("""
- **일간/주간/월간**: 단기/추세/사이클 확인  
- **정규화 100**: “누가 더 강한가(상대강도)” 비교  
- **DXY/환율↑**: 위험자산에 부담일 때가 많음  
- **WTI↑**: 물가/금리 압력 체크  
- **부동산 지수**: 보통 주식 대비 후행, 흐름 확인용
""")
        if st.button("가이드 닫기"):
            st.session_state.show_guide = False
            st.rerun()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# Main Tabs (인덱스 기반)
# =========================
tab_titles = ["일간", "주간", "월간"]
freqs = ["D", "W", "M"]
tabs = st.tabs(tab_titles)

for idx, tab in enumerate(tabs):
    with tab:
        freq = freqs[idx]
        start = (now() - timedelta(days=days(freq))).strftime("%Y-%m-%d")

        # ===== Snapshot (단위 표기) =====
        st.subheader("요약 스냅샷")
        cols = st.columns(4)

        kospi = yf_close("^KS11", start)
        kosdaq = yf_close("^KQ11", start)
        usdkrw = yf_close("KRW=X", start)
        gold_usd_oz = yf_close("GC=F", start)  # USD/oz
        wti = yf_close("CL=F", start)          # USD/bbl
        dxy = yf_close("DX-Y.NYB", start)      # index

        # 한국 금 1돈(원/돈)
        gold_kr_1don = pd.Series(dtype="float64")
        if not gold_usd_oz.empty and not usdkrw.empty:
            aligned = pd.concat([gold_usd_oz, usdkrw], axis=1).dropna()
            if not aligned.empty:
                gold_kr_1don = aligned.iloc[:,0] * aligned.iloc[:,1] * (3.75 / 31.1035)

        with cols[0]:
            kpi_card("KOSPI", f"{kospi.iloc[-1]:.1f}" if not kospi.empty else "-", "pt", kpi_delta(kospi), precision=1)
        with cols[1]:
            kpi_card("USD/KRW", f"{usdkrw.iloc[-1]:.1f}" if not usdkrw.empty else "-", "원", kpi_delta(usdkrw), precision=1)
        with cols[2]:
            kpi_card("금 1돈", f"{gold_kr_1don.iloc[-1]:,.0f}" if not gold_kr_1don.empty else "-", "원/돈")
        with cols[3]:
            kpi_card("WTI", f"{wti.iloc[-1]:.2f}" if not wti.empty else "-", "USD/배럴", kpi_delta(wti), precision=2)

        cols2 = st.columns(4)
        with cols2[0]:
            kpi_card("KOSDAQ", f"{kosdaq.iloc[-1]:.1f}" if not kosdaq.empty else "-", "pt", kpi_delta(kosdaq), precision=1)
        with cols2[1]:
            kpi_card("DXY", f"{dxy.iloc[-1]:.2f}" if not dxy.empty else "-", "index", kpi_delta(dxy), precision=2)
        with cols2[2]:
            kpi_card("Gold", f"{gold_usd_oz.iloc[-1]:.2f}" if not gold_usd_oz.empty else "-", "USD/oz", kpi_delta(gold_usd_oz), precision=2)
        with cols2[3]:
            st.markdown('<div class="card"><div class="ct">단위 안내</div><div style="color:rgba(0,0,0,.55);font-weight:650">pt/원/index/USD</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # ===== 관심종목 (정규화 100) =====
        st.subheader("내 관심종목 (정규화 100 비교)")
        if not st.session_state.watchlist:
            st.info("좌측에서 관심종목을 추가해주세요. (최대 10개)")
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
                st.warning("관심종목 데이터를 가져오지 못했습니다. 티커를 확인해주세요.")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # ===== 부동산(선택1: 주간 아파트 매매가격지수) =====
        render_realestate_kosis(freq)

# =========================
# Footer (최하단 중앙)
# =========================
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
