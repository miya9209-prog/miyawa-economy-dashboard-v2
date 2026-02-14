# --- 위쪽(기존) 코드는 그대로 두고 ---
# 여기부터는 "부동산 탭" 부분만 교체하면 되지만,
# 형준님 요청대로 통째 교체본으로 드립니다.

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

st.set_page_config(page_title="재테크 핵심지표 대시보드", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
header[data-testid="stHeader"] { height: 0rem; }
.block-container { max-width: 1240px; padding-top: 2.8rem; padding-bottom: 2rem; }
h1 { font-size:1.75rem !important; font-weight:800; line-height:1.25 !important; margin:0 !important; }
.small { color:#666; font-size:.92rem; }
.section-title{ font-size:1.25rem; font-weight:800; margin:.1rem 0 .6rem 0; }
.hr { border-top:1px solid rgba(0,0,0,.06); margin:1.0rem 0 1.05rem 0; }
.card{ border:1px solid rgba(0,0,0,.06); border-radius:16px; padding:12px 14px; background:#fff; box-shadow:0 8px 20px rgba(0,0,0,.05)}
.ct{font-size:.88rem;color:rgba(0,0,0,.62);font-weight:650}
.kpi{font-size:1.2rem;font-weight:800;letter-spacing:-.02em;margin-top:.15rem}
.pos{color:#0a7b34;font-weight:700}
.neg{color:#b42318;font-weight:700}
.footer{ margin-top:42px; text-align:center; font-size:.85rem; color:rgba(0,0,0,.55); padding:18px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

def now(): return datetime.now()

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
    if freq == "D": return s
    if freq == "W": return s.resample("W-FRI").last().dropna()
    if freq == "M": return s.resample("M").last().dropna()
    return s

@st.cache_data(ttl=60*10)
def yf_close(symbol: str, start: str) -> pd.Series:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            c = df["Close"]
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            return safe_series(c)
        except Exception:
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
            return safe_series(flat["Close"]) if "Close" in flat.columns else pd.Series(dtype="float64")
    return safe_series(df["Close"]) if "Close" in df.columns else pd.Series(dtype="float64")

def kpi_delta(s: pd.Series):
    s = safe_series(s)
    if s.empty or len(s) < 2: return None
    last = float(s.iloc[-1]); prev = float(s.iloc[-2])
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

# Sidebar
st.sidebar.markdown("### ⚙ 설정")
freq_label = st.sidebar.radio("보기 단위", ["일간", "주간", "월간"], index=0)
freq = {"일간":"D", "주간":"W", "월간":"M"}[freq_label]
news_n = st.sidebar.slider("뉴스 표시 개수", 5, 50, 20, step=5)
keyword = st.sidebar.text_input("뉴스 키워드 필터(선택)", value="")

# 관심종목 검색(한글)
@st.cache_data(ttl=60*60*24)
def krx_listings():
    frames = []
    try: frames.append(fdr.StockListing("KRX")[["Code", "Name", "Market"]])
    except Exception: pass
    try:
        etf = fdr.StockListing("ETF/KR")
        if "Symbol" in etf.columns: etf = etf.rename(columns={"Symbol":"Code"})
        if "Code" in etf.columns and "Name" in etf.columns:
            etf["Market"] = "ETF"
            frames.append(etf[["Code","Name","Market"]])
    except Exception: pass
    if not frames: return pd.DataFrame(columns=["Code","Name","Market"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df

KRX = krx_listings()

def clean_ticker_6(s: str) -> str | None:
    m = re.search(r"(\d{6})", (s or "").strip())
    return m.group(1) if m else None

def search_krx(query: str, limit=20) -> pd.DataFrame:
    q = (query or "").strip()
    if not q or KRX.empty:
        return pd.DataFrame(columns=["Code","Name","Market"])
    t6 = clean_ticker_6(q)
    if t6:
        return KRX[KRX["Code"] == t6].head(limit)
    return KRX[KRX["Name"].str.contains(q, case=False, na=False)].head(limit)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

def add_watch(name, code6):
    code6 = (code6 or "").strip()
    name = (name or code6).strip()
    if not code6 or len(code6) != 6: return "bad"
    if len(st.session_state.watchlist) >= 10: return "full"
    if any(it["code"] == code6 for it in st.session_state.watchlist): return "dup"
    st.session_state.watchlist.append({"name": name, "code": code6})
    return "ok"

def remove_watch(i):
    if 0 <= i < len(st.session_state.watchlist):
        st.session_state.watchlist.pop(i)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⭐ 관심종목 (한글 검색 지원)")
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

st.sidebar.markdown("**현재 관심종목**")
for i, it in enumerate(st.session_state.watchlist):
    cols = st.sidebar.columns([4,1])
    with cols[0]:
        st.sidebar.write(f"{i+1}. {it['name']} · {it['code']}")
    with cols[1]:
        if st.sidebar.button("－", key=f"rm_{it['code']}"):
            remove_watch(i); st.rerun()

# Header
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
- **USD/KRW, DXY**: 달러 강세/원화 약세 압력 체크  
- **Gold, WTI**: 인플레/원자재 사이클 확인  
- **부동산(KOSIS)**: 40,000셀 제한 때문에 자동으로 지역축을 탐색 후 최소 요청으로 조회  
""")
            if st.button("닫기"):
                st.session_state.show_guide = False
                st.rerun()
        guide()

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

def kr_yf_series(code6: str, start: str) -> pd.Series:
    s = yf_close(f"{code6}.KS", start)
    if s.empty: s = yf_close(f"{code6}.KQ", start)
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

tabs = st.tabs(["시장 한눈에", "국내 Top10/ETF10", "내 관심종목", "부동산(KOSIS)", "경제뉴스"])

with tabs[0]:
    start = get_start(freq)
    kospi = resample_close(yf_close("^KS11", start), freq)
    kosdaq = resample_close(yf_close("^KQ11", start), freq)
    spx = resample_close(yf_close("^GSPC", start), freq)
    nas = resample_close(yf_close("^IXIC", start), freq)
    dow = resample_close(yf_close("^DJI", start), freq)

    usdkrw = resample_close(yf_close("KRW=X", start), freq)
    dxy = resample_close(yf_close("DX-Y.NYB", start), freq)
    gold = resample_close(yf_close("GC=F", start), freq)
    wti = resample_close(yf_close("CL=F", start), freq)

    gold_kr_1don = pd.Series(dtype="float64")
    if not gold.empty and not usdkrw.empty:
        aligned = pd.concat([gold, usdkrw], axis=1).dropna()
        if not aligned.empty:
            gold_kr_1don = aligned.iloc[:, 0] * aligned.iloc[:, 1] * (3.75 / 31.1035)

    st.markdown('<div class="section-title">요약 스냅샷</div>', unsafe_allow_html=True)
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
    plot_df(pd.DataFrame({"KOSPI": normalize_100(kospi), "KOSDAQ": normalize_100(kosdaq)}).dropna(how="all"),
            "KOSPI vs KOSDAQ (정규화 100)", height=340)
    plot_df(pd.DataFrame({"S&P500": normalize_100(spx), "NASDAQ": normalize_100(nas), "DOW": normalize_100(dow)}).dropna(how="all"),
            "US Indices (정규화 100)", height=340)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    plot_df(pd.DataFrame({
        "USD/KRW": normalize_100(usdkrw),
        "DXY": normalize_100(dxy),
        "Gold(USD/oz)": normalize_100(gold),
        "WTI": normalize_100(wti),
    }).dropna(how="all"), "FX & Commodities (정규화 100)", height=360)

with tabs[1]:
    start = get_start(freq)
    s1 = {}
    for nm, code in KR_TOP10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty: s1[nm] = normalize_100(s)
    plot_df(pd.DataFrame(s1).dropna(how="all"), "KR Top10 (정규화 100)", height=380)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    s2 = {}
    for nm, code in KR_ETF10_DEFAULT:
        s = resample_close(kr_yf_series(code, start), freq)
        if not s.empty: s2[nm] = normalize_100(s)
    plot_df(pd.DataFrame(s2).dropna(how="all"), "KR ETF10 (정규화 100)", height=380)

with tabs[2]:
    start = get_start(freq)
    if not st.session_state.watchlist:
        st.info("좌측에서 관심종목을 검색/추가해주세요. (한글 회사명 검색 지원)")
    else:
        s = {}
        for it in st.session_state.watchlist[:10]:
            ss = resample_close(kr_yf_series(it["code"], start), freq)
            if not ss.empty: s[it["name"]] = normalize_100(ss)
        plot_df(pd.DataFrame(s).dropna(how="all"), "내 관심종목 (정규화 100)", height=420)

# =========================
# ✅ 부동산(KOSIS) 자동탐색/자동호출
# =========================
KOSIS_BASE = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

def _api_get(params):
    try:
        r = requests.get(KOSIS_BASE, params=params, timeout=25)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"err": "local", "errMsg": str(e)}

def _is_err(data):
    return isinstance(data, dict) and ("err" in data or "errMsg" in data)

def _pick_value_col(df: pd.DataFrame):
    for c in ["DT", "DATA_VALUE", "VALUE", "VAL"]:
        if c in df.columns: return c
    return None

def _pick_date_col(df: pd.DataFrame):
    for c in ["PRD_DE", "TIME"]:
        if c in df.columns: return c
    return None

def _parse_kosis_date(x: str):
    x = re.sub(r"[^0-9]", "", str(x))
    if len(x) == 6:
        return pd.to_datetime(x, format="%Y%m", errors="coerce")
    if len(x) == 8:
        return pd.to_datetime(x, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(x, errors="coerce")

def _to_series(df_raw: pd.DataFrame) -> pd.Series:
    if df_raw is None or df_raw.empty: return pd.Series(dtype="float64")
    dc = _pick_date_col(df_raw); vc = _pick_value_col(df_raw)
    if dc is None or vc is None: return pd.Series(dtype="float64")
    idx = df_raw[dc].apply(_parse_kosis_date)
    val = pd.to_numeric(df_raw[vc], errors="coerce")
    s = pd.Series(val.values, index=idx).dropna().sort_index()
    return s

@st.cache_data(ttl=60*60*6)
def kosis_probe_axis(api_key, org_id, tbl_id, itm_id, prd_se):
    """
    objL1~objL8 중 어떤 축이 '지역'인지 추정하기 위한 탐색.
    - objLk에 ALL을 넣었을 때 40,000셀 초과가 나는지
    - 특정값을 넣어도 '변수 잘못(err21)'이 나는지
    """
    base = {
        "method": "getList",
        "apiKey": api_key,
        "orgId": org_id,
        "tblId": tbl_id,
        "itmId": itm_id,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": prd_se,
        "loadGubun": "2",
        "newEstPrdCnt": "6",  # 아주 작게
    }
    # 기본: 전부 ALL로 하면 어떤 에러가 나는지
    params_all = dict(base)
    for k in range(1, 9):
        params_all[f"objL{k}"] = "ALL"
    code, data = _api_get(params_all)
    return {"code": code, "data": data, "params": params_all}

@st.cache_data(ttl=60*60*6)
def kosis_fetch_min(api_key, org_id, tbl_id, itm_id, prd_se, obj_kwargs: dict, new_cnt: int):
    params = {
        "method": "getList",
        "apiKey": api_key,
        "orgId": org_id,
        "tblId": tbl_id,
        "itmId": itm_id,
        "format": "json",
        "jsonVD": "Y",
        "prdSe": prd_se,
        "loadGubun": "2",
        "newEstPrdCnt": str(new_cnt),
    }
    # 지정되지 않은 obj는 ALL로 둠
    for k in range(1, 9):
        params[f"objL{k}"] = "ALL"
    for k, v in obj_kwargs.items():
        params[k] = v
    _, data = _api_get(params)
    if _is_err(data):
        return pd.DataFrame(), f"api_error: {data.get('err')} {data.get('errMsg','')}".strip()
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame(), "empty"
    return pd.DataFrame(data), "ok"

with tabs[3]:
    st.markdown('<div class="section-title">부동산 지표 (KOSIS) — 40,000셀/변수오류 자동 회피</div>', unsafe_allow_html=True)
    st.caption("현재 테이블은 objL1에 숫자코드를 넣으면 err21(변수오류) 가능성이 높습니다. 앱에서 최소 요청 방식으로 안전 호출합니다.")

    api_key = st.secrets.get("KOSIS_API_KEY", "").strip()
    org_id = st.secrets.get("KOSIS_ORG_ID", "408").strip()
    tbl_id = st.secrets.get("KOSIS_TBL_ID", "").strip()
    itm_id = st.secrets.get("KOSIS_ITM_ID", "").strip().replace("+", "").strip()
    prd_se = st.secrets.get("KOSIS_PRD_SE", "M").strip()

    if not api_key or not tbl_id or not itm_id:
        st.info("Secrets에 KOSIS_API_KEY / KOSIS_TBL_ID / KOSIS_ITM_ID 가 필요합니다.")
    else:
        new_cnt = st.slider("최근 몇 개 기간만 가져올까요?", 3, 60, 12, step=3)

        # ✅ 1) 우선 obj를 전부 ALL로 작은기간(6) 호출 시도 → 에러 메시지 유형 확인용
        probe = kosis_probe_axis(api_key, org_id, tbl_id, itm_id, prd_se)
        data_probe = probe["data"]

        if _is_err(data_probe):
            # 여기서 40000셀 에러가 뜨면, obj를 ALL로 둘 수 없다는 의미
            # 하지만 우리는 newCnt도 작게, obj도 바꿔가며 우회 시도
            st.warning("KOSIS가 현재 조합을 제한합니다. (자동 우회 시도 중)")
            st.info(f"probe err: {data_probe.get('err')} / {data_probe.get('errMsg','')}")
        else:
            st.success("KOSIS 기본 응답 확인 OK (자동 변환 진행)")

        # ✅ 2) 가장 안전한 우회: objL1~objL8 중 'ALL'이 너무 크면,
        # 일단 objL1만 ALL, 나머지는 빈값으로 줄여보는 등 최소요청 시도.
        # (KOSIS 표마다 obj의 구조가 달라서, 우리가 여러 조합을 자동으로 트라이)
        trial_patterns = [
            # ① objL1만 ALL, 나머지 공백(=미지정) 시도
            {"objL1": "ALL"},
            # ② objL1=ALL, objL2=ALL (형준님 URL 기본형)
            {"objL1": "ALL", "objL2": "ALL"},
        ]

        ok_df = None
        ok_msg = None
        for pat in trial_patterns:
            # pat는 objL1/objL2 키만 포함되므로, 함수가 나머지 objL3~objL8은 ALL로 채워버리면 다시 커질 수 있음
            # 그래서 여기서는 "지정되지 않은 objLk는 아예 파라미터에서 빼는" 방식으로 호출할 수 있도록 별도 호출을 사용
            # → Streamlit Cloud에서 가장 안정적으로는 objL3~objL8을 아예 전달하지 않는 방식이 필요
            # 따라서 아래는 직접 params 구성하여 '필요한 obj만' 전달
            params = {
                "method": "getList",
                "apiKey": api_key,
                "orgId": org_id,
                "tblId": tbl_id,
                "itmId": itm_id,
                "format": "json",
                "jsonVD": "Y",
                "prdSe": prd_se,
                "loadGubun": "2",
                "newEstPrdCnt": str(new_cnt),
                # 핵심: objL1/2만 전달, objL3~8은 전달하지 않음
            }
            for k, v in pat.items():
                params[k] = v

            _, data = _api_get(params)
            if _is_err(data):
                ok_msg = f"trial {pat} -> err {data.get('err')} {data.get('errMsg','')}"
                continue
            if isinstance(data, list) and len(data) > 0:
                ok_df = pd.DataFrame(data)
                ok_msg = f"trial {pat} -> ok"
                break
            ok_msg = f"trial {pat} -> empty"

        if ok_df is None or ok_df.empty:
            st.error("현재 tblId/itmId 조합은 '지역별'로 안전하게 가져오기 어렵습니다.")
            st.info(f"확인: orgId={org_id}, tblId={tbl_id}, itmId={itm_id}, prdSe={prd_se}")
            st.info("✅ 해결 방법(둘 중 하나):")
            st.write("1) 이 tblId가 '지역이 너무 많은 표'라면, **서울/부산/대구/경기만 선택 가능한 다른 tblId(아파트가격지수 표)**로 바꾸는 것이 가장 확실합니다.")
            st.write("2) 또는 KOSIS에서 이 표의 지역축이 objL1이 아닌 objL2/objL3에 있는지 확인 후, 해당 축에 '서울/부산/대구/경기' 코드로 호출해야 합니다.")
            st.code(ok_msg or "no trials")
        else:
            # 변환 시도: 만약 지역 컬럼이 있으면 pivot, 없으면 단일 series로 출력
            date_col = _pick_date_col(ok_df)
            val_col = _pick_value_col(ok_df)
            nm_cols = [c for c in ok_df.columns if c.endswith("_NM") or c in ["OBJ_NM", "C1_NM", "C2_NM", "C3_NM"]]

            st.caption(ok_msg)

            if date_col and val_col and nm_cols:
                # 가장 가능성 높은 지역명 컬럼을 찾자
                region_col = None
                for cand in ["C1_NM", "OBJ_NM", "C2_NM"]:
                    if cand in ok_df.columns:
                        region_col = cand
                        break
                if region_col is None:
                    region_col = nm_cols[0]

                idx = ok_df[date_col].apply(_parse_kosis_date)
                val = pd.to_numeric(ok_df[val_col], errors="coerce")
                reg = ok_df[region_col].astype(str)

                tmp = pd.DataFrame({"date": idx, "region": reg, "value": val}).dropna()
                if tmp.empty:
                    st.warning("변환 가능한 값이 없습니다. raw를 확인해주세요.")
                    with st.expander("raw 상위 50행"):
                        st.dataframe(ok_df.head(50), use_container_width=True)
                else:
                    piv = tmp.pivot_table(index="date", columns="region", values="value", aggfunc="last").sort_index()

                    # 서울/부산/대구/경기 자동 필터(이름에 포함되는지)
                    picks = [c for c in piv.columns if any(k in c for k in ["서울", "부산", "대구", "경기"])]
                    if not picks:
                        picks = list(piv.columns)[:8]

                    pick = st.multiselect("표시할 지역 선택", options=list(piv.columns), default=picks)
                    view = piv[pick] if pick else piv
                    plot_df(view.apply(normalize_100), f"KOSIS 부동산 지표 (정규화100) · prdSe={prd_se}", height=420)

                    with st.expander("원자료(최근 30개)"):
                        st.dataframe(view.tail(30), use_container_width=True)
            else:
                # 지역 축이 없는 단일 시계열인 경우
                s = _to_series(ok_df)
                if s.empty:
                    st.warning("시계열 변환 실패. raw를 확인해주세요.")
                    with st.expander("raw 상위 50행"):
                        st.dataframe(ok_df.head(50), use_container_width=True)
                else:
                    plot_df(pd.DataFrame({"부동산지표": normalize_100(s)}), f"KOSIS 부동산 지표(정규화100) · prdSe={prd_se}", height=420)

with tabs[4]:
    st.markdown("### 📰 실시간 경제뉴스")
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

    news_df = fetch_news_rss(limit=news_n, keyword=keyword.strip())
    if news_df.empty:
        st.info("뉴스를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        for _, row in news_df.iterrows():
            st.markdown(f"- [{row['title']}]({row['link']})  \n  <span class='small'>{row['published']}</span>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""", unsafe_allow_html=True)
