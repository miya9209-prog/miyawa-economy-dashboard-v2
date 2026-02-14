import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
import feedparser
import re

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

h1 { font-size:1.7rem !important; font-weight:800; }
.small { color:#666; font-size:.92rem; }

.card {
 border:1px solid #eee; border-radius:16px;
 padding:12px; background:#fff;
 box-shadow:0 8px 20px rgba(0,0,0,.05)
}
.ct{font-size:.9rem;color:#666}
.kpi{font-size:1.2rem;font-weight:800}
.pos{color:#0a7b34} .neg{color:#b42318}
.footer{margin-top:40px;text-align:center;font-size:.85rem;color:#777}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def now(): return datetime.now()
def days(freq): return {"D":180,"W":900,"M":3650}[freq]

def kpi_delta(df):
    if df.empty or len(df)<2: return None,None,None
    last=df.iloc[-1]; prev=df.iloc[-2]
    return last, last-prev, (last-prev)/prev*100

def kpi_card(title,value,unit,delta=None):
    d=""
    cls=""
    if delta:
        cls="pos" if delta[1]>0 else "neg"
        d=f"{delta[1]:+.2f} ({delta[2]:+.2f}%)"
    st.markdown(f"""
    <div class="card">
      <div class="ct">{title}</div>
      <div class="kpi">{value} <span style="font-size:.9rem">{unit}</span></div>
      <div class="{cls}">{d}</div>
    </div>
    """, unsafe_allow_html=True)

def yf_close(sym,start):
    df=yf.download(sym,start=start,progress=False)
    if df.empty: return df
    return df["Close"]

def plot(df,title):
    fig=go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df[c],name=c))
    fig.update_layout(height=320,title=title)
    st.plotly_chart(fig,use_container_width=True)

# =========================
# Header
# =========================
l,r=st.columns([5,1.5])
with l:
    st.title("재테크 핵심지표 대시보드")
    st.markdown('<div class="small">주식 · 환율 · 금 · 유가 · 부동산 · 관심종목 · 경제뉴스</div>',unsafe_allow_html=True)
with r:
    if st.button("활용법 가이드"):
        st.session_state.show_guide=True

if st.session_state.get("show_guide"):
    st.info("""
    - 지표는 **방향성**을 보는 용도입니다  
    - 정규화 100 = 상대 강도 비교  
    - WTI·DXY·환율은 주식보다 **선행**
    - 부동산은 **주식 대비 후행**
    """)
    if st.button("닫기"):
        st.session_state.show_guide=False

# =========================
# Tabs
# =========================
tabs=st.tabs(["일간","주간","월간"])
freq_map={"일간":"D","주간":"W","월간":"M"}

for t in tabs:
    with t:
        freq=freq_map[t.label]
        start=(now()-timedelta(days=days(freq))).strftime("%Y-%m-%d")

        # =========================
        # Snapshot
        # =========================
        cols=st.columns(4)
        kospi=yf_close("^KS11",start)
        usd=yf_close("KRW=X",start)
        gold=yf_close("GC=F",start)
        wti=yf_close("CL=F",start)
        dxy=yf_close("DX-Y.NYB",start)

        gold_kr=(gold*usd*3.75/31.1035).dropna()

        with cols[0]:
            kpi_card("KOSPI",f"{kospi.iloc[-1]:.1f}","pt",kpi_delta(kospi))
        with cols[1]:
            kpi_card("USD/KRW",f"{usd.iloc[-1]:.1f}","원",kpi_delta(usd))
        with cols[2]:
            kpi_card("금 1돈",f"{gold_kr.iloc[-1]:,.0f}","원")
        with cols[3]:
            kpi_card("WTI",f"{wti.iloc[-1]:.2f}","USD/배럴")

        # =========================
        # 부동산
        # =========================
        st.subheader("🏠 주요 도시 아파트 매매가격지수")
        url="https://raw.githubusercontent.com/FinanceData/RealEstateKorea/main/apartment_price_index.csv"
        reb=pd.read_csv(url,index_col=0,parse_dates=True)
        cities=["서울","경기","부산","대구","인천","광주","대전","울산","세종"]
        reb=reb[cities]
        reb=reb/reb.iloc[0]*100
        plot(reb,"아파트 매매가격지수 (정규화 100)")

# =========================
# Footer
# =========================
st.markdown("""
<div class="footer">
© 미샵컴퍼니(MISHARP COMPANY). 무단 전재·복사·배포를 금합니다.<br>
© MISHARP COMPANY. Unauthorized reproduction, copying, or distribution is prohibited.
</div>
""",unsafe_allow_html=True)
