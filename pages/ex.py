# pages/Explainability.py
# ------------------------------------------------------------
# 월세 설명력의 한계 분석 (B 분석)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# 0) 기본 설정
# =========================
st.set_page_config(layout="wide", page_title="SweetHome - 월세 설명력 분석")
st.title("📊 월세 설명력의 한계 분석")
st.caption(
    "환경 데이터(CCTV, 가로등, 소음원 등)로 월세를 어디까지 설명할 수 있는지, "
    "그리고 설명되지 않는 영역은 무엇인지 확인합니다."
)

# =========================
# 1) 데이터 로드
# =========================
DATA_PATH = "./data/block_stats.csv"

if not st.session_state.get("block_stats_loaded"):
    if not pd.io.common.file_exists(DATA_PATH):
        st.error("❌ block_stats.csv가 없습니다. Home 페이지를 먼저 실행하세요.")
        st.stop()

df = pd.read_csv(DATA_PATH)

# =========================
# 2) 분석에 사용할 변수 선택
# =========================
TARGET = "월세"
FEATURES = [
    "cctv_count",
    "lamp_count",
    "conv_count",
    "noise_count",
    "store_count",
]

df = df[[TARGET] + FEATURES].dropna().copy()

X = df[FEATURES]
y = df[TARGET]

# =========================
# 3) 선형 회귀 + 교차검증
# =========================
model = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

model.fit(X, y)
y_pred = model.predict(X)

# =========================
# 4) 핵심 지표
# =========================
st.subheader("📌 핵심 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 설명력 (R²)", f"{r2_score(y, y_pred):.3f}")
col2.metric("교차검증 평균 R²", f"{cv_scores.mean():.3f}")
col3.metric("설명되지 않은 비율", f"{1 - cv_scores.mean():.1%}")

st.markdown(
    """
**해석**
- R²는 *환경 변수로 월세 변동을 얼마나 설명할 수 있는지*를 의미합니다.  
- 값이 1에 가까울수록 설명이 잘 되고, 0에 가까울수록 설명이 어렵습니다.
"""
)

st.markdown("---")

# =========================
# 5) 실제값 vs 예측값
# =========================
st.subheader("📈 실제 월세 vs 예측 월세")

df_plot = pd.DataFrame({
    "실제 월세": y,
    "예측 월세": y_pred
})

fig_scatter = px.scatter(
    df_plot,
    x="실제 월세",
    y="예측 월세",
    title="CCTV 및 가로등개수로 인한 예측 월세 vs 실제 월세"
)
fig_scatter.add_shape(
    type="line",
    x0=y.min(), y0=y.min(),
    x1=y.max(), y1=y.max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "점선에 가까울수록 예측이 잘 맞은 경우이며, "
    "점선에서 멀수록 환경 변수로 설명하기 어려운 매물입니다."
)

st.markdown("---")

# =========================
# 6) 잔차(설명되지 않은 부분) 분석
# =========================
st.subheader("📉 설명되지 않은 월세(잔차) 분포")

residuals = y - y_pred

fig_res = px.histogram(
    residuals,
    nbins=40,
    title="환경 변수로 설명되지 않은 월세 차이"
)
fig_res.update_xaxes(title="실제 월세 - 예측 월세")
fig_res.update_yaxes(title="개수")

st.plotly_chart(fig_res, use_container_width=True)

st.caption(
    "이 분포는 환경 데이터만으로는 설명할 수 없는 영역을 의미합니다. "
    "해당 차이는 건물 상태, 신축 여부, 옵션, 관리비 등의 숨은 요인일 가능성이 큽니다."
)

st.markdown("---")

# =========================
# 7) 최종 결론
# =========================
st.subheader("🧠 최종 결론")

st.markdown(
    f"""
- 본 분석에서 환경 변수만으로 월세 변동의 **약 {cv_scores.mean():.1%}** 정도를 설명할 수 있었습니다.
- 나머지 **{1 - cv_scores.mean():.1%}**는 본 데이터에 포함되지 않은 요인에 의해 결정됩니다.
- 따라서 월세는 **환경 요인 + 비가시적 요인의 결합 결과**임을 확인할 수 있습니다.
"""
)
