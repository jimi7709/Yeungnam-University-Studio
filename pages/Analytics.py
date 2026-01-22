import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
import platform   # ← 이 줄 추가
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import os
from sklearn.model_selection import KFold, cross_val_score


# 🔽 바로 여기
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(page_title="데이터 분석 및 시각화", layout="wide")
st.title("데이터 분석 및 시각화")

# =========================
# 기본 설정
# =========================
STATION_LAT = 35.8363
STATION_LON = 128.7529

BASE_DIR = Path(__file__).resolve().parent.parent  # pages/Analytics.py 기준 -> 프로젝트 루트

ZIGBANG_PATH = BASE_DIR / "./data/zigbang.csv"
LAMPS_PATH   = BASE_DIR / "./data/lamp.csv"
CCTV_PATH    = BASE_DIR / "./data/cctv.csv"
BUILDINGS_PATH = BASE_DIR / "./data/buildings.csv"

# 업로드 환경 실패 시 fallback
if not ZIGBANG_PATH.exists():
    ZIGBANG_PATH = Path("/mnt/data/zigbang.csv")
if not LAMPS_PATH.exists():
    LAMPS_PATH = Path("/mnt/data/lampost.csv")
if not CCTV_PATH.exists():
    CCTV_PATH = Path("/mnt/data/cctv.csv")
if not BUILDINGS_PATH.exists():
    BUILDINGS_PATH = Path("/mnt/data/buildings.csv")

# =========================
# 사이드바 옵션
# =========================
RADIUS_M = st.sidebar.slider("원룸 기준 집계 반경(m)", 50, 800, 200, 50)

bbox_on = st.sidebar.checkbox("역 주변 bbox로 축소", value=True)
do_sample = st.sidebar.checkbox("성능을 위해 원룸(zigbang) 샘플링", value=True)

lat_delta = st.sidebar.slider("bbox 위도 범위(+-)", 0.01, 0.10, 0.04, 0.01)
lon_delta = st.sidebar.slider("bbox 경도 범위(+-)", 0.01, 0.15, 0.05, 0.01)

# 박스플롯 구간 수(기본 3)
BIN_Q = st.sidebar.selectbox("박스플롯 구간 수(qcut)", [3, 4, 5], index=0)

# 2D 그래프 색상 표현
color_mode = st.sidebar.selectbox(
    "2D 그래프 색상 기준(CCTV)",
    ["연속값(스케일)", "구간(low/mid/high)"],
    index=1
)

# =========================
# 유틸 함수
# =========================
def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949", low_memory=False)

def find_lat_lon_cols(cols):
    cols = list(cols)
    lat_keys = ["위도", "latitude", "lat", "LAT", "Latitude", "y", "Y"]
    lon_keys = ["경도", "longitude", "lon", "LON", "Longitude", "x", "X"]

    lat_col = next((c for c in cols if c in lat_keys), None)
    lon_col = next((c for c in cols if c in lon_keys), None)

    if lat_col is None:
        lat_col = next((c for c in cols if ("위도" in c) or ("latitude" in c.lower())), None)
    if lon_col is None:
        lon_col = next((c for c in cols if ("경도" in c) or ("longitude" in c.lower())), None)

    return lat_col, lon_col

def find_rent_col(cols):
    cols = list(cols)
    rent_candidates = ["월세", "rent", "월임대료", "월세(만원)", "월세(원)"]
    rent = next((c for c in cols if c in rent_candidates), None)
    if rent is None:
        rent = next((c for c in cols if ("월세" in c) or ("rent" in c.lower())), None)
    return rent

def find_age_col(cols):
    cols = list(cols)
    age_candidates = ["노후도", "연식", "건축년도", "준공년도", "사용승인일", "준공연도", "build_year", "year_built"]
    age = next((c for c in cols if c in age_candidates), None)
    if age is None:
        age = next((c for c in cols if ("노후" in c) or ("연식" in c) or ("건축" in c) or ("준공" in c) or ("year" in c.lower())), None)
    return age


def parse_monthly_rent_from_text(s):
    """'보증금/월세' 등 텍스트에서 월세만 최대한 숫자로 추출"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()

    if "/" in s:
        right = s.split("/")[-1].strip().replace(",", "")
        try:
            return float(right)
        except:
            return np.nan

    t = s.replace(",", "")
    nums = []
    cur = ""
    for ch in t:
        if ch.isdigit() or ch == ".":
            cur += ch
        else:
            if cur:
                nums.append(cur); cur = ""
    if cur:
        nums.append(cur)

    if not nums:
        return np.nan
    try:
        return float(nums[-1])
    except:
        return np.nan

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def count_points_within_radius(room_lat, room_lon, pts_lat_arr, pts_lon_arr, radius_m):
    cnt = 0
    for la, lo in zip(pts_lat_arr, pts_lon_arr):
        if haversine_m(room_lat, room_lon, la, lo) <= radius_m:
            cnt += 1
    return cnt

def make_bins(series: pd.Series, q: int):
    """qcut 우선, 실패하거나 unique 적으면 cut로 fallback"""
    s = series.copy()
    if s.nunique() >= q:
        try:
            return pd.qcut(s, q=q, labels=[f"q{i+1}" for i in range(q)])
        except ValueError:
            pass
    # fallback
    return pd.cut(s, bins=q, labels=[f"q{i+1}" for i in range(q)])

# =========================
# 데이터 로드
# =========================
for p, name in [(ZIGBANG_PATH, "zigbang.csv"), (LAMPS_PATH, "lampost_v2.csv"),
                 (CCTV_PATH, "cctv.csv"),(BUILDINGS_PATH, "buildings.csv"), ]:
    if not p.exists():
        st.error(f"{name}를 찾지 못했습니다: {p}")
        st.stop()

zig = read_csv_safely(ZIGBANG_PATH)
lamps = read_csv_safely(LAMPS_PATH)
cctv = read_csv_safely(CCTV_PATH)
buildings = read_csv_safely(BUILDINGS_PATH)

z_lat, z_lon = find_lat_lon_cols(zig.columns)
l_lat, l_lon = find_lat_lon_cols(lamps.columns)
c_lat, c_lon = find_lat_lon_cols(cctv.columns)

if z_lat is None or z_lon is None:
    st.error("zigbang.csv에서 위도/경도 컬럼을 찾지 못했습니다.")
    st.stop()
if l_lat is None or l_lon is None:
    st.error("lampost_v2.csv에서 위도/경도 컬럼을 찾지 못했습니다.")
    st.stop()
if c_lat is None or c_lon is None:
    st.error("cctv.csv에서 위도/경도 컬럼을 찾지 못했습니다.")
    st.stop()

rent_col = find_rent_col(zig.columns)
if rent_col is None:
    st.warning("zigbang.csv에서 '월세' 컬럼을 자동으로 못 찾았습니다. 수동 선택하세요.")
    rent_col = st.selectbox("월세 컬럼 선택", options=list(zig.columns))

# 숫자 변환
zig[z_lat] = pd.to_numeric(zig[z_lat], errors="coerce")
zig[z_lon] = pd.to_numeric(zig[z_lon], errors="coerce")

rent_numeric = pd.to_numeric(zig[rent_col], errors="coerce")
if rent_numeric.notna().mean() < 0.5:
    zig["월세_파싱"] = zig[rent_col].apply(parse_monthly_rent_from_text)
    use_rent_col = "월세_파싱"
else:
    zig[rent_col] = rent_numeric
    use_rent_col = rent_col

lamps[l_lat] = pd.to_numeric(lamps[l_lat], errors="coerce")
lamps[l_lon] = pd.to_numeric(lamps[l_lon], errors="coerce")
cctv[c_lat] = pd.to_numeric(cctv[c_lat], errors="coerce")
cctv[c_lon] = pd.to_numeric(cctv[c_lon], errors="coerce")

# 결측 제거
zig = zig.dropna(subset=[z_lat, z_lon, use_rent_col]).copy()
lamps = lamps.dropna(subset=[l_lat, l_lon]).copy()
cctv = cctv.dropna(subset=[c_lat, c_lon]).copy()

# =========================
# bbox 축소
# =========================
if bbox_on:
    zig = zig[
        zig[z_lat].between(STATION_LAT - lat_delta, STATION_LAT + lat_delta) &
        zig[z_lon].between(STATION_LON - lon_delta, STATION_LON + lon_delta)
    ].copy()

    lamps = lamps[
        lamps[l_lat].between(STATION_LAT - lat_delta, STATION_LAT + lat_delta) &
        lamps[l_lon].between(STATION_LON - lon_delta, STATION_LON + lon_delta)
    ].copy()

    cctv = cctv[
        cctv[c_lat].between(STATION_LAT - lat_delta, STATION_LAT + lat_delta) &
        cctv[c_lon].between(STATION_LON - lon_delta, STATION_LON + lon_delta)
    ].copy()

if len(zig) == 0:
    st.error("전처리/bbox 이후 zigbang 표본이 0개입니다. bbox 범위/좌표를 확인하세요.")
    st.stop()

# 샘플링
sample_n = st.sidebar.slider("샘플 크기(zigbang)", 200, min(5000, len(zig)), min(1200, len(zig)), 100)
zig_an = zig.sample(sample_n, random_state=42).copy() if (do_sample and len(zig) > sample_n) else zig.copy()

# =========================
# 반경 내 개수 집계 (가로등 + CCTV 둘 다)
# =========================
lamp_lat_arr = lamps[l_lat].to_numpy()
lamp_lon_arr = lamps[l_lon].to_numpy()
cctv_lat_arr = cctv[c_lat].to_numpy()
cctv_lon_arr = cctv[c_lon].to_numpy()

lamp_col = f"lamp_count_{RADIUS_M}m"
cctv_col = f"cctv_count_{RADIUS_M}m"

with st.spinner(f"원룸별 반경 {RADIUS_M}m 내 가로등/CCTV 개수 계산 중..."):
    zig_an[lamp_col] = zig_an.apply(
        lambda r: count_points_within_radius(float(r[z_lat]), float(r[z_lon]),
                                             lamp_lat_arr, lamp_lon_arr, RADIUS_M),
        axis=1
    )
    zig_an[cctv_col] = zig_an.apply(
        lambda r: count_points_within_radius(float(r[z_lat]), float(r[z_lon]),
                                             cctv_lat_arr, cctv_lon_arr, RADIUS_M),
        axis=1
    )

# =========================
# 구간 만들기 (박스플롯용)
# =========================
zig_an["lamp_bin"] = make_bins(zig_an[lamp_col], q=BIN_Q)
zig_an["cctv_bin"] = make_bins(zig_an[cctv_col], q=BIN_Q)

# =========================
# 레이아웃: 탭 3개
# =========================
# tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
#     "월세 × 가로등 (Boxplot)",
#     "월세 × CCTV (Boxplot)",
#     "2D: 가로등 × 월세",
#     "월세 × 지하철역 거리",
#     "상관관계 분석",
#     "월세 × 생활 인프라(선택)",
#     "다변량 회귀 분석",
#     "월세 × 노후도(건물) 상관관계",
#     "tab9: 회귀(sklearn)",
#     "tab10: 2D(거리-월세, 색=노후도)",
#     "tab11: 통합 회귀(노후도+역세권+생활인프라)"
# ])
tab4,  tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "월세 × 지하철역 거리",
    "월세 × 생활 인프라(선택)",
    "다변량 회귀 분석",
    "월세 × 노후도(건물) 상관관계",
    "tab9: 회귀(sklearn)",
    "tab10: 2D(거리-월세, 색=노후도)",
    "tab11: 통합 회귀(노후도+역세권+생활인프라)",
    "월세 설명력의 한계 분석",
])


# # -------------------------
# # 탭1: 월세 × 가로등 박스플롯
# # -------------------------
# with tab1:
#     st.subheader("박스플롯: 가로등 밀도(구간)별 월세 분포")

#     groups = []
#     labels = []
#     sample_counts = {}

#     for label in zig_an["lamp_bin"].cat.categories if hasattr(zig_an["lamp_bin"], "cat") else sorted(zig_an["lamp_bin"].dropna().unique()):
#         vals = zig_an.loc[zig_an["lamp_bin"] == label, use_rent_col].dropna().tolist()
#         sample_counts[str(label)] = len(vals)
#         if len(vals) > 0:
#             groups.append(vals)
#             labels.append(str(label))

#     labels_with_n = [f"{lb}\n(n={sample_counts.get(lb,0)})" for lb in labels]

#     fig = plt.figure()
#     plt.boxplot(groups, labels=labels_with_n)
#     plt.xlabel(f"Lamp count bin (within {RADIUS_M}m)")
#     plt.ylabel("Monthly rent")
#     st.pyplot(fig)

#     MIN_N = 30
#     small_bins = [f"{k}({v})" for k, v in sample_counts.items() if v < MIN_N]
#     if small_bins:
#         st.warning("⚠️ 표본 수가 적은 구간이 있습니다 (권장: 30+): " + ", ".join(small_bins))

#     c1, c2, c3 = st.columns(3)
#     keys = list(sample_counts.keys())[:3]
#     c1.metric("bin1 n", sample_counts.get(keys[0], 0) if len(keys) > 0 else 0)
#     c2.metric("bin2 n", sample_counts.get(keys[1], 0) if len(keys) > 1 else 0)
#     c3.metric("bin3 n", sample_counts.get(keys[2], 0) if len(keys) > 2 else 0)

# # -------------------------
# # 탭2: 월세 × CCTV 박스플롯
# # -------------------------
# with tab2:
#     st.subheader("박스플롯: CCTV 밀도(구간)별 월세 분포")

#     groups = []
#     labels = []
#     sample_counts = {}

#     for label in zig_an["cctv_bin"].cat.categories if hasattr(zig_an["cctv_bin"], "cat") else sorted(zig_an["cctv_bin"].dropna().unique()):
#         vals = zig_an.loc[zig_an["cctv_bin"] == label, use_rent_col].dropna().tolist()
#         sample_counts[str(label)] = len(vals)
#         if len(vals) > 0:
#             groups.append(vals)
#             labels.append(str(label))

#     labels_with_n = [f"{lb}\n(n={sample_counts.get(lb,0)})" for lb in labels]

#     fig = plt.figure()
#     plt.boxplot(groups, labels=labels_with_n)
#     plt.xlabel(f"CCTV count bin (within {RADIUS_M}m)")
#     plt.ylabel("Monthly rent")
#     st.pyplot(fig)

#     MIN_N = 30
#     small_bins = [f"{k}({v})" for k, v in sample_counts.items() if v < MIN_N]
#     if small_bins:
#         st.warning("⚠️ 표본 수가 적은 구간이 있습니다 (권장: 30+): " + ", ".join(small_bins))

#     c1, c2, c3 = st.columns(3)
#     keys = list(sample_counts.keys())[:3]
#     c1.metric("bin1 n", sample_counts.get(keys[0], 0) if len(keys) > 0 else 0)
#     c2.metric("bin2 n", sample_counts.get(keys[1], 0) if len(keys) > 1 else 0)
#     c3.metric("bin3 n", sample_counts.get(keys[2], 0) if len(keys) > 2 else 0)

# # -------------------------
# # 탭3: 2D 그래프 (x=가로등, y=월세, 색=CCTV)
# # -------------------------
# with tab3:
#     st.subheader("2D: 가로등(x) × 월세(y), CCTV(q1/q2/q3)별 추세선(한 그래프)")

#     x = zig_an[lamp_col].astype(float)
#     y = zig_an[use_rent_col].astype(float)

#     # CCTV를 q1/q2/q3로 구간화
#     zig_an["cctv_q3"] = make_bins(zig_an[cctv_col].astype(float), q=3).astype(str)

#     # (선) 가로등을 bin으로 나눠서, 각 bin에서 월세 중앙값을 이어서 추세선 생성
#     # - 너무 촘촘하면 노이즈가 많고, 너무 거칠면 정보가 적어서 기본 12 추천
#     n_xbins = st.sidebar.slider("추세선용 가로등 bin 개수", 6, 30, 12, 1)

#     # 가로등 값이 모두 같거나 bin을 못 나누는 경우 대비
#     if x.nunique() < 2:
#         st.warning("가로등 개수가 거의 변하지 않아(유니크 값 부족) 추세선을 그리기 어렵습니다.")
#         st.stop()

#     # x bin 만들기: qcut 우선, 실패하면 cut
#     zig_an["_lamp_xbin"] = make_bins(x, q=min(n_xbins, max(2, x.nunique())))

#     fig = plt.figure()

#     # (선택) 산점도: 점이 너무 많으면 끄기
#     show_scatter = st.checkbox("산점도(점) 같이 보기", value=True)
#     if show_scatter:
#         # 점 색은 CCTV 연속값으로 두고 싶으면 c=zig_an[cctv_col]로 바꾸면 됨
#         plt.scatter(x, y, s=10, alpha=0.35)

#     # CCTV q1/q2/q3별 추세선(중앙값 라인)
#     for cat in sorted(zig_an["cctv_q3"].dropna().unique()):
#         sub = zig_an[zig_an["cctv_q3"] == cat].copy()

#         # lamp xbin별 월세 중앙값
#         line = (
#             sub.groupby("_lamp_xbin", observed=True)[use_rent_col]
#                .median()
#                .reset_index()
#         )

#         # x축 위치: 각 bin에서 가로등 값의 중앙값을 x좌표로 사용
#         x_mid = (
#             sub.groupby("_lamp_xbin", observed=True)[lamp_col]
#                .median()
#                .reset_index(name="x_mid")
#         )

#         line = line.merge(x_mid, on="_lamp_xbin", how="inner").dropna()

#         # 너무 표본 적은 구간이 많으면 선이 끊겨 보임 -> 최소 점수 체크
#         if len(line) < 2:
#             continue

#         # matplotlib 기본 색 사용(색 직접 지정 안 함)
#         plt.plot(line["x_mid"], line[use_rent_col], marker="o", linewidth=2, label=f"CCTV {cat}")

#     plt.xlabel(f"Lamp count (within {RADIUS_M}m)")
#     plt.ylabel("Monthly rent")
#     plt.legend(title="CCTV q3", loc="best")
#     #st.pyplot(fig)
#     st.plotly_chart(fig)
#     st.caption("※ 선은 CCTV q1/q2/q3별로, 가로등 bin마다 월세 중앙값을 이어 만든 추세선입니다.")

with tab4:
    st.subheader("산점도: 월세(y) × 지하철역_거리(m)(x)")

    # 1) '지하철역_거리(m)' 컬럼 찾기 (없으면 수동 선택)
    dist_candidates = ["지하철역_거리(m)", "지하철역거리(m)", "지하철역_거리", "역_거리(m)", "역거리(m)", "distance_to_station_m"]
    dist_col = next((c for c in zig.columns if c in dist_candidates), None)

    if dist_col is None:
        st.warning("zigbang.csv에서 '지하철역_거리(m)' 컬럼을 자동으로 찾지 못했습니다. 수동으로 선택하세요.")
        dist_col = st.selectbox("지하철 거리 컬럼 선택", options=list(zig.columns), key="tab4_dist_col")

    # 2) 전체 zig 기준으로 산점도 데이터 구성
    df_sc = zig[[dist_col, use_rent_col]].copy()
    df_sc[dist_col] = pd.to_numeric(df_sc[dist_col], errors="coerce")
    df_sc[use_rent_col] = pd.to_numeric(df_sc[use_rent_col], errors="coerce")
    df_sc = df_sc.dropna(subset=[dist_col, use_rent_col]).copy()

    st.caption(f"사용 표본 수: {len(df_sc)}개 (결측 제거 후)")

    # # 3) 산점도 옵션
    # x_max = float(df_sc[dist_col].max()) if len(df_sc) else 1000.0
    # x_limit = st.slider(
    #     "x축 최대값(거리 m) 제한",
    #     100.0,
    #     max(500.0, x_max),
    #     min(2000.0, x_max),
    #     50.0,
    #     key="tab4_x_limit"
    # )
    # df_plot = df_sc[df_sc[dist_col] <= x_limit].copy()
     # 🔒 x축 최대값 고정
    X_MAX = 950.0
    df_plot = df_sc[df_sc[dist_col] <= X_MAX].copy()

    show_trend = st.checkbox("추세선(단순 선형 회귀) 표시", value=True, key="tab4_trend")
    show_corr = st.checkbox("상관계수(Pearson) 표시", value=True, key="tab4_corr")

    # 4) Plotly 산점도 + (옵션) 추세선
    import plotly.express as px

    # trendline="ols"는 statsmodels가 설치돼 있으면 자동 OLS 추세선 생성
    trend_opt = "ols" if (show_trend and len(df_plot) >= 2) else None

    fig = px.scatter(
        df_plot,
        x=dist_col,
        y=use_rent_col,
        opacity=0.6,
        title="월세 × 지하철역 거리",
        trendline=trend_opt,
        labels={
            dist_col: "Distance to subway station (m)",
            use_rent_col: "Monthly rent"
        }
    )

    # 보기 편하게 약간 설정
    fig.update_traces(marker=dict(size=6))
    fig.update_yaxes(nticks=15)
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # 5) 상관계수(옵션)
    if show_corr and len(df_plot) >= 2:
        corr = df_plot[[dist_col, use_rent_col]].corr(method="pearson").iloc[0, 1]
        st.info(f"Pearson 상관계수 r = {corr:.4f}  (x=거리, y=월세)")

    st.caption("※ 거리가 멀수록 월세가 내려가는 경향이면 음(-)의 상관이 나오는 게 일반적입니다.")



# with tab4:
#     st.subheader("산점도: 월세(y) × 지하철역_거리(m)(x)")

#     # 1) '지하철역_거리(m)' 컬럼 찾기 (없으면 수동 선택)
#     dist_candidates = ["지하철역_거리(m)", "지하철역거리(m)", "지하철역_거리", "역_거리(m)", "역거리(m)", "distance_to_station_m"]
#     dist_col = next((c for c in zig.columns if c in dist_candidates), None)

#     if dist_col is None:
#         st.warning("zigbang.csv에서 '지하철역_거리(m)' 컬럼을 자동으로 찾지 못했습니다. 수동으로 선택하세요.")
#         dist_col = st.selectbox("지하철 거리 컬럼 선택", options=list(zig.columns))

#     # 2) 전체 zig 기준(=397개를 목표)으로 산점도 데이터 구성
#     df_sc = zig[[dist_col, use_rent_col]].copy()
#     df_sc[dist_col] = pd.to_numeric(df_sc[dist_col], errors="coerce")
#     df_sc[use_rent_col] = pd.to_numeric(df_sc[use_rent_col], errors="coerce")
#     df_sc = df_sc.dropna(subset=[dist_col, use_rent_col]).copy()

#     st.caption(f"사용 표본 수: {len(df_sc)}개 (결측 제거 후)")

#     # 3) 산점도 옵션
#     x_max = float(df_sc[dist_col].max()) if len(df_sc) else 1000.0
#     x_limit = st.slider("x축 최대값(거리 m) 제한", 100.0, max(500.0, x_max), min(2000.0, x_max), 50.0)
#     df_plot = df_sc[df_sc[dist_col] <= x_limit].copy()

#     show_trend = st.checkbox("추세선(단순 선형 회귀) 표시", value=True)
#     show_corr = st.checkbox("상관계수(Pearson) 표시", value=True)

#     # 4) 그래프
#     fig = plt.figure()
#     plt.scatter(df_plot[dist_col], df_plot[use_rent_col], s=12, alpha=0.6)

#     plt.xlabel("Distance to subway station (m)")
#     plt.ylabel("Monthly rent")

#     # 5) 추세선(옵션) - numpy polyfit
#     if show_trend and len(df_plot) >= 2:
#         x = df_plot[dist_col].to_numpy(dtype=float)
#         y = df_plot[use_rent_col].to_numpy(dtype=float)

#         # x가 모두 같은 경우 방지
#         if np.nanstd(x) > 0:
#             coef = np.polyfit(x, y, deg=1)   # y = a*x + b
#             a, b = coef[0], coef[1]
#             x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
#             y_line = a * x_line + b
#             plt.plot(x_line, y_line, linewidth=2, label=f"trend: y = {a:.4f}x + {b:.2f}")
#             plt.legend()

#     st.pyplot(fig)

#     # 6) 상관계수(옵션)
#     if show_corr and len(df_plot) >= 2:
#         corr = df_plot[[dist_col, use_rent_col]].corr(method="pearson").iloc[0, 1]
#         st.info(f"Pearson 상관계수 r = {corr:.4f}  (x=거리, y=월세)")

#     st.caption("※ 거리가 멀수록 월세가 내려가는 경향이면 음(-)의 상관이 나오는 게 일반적입니다.")

# with tab5:
#     st.subheader("상관관계 분석: 가로등(x) × 월세(y), CCTV q1/q2/q3별")

#     # 분석에 사용할 컬럼
#     x_col = lamp_col        # 가로등 개수
#     y_col = use_rent_col    # 월세
#     g_col = "cctv_q3"       # CCTV q1/q2/q3

#     # 결측 제거
#     df_corr = zig_an[[x_col, y_col, g_col]].dropna().copy()

#     st.caption(f"사용 표본 수: {len(df_corr)}개 (결측 제거 후)")

#     # -----------------------------
#     # 1) 그룹별 상관계수 계산
#     # -----------------------------
#     corr_rows = []

#     for g in sorted(df_corr[g_col].unique()):
#         sub = df_corr[df_corr[g_col] == g]

#         if len(sub) >= 5:  # 최소 표본 수 방어
#             pearson_r = sub[[x_col, y_col]].corr(method="pearson").iloc[0, 1]
#             spearman_r = sub[[x_col, y_col]].corr(method="spearman").iloc[0, 1]

#             corr_rows.append({
#                 "CCTV 그룹": g,
#                 "표본 수(n)": len(sub),
#                 "Pearson r (선형)": round(pearson_r, 4),
#                 "Spearman ρ (순위)": round(spearman_r, 4)
#             })

#     corr_df = pd.DataFrame(corr_rows)

#     st.markdown("### 📊 CCTV 그룹별 상관계수")
#     st.dataframe(corr_df, use_container_width=True)

#     # -----------------------------
#     # 2) 상관관계 산점도 + 회귀선
#     # -----------------------------
#     fig = plt.figure()

#     show_reg = st.checkbox("그룹별 선형 회귀선 표시", value=True)

#     for g in sorted(df_corr[g_col].unique()):
#         sub = df_corr[df_corr[g_col] == g]

#         # 산점도
#         plt.scatter(
#             sub[x_col],
#             sub[y_col],
#             s=18,
#             alpha=0.6,
#             label=f"CCTV {g} (n={len(sub)})"
#         )

#         # 회귀선 (상관관계 시각화용)
#         if show_reg and len(sub) >= 5:
#             x = sub[x_col].to_numpy(dtype=float)
#             y = sub[y_col].to_numpy(dtype=float)

#             if np.nanstd(x) > 0:
#                 a, b = np.polyfit(x, y, 1)
#                 x_line = np.linspace(x.min(), x.max(), 100)
#                 y_line = a * x_line + b
#                 plt.plot(x_line, y_line, linewidth=2)

#         # -----------------------------
#     # 2) 그래프 안에 상관계수 텍스트 표시
#     # -----------------------------
#     y_text_pos = 0.95  # 그래프 위쪽부터 아래로 내려오게
#     for i, row in corr_df.iterrows():
#         txt = (
#             f"{row['CCTV 그룹']} : "
#             f"Pearson r = {row['Pearson r (선형)']}, "
#             f"Spearman ρ = {row['Spearman ρ (순위)']}"
#         )
#         plt.gca().text(
#             0.02, y_text_pos,
#             txt,
#             transform=plt.gca().transAxes,
#             fontsize=10,
#             verticalalignment="top"
#         )
#         y_text_pos -= 0.07

#     plt.xlabel(f"Lamp count (within {RADIUS_M}m)")
#     plt.ylabel("Monthly rent")
#     plt.legend(title="CCTV group")
#     st.pyplot(fig)

#     # -----------------------------
#     # 3) 해석 가이드
#     # -----------------------------
#     st.info(
#         "Pearson r 해석 기준(일반적 가이드)\n"
#         "- |r| < 0.1 : 거의 상관 없음\n"
#         "- 0.1 ≤ |r| < 0.3 : 약한 상관\n"
#         "- 0.3 ≤ |r| < 0.5 : 중간 상관\n"
#         "- |r| ≥ 0.5 : 강한 상관\n\n"
#         "※ 상관관계는 인과관계를 의미하지 않습니다."
#     )

with tab6:
    st.subheader("상관관계 분석: 월세(y) × 생활 인프라 거리(x)")

    # =========================
    # 1) 분석 대상 변수 정의
    # =========================
    infra_cols = [
        "세탁소_거리(m)",
        "카페_거리(m)",
        "약국_거리(m)",
        "대형마트_거리(m)",
        "편의점_거리(m)",
        "버스정류장_거리(m)"
    ]

    # 실제 zigbang.csv에 존재하는 컬럼만 사용
    infra_cols = [c for c in infra_cols if c in zig.columns]

    if not infra_cols:
        st.error("zigbang.csv에 지정한 생활 인프라 거리 컬럼이 없습니다.")
        st.stop()

    # =========================
    # 2) UI: 클릭으로 변수 선택
    # =========================
    selected_col = st.selectbox(
        "📌 확인할 생활 인프라 거리 변수를 선택하세요",
        infra_cols,
        key="tab6_infra_select"
    )

    # =========================
    # 3) 데이터 준비
    # =========================
    x = pd.to_numeric(zig[selected_col], errors="coerce")
    y = pd.to_numeric(zig[use_rent_col], errors="coerce")

    df_plot = pd.DataFrame({"x": x, "y": y}).dropna()

    st.caption(f"사용 표본 수: {len(df_plot)}개")

    if len(df_plot) < 10:
        st.warning("표본 수가 부족하여 분석이 어렵습니다.")
        st.stop()

    # =========================
    # 4) 그래프 옵션
    # =========================
    show_reg = st.checkbox("선형 회귀선 표시", value=True, key="tab6_show_reg")

    # =========================
    # 5) Plotly 산점도 (+ 회귀선)
    # =========================
    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        opacity=0.6,
        title=f"월세 × {selected_col}",
        labels={"x": f"{selected_col} (m)", "y": "Monthly rent"},
    )

    fig.update_yaxes(nticks=15)
    # 점 크기 조절
    fig.update_traces(marker=dict(size=6))

    # 회귀선 (numpy polyfit로 직접 추가 → statsmodels 불필요)
    if show_reg and df_plot["x"].nunique() > 1:
        a, b = np.polyfit(df_plot["x"].to_numpy(), df_plot["y"].to_numpy(), 1)
        x_line = np.linspace(df_plot["x"].min(), df_plot["x"].max(), 100)
        y_line = a * x_line + b

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"trend: y = {a:.4f}x + {b:.2f}",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 6) 상관계수 계산
    # =========================
    pearson_r = df_plot.corr(method="pearson").iloc[0, 1]
    spearman_r = df_plot.corr(method="spearman").iloc[0, 1]

    st.info(
        f"📊 상관계수 결과\n\n"
        f"- Pearson r (선형 상관): **{pearson_r:.4f}**\n"
        f"- Spearman ρ (순위 상관): **{spearman_r:.4f}**\n\n"
        f"※ r < 0 : 거리가 가까울수록 월세가 높은 경향\n"
        f"※ r > 0 : 거리가 멀수록 월세가 높은 경향"
    )



# -------------------------
# 탭7: 다변량 회귀 분석 (scikit-learn)
# -------------------------
with tab7:
    st.subheader("다변량 회귀 분석 (scikit-learn)")

    # =========================
    # 1) 종속변수(y): 월세
    # =========================
    df = zig_an.copy()
    df["월세_y"] = pd.to_numeric(df[use_rent_col], errors="coerce")
    df = df.dropna(subset=["월세_y", z_lat, z_lon]).copy()

    # =========================
    # 2) 파생변수: 역세권거리
    # =========================
    df["역세권거리_m"] = df.apply(
        lambda r: haversine_m(float(r[z_lat]), float(r[z_lon]), STATION_LAT, STATION_LON),
        axis=1
    )

    # =========================
    # 3) 후보 설명변수 자동 구성
    # =========================
    candidate_cols = []

    # 가로등/CCTV count 컬럼이 있다면 추가
    if "lamp_col" in globals() and lamp_col in df.columns:
        candidate_cols.append(lamp_col)
    if "cctv_col" in globals() and cctv_col in df.columns:
        candidate_cols.append(cctv_col)

    # 거리 기본 포함 후보
    candidate_cols.append("역세권거리_m")

    # 숫자형 컬럼 중에서도 추가 후보 제공(너무 많으면 위험해서 상위 일부만)
    numeric_extra = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # 월세_y, 좌표, 거리, count 제외
    exclude = { "월세_y", z_lat, z_lon, "역세권거리_m" }
    if "lamp_col" in globals():
        exclude.add(lamp_col)
    if "cctv_col" in globals():
        exclude.add(cctv_col)

    numeric_extra = [c for c in numeric_extra if c not in exclude]
    # 너무 길면 앞부분만(원하면 늘려도 됨)
    numeric_extra = numeric_extra[:30]

    candidate_cols += numeric_extra
    # 중복 제거
    candidate_cols = list(dict.fromkeys(candidate_cols))

    st.caption("설명변수(X)를 선택하세요. (기본: 역세권거리 + 가로등/CCTV count)")

    selected_X = st.multiselect(
        "설명변수 선택",
        options=candidate_cols,
        default=[c for c in [lamp_col if "lamp_col" in globals() else None,
                             cctv_col if "cctv_col" in globals() else None,
                             "역세권거리_m"] if c is not None and c in candidate_cols]
    )

    if len(selected_X) == 0:
        st.warning("설명변수를 최소 1개 이상 선택해야 합니다.")
        st.stop()

    # =========================
    # 4) 데이터 정리 (X/y 결측 제거)
    # =========================
    work = df[["월세_y"] + selected_X].copy()
    # 숫자 변환
    for c in selected_X:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()

    st.write(f"✅ 회귀에 사용되는 표본 수: **{len(work)}**")

    if len(work) < 30:
        st.warning("표본 수가 적습니다(권장 30+). 샘플링 옵션/전처리를 조정해보세요.")
        st.dataframe(work.head(50))

    # =========================
    # 5) 학습/평가 split
    # =========================
    test_size = st.slider("테스트 비율(test_size)", 0.1, 0.5, 0.2, 0.05)
    random_state = 42

    X = work[selected_X].to_numpy()
    y = work["월세_y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # =========================
    # 6) 모델 선택 (OLS/Ridge/Lasso)
    # =========================
    model_type = st.selectbox("모델 선택", ["LinearRegression(OLS)", "Ridge", "Lasso"], index=0)

    # 표준화 옵션: 변수 스케일이 다를 때 유용
    use_scaler = st.checkbox("표준화(StandardScaler) 사용", value=True)

    alpha = None
    if model_type in ["Ridge", "Lasso"]:
        alpha = st.slider("정규화 강도(alpha)", 0.01, 50.0, 1.0, 0.01)

    if model_type == "LinearRegression(OLS)":
        base_model = LinearRegression()
    elif model_type == "Ridge":
        base_model = Ridge(alpha=alpha)
    else:
        base_model = Lasso(alpha=alpha)

    if use_scaler:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", base_model)
        ])
    else:
        model = base_model

    # =========================
    # 7) 학습
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # 8) 평가
    # =========================
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    r2_tr = r2_score(y_train, pred_train)
    r2_te = r2_score(y_test, pred_test)

    mae_te = mean_absolute_error(y_test, pred_test)
    rmse_te = math.sqrt(mean_squared_error(y_test, pred_test))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (train)", f"{r2_tr:.4f}")
    c2.metric("R² (test)", f"{r2_te:.4f}")
    c3.metric("MAE (test)", f"{mae_te:.3f}")
    c4.metric("RMSE (test)", f"{rmse_te:.3f}")

    # =========================
    # 9) 계수 출력
    # =========================
    # Pipeline이면 reg 단계에서 계수 꺼내기
    reg = model.named_steps["reg"] if hasattr(model, "named_steps") else model
    coef = reg.coef_
    intercept = reg.intercept_

    coef_df = pd.DataFrame({
        "feature": selected_X,
        "coef": coef
    }).sort_values("coef", ascending=False)

    st.markdown("### 회귀 계수(방향성 확인용)")
    st.write(f"Intercept: **{intercept:.4f}**")
    st.dataframe(coef_df)

        # =====================
    # p-value (statsmodels OLS) - OLS일 때만 출력
    # =====================
    if model_type == "LinearRegression(OLS)":
        st.markdown("### p-value (statsmodels OLS)")

        # d는 dropna된 회귀용 데이터프레임 (tab7에서 work 또는 d로 쓰는 것)
        # 여기서는 work를 기준으로:
        d_sm = work[["월세_y"] + selected_X].dropna().copy()
        y_sm = d_sm["월세_y"]
        X_sm = d_sm[selected_X]
        X_sm = sm.add_constant(X_sm)

        ols = sm.OLS(y_sm, X_sm).fit()

        pv_tbl = pd.DataFrame({
            "coef": ols.params,
            "p_value": ols.pvalues,
            "std_err": ols.bse,
            "t": ols.tvalues
        }).sort_values("p_value")

        st.dataframe(pv_tbl)

        st.caption("※ tab7의 예측 성능 평가는 train/test로 했고, p-value는 전체 표본 기반 OLS 추론 결과입니다.")
    else:
        st.info("Ridge/Lasso는 정규화 회귀라 p-value 해석이 모호합니다. p-value는 OLS(LinearRegression)에서만 제공합니다.")


    st.caption(
        "※ 표준화를 켠 경우 coef는 '표준화된 단위' 기준 영향력으로 해석하면 되고, "
        "표준화를 끈 경우 coef는 '원래 단위' 변화량(예: 거리 m, 개수 등)에 대한 영향으로 해석합니다."
    )

    # =========================
    # 10) 예측 vs 실제 산점도
    # =========================
    fig = plt.figure()
    plt.scatter(y_test, pred_test, alpha=0.7)
    plt.xlabel("실제 월세")
    plt.ylabel("예측 월세")
    plt.title("Test: 실제 vs 예측")
    plt.grid(True)
    st.pyplot(fig)

    # =========================
    # 11) 잔차 플롯
    # =========================
    residual = y_test - pred_test
    fig2 = plt.figure()
    plt.scatter(pred_test, residual, alpha=0.7)
    plt.axhline(0)
    plt.xlabel("예측 월세")
    plt.ylabel("잔차(실제-예측)")
    plt.title("Test: 잔차 플롯")
    plt.grid(True)
    st.pyplot(fig2)

    with st.expander("테스트셋 예측 결과 보기"):
        out = pd.DataFrame({
            "실제월세": y_test,
            "예측월세": pred_test,
            "잔차": residual
        })
        st.dataframe(out.head(200))

# -------------------------
# 탭8: 월세 × 노후도(건물) 상관관계
# -------------------------
with tab8:
    st.subheader("월세 × 노후도(건물) 상관관계 (위경도 매칭 기반)")

    # buildings에서 위도/경도 자동 탐색
    b_lat, b_lon = find_lat_lon_cols(buildings.columns)
    if b_lat is None or b_lon is None:
        st.error("buildings.csv에서 lat/lon(위도/경도) 컬럼을 찾지 못했습니다.")
        st.stop()

    # buildings에서 노후도(연식) 컬럼 탐색
    age_col = find_age_col(buildings.columns)
    if age_col is None:
        st.warning("buildings.csv에서 노후도(연식) 컬럼을 자동으로 못 찾았습니다. 수동 선택하세요.")
        age_col = st.selectbox(
            "노후도(연식) 컬럼 선택",
            options=list(buildings.columns),
            key="tab8_age_col"
        )

    # 숫자 변환
    buildings[b_lat] = pd.to_numeric(buildings[b_lat], errors="coerce")
    buildings[b_lon] = pd.to_numeric(buildings[b_lon], errors="coerce")
    buildings[age_col] = pd.to_numeric(buildings[age_col], errors="coerce")

    # 결측 제거
    b2 = buildings.dropna(subset=[b_lat, b_lon, age_col]).copy()

    # zig_an에서 월세/좌표만 사용
    z2 = zig_an.dropna(subset=[z_lat, z_lon, use_rent_col]).copy()

    # 좌표 반올림
    round_n = 4

    z2["_lat_r"] = z2[z_lat].round(round_n)
    z2["_lon_r"] = z2[z_lon].round(round_n)
    b2["_lat_r"] = b2[b_lat].round(round_n)
    b2["_lon_r"] = b2[b_lon].round(round_n)

    # 좌표 기준 merge
    merged_age = pd.merge(
        z2[["_lat_r", "_lon_r", use_rent_col]],
        b2[["_lat_r", "_lon_r", age_col]],
        on=["_lat_r", "_lon_r"],
        how="inner"
    ).rename(columns={use_rent_col: "월세", age_col: "노후도_raw"})

    st.write(f"✅ 매칭된 표본 수: **{len(merged_age)}**")

    if len(merged_age) < 5:
        st.warning("매칭된 표본이 너무 적습니다. 좌표 반올림 자리수를 조정하세요.")
        st.stop()

    # -------------------------
    # 노후도_raw → 노후도_연차 변환
    # -------------------------
    df_corr = merged_age.copy()
    df_corr["월세"] = pd.to_numeric(df_corr["월세"], errors="coerce")
    df_corr["노후도_raw"] = pd.to_numeric(df_corr["노후도_raw"], errors="coerce")
    df_corr = df_corr.dropna(subset=["월세", "노후도_raw"]).copy()

    CURRENT_YEAR = pd.Timestamp.today().year

    def to_age_years(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")

        # YYYYMMDD → 연도
        year_from_date = (x // 10000).where(x > 10000)

        # 건축년도
        year_like = x.where((x >= 1800) & (x <= CURRENT_YEAR + 1))

        year = year_from_date.combine_first(year_like)
        age_from_year = (CURRENT_YEAR - year).where(year.notna())

        # 이미 연차로 보이는 값
        age_like = x.where((x >= 0) & (x <= 120))

        return age_from_year.combine_first(age_like)

    df_corr["노후도_연차"] = to_age_years(df_corr["노후도_raw"])
    df_corr = df_corr.dropna(subset=["노후도_연차"]).copy()
    df_corr = df_corr[(df_corr["노후도_연차"] >= 0) & (df_corr["노후도_연차"] <= 120)]

    # 이상치 옵션
    remove_outlier = st.checkbox(
        "이상치 제거(월세 상/하위 1%)",
        value=False,
        key="tab8_outlier"
    )
    if remove_outlier and len(df_corr) >= 50:
        lo = df_corr["월세"].quantile(0.01)
        hi = df_corr["월세"].quantile(0.99)
        df_corr = df_corr[df_corr["월세"].between(lo, hi)]

    # -------------------------
    # ✅ 시각화용 필터 (노후도 0~40, 월세>0)
    # -------------------------
    df_plot = df_corr[
        (df_corr["월세"] > 0) &
        (df_corr["노후도_연차"] >= 0) &
        (df_corr["노후도_연차"] <= 40)
    ].copy()

    st.caption(
        f"시각화 표본 수: {len(df_plot)} | "
        f"연차 범위: {df_plot['노후도_연차'].min():.1f} ~ {df_plot['노후도_연차'].max():.1f}"
    )

    # -------------------------
    # 상관계수 (연차 기준)
    # -------------------------
    pearson = df_plot["월세"].corr(df_plot["노후도_연차"], method="pearson")
    spearman = df_plot["월세"].corr(df_plot["노후도_연차"], method="spearman")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pearson r", f"{pearson:.4f}")
    c2.metric("Spearman ρ", f"{spearman:.4f}")
    c3.metric("표본 수(n)", len(df_plot))

    st.caption("※ 노후도는 자동 판별 후 '연차(년)' 기준으로 분석합니다.")

    # -------------------------
    # Plotly 산점도 (x축 0~40 고정)
    # -------------------------
    import plotly.express as px

    show_trend = st.checkbox("추세선(OLS) 표시", value=True, key="tab8_trend")
    trend_opt = "ols" if (show_trend and len(df_plot) >= 2) else None

    fig = px.scatter(
        df_plot,
        x="노후도_연차",
        y="월세",
        opacity=0.6,
        trendline=trend_opt,
        title="월세 vs 노후도(연차) (0~40년)",
        labels={"노후도_연차": "노후도(연차, 년)", "월세": "월세"},
        hover_data={"노후도_raw": True, "노후도_연차": ':.1f', "월세": ':.1f'}
    )

    fig.update_yaxes(nticks=15)
    fig.update_xaxes(range=[0, 40], autorange=False)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=60, b=10))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("매칭 결과 미리보기(연차 변환 포함)"):
        st.dataframe(df_plot[["월세", "노후도_raw", "노후도_연차"]].head(50))

# =========================
# tab9/tab10 공통: "월세 + 거리 + 노후도" 데이터셋 만들기
# =========================
def find_age_col(cols):
    cols = list(cols)
    age_candidates = ["노후도", "연차", "건축년도", "준공년도", "준공연도", "build_year", "year_built"]
    age = next((c for c in cols if c in age_candidates), None)
    if age is None:
        age = next((c for c in cols if ("노후" in c) or ("연차" in c) or ("건축" in c) or ("준공" in c) or ("year" in c.lower())), None)
    return age


def build_rent_dist_age_df(zig_df: pd.DataFrame, buildings_df: pd.DataFrame, round_n: int) -> pd.DataFrame:
    """
    zig_df: zig_an (샘플링/전처리된 원룸 데이터)
    buildings_df: buildings.csv
    round_n: 좌표 반올림 자리수
    return: columns = ['월세', '역세권거리_m', '노후도']
    """
    # buildings lat/lon 찾기
    b_lat, b_lon = find_lat_lon_cols(buildings_df.columns)
    if b_lat is None or b_lon is None:
        st.error("buildings.csv에서 위도/경도(lat/lon) 컬럼을 찾지 못했습니다.")
        return pd.DataFrame()

    # 노후도(연차) 컬럼 찾기
    #age_col = find_age_col(buildings_df.columns)
    age_col = "노후도"

    if age_col is None:
        st.warning("buildings.csv에서 노후도(연차) 컬럼을 자동으로 못 찾았습니다. 수동 선택하세요.")
        age_col = st.selectbox("노후도(연차) 컬럼 선택", options=list(buildings_df.columns), key=f"age_col_sel_{round_n}")

    # 숫자화 + 결측 제거
    b2 = buildings_df.copy()
    b2[b_lat] = pd.to_numeric(b2[b_lat], errors="coerce")
    b2[b_lon] = pd.to_numeric(b2[b_lon], errors="coerce")
    b2[age_col] = pd.to_numeric(b2[age_col], errors="coerce")
    b2 = b2.dropna(subset=[b_lat, b_lon, age_col]).copy()

    z2 = zig_df.dropna(subset=[z_lat, z_lon, use_rent_col]).copy()
    z2["월세"] = pd.to_numeric(z2[use_rent_col], errors="coerce")
    z2 = z2.dropna(subset=["월세"]).copy()

    # 역세권 거리(m) 계산
    z2["역세권거리_m"] = z2.apply(
        lambda r: haversine_m(float(r[z_lat]), float(r[z_lon]), STATION_LAT, STATION_LON),
        axis=1
    )

    # 좌표 반올림 매칭 키
    z2["_lat_r"] = z2[z_lat].round(round_n)
    z2["_lon_r"] = z2[z_lon].round(round_n)
    b2["_lat_r"] = b2[b_lat].round(round_n)
    b2["_lon_r"] = b2[b_lon].round(round_n)

    merged = pd.merge(
        z2[["_lat_r", "_lon_r", "월세", "역세권거리_m"]],
        b2[["_lat_r", "_lon_r", age_col]],
        on=["_lat_r", "_lon_r"],
        how="inner"
    ).rename(columns={age_col: "노후도"})

    merged = merged.dropna(subset=["월세", "역세권거리_m", "노후도"]).copy()
    return merged


# -------------------------
# 탭9: 다변량 회귀 (scikit-learn)
# -------------------------
with tab9:
    st.subheader("다변량 회귀(sklearn): 월세 ~ 노후도(연차) + 역세권 거리")

    round_n = 4
    df_reg = build_rent_dist_age_df(zig_an, buildings, round_n=round_n)

    st.write(f"✅ 매칭된 표본 수: **{len(df_reg)}**")

    if len(df_reg) < 20:
        st.warning(
            "회귀 분석 표본이 적습니다(권장 30+).\n"
            "- 반올림 자리수(6→5) 조정\n"
            "- bbox 범위 확대\n"
            "- 거리 기반 매칭(KDTree) 고려"
        )
        st.dataframe(df_reg.head(50))
    else:
        # 거리 단위 선택
        #dist_unit = st.radio("역세권 거리 단위", ["m", "km"], index=1, horizontal=True, key="tab9_unit_skl")
        #df_reg["역세권거리"] = df_reg["역세권거리_m"] / 1000.0 if dist_unit == "km" else df_reg["역세권거리_m"]
                # =====================
        # 거리 단위 고정: meter
        # =====================
        dist_unit = "m"
        df_reg["역세권거리"] = df_reg["역세권거리_m"]

        st.caption("역세권 거리 단위: **meter(m) 고정**")

        # 이상치 옵션
        remove_outlier = st.checkbox("이상치 제거(월세 상/하위 1%)", value=True, key="tab9_outlier_skl")
        d = df_reg[["월세", "노후도", "역세권거리"]].copy()

        if remove_outlier and len(d) >= 80:
            lo = d["월세"].quantile(0.01)
            hi = d["월세"].quantile(0.99)
            d = d[d["월세"].between(lo, hi)]

        # =====================
        # 회귀 학습
        # =====================
        X = d[["노후도", "역세권거리"]].to_numpy()
        y = d["월세"].to_numpy()

        lr = LinearRegression()
        lr.fit(X, y)
        pred = lr.predict(X)

        # =====================
        # 평가 지표
        # =====================
        r2 = r2_score(y, pred)
        mae = mean_absolute_error(y, pred)
        rmse = math.sqrt(mean_squared_error(y, pred))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("표본 수(n)", len(d))
        c2.metric("R²", f"{r2:.4f}")
        c3.metric("MAE", f"{mae:.3f}")
        c4.metric("RMSE", f"{rmse:.3f}")

        # =====================
        # 계수 표시
        # =====================
        coef_age = lr.coef_[0]
        coef_dist = lr.coef_[1]
        intercept = lr.intercept_

        st.markdown("### 회귀식")
        st.write(f"월세 = {intercept:.4f} + ({coef_age:.4f})·노후도 + ({coef_dist:.4f})·역세권거리({dist_unit})")

        st.markdown("### 계수 해석(단위 주의)")
        st.write(
            f"- 노후도 계수: **{coef_age:.4f}** → 연차가 1 증가할 때 월세가 평균적으로 {coef_age:.4f} 만큼 변하는 방향\n"
            f"- 거리 계수: **{coef_dist:.4f}** → 역세권거리가 1{dist_unit} 증가할 때 월세가 평균적으로 {coef_dist:.4f} 만큼 변하는 방향"
        )

        # =====================
        # p-value (statsmodels OLS) - 같은 d 데이터로 추론용
        # =====================
        st.markdown("### p-value (statsmodels OLS, 추론용)")

        X_sm = d[["노후도", "역세권거리"]].copy()
        X_sm = sm.add_constant(X_sm)         # intercept
        y_sm = d["월세"].copy()

        ols = sm.OLS(y_sm, X_sm).fit()

        pv_tbl = pd.DataFrame({
            "coef": ols.params,
            "p_value": ols.pvalues,
            "std_err": ols.bse,
            "t": ols.tvalues
        })

        st.dataframe(pv_tbl)

        st.caption("※ p-value는 OLS 가정(선형성/등분산/독립 등) 하에서의 계수 유의성 검정 결과입니다.")


        # =====================
        # 예측 vs 실제 / 잔차 플롯
        # =====================
        fig1 = plt.figure()
        plt.scatter(y, pred, alpha=0.6)
        plt.xlabel("실제 월세")
        plt.ylabel("예측 월세")
        plt.title("실제 vs 예측 (suppress p-value)")
        plt.grid(True)
        #st.pyplot(fig1)

        residual = y - pred
        fig2 = plt.figure()
        plt.scatter(pred, residual, alpha=0.6)
        plt.axhline(0)
        plt.xlabel("예측 월세")
        plt.ylabel("잔차(실제-예측)")
        plt.title("잔차 플롯")
        plt.grid(True)
        #st.pyplot(fig2)

        with st.expander("학습 데이터 미리보기"):
            view = d.copy()
            view["예측월세"] = pred
            view["잔차"] = residual
            st.dataframe(view.head(100))

# -------------------------
# 탭10: 2D (거리-월세, 색=노후도)
# -------------------------
with tab10:
    st.subheader("2D 산점도: 역세권 거리(x) vs 월세(y), 색=노후도(연차)")

    round_n = 4
    df_plot = build_rent_dist_age_df(zig_an, buildings, round_n=round_n)

    st.write(f"✅ 매칭된 표본 수: **{len(df_plot)}**")

    if len(df_plot) < 5:
        st.warning("표본이 너무 적어서 2D 시각화가 어렵습니다.")
        st.dataframe(df_plot.head(50))
    else:
        # =====================
        # 거리 단위 고정: meter
        # =====================
        st.caption("역세권 거리 단위: **meter(m) 고정**")

        # 숫자화 + 결측 제거
        df_plot = df_plot.copy()
        df_plot["역세권거리_m"] = pd.to_numeric(df_plot["역세권거리_m"], errors="coerce")
        df_plot["월세"] = pd.to_numeric(df_plot["월세"], errors="coerce")
        df_plot["노후도"] = pd.to_numeric(df_plot["노후도"], errors="coerce")
        df_plot = df_plot.dropna(subset=["역세권거리_m", "월세", "노후도"]).copy()

        # =====================
        # ✅ x축 최대값 상수 고정
        # =====================
        X_MIN = 0
        X_MAX = 1000  # ← 여기서 원하는 최대값으로 고정 (예: 1500, 3000 등)

        import plotly.express as px

        fig = px.scatter(
            df_plot,
            x="역세권거리_m",
            y="월세",
            color="노후도",
            color_continuous_scale="Viridis",
            opacity=0.7,
            title="거리-월세 관계 (색=노후도/연차)",
            labels={
                "역세권거리_m": "역세권 거리 (m)",
                "월세": "월세",
                "노후도": "노후도(연차)"
            }
        )
        fig.update_yaxes(nticks=15)
        fig.update_xaxes(range=[X_MIN, X_MAX], autorange=False)
        fig.update_layout(height=520)

        st.plotly_chart(fig, use_container_width=True)

        # with st.expander("매칭 데이터 미리보기"):
        #     out = df_plot.copy()
        #     out["역세권거리_km"] = out["역세권거리_m"] / 1000.0
        #     st.dataframe(out[["월세", "노후도", "역세권거리_m", "역세권거리_km"]].head(100))


# -------------------------
# 탭11: 통합 다변량 회귀 (노후도 + 역세권거리 + 생활인프라)
# -------------------------
with tab11:
    st.subheader("통합 다변량 회귀: 월세 ~ 노후도(연차) + 역세권거리 + 생활 인프라 거리")

    # -----------------
    # 1) 노후도/거리 포함 데이터셋 만들기 (tab9에서 쓰던 함수 재사용)
    # -----------------
    round_n = 4

    # build_rent_dist_age_df는 tab9/tab10에서 만든 공통 함수가 이미 있다고 가정
    # 없다면, tab9에 넣었던 build_rent_dist_age_df / find_age_col 블록을 위쪽에 공통으로 두면 됨.
    base = build_rent_dist_age_df(zig_an, buildings, round_n=round_n)

    st.write(f"✅ 노후도 매칭된 표본 수(기본): **{len(base)}**")
    if len(base) < 30:
        st.warning("노후도 매칭 표본이 적습니다. 반올림 자리수(6→5) 또는 bbox 범위를 조정해보세요.")
        st.dataframe(base.head(50))
        st.stop()

    # -----------------
    # 2) 생활 인프라 거리 컬럼을 zig_an에서 가져와 base에 붙이기
    #    (좌표 반올림 키를 이용해 병합)
    # -----------------
    # zig_an에 있는 생활 인프라 거리 후보 자동 탐색: '*_거리(m)'
    infra_candidates = [c for c in zig_an.columns if ("거리" in c and "(m)" in c)]
    if len(infra_candidates) == 0:
        st.error("zig_an에 생활 인프라 거리 컬럼('*_거리(m)')이 없습니다. tab6/전처리에서 생성됐는지 확인하세요.")
        st.stop()

    default_infra = [c for c in [
        "편의점_거리(m)", "버스정류장_거리(m)", "약국_거리(m)",
        "카페_거리(m)", "대형마트_거리(m)", "세탁소_거리(m)"
    ] if c in infra_candidates]

    selected_infra = st.multiselect(
        "생활 인프라 거리 변수 선택",
        options=infra_candidates,
        default=default_infra,
        key="tab11_infra_select"
    )

    if len(selected_infra) == 0:
        st.warning("생활 인프라 변수를 최소 1개 이상 선택하세요.")
        st.stop()

    # zig_an 쪽에도 동일한 반올림 키를 만든 뒤 base와 merge
    ztmp = zig_an.copy()
    ztmp = ztmp.dropna(subset=[z_lat, z_lon, use_rent_col]).copy()
    ztmp["_lat_r"] = ztmp[z_lat].round(round_n)
    ztmp["_lon_r"] = ztmp[z_lon].round(round_n)

    # infra 숫자화
    for c in selected_infra:
        ztmp[c] = pd.to_numeric(ztmp[c], errors="coerce")

    # base에도 키가 없으므로 생성 (build_rent_dist_age_df에서 이미 _lat_r/_lon_r 제거했으면 여기서 다시 만듦)
    # 안전하게 base에도 키를 다시 만들기 위해, base는 lat/lon키가 없으니 "재매칭" 방식을 사용:
    # => base를 만들 때 사용했던 키 컬럼이 남아있지 않으면, base 생성 함수에서 _lat_r/_lon_r를 남기도록 수정하는 게 최선.
    #
    # 해결책: base 만들 때 _lat_r/_lon_r 포함하도록 아래처럼 다시 구성:
    # 여기서는 간단히 base를 다시 만들기 위해, build_rent_dist_age_df 대신 키 포함 버전을 한 번 더 만든다.

    # ---- 키 포함 base 재구성 (안전) ----
    # 1) tab9 방식으로 z2,b2를 다시 만들기 위해 최소 로직 복제
    b_lat, b_lon = find_lat_lon_cols(buildings.columns)
    age_col = find_age_col(buildings.columns)
    if age_col is None:
        st.warning("buildings.csv에서 노후도(연차) 컬럼을 자동으로 못 찾았습니다. 수동 선택하세요.")
        age_col = st.selectbox("노후도(연차) 컬럼 선택", options=list(buildings.columns), key="tab11_age_col_sel")

    b2 = buildings.copy()
    b2[b_lat] = pd.to_numeric(b2[b_lat], errors="coerce")
    b2[b_lon] = pd.to_numeric(b2[b_lon], errors="coerce")
    b2[age_col] = pd.to_numeric(b2[age_col], errors="coerce")
    b2 = b2.dropna(subset=[b_lat, b_lon, age_col]).copy()
    b2["_lat_r"] = b2[b_lat].round(round_n)
    b2["_lon_r"] = b2[b_lon].round(round_n)

    z2 = zig_an.copy()
    z2[z_lat] = pd.to_numeric(z2[z_lat], errors="coerce")
    z2[z_lon] = pd.to_numeric(z2[z_lon], errors="coerce")
    z2["월세"] = pd.to_numeric(z2[use_rent_col], errors="coerce")
    z2 = z2.dropna(subset=[z_lat, z_lon, "월세"]).copy()
    z2["_lat_r"] = z2[z_lat].round(round_n)
    z2["_lon_r"] = z2[z_lon].round(round_n)
    z2["역세권거리_m"] = z2.apply(
        lambda r: haversine_m(float(r[z_lat]), float(r[z_lon]), STATION_LAT, STATION_LON),
        axis=1
    )

    # 노후도 붙이기(키 포함)
    base_keyed = pd.merge(
        z2[["_lat_r", "_lon_r", "월세", "역세권거리_m"]],
        b2[["_lat_r", "_lon_r", age_col]],
        on=["_lat_r", "_lon_r"],
        how="inner"
    ).rename(columns={age_col: "노후도"})

    # 생활 인프라 붙이기
    base_all = pd.merge(
        base_keyed,
        ztmp[["_lat_r", "_lon_r"] + selected_infra],
        on=["_lat_r", "_lon_r"],
        how="left"
    )

    # -----------------
    # 3) 회귀용 데이터 정리
    # -----------------
    # 거리 단위 선택(역세권거리만)
    #dist_unit = st.radio("역세권 거리 단위", ["m", "km"], index=1, horizontal=True, key="tab11_unit")
    #base_all["역세권거리"] = base_all["역세권거리_m"] / 1000.0 if dist_unit == "km" else base_all["역세권거리_m"]
    dist_unit = "m"
    base_all["역세권거리"] = base_all["역세권거리_m"]
    st.caption("역세권 거리 단위: **m(미터) 고정**")
    # 사용할 변수 세팅
    X_cols = ["노후도", "역세권거리"] + selected_infra
    work = base_all[["월세"] + X_cols].copy()

    # 숫자화 & 결측 제거
    for c in X_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work["월세"] = pd.to_numeric(work["월세"], errors="coerce")
    work = work.dropna(subset=["월세"] + X_cols).copy()

    st.write(f"✅ 통합 회귀에 사용되는 표본 수: **{len(work)}**")
    if len(work) < 50:
        st.warning("통합 회귀 표본이 적습니다. (노후도 매칭 + 인프라 결측 제거로 표본이 줄어듦)")
        st.dataframe(work.head(50))

    # 이상치 옵션
    remove_outlier = st.checkbox("이상치 제거(월세 상/하위 1%)", value=True, key="tab11_outlier")
    if remove_outlier and len(work) >= 200:
        lo = work["월세"].quantile(0.01)
        hi = work["월세"].quantile(0.99)
        work = work[work["월세"].between(lo, hi)].copy()

    # -----------------
    # 4) Train/Test split + sklearn 회귀
    # -----------------
    #test_size = st.slider("테스트 비율(test_size)", 0.1, 0.5, 0.2, 0.05, key="tab11_test_size")
    test_size = 0.2
    st.caption("테스트 비율: **20% 고정 (Train 80% / Test 20%)**")
    X = work[X_cols].to_numpy()
    y = work["월세"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model_type = st.selectbox("모델 선택", ["LinearRegression(OLS)", "Ridge", "Lasso"], index=0, key="tab11_model")
    use_scaler = st.checkbox("표준화(StandardScaler) 사용", value=True, key="tab11_scaler")

    alpha = None
    if model_type in ["Ridge", "Lasso"]:
        alpha = st.slider("정규화 강도(alpha)", 0.01, 50.0, 1.0, 0.01, key="tab11_alpha")

    if model_type == "LinearRegression(OLS)":
        base_model = LinearRegression()
    elif model_type == "Ridge":
        base_model = Ridge(alpha=alpha)
    else:
        base_model = Lasso(alpha=alpha)

    model = Pipeline([("scaler", StandardScaler()), ("reg", base_model)]) if use_scaler else base_model
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    r2_tr = r2_score(y_train, pred_train)
    r2_te = r2_score(y_test, pred_test)
    mae_te = mean_absolute_error(y_test, pred_test)
    rmse_te = math.sqrt(mean_squared_error(y_test, pred_test))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (train)", f"{r2_tr:.4f}")
    c2.metric("R² (test)", f"{r2_te:.4f}")
    c3.metric("MAE (test)", f"{mae_te:.3f}")
    c4.metric("RMSE (test)", f"{rmse_te:.3f}")

    # -----------------
    # 5) 계수 출력
    # -----------------
    reg = model.named_steps["reg"] if hasattr(model, "named_steps") else model
    coef = reg.coef_
    intercept = reg.intercept_

    coef_df = pd.DataFrame({"feature": X_cols, "coef": coef}).sort_values("coef", ascending=False)

    st.markdown("### 회귀 계수(방향성 확인용)")
    st.write(f"Intercept: **{intercept:.4f}**")
    st.dataframe(coef_df)

    st.caption(
        "※ 표준화를 켠 경우 coef는 '표준화 단위' 영향력 비교에 적합합니다. "
        "표준화를 끈 경우 coef는 원래 단위(m 등) 변화량에 대한 월세 변화량으로 해석합니다."
    )

        # -----------------
    # 회귀 계수 막대그래프 (tab11) - Plotly
    # -----------------
    st.markdown("### 변수별 영향력 비교 (회귀 계수)")

    coef_plot_df = coef_df.copy().sort_values("coef")  # 작은 값 -> 큰 값 (가로 막대에 좋음)

    import plotly.express as px
    import plotly.graph_objects as go

    x_label = "회귀 계수 (표준화 기준)" if use_scaler else "회귀 계수"

    fig_bar = px.bar(
        coef_plot_df,
        x="coef",
        y="feature",
        orientation="h",
        title="통합 다변량 회귀: 변수별 월세 영향 방향/크기",
        labels={"coef": x_label, "feature": "변수"},
    )

    # 0 기준선 추가
    fig_bar.add_vline(x=0, line_width=2, line_color="black")

    # 레이아웃 다듬기
    fig_bar.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis=dict(title="변수"),
        xaxis=dict(title=x_label),
    )

    st.plotly_chart(fig_bar, use_container_width=True)



    # -----------------
    # 6) (선택) p-value: statsmodels가 있으면 출력
    # -----------------
    st.markdown("### p-value (statsmodels OLS, 선택)")
    try:
        import statsmodels.api as sm

        X_sm = work[X_cols].copy()
        X_sm = sm.add_constant(X_sm)
        y_sm = work["월세"].copy()

        ols = sm.OLS(y_sm, X_sm).fit()

        pv_tbl = pd.DataFrame({
            "coef": ols.params,
            "p_value": ols.pvalues,
            "std_err": ols.bse,
            "t": ols.tvalues
        }).sort_values("p_value")

        st.dataframe(pv_tbl)
        st.caption("※ p-value는 OLS 가정 하에서 각 계수가 0인지 검정한 결과입니다. (Ridge/Lasso는 p-value 해석이 모호)")
    except Exception as e:
        st.info(
            "statsmodels가 설치되어 있지 않아서 p-value를 표시할 수 없습니다.\n"
            "해결: VS Code 터미널에서 `python -m pip install statsmodels` 실행 후 재시작하세요.\n"
            f"(에러: {e})"
        )

    # -----------------
    # 7) 실제 vs 예측 / 잔차 플롯
    # -----------------
    fig = plt.figure()
    plt.scatter(y_test, pred_test, alpha=0.7)
    plt.xlabel("실제 월세")
    plt.ylabel("예측 월세")
    plt.title("Test: 실제 vs 예측 (통합 회귀)")
    plt.grid(True)
    #st.pyplot(fig)

    residual = y_test - pred_test
    fig2 = plt.figure()
    plt.scatter(pred_test, residual, alpha=0.7)
    plt.axhline(0)
    plt.xlabel("예측 월세")
    plt.ylabel("잔차(실제-예측)")
    plt.title("Test: 잔차 플롯 (통합 회귀)")
    plt.grid(True)
    #st.pyplot(fig2)


# -------------------------
# 탭12: 월세 설명력의 한계 분석 (Explainability)
# -------------------------
with tab12:
    st.subheader("📊 월세 설명력의 한계 분석")
    st.caption(
        "환경 데이터(CCTV, 가로등, 소음원 등)로 월세를 어디까지 설명할 수 있는지, "
        "그리고 설명되지 않는 영역은 무엇인지 확인합니다."
    )

    # =========================
    # 1) 데이터 로드
    # =========================
    DATA_PATH = "./data/block_stats.csv"

    if not os.path.exists(DATA_PATH):
        st.error("❌ block_stats.csv가 없습니다. Home/SAFE 페이지를 먼저 실행하세요.")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    # =========================
    # 2) 분석 변수 설정
    # =========================
    TARGET = "월세"
    FEATURES = [
        "cctv_count",
        "lamp_count",
        "conv_count",
        "noise_count",
        "store_count",
    ]

    missing = [c for c in [TARGET] + FEATURES if c not in df.columns]
    if missing:
        st.error(f"❌ 필요한 컬럼이 없습니다: {missing}")
        st.stop()

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
    st.markdown("### 📌 핵심 결과")

    col1, col2, col3 = st.columns(3)
    col1.metric("전체 설명력 (R²)", f"{r2_score(y, y_pred):.3f}")
    col2.metric("교차검증 평균 R²", f"{cv_scores.mean():.3f}")
    col3.metric("설명되지 않은 비율", f"{1 - cv_scores.mean():.1%}")

    st.markdown(
        """
**해석**
- R²는 *환경 변수로 월세 변동을 얼마나 설명할 수 있는지*를 의미합니다.  
- 값이 높을수록 환경 요인의 설명력이 크고, 낮을수록 다른 요인의 영향이 큽니다.
"""
    )

    st.divider()

    # =========================
    # 5) 실제값 vs 예측값
    # =========================
    st.markdown("### 📈 실제 월세 vs 예측 월세")

    df_plot = pd.DataFrame({
        "실제 월세": y,
        "예측 월세": y_pred
    })

    fig_scatter = px.scatter(
        df_plot,
        x="실제 월세",
        y="예측 월세",
        title="환경 변수 기반 예측 월세 vs 실제 월세",
        opacity=0.7
    )
    fig_scatter.add_shape(
        type="line",
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max(),
        line=dict(dash="dash", color="gray")
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.caption(
        "점선에 가까울수록 예측이 잘 맞은 경우이며, "
        "점선에서 멀수록 환경 변수만으로 설명하기 어려운 블록입니다."
    )

    st.divider()

    # =========================
    # 6) 잔차(설명되지 않은 부분) 분석
    # =========================
    st.markdown("### 📉 설명되지 않은 월세(잔차) 분포")

    residuals = y - y_pred

    fig_res = px.histogram(
        residuals,
        nbins=40,
        title="환경 변수로 설명되지 않은 월세 차이(잔차)"
    )
    fig_res.update_xaxes(title="실제 월세 − 예측 월세")
    fig_res.update_yaxes(title="블록 수")

    st.plotly_chart(fig_res, use_container_width=True)

    st.caption(
        "이 분포는 환경 데이터만으로는 설명할 수 없는 영역을 의미합니다. "
        "해당 차이는 신축 여부, 건물 옵션, 내부 상태, 관리비 등의 숨은 요인일 가능성이 큽니다."
    )

    st.divider()

    # =========================
    # 7) 최종 결론
    # =========================
    st.markdown("### 🧠 최종 결론")

    st.markdown(
        f"""
- 환경 변수(CCTV, 가로등, 소음원 등)만으로 월세 변동의  
  **약 {cv_scores.mean():.1%}** 정도를 설명할 수 있었습니다.
- 나머지 **{1 - cv_scores.mean():.1%}**는 본 데이터에 포함되지 않은 요인에 의해 결정됩니다.
- 즉, 월세는 **환경 요인 + 건물·내부·시장 요인의 복합 결과**임을 확인할 수 있습니다.
"""
    )
