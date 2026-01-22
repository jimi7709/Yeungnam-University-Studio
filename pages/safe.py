# pages/safe.py
# SAFE 전용 페이지 - Home 지도 구조 + SAFE 범죄주의구간 WMS(전체/절도/폭력) 토글

import streamlit as st
import pandas as pd
import numpy as np
import os

from streamlit_folium import st_folium
import folium
from folium.raster_layers import WmsTileLayer
from sklearn.cluster import DBSCAN

from utils.map_utils import draw_map
from utils.data_loader import (
    get_real_estate_data,
    get_cctv_data,
    get_noise_data,
    get_convenience_data,
    get_store_data,
)

# -----------------------------
# 거리 계산
# -----------------------------
def calculate_distance(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def count_nearby(center_lat, center_lon, target_df, radius=100):
    if target_df is None or target_df.empty:
        return 0
    dists = calculate_distance(center_lat, center_lon, target_df["lat"].values, target_df["lon"].values)
    return int(np.sum(dists <= radius))

def load_csv_any_encoding(path):
    # 프로젝트에서 한글 CSV 인코딩 문제 자주 나서 안전하게 처리
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="cp949")

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(layout="wide", page_title="SweetHome - SAFE")
st.title("🛡️ SweetHome: SAFE 범죄주의구간 오버레이")

FIXED_BOUNDS = {
    "min_lat": 35.835510, "max_lat": 35.842292,
    "min_lon": 128.750314, "max_lon": 128.760809
}

# -----------------------------
# 1) 데이터 로드
# -----------------------------
with st.spinner("주변 시설 데이터를 불러오는 중입니다..."):
    df_price = get_real_estate_data()
    cctv_df = get_cctv_data()

    # CCTV 범위 제한 + 안전한 빈 DF 처리
    if cctv_df is not None and not cctv_df.empty:
        # 혹시 타입이 문자열이면 숫자화
        cctv_df["lat"] = pd.to_numeric(cctv_df["lat"], errors="coerce")
        cctv_df["lon"] = pd.to_numeric(cctv_df["lon"], errors="coerce")
        cctv_df = cctv_df.dropna(subset=["lat", "lon"]).copy()

        cctv_df = cctv_df[
            (cctv_df["lat"] >= FIXED_BOUNDS["min_lat"]) & (cctv_df["lat"] <= FIXED_BOUNDS["max_lat"]) &
            (cctv_df["lon"] >= FIXED_BOUNDS["min_lon"]) & (cctv_df["lon"] <= FIXED_BOUNDS["max_lon"])
        ].copy()
    else:
        cctv_df = pd.DataFrame(columns=["lat", "lon"])

    noise_df = get_noise_data(**FIXED_BOUNDS)
    convenience_df = get_convenience_data(**FIXED_BOUNDS)
    store_df = get_store_data(**FIXED_BOUNDS)

BUILD_PATH = "./data/buildings.csv"
if not os.path.exists(BUILD_PATH):
    st.error("❌ ./data/buildings.csv가 없습니다.")
    st.stop()

df_build = load_csv_any_encoding(BUILD_PATH)

# -----------------------------
# 2) 사이드바
# -----------------------------
with st.sidebar:
    st.header("🔍 설정 (SAFE 페이지)")

    with st.expander("🧩 분석 기준 (고정)", expanded=True):
        st.info("📌 **블록 기준:** 반경 17m / 최소 3개 건물")
        block_eps = 17
        block_min = 3

    st.divider()
    st.subheader("시설 표시")
    show_cctv = st.toggle("CCTV (🎥)", value=True)
    show_conv = st.toggle("편의점 (🛒)", value=True)
    show_noise = st.toggle("소음원 (🍺/🎵)", value=False)
    show_store = st.toggle("상가 (🍴)", value=False)

    st.divider()
    st.subheader("SAFE (범죄주의구간)")
    show_safe_all = st.toggle("전체 (🛡️)", value=True)
    show_safe_theft = st.toggle("경범죄 (🟡)", value=False)
    show_safe_violn = st.toggle("중범죄 (🔴)", value=False)

    st.caption(f"📊 분석 대상 건물: {len(df_build)}개")

# -----------------------------
# 3) 전처리
# -----------------------------
# lat/lon 숫자화
if "lat" not in df_build.columns or "lon" not in df_build.columns:
    st.error("❌ buildings.csv에 lat/lon 컬럼이 없습니다.")
    st.write("현재 컬럼:", list(df_build.columns))
    st.stop()

df_build["lat"] = pd.to_numeric(df_build["lat"], errors="coerce")
df_build["lon"] = pd.to_numeric(df_build["lon"], errors="coerce")
df_build = df_build.dropna(subset=["lat", "lon"]).copy()

# ✅ 노후도 처리 (없으면 0 생성, 있으면 숫자화)
if "노후도" in df_build.columns:
    df_build["노후도"] = pd.to_numeric(df_build["노후도"], errors="coerce").fillna(0)
else:
    df_build["노후도"] = 0  # draw_map에서 요구하는 컬럼 보장

# bounds 필터
df_build = df_build[
    (df_build["lat"] >= FIXED_BOUNDS["min_lat"]) & (df_build["lat"] <= FIXED_BOUNDS["max_lat"]) &
    (df_build["lon"] >= FIXED_BOUNDS["min_lon"]) & (df_build["lon"] <= FIXED_BOUNDS["max_lon"])
].copy()

# 가격 데이터 전처리
if "법정동" not in df_build.columns:
    st.error("❌ buildings.csv에 '법정동' 컬럼이 없습니다.")
    st.write("현재 컬럼:", list(df_build.columns))
    st.stop()

df_build["법정동_정제"] = df_build["법정동"].astype(str).apply(lambda x: x.split()[-1].strip())

if df_price is None or df_price.empty:
    st.error("❌ 가격 데이터(df_price)가 비어있습니다. get_real_estate_data()를 확인하세요.")
    st.stop()

if "법정동" not in df_price.columns:
    st.error("❌ 가격 데이터에 '법정동' 컬럼이 없습니다.")
    st.write("현재 컬럼:", list(df_price.columns))
    st.stop()

df_price["법정동_정제"] = df_price["법정동"].astype(str).apply(lambda x: x.split()[-1].strip())
df_price["보증금"] = pd.to_numeric(df_price.get("보증금", 0), errors="coerce").fillna(0)
df_price["월세"] = pd.to_numeric(df_price.get("월세", 0), errors="coerce").fillna(0)

price_stats = df_price.groupby("법정동_정제")[["보증금", "월세"]].mean().reset_index()

# ✅ buildings + price 병합 (노후도 포함된 df_build가 베이스라 merged_df에도 노후도가 들어감)
merged_df = pd.merge(df_build, price_stats, on="법정동_정제", how="left").fillna(0)

if len(merged_df) == 0:
    st.warning("범위 내 데이터가 없습니다.")
    st.stop()

# -----------------------------
# 4) DBSCAN 군집화
# -----------------------------
coords = np.radians(merged_df[["lat", "lon"]].values)
kms_per_radian = 6371.0088
epsilon = (block_eps / 1000) / kms_per_radian

db = DBSCAN(
    eps=epsilon,
    min_samples=block_min,
    metric="haversine",
    algorithm="ball_tree"
).fit(coords)

merged_df["cluster"] = db.labels_
clustered_df = merged_df[merged_df["cluster"] != -1].copy()

if len(clustered_df) == 0:
    st.warning("블록을 형성할 수 없습니다. (eps/min_samples 조정 필요)")
    st.stop()

# ✅ block_stats에 '노후도' 포함 (KeyError 해결 포인트)
# block_stats = clustered_df.groupby("cluster").agg({
#     "lat": "mean",
#     "lon": "mean",
#     "노후도": "mean",
#     "월세": "mean",
#     "보증금": "mean",
#     "건물명": "count" if "건물명" in clustered_df.columns else "size"
# }).reset_index()
# ✅ block_stats 만들기 (room_count 생성 포함)
if "건물명" in clustered_df.columns:
    block_stats = (
        clustered_df.groupby("cluster")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            노후도=("노후도", "mean"),
            월세=("월세", "mean"),
            보증금=("보증금", "mean"),
            room_count=("건물명", "count"),   # ✅ 새 컬럼 생성
        )
        .reset_index()
    )
else:
    # '건물명'이 없으면, 행 개수(size)로 대체
    block_stats = (
        clustered_df.groupby("cluster")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            노후도=("노후도", "mean"),
            월세=("월세", "mean"),
            보증금=("보증금", "mean"),
            room_count=("lat", "size"),      # ✅ size는 아무 컬럼이나 가능(결측 없을 걸 추천)
        )
        .reset_index()
    )



# 혹시 건물명 컬럼 없어서 size로 들어온 경우 컬럼이 '건물명'이 아닐 수 있음 → 강제 정리
# if "건물명" not in block_stats.columns:
#     # 마지막 컬럼이 count로 들어왔을 가능성이 높음
#     last_col = block_stats.columns[-1]
#     block_stats = block_stats.rename(columns={last_col: "건물명"})

# 주변 시설 카운트
block_stats["cctv_count"] = block_stats.apply(
    lambda row: count_nearby(row["lat"], row["lon"], cctv_df, radius=100), axis=1
)
block_stats["conv_count"] = block_stats.apply(
    lambda row: count_nearby(row["lat"], row["lon"], convenience_df, radius=100), axis=1
)
block_stats["noise_count"] = block_stats.apply(
    lambda row: count_nearby(row["lat"], row["lon"], noise_df, radius=100), axis=1
)
block_stats["store_count"] = block_stats.apply(
    lambda row: count_nearby(row["lat"], row["lon"], store_df, radius=100), axis=1
)

# -----------------------------
# 5) 지도 그리기 + SAFE WMS
# -----------------------------
final_cctv = cctv_df if show_cctv else pd.DataFrame()
final_noise = noise_df if show_noise else pd.DataFrame()
final_conv = convenience_df if show_conv else pd.DataFrame()
final_store = store_df if show_store else pd.DataFrame()

st.success(f"📍 총 **{len(block_stats)}개**의 원룸 블록을 찾았습니다.")
m = draw_map(clustered_df, block_stats, final_cctv, final_noise, final_conv, final_store)

if m is not None:
    SERVICE_KEY = "LZBLDFG6-LZBL-LZBL-LZBL-LZBLDFG6JN"

    # 전체(기존 IF_0087)
    if show_safe_all:
        WMS_ALL = f"https://www.safemap.go.kr/openapi2/IF_0087_WMS?serviceKey={SERVICE_KEY}"
        WmsTileLayer(
            url=WMS_ALL,
            layers="A2SM_CRMNLHSPOT_TOT",
            styles="A2SM_CrmnlHspot_Tot_Tot",
            fmt="image/png",
            transparent=True,
            name="SAFE 범죄주의구간(전체)",
            overlay=True,
            control=True,
        ).add_to(m)

    # 절도(경범죄) IF_0084
    if show_safe_theft:
        WMS_THEFT = f"https://www.safemap.go.kr/openapi2/IF_0084_WMS?serviceKey={SERVICE_KEY}"
        WmsTileLayer(
            url=WMS_THEFT,
            layers="A2SM_CRMNLHSPOT_TOT",
            styles="A2SM_CrmnlHspot_Tot_Theft",
            fmt="image/png",
            transparent=True,
            name="경범죄 주의구간(절도)",
            overlay=True,
            control=True,
        ).add_to(m)

    # 폭력(중범죄) IF_0083
    if show_safe_violn:
        WMS_VIOLN = f"https://www.safemap.go.kr/openapi2/IF_0083_WMS?serviceKey={SERVICE_KEY}"
        WmsTileLayer(
            url=WMS_VIOLN,
            layers="A2SM_CRMNLHSPOT_TOT",
            styles="A2SM_CrmnlHspot_Tot_Violn",
            fmt="image/png",
            transparent=True,
            name="중범죄 주의구간(폭력)",
            overlay=True,
            control=True,
        ).add_to(m)

    if show_safe_all or show_safe_theft or show_safe_violn:
        folium.LayerControl(collapsed=True).add_to(m)

    st_folium(m, width="100%", height=650)
else:
    st.error("지도 생성 실패")
