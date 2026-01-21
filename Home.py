import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_folium import st_folium
from utils.slope_overlay import add_slope_overlay
from sklearn.cluster import DBSCAN 
from utils.map_utils import draw_map
from utils.data_loader import (
    get_real_estate_data, 
    get_cctv_data, 
    get_noise_data, 
    get_convenience_data, 
    get_store_data
)

def calculate_distance(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

st.set_page_config(layout="wide", page_title="SweetHome - 영남대 원룸")
st.title("📍 SweetHome: 영남대 자취방 지도 (블록 분석)")

FIXED_BOUNDS = {
    'min_lat': 35.835510, 'max_lat': 35.842292,
    'min_lon': 128.750314, 'max_lon': 128.760809
}

# 1. 데이터 로드
with st.spinner("주변 시설 데이터를 불러오는 중입니다..."):
    df_price = get_real_estate_data()
    cctv_df = get_cctv_data()
    
    # [수정] CCTV 데이터 범위 제한 (범위 밖 데이터 즉시 제거)
    if not cctv_df.empty:
        cctv_df = cctv_df[
            (cctv_df['lat'] >= FIXED_BOUNDS['min_lat']) & (cctv_df['lat'] <= FIXED_BOUNDS['max_lat']) &
            (cctv_df['lon'] >= FIXED_BOUNDS['min_lon']) & (cctv_df['lon'] <= FIXED_BOUNDS['max_lon'])
        ]
    
    noise_df = get_noise_data(**FIXED_BOUNDS)       
    convenience_df = get_convenience_data(**FIXED_BOUNDS) 
    store_df = get_store_data(**FIXED_BOUNDS)       

BUILD_PATH = r"./data/buildings.csv"
if not os.path.exists(BUILD_PATH):
    st.error("❌ buildings.csv가 없습니다.")
    st.stop()
df_build = pd.read_csv(BUILD_PATH)

# 2. 사이드바 (슬라이더 제거 및 고정값 적용)
with st.sidebar:
    st.header("🔍 설정")
    
    # [수정] 사용자가 조작 불가능하게 텍스트로만 안내
    with st.expander("🧩 분석 기준 (고정됨)", expanded=True):
        st.info("📌 **블록 기준:** 반경 17m / 최소 3개 건물")
        # 내부 변수로 고정
        block_eps = 17
        block_min = 3

    st.divider()
    st.subheader("시설 표시")
    show_cctv = st.toggle("CCTV (🎥)", value=True)
    show_conv = st.toggle("편의점 (🛒)", value=True)
    show_noise = st.toggle("소음원 (🍺/🎵)", value=False)
    show_store = st.toggle("상가 (🍴)", value=False)
    show_slope = st.toggle("경사도 (⛰️)", value=False)
    with st.expander("⛰️ 경사도 색상 해석", expanded=False):
        st.markdown("""
        **경사도 컬러맵 (viridis) 기준 설명**

        🟣 **보라 / 남색**  
        - 매우 완만한 경사 (평지)
        - 보행·자전거 이동에 최적

        🔵 **파랑 / 청록**  
        - 완만한 경사
        - 일반적인 주거 밀집 구간

        🟢 **초록**  
        - 체감되는 경사
        - 짐 이동 시 피로 증가

        🟡 **노랑**  
        - 상대적 급경사
        - 야간·우천 시 보행 불리

        ※ 경사도는 10~90 퍼센타일 기준으로 정규화됨
        """)

    st.divider()
    st.caption(f"📊 분석 대상 건물: {len(df_build)}개")

# 3. 데이터 전처리
df_build['노후도'] = pd.to_numeric(df_build['노후도'], errors='coerce').fillna(0)
df_build['lat'] = pd.to_numeric(df_build['lat'], errors='coerce')
df_build['lon'] = pd.to_numeric(df_build['lon'], errors='coerce')

df_build['법정동_정제'] = df_build['법정동'].astype(str).apply(lambda x: x.split()[-1].strip())
df_price['법정동_정제'] = df_price['법정동'].astype(str).apply(lambda x: x.split()[-1].strip())
df_price['보증금'] = pd.to_numeric(df_price['보증금'], errors='coerce').fillna(0)
df_price['월세'] = pd.to_numeric(df_price['월세'], errors='coerce').fillna(0)

price_stats = df_price.groupby('법정동_정제')[['보증금', '월세']].mean().reset_index()
merged_df = pd.merge(df_build, price_stats, on='법정동_정제', how='left').fillna(0)

merged_df = merged_df[
    (merged_df['lat'] >= FIXED_BOUNDS['min_lat']) & (merged_df['lat'] <= FIXED_BOUNDS['max_lat']) &
    (merged_df['lon'] >= FIXED_BOUNDS['min_lon']) & (merged_df['lon'] <= FIXED_BOUNDS['max_lon'])
].copy()

# 4. DBSCAN 군집화
if len(merged_df) > 0:
    coords = np.radians(merged_df[['lat', 'lon']].values)
    kms_per_radian = 6371.0088
    epsilon = (block_eps / 1000) / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=block_min, metric='haversine', algorithm='ball_tree').fit(coords)
    merged_df['cluster'] = db.labels_
    
    clustered_df = merged_df[merged_df['cluster'] != -1].copy()
    
    block_stats = clustered_df.groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean',
        '월세': 'mean',
        '보증금': 'mean',
        '건물명': 'count'
    }).reset_index()
    
    def count_nearby(center_lat, center_lon, target_df, radius=100):
        if target_df.empty: return 0
        dists = calculate_distance(center_lat, center_lon, target_df['lat'].values, target_df['lon'].values)
        return np.sum(dists <= radius)

    block_stats['cctv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], cctv_df), axis=1)
    block_stats['conv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], convenience_df), axis=1)

else:
    clustered_df = pd.DataFrame()
    block_stats = pd.DataFrame()

# 5. 지도 그리기
final_cctv = cctv_df if show_cctv else pd.DataFrame()
final_noise = noise_df if show_noise else pd.DataFrame()
final_conv = convenience_df if show_conv else pd.DataFrame()
final_store = store_df if show_store else pd.DataFrame()

if len(block_stats) > 0:
    st.success(f"📍 총 **{len(block_stats)}개**의 원룸 블록을 찾았습니다.")
    m = draw_map(clustered_df, block_stats, final_cctv, final_noise, final_conv, final_store)
    if show_slope and m is not None:
        add_slope_overlay(
            m,
            slope_tif_path="./data/yeungnam_slope_in_buffer.tif",
            opacity=0.35
        )
    if m:
        st_folium(m, width="100%", height=600)
    else:
        st.error("지도 생성 실패")
else:
    st.warning("블록을 형성할 수 없습니다.")