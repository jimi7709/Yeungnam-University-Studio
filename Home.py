import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN 
from utils.map_utils import draw_map
from utils.data_loader import (
    get_realtime_zigbang_data, 
    get_cctv_data, 
    get_noise_data, 
    get_convenience_data, 
    get_store_data,
    get_lamp_data,
    get_ors_walking_duration
)

DESTINATION = [35.8337, 128.6843]       # 대구스마트시티센터
SUSUNG_STATION = [35.8427, 128.6799]    # 수성알파시티역
SUSUNG_BUS_STOP = [35.8420, 128.6813]   # 수성알파시티역 정거장
STADIUM_BUS_STOP = [35.8328, 128.6848]  # 경기장네거리1 정거장
YU_STATION = [35.8363, 128.7529]        # 영남대역

if 'calc_result' not in st.session_state:
    st.session_state.calc_result = None

if 'fixed_walk_times' not in st.session_state:
    with st.spinner("🚌 대중교통 환승 구간 정보를 계산 중..."):
        # 고정 구간은 최초 1회만 계산
        w2_fixed = get_ors_walking_duration(SUSUNG_STATION, SUSUNG_BUS_STOP)
        w3_fixed = get_ors_walking_duration(STADIUM_BUS_STOP, DESTINATION)
        
        # 값이 0으로 올 경우를 대비한 최소값(Safe-guard) 설정
        st.session_state.fixed_walk_times = {
            'w2': w2_fixed if w2_fixed > 0 else 2, 
            'w3': w3_fixed if w3_fixed > 0 else 4
        }    

def calculate_distance(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_details_for_visual(block_lat, block_lon):
    # 1. ORS 도보 계산
    walk_to_yu = get_ors_walking_duration([block_lat, block_lon], YU_STATION)
    walk_to_bus = get_ors_walking_duration(SUSUNG_STATION, SUSUNG_BUS_STOP)
    walk_to_center = get_ors_walking_duration(STADIUM_BUS_STOP, DESTINATION)
    
    # 2. 교통수단별 시간 (대기 시간 포함)
    subway_segment = 11 + 5  # 이동 11분 + 대기 5분
    bus_segment = 5 + 5      # 이동 5분 + 대기 5분
    
    # 3. 시각화(가로선)를 위한 구간별 데이터 구조화
    # 각 지점 도달 시점의 누적 시간을 계산하여 타임라인 생성
    timeline = [
        {"지점": "출발", "소요": 0},
        {"지점": "영남대역", "소요": walk_to_yu},
        {"지점": "수성알파시티역", "소요": subway_segment},
        {"지점": "스마트시티센터", "소요": walk_to_bus + bus_segment + walk_to_center}
    ]
    
    total_time = walk_to_yu + subway_segment + walk_to_bus + bus_segment + walk_to_center
    
    return {
        "total": total_time,
        "segments": {
            "도보_총": walk_to_yu + walk_to_bus + walk_to_center,
            "지하철(대기포함)": subway_segment,
            "버스(대기포함)": bus_segment
        },
        "timeline": timeline
    }

st.set_page_config(layout="wide", page_title="영남대 원룸추천")
st.title("📍 SweetHome: 영남대 자취방 지도 (블록 분석)")

FIXED_BOUNDS = {
    'min_lat': 35.835510, 'max_lat': 35.842292,
    'min_lon': 128.750314, 'max_lon': 128.760809
}

# 1. 데이터 로드
with st.spinner("실시간 매물 및 주변 시설 정보를 불러오는 중..."):
    # 자동 통합 로직이 포함된 함수 호출
    df_main = get_realtime_zigbang_data()
    cctv_df = get_cctv_data()
    lamp_df = get_lamp_data()
    
    # [수정] CCTV 데이터 범위 제한 (범위 밖 데이터 즉시 제거)
    if not cctv_df.empty:
        cctv_df = cctv_df[
            (cctv_df['lat'] >= FIXED_BOUNDS['min_lat']) & (cctv_df['lat'] <= FIXED_BOUNDS['max_lat']) &
            (cctv_df['lon'] >= FIXED_BOUNDS['min_lon']) & (cctv_df['lon'] <= FIXED_BOUNDS['max_lon'])
        ]
    
    if not lamp_df.empty:
        lamp_df = lamp_df[
            (lamp_df['lat'] >= FIXED_BOUNDS['min_lat']) & (lamp_df['lat'] <= FIXED_BOUNDS['max_lat']) &
            (lamp_df['lon'] >= FIXED_BOUNDS['min_lon']) & (lamp_df['lon'] <= FIXED_BOUNDS['max_lon'])
        ]

    noise_df = get_noise_data(**FIXED_BOUNDS)       
    convenience_df = get_convenience_data(**FIXED_BOUNDS) 
    store_df = get_store_data(**FIXED_BOUNDS)       
    
# --- [Section 3: 데이터 전처리] 수정 ---
# 1. 수집된 매물 중 지도 범위(영남대역 인근)에 있는 데이터만 추출
merged_df = df_main[
    (df_main['lat'] >= FIXED_BOUNDS['min_lat']) & (df_main['lat'] <= FIXED_BOUNDS['max_lat']) &
    (df_main['lon'] >= FIXED_BOUNDS['min_lon']) & (df_main['lon'] <= FIXED_BOUNDS['max_lon'])
].copy()

# 2. 분석을 위해 가격 및 노후도 데이터를 숫자로 변환 (안전장치)
merged_df['보증금'] = pd.to_numeric(merged_df['보증금'], errors='coerce').fillna(0)
merged_df['월세'] = pd.to_numeric(merged_df['월세'], errors='coerce').fillna(0)
merged_df['노후도'] = pd.to_numeric(merged_df['노후도'], errors='coerce').fillna(0)

# 2. 사이드바 (슬라이더 제거 및 고정값 적용)
# 2. 사이드바
with st.sidebar:
    # [추가] 가격 필터 슬라이더
    st.header("🔍 필터 설정")
    with st.expander("원룸 정보(블록)", expanded=False):
        deposit_range = st.slider(
            "평균 보증금 (만원)", 
            min_value=10, max_value=3000, 
            value=(10, 3000), step=50
        )
        rent_range = st.slider(
            "평균 월세 (만원)", 
            min_value=0, max_value=100, 
            value=(0, 100), step=5
        )
        age_range = st.slider(
            "평균 노후도", 
            min_value=0, max_value=100, 
            value=(0, 100), step=1
        )

    with st.expander(" 편의/안전", expanded=False):
        # 편의점 유무 (체크박스)
        need_conv = st.checkbox("100m 이내 편의점 필수", value=False)
        
        # CCTV 개수 (슬라이더: 0 ~ 10개)
        cctv_min = st.slider(
            "100m 이내 최소 CCTV 개수", 
            min_value=0, max_value=30, 
            value=0, step=1
        )
        lamp_min = st.slider(
            "100m 이내 최소 가로등 개수", 
            min_value=0, max_value=50, 
            value=0, step=1
        )

    # [추가] Expander 3: 생활 조건 필터
    with st.expander("생활 조건", expanded=False):
        subway_max = st.slider(
            "지하철역 도보 거리 (분)", 
            min_value=0, max_value=30, 
            value=30, step=1
        )
        # 소음원 개수 (슬라이더: 0 ~ 100개)
        noise_max = st.slider(
            "최대 소음원 수 (100m)", 
            min_value=0, max_value=50, 
            value=50, step=1
        )
        store_min = st.slider(
            "최소 상가 수 (100m)", 
            min_value=0, max_value=50, 
            value=0, step=1
        )

    st.divider()
    with st.expander("지도 표시 항목", expanded=True):
        show_cctv = st.toggle("CCTV (🎥)", value=False)
        show_lamp_heat = st.toggle("가로등 밀집도(🔥)", value=False) # 히트맵 토글 추가
        show_conv = st.toggle("편의점 (🛒)", value=False)
        show_noise = st.toggle("소음원 (🍺/🎵)", value=False)
        show_store = st.toggle("상가 (🍴)", value=False)
    
    st.divider()
    st.caption(f"📊 분석 대상 건물: {len(merged_df)}개")

# 4. DBSCAN 군집화
@st.cache_data
def get_clustered_block_stats(_df_build, _cctv, _lamp, _noise, _conv, _store):
    """
    DBSCAN 군집화 및 블록별 통계 계산은 데이터가 변하지 않는 한 
    처음 한 번만 실행하고 결과를 메모리에 저장합니다.
    """
    if len(_df_build) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # DBSCAN 설정
    block_eps = 19
    block_min = 1
    coords = np.radians(_df_build[['lat', 'lon']].values)
    kms_per_radian = 6371.0088
    epsilon = (block_eps / 1000) / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=block_min, metric='haversine', algorithm='ball_tree').fit(coords)
    _df_build['cluster'] = db.labels_
    
    clustered_df = _df_build[_df_build['cluster'] != -1].copy()
    if clustered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 블록 통계 계산
    block_stats = clustered_df.groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean',
        '월세': 'mean',
        '보증금': 'mean',
        '노후도': 'mean',
        '지하철역_도보(분)': 'mean',
        '매물번호': 'count'
    }).reset_index()
    block_stats = block_stats.rename(columns={'매물번호': 'room_count', '지하철역_도보(분)': 'subway_walk'})
    
    # 주변 시설 개수 계산 (가장 오래 걸리는 부분)
    def count_nearby(center_lat, center_lon, target_df, radius=100):
        if target_df.empty: return 0
        dists = calculate_distance(center_lat, center_lon, target_df['lat'].values, target_df['lon'].values)
        return np.sum(dists <= radius)

    block_stats['cctv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], _cctv), axis=1)
    block_stats['conv_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], _conv), axis=1)
    block_stats['noise_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], _noise, radius=100), axis=1)
    block_stats['store_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], _store), axis=1)
    block_stats['lamp_count'] = block_stats.apply(lambda row: count_nearby(row['lat'], row['lon'], _lamp), axis=1)

    return block_stats, clustered_df

with st.spinner("블록 분석 및 통계 계산 중..."):
    block_stats, clustered_df = get_clustered_block_stats(
        merged_df, cctv_df, lamp_df, noise_df, convenience_df, store_df
    )

# [유지] 슬라이더 값에 따른 필터링 (이 부분은 캐싱하지 않음 - 실시간 반응 필요)
if not block_stats.empty:
    filtered_block_stats = block_stats[
        (block_stats['보증금'] >= deposit_range[0]) & (block_stats['보증금'] <= deposit_range[1]) &
        (block_stats['월세'] >= rent_range[0]) & (block_stats['월세'] <= rent_range[1]) &
        (block_stats['노후도'] >= age_range[0]) & (block_stats['노후도'] <= age_range[1]) &
        (block_stats['cctv_count'] >= cctv_min) &
        (block_stats['lamp_count'] >= lamp_min) &
        (block_stats['noise_count'] <= noise_max) &
        (block_stats['store_count'] >= store_min) &
        (block_stats['subway_walk'] <= subway_max)
    ]
    
    # [추가] 편의점 필수 체크 시: 위에서 걸러진 데이터 중 편의점이 0개인 블록은 제외
    if need_conv:
        filtered_block_stats = filtered_block_stats[filtered_block_stats['conv_count'] > 0]
    
    # 2. 필터링된 블록들의 cluster ID 목록을 가져옴
    valid_cluster_ids = filtered_block_stats['cluster'].tolist()
    
    # 3. 지도에 표시할 개별 건물 데이터도 해당 블록 ID만 남김
    filtered_clustered_df = clustered_df[clustered_df['cluster'].isin(valid_cluster_ids)]
else:
    clustered_df = pd.DataFrame()
    block_stats = pd.DataFrame()
    # 데이터가 없을 경우를 대비해 필터링 변수도 초기화
    filtered_block_stats = pd.DataFrame()
    filtered_clustered_df = pd.DataFrame()

# 5. 지도 그리기
final_cctv = cctv_df if show_cctv else pd.DataFrame()
final_noise = noise_df if show_noise else pd.DataFrame()
final_conv = convenience_df if show_conv else pd.DataFrame()
final_store = store_df if show_store else pd.DataFrame()
# 5. 지도 그리기 섹션
final_lamps = lamp_df if show_lamp_heat else pd.DataFrame() # 추가

if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None


# [수정] filtered_block_stats를 기준으로 체크
if len(filtered_block_stats) > 0:
    total_rooms = int(filtered_block_stats['room_count'].sum())
    total_blocks = len(filtered_block_stats)

    # 2. 메시지 수정
    st.success(f"📍 조건에 맞는 매물 **{total_rooms}개**, 블록 **{total_blocks}개**를 찾았습니다!")
    col_left, col_right = st.columns([7, 3])


    with col_left:
        # 지도 출력
        m = draw_map(
            filtered_clustered_df, 
            filtered_block_stats, 
            final_cctv, final_noise, final_conv, final_store, final_lamps, selected_id=st.session_state.selected_cluster
        )
        # 나중에 클릭 이벤트를 잡기 위해 변수 output에 저장
        output = st_folium(m, width="100%", height=650, key="main_map")

# 2. 명시적으로 클릭된 객체의 ID 추출 (에러 방지 로직 포함)
        if output and "last_active_drawing" in output:
            drawing = output["last_active_drawing"]
            if drawing is not None:
                clicked_id = drawing.get("properties", {}).get("cluster_id")
                
                # 새로운 블록을 클릭했을 때만 세션 상태 업데이트 및 리런
                if clicked_id is not None and st.session_state.selected_cluster != clicked_id:
                    st.session_state.selected_cluster = clicked_id
                    st.rerun()

        
        if st.session_state.selected_cluster is not None:
            target_id = st.session_state.selected_cluster

            if "last_id" not in st.session_state or st.session_state.last_id != target_id:
                st.session_state.calc_result = None
                st.session_state.last_id = target_id


            with st.container(border=True):    
                col_title, col_btn = st.columns([7, 3])
                with col_title:
                    st.subheader(f"Block #{target_id} → 교육장까지 얼마나 걸릴까요?")
                
                with col_btn:
                    calculate_clicked = st.button("🚀 소요 시간 계산하기", use_container_width=True)

                # 2. 계산 로직 실행
                if calculate_clicked:
                # 가로로 긴 프로그레스 바 생성
                    progress_bar = st.progress(0)
                    status_text = st.empty() # 상태 메시지를 보낼 자리
                    
                    status_text.caption("⏳ 최적 도보 경로 분석 중...")
                    progress_bar.progress(30) # 30% 진행 표시
                    
                    # --- 실제 계산 로직 시작 ---
                    selected_block = filtered_block_stats[filtered_block_stats['cluster'] == target_id].iloc[0]
                    b_lat, b_lon = selected_block['lat'], selected_block['lon']
                    
                    # 도보 구간 계산 (ORS API)
                    w1 = get_ors_walking_duration([b_lat, b_lon], [35.8363, 128.7529])
                    progress_bar.progress(60) # 60% 진행 표시
                    
                    w2 = st.session_state.fixed_walk_times['w2']
                    w3 = st.session_state.fixed_walk_times['w3']
                    
                    subway_total, bus_total = 16, 10
                    total_min = w1 + subway_total + w2 + bus_total + w3
                    
                    progress_bar.progress(100) # 완료!
                    status_text.empty() # 메시지 삭제
                    progress_bar.empty() # 가로 바 삭제
                    # --- 실제 계산 로직 종료 ---

                    st.session_state.calc_result = {
                        'total': total_min, 'w1': w1, 'w2': w2, 'w3': w3,
                        'subway': subway_total, 'bus': bus_total
                    }
                    st.rerun()
                if st.session_state.calc_result:
                    res = st.session_state.calc_result
                    st.markdown(f"""
<div style="background-color: #ffffff; padding: 10px 15px; border-radius: 15px; border: 1px solid #ececec; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 10px;">
<h3 style="margin-top:0; text-align:center; font-family: 'Pretendard', sans-serif;">
총 소요 시간: <span style="color:#e74c3c; font-weight:800;">약 {res['total']}분</span>
</h3>
<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 20px 20px; position: relative;">
<div style="position: absolute; top: 15px; left: 10%; right: 10%; height: 2px; background-color: #e0e0e0; z-index: 1;"></div>
<div style="z-index: 2; text-align: center; width: 20%;">
<div style="width: 35px; height: 35px; background: #3498db; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🏠</div>
<div style="font-size: 13px; font-weight: 700; margin-top: 8px;">내 방</div>
<div style="font-size: 11px; color: #5d6d7e; background: #ebf5fb; border-radius: 10px; padding: 2px 5px; margin-top: 3px;">도보 {res['w1']}분</div>
</div>
<div style="z-index: 2; text-align: center; width: 20%;">
<div style="width: 35px; height: 35px; background: #2ecc71; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🚇</div>
<div style="font-size: 13px; font-weight: 700; margin-top: 8px;">영남대역</div>
<div style="font-size: 11px; color: #1d8348; background: #e9f7ef; border-radius: 10px; padding: 2px 5px; margin-top: 3px;">지하철 {res['subway']}분</div>
</div>
<div style="z-index: 2; text-align: center; width: 20%;">
<div style="width: 35px; height: 35px; background: #3498db; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🏃</div>
<div style="font-size: 13px; font-weight: 700; margin-top: 8px;">알파시티역</div>
<div style="font-size: 11px; color: #5d6d7e; background: #ebf5fb; border-radius: 10px; padding: 2px 5px; margin-top: 3px;">도보 {res['w2']}분</div>
</div>
<div style="z-index: 2; text-align: center; width: 20%;">
<div style="width: 35px; height: 35px; background: #f1c40f; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🚌</div>
<div style="font-size: 13px; font-weight: 700; margin-top: 8px;">버스정거장</div>
<div style="font-size: 11px; color: #9a7d0a; background: #fef9e7; border-radius: 10px; padding: 2px 5px; margin-top: 3px;">버스 {res['bus']}분</div>
</div>
<div style="z-index: 2; text-align: center; width: 20%;">
<div style="width: 35px; height: 35px; background: #e74c3c; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">🏁</div>
<div style="font-size: 13px; font-weight: 700; margin-top: 8px;">시티센터</div>
<div style="font-size: 11px; color: #922b21; background: #fdedec; border-radius: 10px; padding: 2px 5px; margin-top: 3px;">도보 {res['w3']}분</div>
</div>
</div>
<p style="text-align: center; color: #95a5a6; font-size: 11px; margin-top: 25px;">
* 지하철/버스 소요시간에는 평균 대기시간(5분)이 포함되어 있습니다.
</p>
</div>
""", unsafe_allow_html=True)
                st.markdown(f"""  """, unsafe_allow_html=True)

        else:
            st.info("💡 분석하고 싶은 블록을 지도에서 먼저 선택해 주세요.")

    with col_right:
        with st.container(border=True):
            st.subheader("블록 랭킹")
            
            # (1) UI 및 가중치 설정
            priority = st.radio(
                "가장 중요하게 생각하는 조건은?",
                ("🏠신축", "🛡️안전성", "🛒편의"),
                horizontal=True,
                key="rank_priority"
            )
            st.divider()

            if priority == "🏠신축":
                w_age, w_safety, w_conv = 70, 15, 15
            elif priority == "🛡️안전성": # 오타 수정: 🛡️ 안전성 -> 🛡️안전성 (공백 확인)
                w_age, w_safety, w_conv = 15, 70, 15
            else:
                w_age, w_safety, w_conv = 15, 15, 70

            # (2) [먼저] 데이터 점수 계산 로직 (계산이 먼저 와야 합니다)
            def normalize(series, reverse=False):
                if series.max() == series.min(): return series * 0 + 0.5
                norm = (series - series.min()) / (series.max() - series.min())
                return 1 - norm if reverse else norm

            ranking_df = filtered_block_stats.copy()

            score_age = normalize(ranking_df['노후도'], reverse=True)
            ranking_df['safety_total'] = ranking_df['cctv_count'] + ranking_df['lamp_count']
            score_safety = normalize(ranking_df['safety_total'])
            ranking_df['conv_total'] = ranking_df['conv_count'] + ranking_df['store_count']
            score_conv = normalize(ranking_df['conv_total'])

            ranking_df['total_score'] = (
                (score_age * w_age) + 
                (score_safety * w_safety) + 
                (score_conv * w_conv)
            )

            # (3) [그 다음] 정렬하여 top5 생성
            top3 = ranking_df.sort_values(by='total_score', ascending=False).head(3)

            st.write("🔍 **분석된 추천 순위**")

            # (4) [마지막] 계산된 top5를 사용하여 리스트 출력
            for i, (idx, row) in enumerate(top3.iterrows()):
                cluster_id = int(row['cluster'])
                score = round(row['total_score'], 1)
                
                if st.button(f"🥇 {i+1}위: Block #{cluster_id} ({score}점)", key=f"rank_{cluster_id}", use_container_width=True):
                    st.session_state.selected_cluster = cluster_id
                    st.rerun()

        with st.container(border=True):
            st.subheader("블록 매물 정보")
            if st.session_state.selected_cluster is not None:
                target_id = st.session_state.selected_cluster
                
                # [cite_start]해당 블록의 매물만 필터링 [cite: 1, 3]
                rooms_in_block = filtered_clustered_df[filtered_clustered_df['cluster'] == target_id]
                
                with st.container(border=True):
                    st.write(f"Block #{target_id} 매물 목록")
                    
                    if not rooms_in_block.empty:
                        # [cite_start]필요한 정보만 나열 (매물번호, 보증금, 월세 등) [cite: 1, 3]
                        st.dataframe(
                            rooms_in_block[['매물번호', '보증금', '월세', '층', '노후도']],
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("해당 블록에 조건에 맞는 매물이 없습니다.")
                    
                    col_btn_1, col_btn_2 = st.columns([6, 4])
                    with col_btn_2:
                        if st.button("선택 해제", use_container_width=True):
                            st.query_params.clear() # URL 파라미터 삭제
                            st.session_state.selected_cluster = None
                            st.rerun()
            else:
                # 블록이 선택되지 않았을 때 표시되는 안내 메시지
                st.info("👆 지도 또는 위에서 블록을 클릭해 주세요!")
                    
else:
    st.warning("선택하신 조건에 맞는 블록이 없습니다.")