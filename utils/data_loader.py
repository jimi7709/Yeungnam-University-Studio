import requests
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import os
import numpy as np 

# ==========================================
# [설정] API 키 및 경로
# ==========================================
MOLIT_API_KEY = "fba6973ac6f9aed36f2b30b7dcce1fa4f6bef6c6c26cb61aff47144cc68520e5"
KAKAO_API_KEY = "5b71324d3e681cdeaa038e7725055998"
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjQxOTY0MTQ1MTI0MDRlZGZiYWJlMWMxNTYzN2E0NDc2IiwiaCI6Im11cm11cjY0In0="

# 기준 경로 설정
UTILS_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(UTILS_DIR) 

# 폴더 경로 정의
DATA_DIR = os.path.join(BASE_DIR, "data")
ZICBANG_DIR = os.path.join(BASE_DIR, "zicbang")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 개별 파일 경로 정의 (일원화)
ZIGBANG_RAW_PATH = os.path.join(ZICBANG_DIR, "zigbang.csv")
ZIGBANG_FINAL_PATH = os.path.join(DATA_DIR, "zigbang_with_age.csv")
BUILDINGS_CSV_PATH = os.path.join(DATA_DIR, "buildings.csv")

CCTV_CSV_PATH = os.path.join(DATA_DIR, "cctv.csv")
NOISE_CSV_PATH = os.path.join(DATA_DIR, "noise.csv")
CONVENIENCE_CSV_PATH = os.path.join(DATA_DIR, "convenience.csv")
STORE_CSV_PATH = os.path.join(DATA_DIR, "store.csv")
LAMP_CSV_PATH = os.path.join(DATA_DIR, "lamp.csv")

TARGET_DONGS = ["조영동", "대동", "임당동", "부적리"]

# ==========================================
# 1. 국토부 실거래가 / CCTV (기존 유지)
# ==========================================

# utils/data_loader.py 에 추가할 내용

def calculate_distance(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@st.cache_data
def get_ors_walking_duration(start_coords, end_coords):
    # ORS API는 [경도, 위도] 순서를 사용함에 주의
    url = f"https://api.openrouteservice.org/v2/directions/foot-walking"
    headers = {
        'Authorization': ORS_API_KEY,
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8'
    }
    params = {
        'start': f"{start_coords[1]},{start_coords[0]}",
        'end': f"{end_coords[1]},{end_coords[0]}"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        # 소요 시간(초) 추출 후 분 단위로 변환
        duration_seconds = data['features'][0]['properties']['summary']['duration']
        return round(duration_seconds / 60)
    return 0

def get_realtime_zigbang_data():
    """
    1. zigbang_with_age.csv(최종본)가 있으면 바로 리턴
    2. 없으면 zigbang.csv(원본) + buildings.csv(건물)를 합쳐서 생성 후 리턴
    """
    # 1. 최종 파일이 이미 존재하는 경우
    if os.path.exists(ZIGBANG_FINAL_PATH):
        df = pd.read_csv(ZIGBANG_FINAL_PATH, encoding='utf-8-sig')
        return df.rename(columns={'위도': 'lat', '경도': 'lon'}, errors='ignore')

    # 2. 최종 파일이 없을 경우 병합 시작
    st.info("🔄 처음 실행 시 데이터 통합 작업(노후도 매칭)이 필요합니다. 잠시만 기다려주세요...")
    
    if not os.path.exists(ZIGBANG_RAW_PATH) or not os.path.exists(BUILDINGS_CSV_PATH):
        st.error(f"❌ 필수 데이터가 누락되었습니다.\n- 원본: {ZIGBANG_RAW_PATH}\n- 건물: {BUILDINGS_CSV_PATH}")
        st.stop()

    df_zig = pd.read_csv(ZIGBANG_RAW_PATH)
    df_bld = pd.read_csv(BUILDINGS_CSV_PATH)

    b_lats = df_bld['lat'].values
    b_lons = df_bld['lon'].values
    b_ages = df_bld['노후도'].values

    def match_age(row):
        dists = calculate_distance(row['위도'], row['경도'], b_lats, b_lons)
        min_idx = np.argmin(dists)
        return b_ages[min_idx] if dists[min_idx] <= 20 else 0

    df_zig['노후도'] = df_zig.apply(match_age, axis=1)
    
    # 통합 파일 저장
    df_zig.to_csv(ZIGBANG_FINAL_PATH, index=False, encoding='utf-8-sig')
    st.success(f"✅ 통합 완료! '{os.path.basename(ZIGBANG_FINAL_PATH)}' 파일이 생성되었습니다.")
    
    return df_zig.rename(columns={'위도': 'lat', '경도': 'lon'}, errors='ignore')

def get_cctv_data():
    if not os.path.exists(CCTV_CSV_PATH): return pd.DataFrame()
    try:
        try: df = pd.read_csv(CCTV_CSV_PATH, encoding='utf-8-sig')
        except: df = pd.read_csv(CCTV_CSV_PATH, encoding='cp949')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        df = df.rename(columns={'WGS84위도': 'lat', 'WGS84경도': 'lon', '위도': 'lat', '경도': 'lon'})
        if 'lat' in df.columns and 'lon' in df.columns:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            return df.dropna(subset=['lat', 'lon']).drop_duplicates(subset=['lat', 'lon'])
    except: pass
    return pd.DataFrame()

def get_lamp_data():
    if not os.path.exists(LAMP_CSV_PATH): return pd.DataFrame()
    try:
        df = pd.read_csv(LAMP_CSV_PATH, encoding='utf-8-sig')
        # 한글 컬럼명을 lat, lon으로 변경
        df = df.rename(columns={'위도': 'lat', '경도': 'lon'})
        if 'lat' in df.columns and 'lon' in df.columns:
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            return df.dropna(subset=['lat', 'lon'])
    except: pass
    return pd.DataFrame()

# ==========================================
# [수정] 카카오 API 내부 호출 함수 (저장 기능 제거, 데이터 리턴만)
# ==========================================
def _fetch_api(category_code=None, keyword=None, rect_str=None):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json" if keyword else "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"rect": rect_str, "page": 1, "size": 15}
    if category_code: params["category_group_code"] = category_code
    if keyword: params["query"] = keyword

    results = []
    while True:
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=3)
            if resp.status_code != 200: break
            data = resp.json()
            for doc in data.get('documents', []):
                results.append({
                    "name": doc['place_name'],
                    "lat": float(doc['y']),
                    "lon": float(doc['x']),
                    "category": doc.get('category_name', ''),
                    "url": doc['place_url']
                })
            if not data.get('meta', {}).get('is_end'):
                params['page'] += 1
                if params['page'] > 3: break
            else: break
        except: break
        time.sleep(0.1)
    return pd.DataFrame(results)

# ==========================================
# [핵심] 3. 그룹별 데이터 수집 및 병합 저장
# ==========================================
def _get_grouped_data(save_path, fetch_funcs, min_lat, max_lat, min_lon, max_lon, force_update=False):
    # 1. 파일 있으면 로드
    if os.path.exists(save_path) and not force_update:
        try:
            df = pd.read_csv(save_path, encoding='utf-8-sig')
            # 현재 화면 범위 필터링
            mask = (df['lat'] >= min_lat) & (df['lat'] <= max_lat) & \
                   (df['lon'] >= min_lon) & (df['lon'] <= max_lon)
            if len(df[mask]) > 0: return df[mask]
        except: pass

    # 2. API 호출 및 병합
    pad = 0.01
    rect_str = f"{min_lon-pad},{min_lat-pad},{max_lon+pad},{max_lat+pad}"
    
    dfs = []
    print(f"📡 데이터 수집 중... ({os.path.basename(save_path)})")
    
    for func_args, type_name in fetch_funcs:
        # func_args: {'keyword': '...'} or {'category_code': '...'}
        df = _fetch_api(rect_str=rect_str, **func_args)
        if not df.empty:
            df['type'] = type_name # 구분값 추가 (예: 술집, 노래방)
            dfs.append(df)
            
    if dfs:
        merged_df = pd.concat(dfs).drop_duplicates(subset=['lat', 'lon'])
        merged_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ 저장 완료: {save_path} ({len(merged_df)}개)")
        
        # 범위 필터링 반환
        mask = (merged_df['lat'] >= min_lat) & (merged_df['lat'] <= max_lat) & \
               (merged_df['lon'] >= min_lon) & (merged_df['lon'] <= max_lon)
        return merged_df[mask]
    
    return pd.DataFrame()

# --- 외부 호출 함수들 ---

def get_noise_data(min_lat, max_lat, min_lon, max_lon):
    """술집 + 노래방 -> noise.csv"""
    tasks = [
        ({'keyword': '술집'}, '술집'),
        ({'keyword': '노래방'}, '노래방')
    ]
    return _get_grouped_data(NOISE_CSV_PATH, tasks, min_lat, max_lat, min_lon, max_lon)

def get_convenience_data(min_lat, max_lat, min_lon, max_lon):
    """편의점 -> convenience.csv"""
    tasks = [
        ({'category_code': 'CS2'}, '편의점')
    ]
    return _get_grouped_data(CONVENIENCE_CSV_PATH, tasks, min_lat, max_lat, min_lon, max_lon)

def get_store_data(min_lat, max_lat, min_lon, max_lon):
    """음식점 + 카페 -> store.csv (상가 1층 추정용)"""
    tasks = [
        ({'category_code': 'FD6'}, '음식점'),
        ({'category_code': 'CE7'}, '카페')
    ]
    return _get_grouped_data(STORE_CSV_PATH, tasks, min_lat, max_lat, min_lon, max_lon)