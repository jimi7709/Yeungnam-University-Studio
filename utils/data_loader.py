import requests
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import os

# ==========================================
# [설정] API 키 및 경로
# ==========================================
MOLIT_API_KEY = "fba6973ac6f9aed36f2b30b7dcce1fa4f6bef6c6c26cb61aff47144cc68520e5"
KAKAO_API_KEY = "5b71324d3e681cdeaa038e7725055998"

DATA_DIR = r"./data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# [수정] 파일 경로 정리 (3개로 통합)
ROOM_CSV_PATH = os.path.join(DATA_DIR, "room.csv")
CCTV_CSV_PATH = os.path.join(DATA_DIR, "cctv.csv")
NOISE_CSV_PATH = os.path.join(DATA_DIR, "noise.csv")          # 술집 + 노래방
CONVENIENCE_CSV_PATH = os.path.join(DATA_DIR, "convenience.csv") # 편의점
STORE_CSV_PATH = os.path.join(DATA_DIR, "store.csv")           # 음식점 + 카페

TARGET_DONGS = ["조영동", "대동", "임당동", "부적리"]

# ==========================================
# 1. 국토부 실거래가 / CCTV (기존 유지)
# ==========================================
def fetch_one_month_data(lawd_cd, deal_ymd):
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent"
    params = {"serviceKey": MOLIT_API_KEY, "LAWD_CD": lawd_cd, "DEAL_YMD": deal_ymd, "numOfRows": 1000, "pageNo": 1}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200: return []
        root = ET.fromstring(response.content)
        if root.findtext(".//resultCode") != "000": return []
        data_list = []
        for item in root.findall(".//item"):
            if item.findtext("jibun", "").strip().startswith("산"): continue
            data_list.append({
                "법정동": item.findtext("umdNm", "").strip(),
                "건축년도": int(item.findtext("buildYear", "0").strip() or 0),
                "전용면적": float(item.findtext("totalFloorAr", 0)),
                "보증금": int(item.findtext("deposit", "0").replace(",", "")),
                "월세": int(item.findtext("monthlyRent", "0").replace(",", "")),
                "계약일": f"{item.findtext('dealYear')}-{item.findtext('dealMonth')}-{item.findtext('dealDay')}"
            })
        return data_list
    except: return []

def get_real_estate_data(lawd_cd="47290", months=24, force_update=False):
    if os.path.exists(ROOM_CSV_PATH) and not force_update:
        try: df = pd.read_csv(ROOM_CSV_PATH, encoding='utf-8-sig')
        except: df = pd.read_csv(ROOM_CSV_PATH, encoding='cp949')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        if '법정동' in df.columns:
            mask = df['법정동'].apply(lambda x: any(target in str(x) for target in TARGET_DONGS))
            return df[mask]
    
    date_list = [ (datetime.now() - relativedelta(months=i)).strftime("%Y%m") for i in range(months) ]
    all_data = []
    for ymd in date_list:
        all_data.extend(fetch_one_month_data(lawd_cd, ymd))
        time.sleep(0.05)
    
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
    mask = df['법정동'].apply(lambda x: any(target in str(x) for target in TARGET_DONGS))
    df_filtered = df[mask].copy()
    current_year = datetime.now().year
    df_filtered['노후도'] = df_filtered['건축년도'].apply(lambda x: current_year - x if x > 0 else 0)
    df_filtered.to_csv(ROOM_CSV_PATH, index=False, encoding='utf-8-sig')
    return df_filtered

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