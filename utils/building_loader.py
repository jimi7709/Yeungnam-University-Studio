import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import os

# ==========================================
# [설정] API 키 및 타겟 지역
# ==========================================
PUBLIC_DATA_KEY = "fba6973ac6f9aed36f2b30b7dcce1fa4f6bef6c6c26cb61aff47144cc68520e5" 
KAKAO_API_KEY = "5b71324d3e681cdeaa038e7725055998"
SIGUNGU_CODE = "47290"

# 동별 코드 (부적리: 리 단위 코드 적용)
TARGET_DONGS = {
    "조영동": "11800",
    "대동": "11500",
    "임당동": "11400",
    "부적리": "25621" 
}

# ==========================================
# [기능 1] 건축물대장 수집
# ==========================================
def get_building_list(dong_name, dong_code):
    url = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
    
    params = {
        "serviceKey": PUBLIC_DATA_KEY,
        "sigunguCd": SIGUNGU_CODE,
        "bjdongCd": dong_code,
        "numOfRows": 500,
        "pageNo": 1
    }

    # [핵심] 조영동은 블록 번지가 많아 필터 해제, 나머지는 대지(0)만 수집
    if dong_name != "조영동":
        params["platGbCd"] = "0"

    all_items = []
    
    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200: break
            
            try:
                root = ET.fromstring(response.content)
            except:
                break

            if root.findtext(".//resultCode") != "00": break
            items = root.findall(".//item")
            if not items: break 

            print(f"   [{dong_name}] {len(items)}개 로딩 중... (Page {params['pageNo']})")

            for item in items:
                # [필터] 조영동일 경우 '산' 번지 수동 제외
                if dong_name == "조영동":
                    plat_gb = item.findtext("platGbCd")
                    jibun_addr = item.findtext("platPlc") or ""
                    if plat_gb == "1" or jibun_addr.strip().startswith("산"):
                        continue
                
                # 주소 및 용도 추출
                full_address = item.findtext("newPlatPlc") or item.findtext("platPlc") or ""
                if not full_address: continue

                main_purps = item.findtext("mainPurpsCdNm")
                if main_purps in ["단독주택", "공동주택", "제2종근린생활시설", "제1종근린생활시설", "다가구주택"]:
                    all_items.append({
                        "건물명": item.findtext("bldNm"),
                        "주소": full_address,
                        "주용도": main_purps,
                        "세대수": int(item.findtext("hhldCnt") or 0),
                        "사용승인일": item.findtext("useAprDay"),
                        "법정동": dong_name
                    })
            
            params['pageNo'] += 1
            if params['pageNo'] > 50: break 

        except Exception as e:
            print(f"에러: {e}")
            break
            
    return all_items

# ==========================================
# [기능 2] 좌표 변환
# ==========================================
def get_coordinates(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, params={"query": address}, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if data['documents']:
                return float(data['documents'][0]['y']), float(data['documents'][0]['x'])
    except:
        pass
    return None, None

# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("🏗️ 건축물대장 수집 시작...")
    all_data = []
    
    for name, code in TARGET_DONGS.items():
        print(f"➡️ {name} 수집 시작")
        buildings = get_building_list(name, code)
        all_data.extend(buildings)
        time.sleep(0.5)

    if all_data:
        print(f"\n📍 좌표 변환 시작 (총 {len(all_data)}건)...")
        valid_data = []
        for i, b in enumerate(all_data):
            lat, lon = get_coordinates(b['주소'])
            if lat:
                b['lat'] = lat
                b['lon'] = lon
                valid_data.append(b)
            if (i+1) % 100 == 0:
                print(f"   ... {i+1}건 완료")
                time.sleep(0.2)

        # 저장
        save_dir = "room/data"
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        df = pd.DataFrame(valid_data)
        this_year = int(time.strftime("%Y"))
        df['노후도'] = df['사용승인일'].apply(lambda x: this_year - int(str(x)[:4]) if pd.notnull(x) and str(x)[:4].isdigit() else 0)
        
        df.to_csv(f"{save_dir}/buildings.csv", index=False, encoding="utf-8-sig")
        print(f"\n✅ 저장 완료! 총 {len(df)}개 건물")
    else:
        print("❌ 수집된 데이터가 없습니다.")