import folium
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from folium.plugins import HeatMap


def draw_map(clustered_df, block_stats, cctv_df=None, noise_df=None, conv_df=None, store_df=None, lamp_df=None):
    
    m = folium.Map(location=[35.8359, 128.7535], zoom_start=16, tiles='CartoDB positron')

    # (1) 영남대역
    folium.Marker([35.8359, 128.7535], tooltip="<b>영남대역</b>", icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(m)

    # ---------------------------------------------
    # [핵심] 블록 영역 시각화 (GeoJSON 활용)
    # ---------------------------------------------
    
    # 스타일 정의 (기본: 회색 / 호버: 진한 회색)
    style_function = lambda x: {
        'fillColor': '#808080',  # 회색
        'color': '#808080',      # 테두리도 회색
        'fillOpacity': 0.4,      # 반투명
        'weight': 1
    }
    
    highlight_function = lambda x: {
        'fillColor': '#505050',  # 진한 회색 (호버 시)
        'color': '#333333',      # 진한 테두리
        'fillOpacity': 0.7,      # 좀 더 불투명하게
        'weight': 3              # 테두리 강조
    }

    grouped = clustered_df.groupby('cluster')
    
    for cluster_id, group in grouped:
        # 블록 통계 정보
        stats = block_stats[block_stats['cluster'] == cluster_id].iloc[0]
        avg_rent = int(stats['월세'])
        avg_deposit = int(stats['보증금'])
        count = int(stats['건물명'])
        cctv_cnt = int(stats['cctv_count'])
        conv_cnt = int(stats['conv_count'])
        
        # 툴팁 내용 (HTML)
        tooltip_html = f"""
        <div style="font-family:sans-serif; width:160px;">
            <h4 style="margin:0; color: #333;">Block #{cluster_id}</h4>
            <hr style="margin:5px 0;">
            <b>월세:</b> {avg_rent}만원<br>
            <b>보증금:</b> {avg_deposit}만원<br>
            <b>건물:</b> {count}개<br>
            <b>CCTV:</b> {cctv_cnt}개<br>
            <b>편의점:</b> {conv_cnt}개
        </div>
        """
        
        # 좌표 추출
        points = group[['lat', 'lon']].values
        polygon_coords = [] # GeoJSON용 좌표 [[lon, lat], ...]
        
        # 1. 다각형 생성 시도 (Convex Hull)
        try:
            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                # GeoJSON은 [lon, lat] 순서임에 주의!
                polygon_coords = [[p[1], p[0]] for p in hull_points]
                # 폐곡선을 위해 시작점 추가
                polygon_coords.append(polygon_coords[0])
            else:
                raise QhullError("점 부족")
                
        except (QhullError, ValueError):
            # 2. 실패 시 사각형(Bounding Box) 생성 + 패딩(여유공간)
            lat_min, lat_max = points[:, 0].min(), points[:, 0].max()
            lon_min, lon_max = points[:, 1].min(), points[:, 1].max()
            
            # 패딩 (약 15m 정도 여유를 둠)
            pad = 0.00015
            
            polygon_coords = [
                [lon_min - pad, lat_min - pad], # 좌하
                [lon_max + pad, lat_min - pad], # 우하
                [lon_max + pad, lat_max + pad], # 우상
                [lon_min - pad, lat_max + pad], # 좌상
                [lon_min - pad, lat_min - pad]  # 좌하 (닫기)
            ]

        # 3. GeoJSON 데이터 생성
        geo_json_data = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            },
            "properties": {
                "cluster_id": int(cluster_id)
            }
        }
        
        # 4. 지도에 추가 (GeoJSON 레이어)
        folium.GeoJson(
            geo_json_data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=tooltip_html # 레이어 전체에 툴팁 적용
        ).add_to(m)

        # 5. 중심점에 블록 번호 텍스트 표시
        folium.map.Marker(
            [stats['lat'], stats['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10pt; font-weight: bold; color: #333; text-shadow: -1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;">{cluster_id}</div>'
            )
        ).add_to(m)

    # ---------------------------------------------
    # 시설 마커 (아이콘 유지)
    # ---------------------------------------------
    def add_markers_smart(df, default_color, default_icon):
        if df is not None and not df.empty:
            for idx, row in df.head(150).iterrows():
                name = row.get('name', '시설')
                ptype = row.get('type', '')
                color = default_color
                icon = default_icon
                
                if ptype == '술집': color, icon = 'purple', 'beer'
                elif ptype == '노래방': color, icon = 'pink', 'music'
                elif ptype == '음식점': color, icon = 'green', 'cutlery'
                elif ptype == '카페': color, icon = 'darkgreen', 'coffee'
                
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    tooltip=f"<b>{name}</b> ({ptype})",
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)

    if cctv_df is not None and not cctv_df.empty:
        for idx, row in cctv_df.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                tooltip="<b>CCTV</b>",
                icon=folium.Icon(color='blue', icon='video-camera', prefix='fa')
            ).add_to(m)

    if lamp_df is not None and not lamp_df.empty:
        heat_data = lamp_df[['lat', 'lon']].dropna().values.tolist()

        # FeatureGroup로 감싸면 LayerControl로 ON/OFF 가능
        lamp_layer = folium.FeatureGroup(name="가로등 밀도(HeatMap)", show=True)

        HeatMap(
            heat_data,
            name="가로등 밀도",
            radius=18,        # 조도 범위 느낌
            blur=12,
            min_opacity=0.25,
            max_zoom=18
        ).add_to(lamp_layer)
        lamp_layer.add_to(m)
            
    add_markers_smart(conv_df, 'orange', 'shopping-cart')
    add_markers_smart(noise_df, 'purple', 'question') 
    add_markers_smart(store_df, 'green', 'cutlery')

    # folium.LayerControl(collapsed=False).add_to(m)


    return m