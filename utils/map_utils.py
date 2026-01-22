import folium
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from folium.plugins import HeatMap


def draw_map(clustered_df, block_stats, cctv_df=None, noise_df=None, conv_df=None, store_df=None, lamp_df=None, selected_id=None):
    
    m = folium.Map(location=[35.8389, 128.7552], zoom_start=15.5, tiles='CartoDB positron')

    # (1) 영남대역 (가장 아래 레이어)
    folium.Marker([35.8359, 128.7535], tooltip="<b>영남대역</b>", icon=folium.Icon(color='red', icon='flag', prefix='fa')).add_to(m)

    # ---------------------------------------------
    # (2) [순서 변경] 소음 반경 원(Circle) 먼저 그리기 (바닥 레이어)
    # ---------------------------------------------
    if noise_df is not None and not noise_df.empty:
        for idx, row in noise_df.iterrows():
            ptype = row.get('type', '')
            if ptype in ['술집', '노래방']:
                folium.Circle(
                    location=[row['lat'], row['lon']],
                    radius=100,          # 소음 영향권 100m
                    color=None,          # 테두리 없음
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.05,
                    interactive=False    # 마우스 이벤트 통과 (중요)
                ).add_to(m)

    # ---------------------------------------------
    # (3) 시설 마커들 그리기
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

    # CCTV 마커
    if cctv_df is not None and not cctv_df.empty:
        for idx, row in cctv_df.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                tooltip="<b>CCTV</b>",
                icon=folium.Icon(color='blue', icon='video-camera', prefix='fa')
            ).add_to(m)
    
    if lamp_df is not None and not lamp_df.empty:
        # 히트맵용 데이터 준비 [[lat, lon], ...]
        heat_data = lamp_df[['lat', 'lon']].values.tolist()
        
        # 파란색 히트맵 설정 (Gradient 조절)
        HeatMap(
            heat_data,
            name="가로등 밀도",
            radius=18,
            blur=10,
            min_opacity=0.3,
            # 진한 파란색에서 밝은 하늘색으로 이어지는 그라데이션
            gradient={0.2: '#0000FF', 0.5: '#00FFFF', 1.0: '#FFFFFF'} 
        ).add_to(m)            
    # 나머지 시설 마커들
    add_markers_smart(conv_df, 'orange', 'shopping-cart')
    add_markers_smart(noise_df, 'purple', 'question') 
    add_markers_smart(store_df, 'green', 'cutlery')

    # ---------------------------------------------
    # (4) [순서 변경] 블록 영역 시각화 (최상단 레이어)
    # ---------------------------------------------
    # 스타일 및 하이라이트 정의 (기존과 동일)
    style_function = lambda x: {'fillColor': '#808080', 'color': '#808080', 'fillOpacity': 0.4, 'weight': 1}
    highlight_function = lambda x: {'fillColor': '#505050', 'color': '#333333', 'fillOpacity': 0.7, 'weight': 3}

    grouped = clustered_df.groupby('cluster')
    
    for cluster_id, group in grouped:
        stats = block_stats[block_stats['cluster'] == cluster_id].iloc[0]

        # --- [여기서부터 수정 및 추가] ---
        # 현재 블록이 선택된 블록인지 확인
        is_selected = (selected_id is not None and int(cluster_id) == int(selected_id))
        
        # 선택 여부에 따른 색상 및 스타일 결정
        # 선택 시: 노란색 테두리와 짙은 노랑 채우기 / 미선택 시: 기존 회색
        f_color = '#FFFF00' if is_selected else '#808080'  # 채우기 색상
        l_color = '#FFD700' if is_selected else '#808080'  # 테두리 색상
        weight = 4 if is_selected else 1                  # 테두리 두께
        opacity = 0.6 if is_selected else 0.4              # 투명도

        # 개별 블록마다 스타일 함수를 새로 정의
        current_style = lambda x, fc=f_color, lc=l_color, w=weight, o=opacity: {
            'fillColor': fc,
            'color': lc,
            'fillOpacity': o,
            'weight': w
        }
        # 하이라이트(마우스 올렸을 때)는 공통으로 사용하거나 비슷하게 조절
        current_highlight = lambda x: {'fillColor': '#505050', 'color': '#333333', 'fillOpacity': 0.7, 'weight': 3}

        avg_rent, avg_deposit = int(stats['월세']), int(stats['보증금'])
        count, cctv_cnt, conv_cnt = int(stats['room_count']), int(stats['cctv_count']), int(stats['conv_count'])
        avg_age = round(stats['노후도'], 1)
        
        tooltip_html = f"""
        <div style="font-family:sans-serif; width:160px;">
            <h4 style="margin:0; color: #333;">Block #{cluster_id}</h4>
            <hr style="margin:5px 0;">
            <b>월세:</b> {avg_rent}만원<br>
            <b>보증금:</b> {avg_deposit}만원<br>
            <b>평균 노후도:</b> {avg_age}<br><b>현재 매물:</b> {count}개<br>
            <b>CCTV:</b> {cctv_cnt}개<br>
            <b>편의점:</b> {conv_cnt}개
        </div>
        """
        

        points = group[['lat', 'lon']].values
        try:
            if len(points) >= 3:
                hull = ConvexHull(points)
                polygon_coords = [[p[1], p[0]] for p in points[hull.vertices]]
                polygon_coords.append(polygon_coords[0])
            else: raise QhullError("점 부족")
        except:
            lat_min, lat_max = points[:, 0].min(), points[:, 0].max()
            lon_min, lon_max = points[:, 1].min(), points[:, 1].max()
            pad = 0.00015
            polygon_coords = [[lon_min-pad, lat_min-pad], [lon_max+pad, lat_min-pad], [lon_max+pad, lat_max+pad], [lon_min-pad, lat_max+pad], [lon_min-pad, lat_min-pad]]

        geo_json_data = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon_coords]},
            "properties": {"cluster_id": int(cluster_id)}
        }
        
        # 블록 영역을 가장 마지막에 그려서 툴팁이 최상단에 오도록 함
        popup_html = f"""
        <div style="font-family:sans-serif; text-align:center; width:150px;">
            <h4 style="margin:0;">Block #{cluster_id}</h4>
            <p style="font-size:11px; color:gray;">매물 {count}개</p>
            <p style="font-size:12px;"><b>이 블록이 선택되었습니다.</b></p>
        </div>
        """

        folium.GeoJson(
            geo_json_data,
            style_function=current_style,
            highlight_function=current_highlight,
            tooltip=tooltip_html,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)

        # 블록 번호 마커
        folium.map.Marker(
            [stats['lat'], stats['lon']],
            icon=folium.DivIcon(html=f'<div style="font-size: 10pt; font-weight: bold; color: #333; text-shadow: -1px 0 white, 0 1px white, 1px 0 white, 0 -1px white;">{cluster_id}</div>')
        ).add_to(m)

    return m