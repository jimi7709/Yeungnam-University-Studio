# utils/slope_overlay.py

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from io import BytesIO
import base64
import folium


def add_slope_overlay(
    folium_map,
    slope_tif_path,
    opacity=0.35
):
    """
    Folium 지도 위에 경사도(GeoTIFF)를 반투명 오버레이로 추가
    - Home.py 수정 없이 그대로 사용 가능
    - NoData/빈 픽셀/퍼센타일 깨짐 방지
    """

    dst_crs = "EPSG:4326"

    with rasterio.open(slope_tif_path) as src:
        # 1) 재투영 대상 격자 정의
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        # 2) dst를 NaN으로 초기화 (빈 영역/미채움 픽셀 안전)
        dst = np.full((height, width), np.nan, dtype=np.float32)

        # 원본 nodata
        src_nodata = src.nodata

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,        # ✅ 원본 nodata 전달
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,            # ✅ 목적지 nodata는 NaN
            resampling=Resampling.bilinear
        )

        # 3) 4326 bounds 계산
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
        bounds = [[bottom, left], [top, right]]

    # 4) NoData 정리
    arr = dst.astype(np.float32)

    # (A) NaN은 이미 NoData로 취급
    # (B) 만약 원본 nodata가 -9999처럼 들어온 경우, 재투영 후에도 잔존 가능 → 안전망 처리
    #     slope에서 말도 안되는 큰 음수는 NoData로 보는게 안전
    arr[(arr < -1e6) | (arr > 1e6)] = np.nan
    # 흔한 nodata(-9999) 안전망
    arr[arr <= -9990] = np.nan

    # 유효 픽셀 없으면 종료
    if np.all(np.isnan(arr)):
        # Home.py는 수정 금지니까, 예외 대신 조용히 리턴하는 편이 안전
        return

    # 5) 퍼센타일 기반 정규화 범위
    vmin = float(np.nanpercentile(arr, 10))
    vmax = float(np.nanpercentile(arr, 90))

    # vmin==vmax 방지
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    # 6) 컬러맵 적용 (NaN은 투명)
    rgba = (cmap(norm(arr)) * 255).astype(np.uint8)
    rgba[np.isnan(arr), 3] = 0  # alpha=0

    # 7) PNG → base64
    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    png_url = f"data:image/png;base64,{b64}"

    folium.raster_layers.ImageOverlay(
        image=png_url,
        bounds=bounds,
        opacity=opacity,
        name="Slope (경사도)",
        interactive=False,
        zindex= 900
    ).add_to(folium_map)


