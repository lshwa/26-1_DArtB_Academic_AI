"""
데이터센터 입지 선정 — 미시분석 (Top 10 시군구)
Hard Filter → Soft Score → 후보지 클러스터링 → 지도 시각화
"""
import subprocess, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box, Point, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import KDTree
import folium
from folium.plugins import MarkerCluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as ctx
import json

# ── 경로 설정 ──────────────────────────────────────
def find_base():
    r = subprocess.run(
        ["find", "/Users/lshwa/Desktop/이승화/학교/동아리/DArt-B/학술제/26-1학기",
         "-maxdepth", "3", "-name", "09_landslide_risk", "-type", "d"],
        capture_output=True, text=True)
    return os.path.dirname(r.stdout.strip())

DS = find_base()
RES_DATA = os.path.join(os.path.dirname(DS), "results", "data")
RES_MAPS = os.path.join(os.path.dirname(DS), "results", "maps")
print(f"데이터셋: {DS}")

# ── 상수 ────────────────────────────────────────────
CRS = "EPSG:5179"   # Korea 2000 Unified (미터)
GRID_M = 250        # 격자 크기 (m)
SLOPE_MAX = 5.0     # Hard Filter: 경사도 최대 (°)
SCORE_W = {"power": 0.35, "transport": 0.25, "slope": 0.20,
           "water": 0.12, "periphery": 0.08}

TOP10 = [
    ("해남군","전남",1), ("영광군","전남",2), ("익산시","전북",3),
    ("정선군","강원",4), ("단양군","충북",5), ("원주시","강원",6),
    ("고흥군","전남",7), ("영월군","강원",8), ("영암군","전남",9),
    ("횡성군","강원",10),
]

# ── 데이터 로드 ─────────────────────────────────────
print("\n[1/5] 데이터 로드...")

admin = gpd.read_file(os.path.join(DS, "01_admin_boundaries", "sigungu_top10.geojson")).to_crs(CRS)
admin["name"] = admin["name"].fillna(admin.get("NAME_KOR", admin.get("name_kor", "")))

power = gpd.read_file(os.path.join(DS, "02_power_infrastructure", "power_infrastructure_top10.geojson")).to_crs(CRS)
substations = power[power["power"] == "substation"].copy()
print(f"  변전소: {len(substations)}개")

transport = gpd.read_file(os.path.join(DS, "03_transport", "roads_highways_top10.geojson")).to_crs(CRS)
ic = transport[transport["highway"] == "motorway_junction"].copy()
print(f"  고속도로IC: {len(ic)}개")

waterways = gpd.read_file(os.path.join(DS, "04_waterways", "waterways_top10.geojson")).to_crs(CRS)
rivers = waterways[waterways["waterway"].isin(["river","stream"]) | (waterways["natural"] == "water")].copy()
print(f"  하천/수계: {len(rivers)}개")

farmland = gpd.read_file(os.path.join(DS, "10_farmland", "farmland_top10.geojson")).to_crs(CRS)
cons_forest = gpd.read_file(os.path.join(DS, "11_conservation_forest", "conservation_forest_top10.geojson")).to_crs(CRS)
ecology = gpd.read_file(os.path.join(DS, "12_ecology_core", "ecology_core_top10.geojson")).to_crs(CRS)
print(f"  농업진흥: {len(farmland)}, 보전산지: {len(cons_forest)}, 생태핵심: {len(ecology)}")

# ── KDTree 인덱스 ───────────────────────────────────
def pts_xy(gdf):
    coords = np.array([[g.centroid.x, g.centroid.y] if g.geom_type != "Point"
                       else [g.x, g.y] for g in gdf.geometry])
    return coords

sub_xy  = pts_xy(substations)  if len(substations)  > 0 else None
ic_xy   = pts_xy(ic)           if len(ic)           > 0 else None
riv_xy  = pts_xy(rivers)       if len(rivers)       > 0 else None

tree_sub = KDTree(sub_xy)  if sub_xy  is not None else None
tree_ic  = KDTree(ic_xy)   if ic_xy   is not None else None
tree_riv = KDTree(riv_xy)  if riv_xy  is not None else None

# ── DEM 경사도 함수 ─────────────────────────────────
def get_slope_at_points(lons, lats):
    """WGS84 경위도 좌표에서 SRTM 기반 경사도(°) 반환"""
    dem_files = [
        os.path.join(DS, "05_dem", "srtm_62_05.tif"),
        os.path.join(DS, "05_dem", "srtm_62_06.tif"),
    ]
    slopes = np.full(len(lons), np.nan)
    for dem_path in dem_files:
        if not os.path.exists(dem_path):
            continue
        with rasterio.open(dem_path) as src:
            data = src.read(1).astype(float)
            data[data == src.nodata] = np.nan
            # 경사도 계산 (중앙차분, 위경도 → 미터 변환)
            res_deg = abs(src.transform.a)
            res_m = res_deg * 111320  # 1° ≈ 111.32 km
            dy, dx = np.gradient(data, res_m, res_m)
            slope_arr = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            # 포인트 샘플링
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                if not (src.bounds.left <= lon <= src.bounds.right and
                        src.bounds.bottom <= lat <= src.bounds.top):
                    continue
                row, col = rowcol(src.transform, lon, lat)
                if 0 <= row < slope_arr.shape[0] and 0 <= col < slope_arr.shape[1]:
                    slopes[i] = slope_arr[row, col]
    return slopes

# ── 침수위험 샘플링 ─────────────────────────────────
def get_flood_risk_at_points(xs_5179, ys_5179, region_name):
    """EPSG:5179 좌표에서 침수위험 GeoTIFF 픽셀값 반환"""
    tif = os.path.join(DS, "08_flood_risk", f"flood_risk_{region_name}.tif")
    if not os.path.exists(tif):
        return np.zeros(len(xs_5179))
    risks = np.zeros(len(xs_5179))
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:5179", "EPSG:5186", always_xy=True)
    xs_5186, ys_5186 = t.transform(xs_5179, ys_5179)
    with rasterio.open(tif) as src:
        for i, (x, y) in enumerate(zip(xs_5186, ys_5186)):
            if not (src.bounds.left <= x <= src.bounds.right and
                    src.bounds.bottom <= y <= src.bounds.top):
                continue
            row, col = rowcol(src.transform, x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1)[row, col]
                risks[i] = float(val) if val != src.nodata else 0.0
    return risks

# ── 점수 계산 ───────────────────────────────────────
def dist_score(dists_m, thresholds, scores):
    """거리 → 점수 (thresholds: 구간 경계, scores: 각 구간 점수)"""
    result = np.zeros(len(dists_m))
    for i, d in enumerate(dists_m):
        for j, t in enumerate(thresholds):
            if d <= t:
                result[i] = scores[j]
                break
    return result

# ── 메인 분석 루프 ───────────────────────────────────
print("\n[2/5] 시군구별 격자 생성 + Hard Filter + Soft Score...")

all_results = []
region_summaries = []

for sigg_name, sido, rank in TOP10:
    print(f"\n  [{rank}위] {sigg_name}({sido})")

    # 행정 경계
    mask = admin.geometry.apply(lambda g: sigg_name in str(g))
    boundary_row = admin[admin.apply(lambda r: sigg_name in str(r.to_dict()), axis=1)]
    if len(boundary_row) == 0:
        boundary_row = admin.iloc[[0]]  # fallback
        print(f"    ⚠️ 경계 없음, 첫번째 폴리곤 사용")

    boundary = boundary_row.geometry.iloc[0]
    minx, miny, maxx, maxy = boundary.bounds

    # 250m 격자 생성
    xs = np.arange(minx + GRID_M/2, maxx, GRID_M)
    ys = np.arange(miny + GRID_M/2, maxy, GRID_M)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    # 경계 내부 필터
    grid_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in pts],
        crs=CRS
    )
    inside = grid_gdf[grid_gdf.within(boundary.buffer(0))].copy()
    inside["x"] = inside.geometry.x
    inside["y"] = inside.geometry.y
    n_total = len(inside)
    print(f"    격자: {n_total}개 ({GRID_M}m × {GRID_M}m)")

    if n_total == 0:
        continue

    # ── Hard Filter ────────────────────────────
    keep = np.ones(n_total, dtype=bool)
    filter_log = {}

    # 1) 경사도
    from pyproj import Transformer
    t_to_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lons, lats = t_to_wgs.transform(inside["x"].values, inside["y"].values)
    slopes = get_slope_at_points(lons, lats)
    slope_mask = ~np.isnan(slopes) & (slopes > SLOPE_MAX)
    filter_log["경사도>5°"] = int(slope_mask.sum())
    keep &= ~slope_mask
    inside["slope"] = slopes

    # 2) 침수위험
    flood = get_flood_risk_at_points(inside["x"].values, inside["y"].values, sigg_name)
    flood_mask = flood > 0
    filter_log["침수위험"] = int(flood_mask.sum())
    keep &= ~flood_mask

    # 3) 농업진흥지역
    farm_union = unary_union(farmland.geometry)
    farm_mask = np.array([farm_union.contains(Point(x, y))
                          for x, y in zip(inside["x"].values, inside["y"].values)])
    filter_log["농업진흥지역"] = int(farm_mask.sum())
    keep &= ~farm_mask

    # 4) 보전산지
    cf_union = unary_union(cons_forest.clip(boundary.buffer(1000)).geometry)
    cf_mask = np.array([cf_union.contains(Point(x, y))
                        for x, y in zip(inside["x"].values, inside["y"].values)])
    filter_log["보전산지"] = int(cf_mask.sum())
    keep &= ~cf_mask

    # 5) 생태경관핵심
    eco_union = unary_union(ecology.geometry) if len(ecology) > 0 else None
    if eco_union:
        eco_mask = np.array([eco_union.contains(Point(x, y))
                             for x, y in zip(inside["x"].values, inside["y"].values)])
        filter_log["생태경관핵심"] = int(eco_mask.sum())
        keep &= ~eco_mask

    # 6) 산사태 1-2등급
    ls_path = os.path.join(DS, "09_landslide_risk", f"landslide_{sigg_name}.geojson")
    if os.path.exists(ls_path):
        ls_gdf = gpd.read_file(ls_path)
        if len(ls_gdf) > 0 and "hazard_grade" in ls_gdf.columns:
            from pyproj import Transformer as T2
            t5181 = T2.from_crs("EPSG:5181", CRS, always_xy=True)
            ls_danger = ls_gdf[ls_gdf["hazard_grade"].isin([1, 2])].copy()
            if len(ls_danger) > 0:
                x5179, y5179 = t5181.transform(
                    ls_danger["x_5181"].values, ls_danger["y_5181"].values)
                ls_tree = KDTree(np.column_stack([x5179, y5179]))
                dists_ls, _ = ls_tree.query(np.column_stack([inside["x"].values, inside["y"].values]))
                ls_mask = dists_ls < 500
                filter_log["산사태1-2등급"] = int(ls_mask.sum())
                keep &= ~ls_mask

    n_pass = int(keep.sum())
    pass_rate = n_pass / n_total * 100
    print(f"    Hard Filter: {n_total}→{n_pass}개 ({pass_rate:.1f}% 통과)")
    for k, v in filter_log.items():
        if v > 0:
            print(f"      제거 {k}: {v}개")

    # ── Soft Score ─────────────────────────────
    survived = inside[keep].copy()
    if len(survived) == 0:
        print(f"    ⚠️ {sigg_name}: Hard Filter 후 후보 없음")
        region_summaries.append({
            "rank": rank, "name": sigg_name, "sido": sido,
            "grid_total": n_total, "grid_pass": 0, "pass_rate": 0,
            "top_candidate": None
        })
        continue

    sv_xy = np.column_stack([survived["x"].values, survived["y"].values])

    # 전력(35%): 변전소 거리
    if tree_sub:
        d_sub, _ = tree_sub.query(sv_xy)
        s_power = dist_score(d_sub/1000, [5,10,20,999], [100,70,40,0])
    else:
        s_power = np.full(len(survived), 50.0)

    # 교통(25%): IC 거리
    if tree_ic:
        d_ic, _ = tree_ic.query(sv_xy)
        s_transport = dist_score(d_ic/1000, [5,15,30,999], [100,70,40,0])
    else:
        s_transport = np.full(len(survived), 50.0)

    # 경사도(20%): 낮을수록 좋음
    sl = survived["slope"].fillna(3.0).values
    s_slope = np.where(sl<=1, 100, np.where(sl<=3, 80, 60))

    # 냉각수(12%): 하천 거리
    if tree_riv:
        d_riv, _ = tree_riv.query(sv_xy)
        s_water = dist_score(d_riv/1000, [1,3,5,999], [100,70,40,10])
    else:
        s_water = np.full(len(survived), 30.0)

    # 민원회피(8%): 시군구 중심에서 멀수록 좋음
    cx, cy = boundary.centroid.x, boundary.centroid.y
    d_center = np.sqrt((survived["x"].values - cx)**2 + (survived["y"].values - cy)**2)
    max_d = d_center.max() if d_center.max() > 0 else 1
    s_peri = (d_center / max_d) * 100

    total_score = (s_power   * SCORE_W["power"] +
                   s_transport * SCORE_W["transport"] +
                   s_slope   * SCORE_W["slope"] +
                   s_water   * SCORE_W["water"] +
                   s_peri    * SCORE_W["periphery"])

    survived["score_total"]     = total_score
    survived["score_power"]     = s_power
    survived["score_transport"] = s_transport
    survived["score_slope"]     = s_slope
    survived["score_water"]     = s_water
    survived["score_periphery"] = s_peri
    survived["sigg_name"]       = sigg_name
    survived["sido"]            = sido
    survived["rank"]            = rank

    # 상위 10% 후보 클러스터링
    thresh = np.percentile(total_score, 90)
    top_cells = survived[survived["score_total"] >= thresh].copy()

    # 결과 저장
    survived.to_file(os.path.join(RES_DATA, f"scored_{sigg_name}.geojson"), driver="GeoJSON")
    top_cells.to_file(os.path.join(RES_DATA, f"top_candidates_{sigg_name}.geojson"), driver="GeoJSON")

    top_score = total_score.max()
    top_idx = total_score.argmax()
    top_lon, top_lat = t_to_wgs.transform(sv_xy[top_idx, 0], sv_xy[top_idx, 1])

    print(f"    Soft Score 최고: {top_score:.1f}점 @ ({top_lat:.4f}N, {top_lon:.4f}E)")
    print(f"    상위 10% 후보: {len(top_cells)}개 격자")

    region_summaries.append({
        "rank": rank, "name": sigg_name, "sido": sido,
        "grid_total": n_total, "grid_pass": n_pass, "pass_rate": round(pass_rate, 1),
        "top_score": round(top_score, 1),
        "top_lat": round(top_lat, 5), "top_lon": round(top_lon, 5),
        "top_candidates": len(top_cells),
        "filter_log": filter_log
    })
    all_results.append(survived)

print("\n[3/5] 결과 CSV 저장...")
summary_df = pd.DataFrame(region_summaries)
summary_df.to_csv(os.path.join(RES_DATA, "analysis_summary.csv"), index=False, encoding="utf-8-sig")
print(f"  → analysis_summary.csv 저장")

# ── 전국 통합 지도 (Folium) ─────────────────────────
print("\n[4/5] 인터랙티브 지도 생성 (folium)...")

m = folium.Map(location=[36.5, 127.8], zoom_start=7,
               tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
               attr="Esri World Imagery")

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Labels", overlay=True
).add_to(m)

colors = ["#FF4444","#FF8800","#FFCC00","#88DD00","#44AA00",
          "#00AA88","#0088FF","#4444FF","#8800FF","#FF44AA"]

for i, row in summary_df.iterrows():
    if row.get("top_lat") is None:
        continue
    c = colors[i % len(colors)]
    folium.CircleMarker(
        location=[row["top_lat"], row["top_lon"]],
        radius=18, color=c, fill=True, fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['rank']}위 {row['name']} ({row['sido']})</b><br>"
            f"Hard Filter 통과율: {row.get('pass_rate','?')}%<br>"
            f"최고 Soft Score: {row.get('top_score','?')}점<br>"
            f"상위 10% 후보: {row.get('top_candidates','?')}개 격자<br>"
            f"좌표: {row['top_lat']:.4f}N, {row['top_lon']:.4f}E",
            max_width=300
        ),
        tooltip=f"{row['rank']}위 {row['name']}"
    ).add_to(m)

# 상위 후보 히트맵
if all_results:
    merged = pd.concat([r[["x","y","score_total","sigg_name"]].copy() for r in all_results])
    from pyproj import Transformer
    t_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    lons_all, lats_all = t_wgs.transform(merged["x"].values, merged["y"].values)
    merged["lon"] = lons_all
    merged["lat"] = lats_all

    # 상위 5% 격자 포인트
    top5_pct = merged[merged["score_total"] >= merged["score_total"].quantile(0.95)]
    for _, r in top5_pct.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=3,
            color="#FFFF00", fill=True, fill_opacity=0.6, weight=0
        ).add_to(m)

folium.LayerControl().add_to(m)
map_path = os.path.join(RES_MAPS, "datacenter_candidates_map.html")
m.save(map_path)
print(f"  → {map_path}")

# ── 시군구별 개별 지도 ──────────────────────────────
print("\n[5/5] 시군구별 결과 지도 생성...")

for res_gdf, summ in zip(all_results, [s for s in region_summaries if s.get("grid_pass",0) > 0]):
    sigg = summ["name"]
    print(f"  {sigg} 지도 생성 중...")

    from pyproj import Transformer
    t_wgs = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

    # 중심 좌표
    cx = res_gdf["x"].mean(); cy = res_gdf["y"].mean()
    clat, clon = t_wgs.transform(cx, cy)[1], t_wgs.transform(cx, cy)[0]

    m2 = folium.Map(location=[clat, clon], zoom_start=11,
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri World Imagery")

    # 점수별 색상
    scores = res_gdf["score_total"].values
    s_min, s_max = scores.min(), scores.max()
    def score_color(s):
        t = (s - s_min) / (s_max - s_min + 1e-9)
        r = int(255 * (1 - t)); g = int(255 * t)
        return f"#{r:02x}{g:02x}00"

    for _, row in res_gdf.iterrows():
        lon_p, lat_p = t_wgs.transform(row["x"], row["y"])
        folium.Rectangle(
            bounds=[[lat_p - 0.0011, lon_p - 0.0011],
                    [lat_p + 0.0011, lon_p + 0.0011]],
            color=score_color(row["score_total"]),
            fill=True, fill_opacity=0.5, weight=0,
            popup=(f"Score: {row['score_total']:.1f}<br>"
                   f"전력: {row['score_power']:.0f} / "
                   f"교통: {row['score_transport']:.0f} / "
                   f"경사: {row['score_slope']:.0f}")
        ).add_to(m2)

    # 최고점 마커
    best = res_gdf.nlargest(1, "score_total").iloc[0]
    b_lon, b_lat = t_wgs.transform(best["x"], best["y"])
    folium.Marker(
        location=[b_lat, b_lon],
        popup=f"<b>최적 후보지</b><br>Score: {best['score_total']:.1f}",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(m2)

    p = os.path.join(RES_MAPS, f"map_{sigg}.html")
    m2.save(p)

print("\n" + "="*60)
print("분석 완료!")
print("="*60)
print(f"\n결과 파일:")
print(f"  지도: {RES_MAPS}/")
print(f"  데이터: {RES_DATA}/")
print("\n시군구별 요약:")
for s in sorted(region_summaries, key=lambda x: x["rank"]):
    if s.get("grid_pass",0) > 0:
        print(f"  {s['rank']}위 {s['name']}: 통과율 {s.get('pass_rate','?')}% | "
              f"최고점 {s.get('top_score','?')}점 | 상위후보 {s.get('top_candidates','?')}개")
    else:
        print(f"  {s['rank']}위 {s['name']}: Hard Filter 후 후보 없음")
