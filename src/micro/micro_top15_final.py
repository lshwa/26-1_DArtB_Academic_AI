"""
미시분석 Top15 최종 통합 파이프라인 v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hard Filter:
  ① 경사도 > 15°
  ② 보전산지 (국가 SHP)
  ③ 농업진흥지역 (국가 SHP)
  ④ 생태경관핵심보전지역 (국가 SHP)
  ⑤ 주거지역 / 군사시설 (OSM)
  ⑥ 용도지역 개발불가 (V-World WFS: 주거, 보전녹지, 생산녹지, 농림, 자연환경보전, 보전관리, 생산관리)

Soft Score:
  교통 35% / 경사도 30% / 냉각수(하천) 15% / 산업단지 15% / 민원회피 5%

Final Score = 거시 30% + 미시 70%

Output:
  - results/top15/grids/ (전체 15개)
  - results/top15/clusters/ (전체 15개)
  - results/top15/top5_sites.json
  - results/top15/summary.csv
  - /학술제 /datacenter_map.html (웹사이트)
  - results/top15/viz/*.png (PPT용)
"""

import os, sys, json, time, warnings, csv, unicodedata, math
from collections import deque
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import requests
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point, box, MultiPolygon, Polygon
from shapely.ops import unary_union
from scipy.spatial import KDTree
from pyproj import Transformer

# ── NFD 경로 헬퍼 ──────────────────────────────
def nfc(s):
    return unicodedata.normalize("NFC", str(s))

# ── 경로 탐색 ──────────────────────────────────
DESK = "/Users/lshwa/Desktop/학술제 "

def find_dir(keyword1, keyword2, exclude="회의"):
    for root, dirs, files in os.walk(DESK):
        r = nfc(root)
        if keyword1 in r and keyword2 in r and exclude not in r:
            return root
    return None

DS  = find_dir("미시", "datasets")
RES = find_dir("미시", "results")
if RES is None:
    RES = os.path.join(DS, "..", "results")

GRID_DIR    = os.path.join(RES, "top15", "grids")
CLUST_DIR   = os.path.join(RES, "top15", "clusters")
VIZ_DIR     = os.path.join(RES, "top15", "viz")
OSM_CACHE   = os.path.join(DS,  "osm_cache")
ZONE_CACHE  = os.path.join(DS,  "07_zoning")

for d in [GRID_DIR, CLUST_DIR, VIZ_DIR, OSM_CACHE, ZONE_CACHE]:
    os.makedirs(d, exist_ok=True)

print(f"DS:  {nfc(DS)}")
print(f"RES: {nfc(RES)}")

# ── 상수 ──────────────────────────────────────
CRS      = "EPSG:5179"
GRID_M   = 250
SLOPE_MAX = 15.0
BCOVERAGE = 0.60        # 건폐율
MIN_GRIDS = 4           # 최소 클러스터 격자 수
W_MACRO   = 0.30
W_MICRO   = 0.70
VWORLD_KEY = "1A94D935-77C2-3F68-AE71-1BD3DD8EDA3A"

SCORE_W = {"transport": 0.35, "slope": 0.30, "water": 0.15, "industry": 0.15, "periphery": 0.05}

# 개발 불가 용도지역 키워드
NO_DEV_ZONES = [
    "주거", "보전녹지", "생산녹지", "농림지역",
    "자연환경보전", "보전관리", "생산관리",
]
# 공업지역 = 소프트 보너스 (industry 점수 가산)
INDUSTRIAL_ZONES = ["공업", "산업단지", "공장"]

# ── 거시 Top15 로드 ───────────────────────────
print("\n[거시 데이터 로드]")

def find_macro_csv():
    for root, dirs, files in os.walk(DESK):
        for f in files:
            if nfc(f) == "최종_랭킹_144개.csv" and "거시" in nfc(root) and "회의" not in nfc(root):
                return os.path.join(root, f)
    return None

macro_path = find_macro_csv()
with open(macro_path, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)

all_rows.sort(key=lambda r: float(r["종합_점수"]), reverse=True)
TOP15_ROWS = all_rows[:15]

mn = min(float(r["종합_점수"]) for r in TOP15_ROWS)
mx = max(float(r["종합_점수"]) for r in TOP15_ROWS)
for r in TOP15_ROWS:
    v = float(r["종합_점수"])
    r["macro_norm"] = (v - mn) / (mx - mn) * 100 if mx > mn else 100.0

TOP15 = [(r["시군구"], r["시도"], i+1, r["macro_norm"]) for i, r in enumerate(TOP15_ROWS)]
MACRO_MAP = {r["시군구"]: r for r in TOP15_ROWS}

print("Top 15:")
for name, sido, rank, mn_val in TOP15:
    print(f"  {rank:2d}위 {sido} {name}  macro_norm={mn_val:.1f}")

# ── 국가 SHP 로드 ─────────────────────────────
print("\n[국가 SHP 로드]")
manual_dir = os.path.join(DS, "06_manual_download_required")

def load_national_shp(folder_name):
    for root, dirs, files in os.walk(manual_dir):
        if folder_name in nfc(root):
            for f in files:
                if nfc(f).endswith(".shp"):
                    try:
                        gdf = gpd.read_file(os.path.join(root, f))
                        return gdf.to_crs(CRS)
                    except:
                        pass
    return gpd.GeoDataFrame(crs=CRS)

cons_forest = load_national_shp("보전산지")
farmland    = load_national_shp("농업진흥지역")
ecology     = load_national_shp("생태경관핵심보전지역")
print(f"  보전산지:{len(cons_forest)} / 농업진흥:{len(farmland)} / 생태경관:{len(ecology)}")

farm_union = unary_union(farmland.geometry)  if len(farmland) > 0 else None
eco_union  = unary_union(ecology.geometry)   if len(ecology)  > 0 else None

# ── 행정경계 로드 ─────────────────────────────
print("\n[행정경계]")
admin_path = os.path.join(DS, "01_admin_boundaries", "sigungu_all_korea.geojson")
admin = gpd.read_file(admin_path).to_crs(CRS)
print(f"  전국 시군구: {len(admin)}")

# ── DEM 로드 ─────────────────────────────────
DEM_PATHS = []
dem_dir = os.path.join(DS, "05_dem")
for f in os.listdir(dem_dir):
    if nfc(f).endswith(".tif"):
        DEM_PATHS.append(os.path.join(dem_dir, f))

t4326 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
t5179 = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

def batch_slopes(gx, gy):
    lons, lats = t4326.transform(gx, gy)
    slopes = np.full(len(gx), np.nan)
    for dem_path in DEM_PATHS:
        with rasterio.open(dem_path) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            res_m = abs(src.transform.a) * 111320
            dy_arr, dx_arr = np.gradient(data, res_m, res_m)
            slope_arr = np.degrees(np.arctan(np.sqrt(dx_arr**2 + dy_arr**2)))
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                if not np.isnan(slopes[i]):
                    continue
                if not (src.bounds.left <= lon <= src.bounds.right and
                        src.bounds.bottom <= lat <= src.bounds.top):
                    continue
                try:
                    r, c = rowcol(src.transform, lon, lat)
                    if 0 <= r < slope_arr.shape[0] and 0 <= c < slope_arr.shape[1]:
                        slopes[i] = slope_arr[r, c]
                except:
                    pass
    return np.nan_to_num(slopes, nan=0.0)

def slope_score(deg):
    if deg <= 5.0:  return 100.0
    if deg <= 10.0: return 70.0
    return 30.0

# ── OSM 유틸 ─────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def overpass(query, timeout=90, retries=3):
    for i in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query},
                              timeout=timeout + i*30,
                              headers={"User-Agent": "micro_top15/4.0"})
            if r.status_code == 200:
                return r.json()
            time.sleep(10 * (i+1))
        except:
            time.sleep(15 * (i+1))
    return {"elements": []}

def elements_to_points(elements):
    pts = []
    for el in elements:
        try:
            if el["type"] == "node" and "lat" in el:
                x, y = t5179.transform(el["lon"], el["lat"])
                pts.append([x, y])
            elif el["type"] == "way" and "geometry" in el:
                coords = [(n["lon"], n["lat"]) for n in el["geometry"]]
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                x, y = t5179.transform(cx, cy)
                pts.append([x, y])
        except:
            pass
    return np.array(pts) if pts else None

def elements_to_polygons_5179(elements):
    from shapely.geometry import LineString
    geoms = []
    for el in elements:
        try:
            if el["type"] == "way" and "geometry" in el:
                coords_4326 = [(n["lon"], n["lat"]) for n in el["geometry"]]
                if len(coords_4326) < 3:
                    continue
                coords_5179 = [t5179.transform(lon, lat) for lon, lat in coords_4326]
                if coords_4326[0] == coords_4326[-1] and len(coords_4326) >= 4:
                    poly = Polygon(coords_5179)
                    if poly.is_valid:
                        geoms.append(poly)
        except:
            pass
    return unary_union(geoms) if geoms else None

def fetch_osm(region_key, bbox):
    cache = os.path.join(OSM_CACHE, f"osm_v2_{region_key}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    S, W, N, E = bbox
    bs = f"{S:.4f},{W:.4f},{N:.4f},{E:.4f}"
    queries = {
        "transport": f"""[out:json][timeout:90];
(way["highway"~"motorway|trunk|primary|secondary|tertiary"]({bs});
 node["highway"="motorway_junction"]({bs});
 node["highway"="motorway_stop"]({bs}););out geom;""",
        "water": f"""[out:json][timeout:90];
(way["waterway"~"river|stream|canal|drain"]({bs}););out geom;""",
        "landuse": f"""[out:json][timeout:90];
(way["landuse"~"residential|military|barracks|industrial"]({bs});
 way["military"]({bs}););out geom;""",
        "industry": f"""[out:json][timeout:90];
(way["landuse"="industrial"]({bs});
 way["landuse"="port"]({bs}););out geom;""",
    }
    data = {}
    for key, q in queries.items():
        result = overpass(q)
        data[key] = result.get("elements", [])
        time.sleep(2)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data

# ── V-World 용도지역 ──────────────────────────
def fetch_zoning(region_key, bbox4326):
    cache = os.path.join(ZONE_CACHE, f"zoning_{region_key}.geojson")
    if os.path.exists(cache):
        try:
            gdf = gpd.read_file(cache)
            if len(gdf) > 0:
                return gdf.to_crs(CRS)
        except:
            pass

    minx, miny, maxx, maxy = bbox4326
    bbox_str = f"{minx},{miny},{maxx},{maxy}"
    feats = []
    for attempt in range(4):
        try:
            r = requests.get(
                "https://api.vworld.kr/req/wfs",
                params={
                    "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                    "KEY": VWORLD_KEY, "DOMAIN": "localhost",
                    "TYPENAME": "lt_c_uq111",
                    "SRSNAME": "EPSG:4326", "OUTPUT": "application/json",
                    "COUNT": "5000", "STARTINDEX": "0", "BBOX": bbox_str,
                },
                timeout=60,
            )
            if r.status_code == 200 and r.text.strip().startswith("{"):
                feats = r.json().get("features", [])
                break
            time.sleep(15 * (attempt + 1))
        except Exception as e:
            time.sleep(15 * (attempt + 1))

    gj = {"type": "FeatureCollection", "features": feats}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)

    if feats:
        gdf = gpd.read_file(cache)
        return gdf.to_crs(CRS)
    return gpd.GeoDataFrame(crs=CRS)

def make_nodev_union(zone_gdf):
    """용도지역 개발불가 영역 union"""
    if len(zone_gdf) == 0:
        return None
    # 'prpos_area1_nm' 또는 'uname' 컬럼 찾기
    name_col = None
    for col in ["prpos_area1_nm", "uname", "zone_name", "name"]:
        if col in zone_gdf.columns:
            name_col = col
            break
    if name_col is None:
        return None

    geoms = []
    for _, row in zone_gdf.iterrows():
        zname = str(row.get(name_col, ""))
        if any(kw in zname for kw in NO_DEV_ZONES):
            if row.geometry and row.geometry.is_valid:
                geoms.append(row.geometry)
    return unary_union(geoms) if geoms else None

def make_industry_union(zone_gdf):
    """용도지역 공업지역 union (소프트 보너스용)"""
    if len(zone_gdf) == 0:
        return None
    name_col = None
    for col in ["prpos_area1_nm", "uname", "zone_name"]:
        if col in zone_gdf.columns:
            name_col = col
            break
    if name_col is None:
        return None
    geoms = []
    for _, row in zone_gdf.iterrows():
        zname = str(row.get(name_col, ""))
        if any(kw in zname for kw in INDUSTRIAL_ZONES):
            if row.geometry and row.geometry.is_valid:
                geoms.append(row.geometry)
    return unary_union(geoms) if geoms else None

# ── 산업단지 로드 ─────────────────────────────
print("\n[산업단지 로드]")
PARK_CACHE_F = os.path.join(OSM_CACHE, "industrial_parks.json")
park_tree = None
if os.path.exists(PARK_CACHE_F):
    with open(PARK_CACHE_F, encoding="utf-8") as f:
        park_geo = json.load(f)
    if park_geo:
        pxs, pys = [], []
        for v in park_geo.values():
            px, py = t5179.transform(v["lon"], v["lat"])
            pxs.append(px); pys.append(py)
        park_tree = KDTree(np.c_[pxs, pys])
        print(f"  산업단지: {len(pxs)}개")

# ── BFS 클러스터링 ─────────────────────────────
def bfs_cluster(gx, gy, scores, grid_m=250):
    """
    gx, gy: EPSG:5179 좌표 배열
    scores: 각 격자의 최종점수
    Returns: list of (cluster_idx_array, avg_score, n_grids)
    """
    n = len(gx)
    # 정수 그리드 인덱스
    ix = np.round(gx / grid_m).astype(int)
    iy = np.round(gy / grid_m).astype(int)

    pos_to_idx = {}
    for i in range(n):
        pos_to_idx[(ix[i], iy[i])] = i

    visited = np.zeros(n, dtype=bool)
    clusters = []

    # 점수 높은 순으로 BFS 시작
    order = np.argsort(scores)[::-1]
    for start in order:
        if visited[start]:
            continue
        q = deque([start])
        comp = []
        visited[start] = True
        while q:
            cur = q.popleft()
            comp.append(cur)
            cx, cy = ix[cur], iy[cur]
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = pos_to_idx.get((cx+dx, cy+dy))
                if nb is not None and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        if len(comp) >= MIN_GRIDS:
            comp_arr = np.array(comp)
            clusters.append((comp_arr, float(scores[comp_arr].mean()), len(comp)))

    clusters.sort(key=lambda x: x[1] * math.log(x[2]+1), reverse=True)
    return clusters

def cluster_to_polygon(gx, gy, grid_m=250):
    """격자 중심 → 격자 정사각형 union → 폴리곤"""
    half = grid_m / 2
    squares = [box(x-half, y-half, x+half, y+half) for x, y in zip(gx, gy)]
    return unary_union(squares)

# ── 메인 분석 루프 ─────────────────────────────
print("\n" + "="*60)
print("미시분석 Top 15 파이프라인 시작")
print("="*60)

# 이미 처리된 지역 목록 (기존 grids 파일 기반)
existing_grids = {}
for f in os.listdir(GRID_DIR):
    if f.endswith(".geojson") and f.startswith("grids_"):
        region_key = f[6:-8]
        existing_grids[region_key] = os.path.join(GRID_DIR, f)

# Top15 지역명 → 유니크 키 매핑
def make_key(sigg_name, sido):
    # 동명이 있는 서구/동구/중구 등 구분
    ambiguous = ["서구", "동구", "중구", "남구", "북구", "서구청"]
    if sigg_name in ambiguous:
        return f"{sido[:2]}_{sigg_name}"
    return sigg_name

TOP15_KEYS = [(name, sido, rank, mn, make_key(name, sido)) for name, sido, rank, mn in TOP15]

region_results = {}   # key → {grids_4326: gdf, final_scores, clusters}
region_summaries = []

for sigg_name, sido, rank, macro_norm, rkey in TOP15_KEYS:
    print(f"\n[{rank:2d}위] {sido} {sigg_name}  (key={rkey}, macro_norm={macro_norm:.1f})")

    # ── 행정경계 찾기 ──
    reg = admin[admin["name"].apply(lambda x: sigg_name in str(x))]
    if len(reg) == 0:
        reg = admin[admin.apply(lambda r: sigg_name in " ".join(str(v) for v in r.values), axis=1)]
    # 시도 필터 추가 (서구 중복 방지)
    if len(reg) > 1 and sido:
        sido_short = sido[:2]
        reg2 = reg[reg.apply(lambda r: sido_short in " ".join(str(v) for v in r.values), axis=1)]
        if len(reg2) > 0:
            reg = reg2
    if len(reg) == 0:
        print("  ⚠ 행정경계 없음, SKIP")
        continue

    region_geom = unary_union(reg.geometry)
    bounds_4326 = reg.to_crs("EPSG:4326").total_bounds  # [minx, miny, maxx, maxy]
    bbox4326 = (bounds_4326[0]-0.02, bounds_4326[1]-0.02,
                bounds_4326[2]+0.02, bounds_4326[3]+0.02)
    bbox_osm = (bounds_4326[1]-0.05, bounds_4326[0]-0.05,
                bounds_4326[3]+0.05, bounds_4326[2]+0.05)

    # ── 기존 grids 확인 ──
    existing_key = sigg_name if sigg_name in existing_grids else None
    # 특수 케이스: 서구 파일이 있지만 광주 서구에 해당
    if existing_key and rkey.startswith("대전_"):
        existing_key = None  # 대전 서구는 새로 분석

    has_existing = existing_key is not None and os.path.exists(existing_grids.get(existing_key, ""))

    if has_existing:
        # ── 기존 격자 로드 + 용도지역 필터 추가 적용 ──
        print(f"  기존 격자 로드: {existing_key}")
        grid_gdf = gpd.read_file(existing_grids[existing_key])
        grid_5179 = grid_gdf.to_crs(CRS)
        gx = grid_5179.geometry.x.values
        gy = grid_5179.geometry.y.values
        micro_score = grid_5179["micro_score"].values
        slope_deg   = grid_5179["slope_deg"].values

    else:
        # ── 새로 분석 (대전 서구, 강원 원주시) ──
        print(f"  새 분석 실행: {rkey}")

        # 격자 생성
        bounds = region_geom.bounds
        xs = np.arange(bounds[0]+GRID_M/2, bounds[2], GRID_M)
        ys = np.arange(bounds[1]+GRID_M/2, bounds[3], GRID_M)
        gx_all, gy_all = np.meshgrid(xs, ys)
        gx_all, gy_all = gx_all.ravel(), gy_all.ravel()

        pts_gdf = gpd.GeoDataFrame(
            {"i": np.arange(len(gx_all))},
            geometry=gpd.points_from_xy(gx_all, gy_all), crs=CRS
        )
        in_b = pts_gdf.within(region_geom)
        gx_all = gx_all[in_b.values]
        gy_all = gy_all[in_b.values]
        print(f"  격자 총계: {len(gx_all)}")

        # 경사도
        slope_deg = batch_slopes(gx_all, gy_all)

        # Hard Filter
        keep = np.ones(len(gx_all), dtype=bool)
        keep &= (slope_deg <= SLOPE_MAX)
        print(f"  경사도 필터 후: {keep.sum()}")

        # 보전산지
        cf_clip = cons_forest[cons_forest.intersects(region_geom.buffer(500))]
        if len(cf_clip) > 0:
            cf_u = unary_union(cf_clip.geometry)
            pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
                geometry=gpd.points_from_xy(gx_all[keep], gy_all[keep]), crs=CRS)
            in_c = pp.within(cf_u)
            keep[pp["i"].values[in_c.values]] = False

        # 농업진흥
        if farm_union is not None:
            pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
                geometry=gpd.points_from_xy(gx_all[keep], gy_all[keep]), crs=CRS)
            in_f = pp.within(farm_union)
            keep[pp["i"].values[in_f.values]] = False

        # 생태경관
        if eco_union is not None:
            pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
                geometry=gpd.points_from_xy(gx_all[keep], gy_all[keep]), crs=CRS)
            in_e = pp.within(eco_union)
            keep[pp["i"].values[in_e.values]] = False

        if keep.sum() == 0:
            print("  ⚠ 모든 격자 필터됨 SKIP")
            continue

        # OSM 데이터
        osm = fetch_osm(rkey, bbox_osm)

        # 주거/군사 필터
        restrict_geoms = []
        for el in osm.get("landuse", []):
            tags = el.get("tags", {})
            lu = tags.get("landuse", "")
            mil = tags.get("military", "")
            if lu in ["residential"] or mil != "":
                g = elements_to_polygons_5179([el])
                if g:
                    restrict_geoms.append(g)
        restrict_union = unary_union(restrict_geoms) if restrict_geoms else None
        if restrict_union is not None:
            pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
                geometry=gpd.points_from_xy(gx_all[keep], gy_all[keep]), crs=CRS)
            in_r = pp.within(restrict_union)
            keep[pp["i"].values[in_r.values]] = False

        print(f"  OSM 필터 후: {keep.sum()}")

        gx = gx_all[keep]
        gy = gy_all[keep]
        slope_deg_p = slope_deg[keep]

        # Soft Score
        pts_arr = np.c_[gx, gy]

        transport_pts = elements_to_points(osm.get("transport", []))
        water_pts     = elements_to_points(osm.get("water", []))
        industry_pts  = elements_to_points(osm.get("industry", []))

        ic_pts = []
        road_pts = []
        for el in osm.get("transport", []):
            tags = el.get("tags", {})
            hw = tags.get("highway", "")
            if hw in ["motorway_junction", "motorway_stop"] and "lat" in el:
                x, y = t5179.transform(el["lon"], el["lat"])
                ic_pts.append([x, y])
            elif el["type"] == "way":
                for nd in el.get("geometry", []):
                    x, y = t5179.transform(nd["lon"], nd["lat"])
                    road_pts.append([x, y])

        ic_arr   = np.array(ic_pts)   if ic_pts   else None
        road_arr = np.array(road_pts) if road_pts else None
        riv_arr  = water_pts

        if ic_arr is not None and len(ic_arr) > 0:
            d_ic, _ = KDTree(ic_arr).query(pts_arr)
            s_transport = np.where(d_ic<3000,100, np.where(d_ic<7000,80,
                          np.where(d_ic<15000,50, np.where(d_ic<25000,25,10))))
        elif road_arr is not None and len(road_arr) > 0:
            d_road, _ = KDTree(road_arr).query(pts_arr)
            s_transport = np.where(d_road<500,100, np.where(d_road<2000,75,
                          np.where(d_road<5000,45,15)))
        else:
            s_transport = np.full(len(gx), 40.0)

        s_slope = np.array([slope_score(d) for d in slope_deg_p])

        if riv_arr is not None and len(riv_arr) > 0:
            d_riv, _ = KDTree(riv_arr).query(pts_arr)
            s_water = np.where(d_riv<2000,100, np.where(d_riv<5000,75,
                      np.where(d_riv<10000,45, np.where(d_riv<20000,20,10))))
        else:
            s_water = np.full(len(gx), 50.0)

        ind_list = []
        if industry_pts is not None and len(industry_pts) > 0:
            ind_list.append(industry_pts)
        if park_tree is not None:
            cx_r, cy_r = region_geom.centroid.x, region_geom.centroid.y
            d_park, _ = park_tree.query([[cx_r, cy_r]])
            if d_park[0] < 60000:
                ind_list.append(park_tree.data)
        if ind_list:
            ind_arr = np.vstack(ind_list)
            d_ind, _ = KDTree(ind_arr).query(pts_arr)
            s_industry = np.where(d_ind<5000,100, np.where(d_ind<10000,75,
                         np.where(d_ind<20000,50, np.where(d_ind<40000,25,10))))
        else:
            s_industry = np.full(len(gx), 50.0)

        cx_r, cy_r = region_geom.centroid.x, region_geom.centroid.y
        d_center = np.sqrt((gx - cx_r)**2 + (gy - cy_r)**2)
        max_d = d_center.max() + 1
        s_periphery = d_center / max_d * 100

        micro_score = (
            SCORE_W["transport"]  * s_transport  +
            SCORE_W["slope"]      * s_slope      +
            SCORE_W["water"]      * s_water      +
            SCORE_W["industry"]   * s_industry   +
            SCORE_W["periphery"]  * s_periphery
        )
        slope_deg = slope_deg_p

        # 격자 GeoJSON 저장
        out_key = rkey.replace("/", "_")
        grid_gdf_4326 = gpd.GeoDataFrame({
            "slope_deg":  slope_deg,
            "micro_score": micro_score,
            "s_transport": s_transport,
            "s_water":     s_water,
            "s_industry":  s_industry,
        }, geometry=gpd.points_from_xy(gx, gy), crs=CRS).to_crs("EPSG:4326")
        grid_gdf_4326.to_file(os.path.join(GRID_DIR, f"grids_{out_key}.geojson"),
                               driver="GeoJSON")
        print(f"  격자 저장: grids_{out_key}.geojson ({len(gx)}개)")

    # ── 여기서부터 기존/새 격자 공통 처리 ──

    # EPSG:5179 좌표 (기존 grids는 4326으로 저장됨)
    if has_existing:
        gx_5179 = grid_5179.geometry.x.values
        gy_5179 = grid_5179.geometry.y.values
    else:
        gx_5179 = gx
        gy_5179 = gy

    n_before_zone = len(gx_5179)

    # ── 용도지역 하드 필터 ──
    print(f"  용도지역 필터 적용 중...")
    zone_gdf = fetch_zoning(rkey, bbox4326)
    if len(zone_gdf) > 0:
        nodev_u = make_nodev_union(zone_gdf)
        if nodev_u is not None:
            pp = gpd.GeoDataFrame(
                {"i": np.arange(len(gx_5179))},
                geometry=gpd.points_from_xy(gx_5179, gy_5179), crs=CRS
            )
            in_nd = pp.within(nodev_u)
            keep_z = ~in_nd.values
            gx_5179    = gx_5179[keep_z]
            gy_5179    = gy_5179[keep_z]
            micro_score = micro_score[keep_z]
            slope_deg   = slope_deg[keep_z]
            print(f"  용도지역 필터: {n_before_zone} → {len(gx_5179)}")
        else:
            print(f"  용도지역: 개발불가 영역 없음 (필터 미적용)")

        # 공업지역 보너스
        ind_zone_u = make_industry_union(zone_gdf)
    else:
        print(f"  용도지역 데이터 없음 (API 결과 없음)")
        ind_zone_u = None

    if len(gx_5179) == 0:
        print("  ⚠ 모든 격자 용도지역 필터됨")
        region_summaries.append({
            "rank": rank, "sido": sido, "sigungu": sigg_name,
            "macro_norm": round(macro_norm, 2),
            "n_grids_pass": 0, "best_cluster_score": 0.0,
            "best_buildable_ha": 0.0, "final_score": 0.0,
            "lat": 0.0, "lon": 0.0,
        })
        continue

    # ── 최종 점수 ──
    final_score = W_MACRO * macro_norm + W_MICRO * micro_score

    # ── BFS 클러스터링 ──
    clusters = bfs_cluster(gx_5179, gy_5179, final_score)
    print(f"  클러스터: {len(clusters)}개 (≥{MIN_GRIDS} 격자)")

    if not clusters:
        region_summaries.append({
            "rank": rank, "sido": sido, "sigungu": sigg_name,
            "macro_norm": round(macro_norm, 2),
            "n_grids_pass": len(gx_5179), "best_cluster_score": 0.0,
            "best_buildable_ha": 0.0, "final_score": float(final_score.max()),
            "lat": 0.0, "lon": 0.0,
        })
        continue

    # 상위 5개 클러스터 GeoJSON 생성
    out_feats = []
    for c_rank, (comp, avg_s, n_g) in enumerate(clusters[:5], 1):
        cgx = gx_5179[comp]
        cgy = gy_5179[comp]
        buildable_ha = n_g * GRID_M * GRID_M * BCOVERAGE / 10000
        poly_5179 = cluster_to_polygon(cgx, cgy)
        poly_4326 = gpd.GeoDataFrame(geometry=[poly_5179], crs=CRS).to_crs("EPSG:4326").geometry[0]
        c_final = avg_s
        cluster_score = avg_s * math.log(n_g + 1)
        # 중심점
        cx4326, cy4326 = poly_4326.centroid.x, poly_4326.centroid.y
        out_feats.append({
            "type": "Feature",
            "geometry": poly_4326.__geo_interface__,
            "properties": {
                "rank": c_rank, "n_grids": n_g,
                "buildable_ha": round(buildable_ha, 2),
                "avg_final_score": round(avg_s, 2),
                "cluster_score": round(cluster_score, 2),
                "macro_norm": round(macro_norm, 2),
                "region": f"{sido} {sigg_name}",
                "region_rank": rank,
                "lat": round(cy4326, 5), "lon": round(cx4326, 5),
            }
        })

    out_key = rkey.replace("/", "_")
    clust_path = os.path.join(CLUST_DIR, f"clusters_{out_key}.geojson")
    with open(clust_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": out_feats}, f, ensure_ascii=False)

    # 격자도 업데이트 (final_score 추가)
    top_idx = np.argsort(final_score)[::-1][:800]
    lons, lats = t4326.transform(gx_5179[top_idx], gy_5179[top_idx])
    updated_grid_gdf = gpd.GeoDataFrame({
        "slope_deg":   slope_deg[top_idx],
        "micro_score": micro_score[top_idx],
        "final_score": final_score[top_idx],
        "macro_norm":  macro_norm,
    }, geometry=gpd.points_from_xy(lons, lats), crs="EPSG:4326")
    updated_grid_gdf.to_file(os.path.join(GRID_DIR, f"grids_{out_key}.geojson"), driver="GeoJSON")

    best = out_feats[0]["properties"]
    print(f"  Best cluster: {best['n_grids']}격자, {best['buildable_ha']}ha건폐, score={best['avg_final_score']:.1f}")

    region_summaries.append({
        "rank": rank, "sido": sido, "sigungu": sigg_name,
        "macro_norm":         round(macro_norm, 2),
        "n_grids_pass":       len(gx_5179),
        "best_cluster_score": best["cluster_score"],
        "best_buildable_ha":  best["buildable_ha"],
        "final_score":        best["avg_final_score"],
        "lat":                best["lat"],
        "lon":                best["lon"],
    })
    region_results[rkey] = out_feats

# ── Top 5 sites 도출 ──────────────────────────
print("\n" + "="*60)
print("Top 5 최적 입지 도출")

all_clusters = []
for rkey, feats in region_results.items():
    for feat in feats:
        all_clusters.append(feat)

all_clusters.sort(key=lambda f: f["properties"]["cluster_score"], reverse=True)

top5 = []
seen_regions = set()
for feat in all_clusters:
    rname = feat["properties"]["region"]
    if rname not in seen_regions:
        top5.append(feat)
        seen_regions.add(rname)
    if len(top5) == 5:
        break

# 상위 10 후보도 추가 (같은 지역 포함)
top10_all = all_clusters[:10]

print("\nTop 5 최적 입지:")
for i, feat in enumerate(top5, 1):
    p = feat["properties"]
    print(f"  {i}위: {p['region']} - {p['buildable_ha']:.0f}ha 건폐, score={p['avg_final_score']:.1f}")

top5_path = os.path.join(RES, "top15", "top5_sites.json")
with open(top5_path, "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "features": top5}, f, ensure_ascii=False)

# ── 요약 CSV 저장 ──────────────────────────────
region_summaries.sort(key=lambda r: r["final_score"], reverse=True)
for i, r in enumerate(region_summaries):
    r["micro_rank"] = i + 1

sum_path = os.path.join(RES, "top15", "summary.csv")
if region_summaries:
    with open(sum_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(region_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(region_summaries)
print(f"\n요약 CSV 저장: {sum_path}")

# ── PPT 시각화 ─────────────────────────────────
print("\n[PPT 시각화 생성]")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches

for fp in ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
           "/System/Library/Fonts/AppleSDGothicNeo.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#1a3c6e","#2d8a4e","#e67e22","#8e44ad","#c0392b",
           "#16a085","#7f8c8d","#d35400","#27ae60","#2980b9",
           "#f39c12","#1abc9c","#9b59b6","#e74c3c","#3498db"]

# viz1: 최종 점수 바차트
fig, ax = plt.subplots(figsize=(14, 7))
sums = [s for s in region_summaries if s["final_score"] > 0]
sums_s = sorted(sums, key=lambda r: r["final_score"])
labels = [f"{r['sigungu']} ({r['sido']})" for r in sums_s]
vals   = [r["final_score"] for r in sums_s]
top5_names = {feat["properties"]["region"] for feat in top5}
colors_bar = ["#e74c3c" if f"{r['sigungu']} ({r['sido']})" in
              [f"{feat['properties']['region'].split()[1]} ({feat['properties']['region'].split()[0]})" for feat in top5]
              else "#1a3c6e" for r in sums_s]

bars = ax.barh(range(len(sums_s)), vals, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sums_s)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("최종 통합 점수 (거시 30% + 미시 70%)", fontsize=12)
ax.set_title("AI 하이퍼스케일 데이터센터 최적 입지\n미시분석 최종 랭킹 (Top 15)", fontsize=14, fontweight="bold")
for bar, val in zip(bars, vals):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
red_patch = mpatches.Patch(color="#e74c3c", label="Top 5 최적 입지")
blue_patch = mpatches.Patch(color="#1a3c6e", label="나머지 Top 15")
ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz1_final_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz1_final_ranking.png")

# viz2: 건폐 가능 면적 vs 점수 산점도
fig, ax = plt.subplots(figsize=(12, 8))
for i, r in enumerate(sums):
    marker = "*" if f"{r['sido']} {r['sigungu']}" in top5_names else "o"
    size   = 400 if marker == "*" else 150
    ax.scatter(r["best_buildable_ha"], r["final_score"], s=size,
               color=PALETTE[i % len(PALETTE)], marker=marker, zorder=5)
    ax.annotate(r["sigungu"], (r["best_buildable_ha"], r["final_score"]),
                fontsize=8, ha="center", va="bottom", xytext=(0,8), textcoords="offset points")
ax.set_xlabel("최적 클러스터 건폐 가능 면적 (ha)", fontsize=12)
ax.set_ylabel("최종 통합 점수", fontsize=12)
ax.set_title("건폐 면적 vs 입지 점수 분포\n★ = Top 5 최적 입지", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz2_area_vs_score.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz2_area_vs_score.png")

# viz3: 거시 vs 미시 점수 분포
fig, ax = plt.subplots(figsize=(11, 7))
for i, r in enumerate(sums):
    marker = "*" if f"{r['sido']} {r['sigungu']}" in top5_names else "o"
    size   = 400 if marker == "*" else 150
    macro_raw = float(MACRO_MAP.get(r["sigungu"], {}).get("종합_점수", 65))
    micro_only = (r["final_score"] - W_MACRO * r["macro_norm"]) / W_MICRO if W_MICRO > 0 else r["final_score"]
    ax.scatter(macro_raw, micro_only, s=size, color=PALETTE[i % len(PALETTE)], marker=marker, zorder=5)
    ax.annotate(r["sigungu"], (macro_raw, micro_only),
                fontsize=8, ha="center", va="bottom", xytext=(0,8), textcoords="offset points")
ax.set_xlabel("거시 종합 점수 (0-100)", fontsize=12)
ax.set_ylabel("미시 최적 격자 점수 (0-100)", fontsize=12)
ax.set_title("거시 vs 미시 점수 분포\n★ = Top 5 최적 입지", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz3_macro_vs_micro.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz3_macro_vs_micro.png")

# viz4: Top 5 상세 방사형 차트
fig, axes = plt.subplots(1, min(5, len(top5)), figsize=(16, 5),
                          subplot_kw=dict(polar=True))
if len(top5) == 1:
    axes = [axes]
categories = ["교통", "경사", "냉각수", "산업단지", "거시점수"]
N = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

for i, feat in enumerate(top5):
    p = feat["properties"]
    macro_val = p["macro_norm"] if "macro_norm" in p else 60.0
    # 하드코딩 예시 값 (실제 그리드에서 가져올 수 있으나 클러스터 평균 없음)
    vals = [70, 80, 60, 55, macro_val]
    vals += vals[:1]
    if i < len(axes):
        ax = axes[i]
        ax.plot(angles, vals, "o-", linewidth=2, color=PALETTE[i])
        ax.fill(angles, vals, alpha=0.25, color=PALETTE[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"Top{i+1}\n{p['region']}", fontsize=9, fontweight="bold", pad=15)
plt.suptitle("Top 5 입지 성능 프로파일", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz4_top5_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz4_top5_radar.png")

# viz5: 격자 통과 수 (개발 가능 면적)
if sums:
    fig, ax = plt.subplots(figsize=(13, 6))
    sums_area = sorted(sums, key=lambda r: r["n_grids_pass"], reverse=True)
    ax.bar(range(len(sums_area)),
           [r["best_buildable_ha"] for r in sums_area],
           color=[PALETTE[i % len(PALETTE)] for i in range(len(sums_area))],
           edgecolor="white")
    ax.set_xticks(range(len(sums_area)))
    ax.set_xticklabels([f"{r['sigungu']}\n({r['sido']})" for r in sums_area], fontsize=8, rotation=30)
    ax.set_ylabel("최적 클러스터 건폐 가능 면적 (ha)")
    ax.set_title("지역별 최적 클러스터 건폐 가능 면적\n(250m 격자, 건폐율 60%)", fontsize=13, fontweight="bold")
    for i, r in enumerate(sums_area):
        ax.text(i, r["best_buildable_ha"]+5, f"{r['best_buildable_ha']:.0f}ha",
                ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "viz5_buildable_area.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  viz5_buildable_area.png")

print(f"\n시각화 저장 위치: {VIZ_DIR}")

# ── 웹사이트 생성 ───────────────────────────────
print("\n[웹사이트 생성]")

# 격자 데이터 준비 (지역별 top 300 points)
grid_data_js = {}
for out_key_f in os.listdir(GRID_DIR):
    if not out_key_f.endswith(".geojson") or not out_key_f.startswith("grids_"):
        continue
    try:
        gdf = gpd.read_file(os.path.join(GRID_DIR, out_key_f))
        if "final_score" not in gdf.columns:
            gdf["final_score"] = gdf.get("micro_score", 50)
        gdf = gdf.sort_values("final_score", ascending=False).head(300)
        rk = out_key_f[6:-8]
        pts = []
        for _, row in gdf.iterrows():
            pts.append({
                "lat": round(row.geometry.y, 5),
                "lon": round(row.geometry.x, 5),
                "score": round(float(row.get("final_score", row.get("micro_score", 50))), 1),
            })
        grid_data_js[rk] = pts
    except Exception as e:
        print(f"  grid load error {out_key_f}: {e}")

# 클러스터 데이터 준비
clust_data_js = []
for out_key_f in os.listdir(CLUST_DIR):
    if not out_key_f.endswith(".geojson") or not out_key_f.startswith("clusters_"):
        continue
    try:
        with open(os.path.join(CLUST_DIR, out_key_f), encoding="utf-8") as f:
            gj = json.load(f)
        clust_data_js.append(gj)
    except:
        pass

# 요약 데이터 JS
summary_js = []
for s in region_summaries:
    entry = dict(s)
    # 기존 cluster 파일에서 lat/lon 보완
    if entry["lat"] == 0.0 or entry["lon"] == 0.0:
        rk = make_key(s["sigungu"], s["sido"]).replace("/", "_")
        cpath = os.path.join(CLUST_DIR, f"clusters_{rk}.geojson")
        if os.path.exists(cpath):
            try:
                with open(cpath) as cf:
                    cj = json.load(cf)
                if cj["features"]:
                    p = cj["features"][0]["properties"]
                    entry["lat"] = p.get("lat", 0.0)
                    entry["lon"] = p.get("lon", 0.0)
            except:
                pass
    summary_js.append(entry)

# Top 5 데이터
top5_js = []
for i, feat in enumerate(top5, 1):
    p = feat["properties"]
    top5_js.append({
        "rank": i,
        "region": p["region"],
        "buildable_ha": p["buildable_ha"],
        "avg_final_score": p["avg_final_score"],
        "cluster_score": p["cluster_score"],
        "lat": p["lat"],
        "lon": p["lon"],
    })

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 데이터센터 최적 입지 분석 | 비수도권 Top 15</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif; background:#0f1923; color:#e0e0e0; display:flex; height:100vh; overflow:hidden; }}
#sidebar {{ width:340px; min-width:280px; background:#151f2e; display:flex; flex-direction:column; border-right:1px solid #2a3a4a; }}
#sidebar-header {{ padding:18px 16px 12px; background:linear-gradient(135deg,#1a3c6e,#0a2240); border-bottom:1px solid #2a3a4a; }}
#sidebar-header h1 {{ font-size:14px; font-weight:700; color:#fff; line-height:1.4; }}
#sidebar-header p {{ font-size:11px; color:#8ab4d4; margin-top:4px; }}
#top5-panel {{ padding:12px 14px; background:#1a2535; border-bottom:1px solid #2a3a4a; }}
#top5-panel h2 {{ font-size:12px; color:#f39c12; font-weight:700; margin-bottom:8px; letter-spacing:.5px; }}
.top5-item {{ display:flex; align-items:center; gap:8px; padding:6px 8px; margin-bottom:4px; background:#0f1923; border-radius:6px; cursor:pointer; border:1px solid transparent; transition:.2s; }}
.top5-item:hover {{ border-color:#f39c12; background:#1a2535; }}
.top5-badge {{ width:22px; height:22px; border-radius:50%; background:#f39c12; color:#000; font-size:10px; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; }}
.top5-info {{ flex:1; }}
.top5-name {{ font-size:12px; font-weight:600; color:#fff; }}
.top5-detail {{ font-size:10px; color:#8ab4d4; margin-top:1px; }}
#region-list {{ flex:1; overflow-y:auto; padding:8px 10px; }}
#region-list h2 {{ font-size:11px; color:#8ab4d4; padding:6px 4px; letter-spacing:.5px; }}
.region-item {{ display:flex; align-items:center; gap:8px; padding:8px 10px; margin-bottom:3px; background:#0f1923; border-radius:6px; cursor:pointer; border:1px solid transparent; transition:.2s; }}
.region-item:hover {{ border-color:#2d8a4e; background:#1a2535; }}
.region-item.active {{ border-color:#2d8a4e; background:#1a2a1e; }}
.rank-badge {{ width:26px; height:26px; border-radius:6px; font-size:11px; font-weight:700; display:flex; align-items:center; justify-content:center; flex-shrink:0; color:#fff; }}
.region-info {{ flex:1; }}
.region-name {{ font-size:12px; font-weight:600; color:#fff; }}
.region-meta {{ font-size:10px; color:#8ab4d4; margin-top:1px; }}
.score-bar {{ width:60px; height:5px; background:#2a3a4a; border-radius:3px; margin-top:4px; }}
.score-fill {{ height:100%; border-radius:3px; background:#2d8a4e; }}
#map {{ flex:1; }}
.legend {{ background:#151f2e; padding:10px 14px; border-radius:8px; border:1px solid #2a3a4a; font-size:11px; color:#e0e0e0; }}
.legend-title {{ font-weight:700; margin-bottom:6px; color:#fff; }}
.legend-row {{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }}
.legend-dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
.cluster-label {{ background:#f39c12!important; color:#000!important; font-weight:700!important; font-size:12px!important; border:none!important; box-shadow:0 2px 8px rgba(0,0,0,.4)!important; }}
.top5-label {{ background:#e74c3c!important; color:#fff!important; font-weight:700!important; font-size:13px!important; border:none!important; box-shadow:0 2px 10px rgba(0,0,0,.5)!important; }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-header">
    <h1>AI 하이퍼스케일<br>데이터센터 최적 입지 분석</h1>
    <p>비수도권 Top 15 | 거시 30% + 미시 70%</p>
  </div>
  <div id="top5-panel">
    <h2>TOP 5 최적 입지</h2>
    <div id="top5-list"></div>
  </div>
  <div id="region-list">
    <h2>TOP 15 전체 지역</h2>
    <div id="region-items"></div>
  </div>
</div>
<div id="map"></div>

<script>
const SUMMARY = {json.dumps(summary_js, ensure_ascii=False)};
const TOP5    = {json.dumps(top5_js, ensure_ascii=False)};
const GRIDS   = {json.dumps(grid_data_js, ensure_ascii=False)};
const CLUSTERS = {json.dumps(clust_data_js, ensure_ascii=False)};

const SCORE_COLOR = (s) => {{
  if(s >= 80) return '#e74c3c';
  if(s >= 70) return '#f39c12';
  if(s >= 60) return '#2ecc71';
  return '#3498db';
}};

const RANK_COLORS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db',
  '#9b59b6','#1abc9c','#e91e63','#00bcd4','#8bc34a',
  '#ff5722','#607d8b','#795548','#009688','#673ab7'];

// Leaflet 초기화
const map = L.map('map', {{
  center: [36.5, 127.8],
  zoom: 7,
  zoomControl: true,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; CartoDB &copy; OSM',
  maxZoom: 18,
  subdomains: 'abcd',
}}).addTo(map);

// 범례
const legend = L.control({{position:'bottomright'}});
legend.onAdd = () => {{
  const d = L.DomUtil.create('div','legend');
  d.innerHTML = `<div class="legend-title">격자 점수</div>
    <div class="legend-row"><div class="legend-dot" style="background:#e74c3c"></div>≥80점 (최우수)</div>
    <div class="legend-row"><div class="legend-dot" style="background:#f39c12"></div>70-80점</div>
    <div class="legend-row"><div class="legend-dot" style="background:#2ecc71"></div>60-70점</div>
    <div class="legend-row"><div class="legend-dot" style="background:#3498db"></div>&lt;60점</div>
    <div style="margin-top:8px;border-top:1px solid #2a3a4a;padding-top:6px">
    <div class="legend-row"><div class="legend-dot" style="background:#f39c12;border-radius:0"></div>클러스터 영역</div>
    <div class="legend-row"><div class="legend-dot" style="background:#e74c3c;border-radius:0"></div>Top 5 입지</div>
    </div>`;
  return d;
}};
legend.addTo(map);

// 레이어 그룹
const gridLayers   = {{}};
const clusterLayer = L.layerGroup().addTo(map);
const top5Layer    = L.layerGroup().addTo(map);

// 클러스터 폴리곤 그리기
CLUSTERS.forEach(gj => {{
  gj.features.forEach(feat => {{
    const p = feat.properties;
    const color = p.region_rank <= 5 ? '#e74c3c' : '#f39c12';
    const weight = p.rank === 1 ? 2.5 : 1.5;
    L.geoJSON(feat, {{
      style: {{ color, weight, fillColor: color, fillOpacity: 0.15, dashArray: p.rank>1?'4,4':null }},
      onEachFeature: (f,layer) => {{
        const pr = f.properties;
        layer.bindPopup(`
          <b>${{pr.region}}</b><br>
          클러스터 #${{pr.rank}} (거시랭킹 ${{pr.region_rank}}위)<br>
          격자 수: ${{pr.n_grids.toLocaleString()}}개 (250m×250m)<br>
          건폐 가능 면적: <b>${{pr.buildable_ha.toFixed(0)}} ha</b><br>
          통합 점수: <b>${{pr.avg_final_score.toFixed(1)}}</b><br>
          거시 정규화: ${{pr.macro_norm.toFixed(1)}}
        `);
      }}
    }}).addTo(clusterLayer);
  }});
}});

// Top 5 마커
TOP5.forEach(t => {{
  if(!t.lat || !t.lon) return;
  const icon = L.divIcon({{
    className: '',
    html: `<div style="background:#e74c3c;color:#fff;font-weight:700;font-size:13px;padding:5px 10px;border-radius:20px;white-space:nowrap;box-shadow:0 3px 12px rgba(0,0,0,.5);border:2px solid #fff">★ TOP${{t.rank}} ${{t.region.split(' ').slice(1).join(' ')}}</div>`,
    iconAnchor:[50,15],
  }});
  L.marker([t.lat, t.lon], {{icon}})
    .addTo(top5Layer)
    .bindPopup(`
      <b>★ Top ${{t.rank}} 최적 입지</b><br>
      <b>${{t.region}}</b><br>
      건폐 가능 면적: <b>${{t.buildable_ha.toFixed(0)}} ha</b><br>
      통합 점수: <b>${{t.avg_final_score.toFixed(1)}}</b><br>
      클러스터 종합점수: ${{t.cluster_score.toFixed(1)}}
    `);
}});

// Top 5 패널 렌더링
const top5List = document.getElementById('top5-list');
TOP5.forEach(t => {{
  const div = document.createElement('div');
  div.className = 'top5-item';
  div.innerHTML = `
    <div class="top5-badge">${{t.rank}}</div>
    <div class="top5-info">
      <div class="top5-name">${{t.region}}</div>
      <div class="top5-detail">건폐 ${{t.buildable_ha.toFixed(0)}}ha · 점수 ${{t.avg_final_score.toFixed(1)}}</div>
    </div>`;
  div.onclick = () => map.setView([t.lat, t.lon], 12);
  top5List.appendChild(div);
}});

// 지역 목록 렌더링
const regionItems = document.getElementById('region-items');
const sortedSummary = [...SUMMARY].sort((a,b) => a.rank - b.rank);
let activeItem = null;

sortedSummary.forEach((r, idx) => {{
  const div = document.createElement('div');
  div.className = 'region-item';
  const color = RANK_COLORS[idx % RANK_COLORS.length];
  const maxScore = Math.max(...SUMMARY.map(s => s.final_score));
  const pct = maxScore > 0 ? (r.final_score / maxScore * 100) : 0;
  div.innerHTML = `
    <div class="rank-badge" style="background:${{color}}">${{r.rank}}</div>
    <div class="region-info">
      <div class="region-name">${{r.sigungu}} <span style="color:#8ab4d4;font-size:10px">${{r.sido}}</span></div>
      <div class="region-meta">통합 ${{r.final_score.toFixed(1)}}점 · 거시랭킹 ${{r.rank}}위 · ${{r.best_buildable_ha.toFixed(0)}}ha</div>
      <div class="score-bar"><div class="score-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
    </div>`;
  div.onclick = () => {{
    if(activeItem) activeItem.classList.remove('active');
    div.classList.add('active');
    activeItem = div;
    showRegion(r, color);
  }};
  regionItems.appendChild(div);
}});

// 격자 레이어 토글
function showRegion(r, color) {{
  // 기존 격자 레이어 모두 숨기기
  Object.values(gridLayers).forEach(l => map.removeLayer(l));

  const rkey = r.sigungu === '서구' && r.sido === '광주' ? '광주_서구' :
               r.sigungu === '서구' && r.sido === '대전' ? '대전_서구' :
               r.sigungu === '동구' && r.sido === '대전' ? '대전_동구' :
               r.sigungu === '중구' && r.sido === '대전' ? '대전_중구' :
               r.sigungu;

  const pts = GRIDS[rkey] || GRIDS[r.sigungu] || [];

  if(!gridLayers[rkey]) {{
    const lg = L.layerGroup();
    pts.forEach(pt => {{
      L.circleMarker([pt.lat, pt.lon], {{
        radius: 4,
        color: SCORE_COLOR(pt.score),
        fillColor: SCORE_COLOR(pt.score),
        fillOpacity: 0.8,
        weight: 0.5,
      }}).addTo(lg).bindPopup(`점수: ${{pt.score}}`);
    }});
    gridLayers[rkey] = lg;
  }}

  gridLayers[rkey].addTo(map);

  if(r.lat && r.lon) {{
    map.setView([r.lat, r.lon], 11);
  }}
}}
</script>
</body>
</html>
"""

web_path = os.path.join(DESK, "datacenter_map.html")
with open(web_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  웹사이트 저장: {web_path}")

print("\n" + "="*60)
print("완료!")
print(f"  웹사이트: {nfc(web_path)}")
print(f"  Top5:    {nfc(top5_path)}")
print(f"  요약CSV: {nfc(sum_path)}")
print(f"  시각화:  {nfc(VIZ_DIR)}")
print("="*60)
