"""
1) SHP → GeoJSON 변환 및 시군구 클리핑
   10_farmland       : 농업진흥지역
   11_conservation_forest : 보전산지
   12_ecology_core   : 생태경관핵심보전지역

2) 07_zoning : V-World WFS BBOX 방식 (요청 간 8초 딜레이)

3) 09_landslide_risk : sansatai EPSG:5181 격자 샘플링
"""
import requests, json, os, time
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer

BASE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(BASE, "06_manual_download_required")
VWORLD_KEY = '1A94D935-77C2-3F68-AE71-1BD3DD8EDA3A'

TOP10 = [
    {"name":"해남군",  "sido":"전남","rank":1, "bbox4326":[126.17,34.26,126.92,34.72]},
    {"name":"영광군",  "sido":"전남","rank":2, "bbox4326":[126.38,35.20,126.90,35.47]},
    {"name":"익산시",  "sido":"전북","rank":3, "bbox4326":[126.84,35.86,127.09,36.14]},
    {"name":"정선군",  "sido":"강원","rank":4, "bbox4326":[128.56,37.19,128.96,37.46]},
    {"name":"단양군",  "sido":"충북","rank":5, "bbox4326":[128.19,36.83,128.56,37.17]},
    {"name":"원주시",  "sido":"강원","rank":6, "bbox4326":[127.79,37.17,128.25,37.54]},
    {"name":"고흥군",  "sido":"전남","rank":7, "bbox4326":[127.12,34.40,127.52,34.83]},
    {"name":"영월군",  "sido":"강원","rank":8, "bbox4326":[128.31,37.04,128.97,37.37]},
    {"name":"영암군",  "sido":"전남","rank":9, "bbox4326":[126.50,34.69,126.93,34.94]},
    {"name":"횡성군",  "sido":"강원","rank":10,"bbox4326":[127.79,37.36,128.29,37.74]},
]

# Top10 전체 커버하는 단일 BBOX (클리핑용)
ALL_MINX, ALL_MINY = 126.17, 34.26
ALL_MAXX, ALL_MAXY = 129.00, 37.75


# ─────────────────────────────────────────
# 1. SHP 처리
# ─────────────────────────────────────────

def shp_to_geojson(shp_path, out_path, label, clip_crs="EPSG:4326"):
    """SHP 읽기 → Top10 범위 클리핑 → GeoJSON 저장"""
    gdf = gpd.read_file(shp_path, encoding='cp949')
    gdf_wgs = gdf.to_crs("EPSG:4326")
    clip_box = box(ALL_MINX, ALL_MINY, ALL_MAXX, ALL_MAXY)
    clipped = gdf_wgs[gdf_wgs.intersects(clip_box)].copy()
    # geometry를 GeoJSON dict로
    features = []
    for _, row in clipped.iterrows():
        props = row.drop("geometry").to_dict()
        # timestamp 직렬화
        for k, v in props.items():
            if hasattr(v, 'isoformat'):
                props[k] = v.isoformat()
            elif v != v:  # NaN
                props[k] = None
        geom = row.geometry.__geo_interface__
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    gj = {"type": "FeatureCollection", "features": features,
          "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)
    print(f"  [{label}] {len(features)}개 피처 → {os.path.basename(out_path)}")
    return len(features)


def process_shp_files():
    print("\n[SHP 처리] 농업진흥지역 · 보전산지 · 생태경관핵심보전지역")

    # 10_farmland
    shp_to_geojson(
        os.path.join(MAN, "농업진흥지역", "농업진흥지역.shp"),
        os.path.join(BASE, "10_farmland", "farmland_top10.geojson"),
        "농업진흥지역"
    )

    # 11_conservation_forest
    shp_to_geojson(
        os.path.join(MAN, "보전산지", "보전산지.shp"),
        os.path.join(BASE, "11_conservation_forest", "conservation_forest_top10.geojson"),
        "보전산지"
    )

    # 12_ecology_core
    shp_to_geojson(
        os.path.join(MAN, "생태경관핵심보전지역", "생태경관핵심보전지역.shp"),
        os.path.join(BASE, "12_ecology_core", "ecology_core_top10.geojson"),
        "생태경관핵심보전지역"
    )


# ─────────────────────────────────────────
# 2. V-World WFS 07_zoning  (BBOX 방식)
# ─────────────────────────────────────────

def fetch_zoning():
    print("\n[07_zoning] V-World WFS BBOX 방식 (요청 간 8초 딜레이)...")
    out_dir = os.path.join(BASE, "07_zoning")
    WFS_URL = "https://api.vworld.kr/req/wfs"

    for region in TOP10:
        name = region["name"]
        bb = region["bbox4326"]
        bbox_str = f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}"

        feats = []
        for attempt in range(5):
            try:
                r = requests.get(WFS_URL, params={
                    "SERVICE":"WFS","VERSION":"2.0.0","REQUEST":"GetFeature",
                    "KEY":VWORLD_KEY,"DOMAIN":"localhost","TYPENAME":"lt_c_uq111",
                    "SRSNAME":"EPSG:4326","OUTPUT":"application/json",
                    "COUNT":"2000","STARTINDEX":"0","BBOX":bbox_str
                }, timeout=60)
                body = r.text.strip()
                if r.status_code == 200 and body.startswith('{'):
                    feats = r.json().get("features", [])
                    break
                else:
                    wait = 15 * (attempt + 1)
                    print(f"  {name} 실패(시도{attempt+1}), {wait}s 대기...")
                    time.sleep(wait)
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"  {name} 예외({attempt+1}): {e}, {wait}s 대기")
                time.sleep(wait)

        gj = {"type": "FeatureCollection", "features": feats}
        path = os.path.join(out_dir, f"zoning_{name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gj, f, ensure_ascii=False)

        # 용도지역 분포 요약
        zone_cnt = {}
        for feat in feats:
            z = feat.get("properties", {}).get("prpos_area1_nm", "?")
            zone_cnt[z] = zone_cnt.get(z, 0) + 1
        top3 = sorted(zone_cnt.items(), key=lambda x:-x[1])[:3]
        print(f"  {name}: {len(feats)}개 → {dict(top3)}")
        time.sleep(8)  # rate limit 회피


# ─────────────────────────────────────────
# 3. 산사태위험 09_landslide_risk
# ─────────────────────────────────────────

def fetch_landslide():
    print("\n[09_landslide_risk] 산사태위험 격자 샘플링 (EPSG:5181, 500m 간격)...")
    out_dir = os.path.join(BASE, "09_landslide_risk")
    GRID_URL = ("https://sansatai.forest.go.kr/gis1/ictout/iserver"
                "/services/data-HazardParam2025/rest/data"
                "/datasources/HazardParam2025/datasets/hazard2025/gridValue.json")
    STEP = 500

    t = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)

    for region in TOP10:
        name = region["name"]
        lon0, lat0, lon1, lat1 = region["bbox4326"]
        x0, y0 = t.transform(lon0, lat0)
        x1, y1 = t.transform(lon1, lat1)
        xs = list(range(int(x0), int(x1), STEP))
        ys = list(range(int(y0), int(y1), STEP))
        total_pts = len(xs) * len(ys)
        print(f"  -> {name}: {len(xs)}×{len(ys)}={total_pts}점")

        features = []
        for xi, x in enumerate(xs):
            for y in ys:
                try:
                    r = requests.get(GRID_URL, params={"x": x, "y": y},
                                     timeout=8, headers={"User-Agent":"Mozilla/5.0"})
                    if r.status_code == 200:
                        val = r.json().get("value", None)
                        try:
                            grade = int(val) if val is not None and int(val) in [1,2,3,4,5] else None
                        except (ValueError, TypeError):
                            grade = None
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [x, y]},
                            "properties": {"x_5181":x,"y_5181":y,"hazard_grade":grade,"region":name}
                        })
                except Exception:
                    pass
            if (xi+1) % 5 == 0 or (xi+1) == len(xs):
                valid = sum(1 for f in features if f["properties"]["hazard_grade"] is not None)
                print(f"     {(xi+1)*len(ys)}/{total_pts}점 완료, 유효등급={valid}")
            time.sleep(0.02)

        out = {"type":"FeatureCollection","features":features,
               "crs":{"type":"name","properties":{"name":"EPSG:5181"}}}
        path = os.path.join(out_dir, f"landslide_{name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        valid_c = sum(1 for feat in features if feat["properties"]["hazard_grade"] is not None)
        print(f"  {name} 저장: {len(features)}점, 유효={valid_c}개")


if __name__ == "__main__":
    print("=" * 60)
    print("데이터 수집·처리 통합 스크립트")
    print("=" * 60)
    process_shp_files()
    fetch_zoning()
    fetch_landslide()
    print("\n전체 완료!")
