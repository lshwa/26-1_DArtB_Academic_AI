"""
미시분석 데이터 자동 수집 모듈
──────────────────────────────────────────────────────────────────
거시분석 결과(최종_랭킹_144개.csv)를 읽어 상위 N개 지역을 확인하고,
데이터가 없는 지역은 REST API로 자동 수집한다.

사용 API:
  - Overpass API  : 전력 인프라, 교통(IC), 하천 (OSM 기반, 인증 불필요)
  - V-World API   : 행정경계 (국토지리정보원, API키 필요)
  - SRTM DEM      : OpenTopography REST API (이미 파일 존재 시 스킵)
  - 농업진흥/보전산지/생태경관: 캐시 없으면 경고 (별도 인증 필요)

실행:
  python3 fetch_region_data.py          # 거시 결과 자동 읽어 Top-10 수집
  python3 fetch_region_data.py --topn 15  # Top-15 수집
  python3 fetch_region_data.py --region 해남군  # 특정 지역만
"""
import os, sys, time, json, unicodedata, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import shape, Point, box
from pyproj import Transformer

# ── 경로 설정 ────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(BASE)
DESK  = os.path.dirname(ROOT)

def _find_dir(parent, keyword):
    for d in os.listdir(parent):
        if keyword in unicodedata.normalize('NFC', d):
            return os.path.join(parent, d)
    return None

MICRO_ROOT  = _find_dir(DESK, "미시") or ROOT
DS          = os.path.join(MICRO_ROOT, "datasets")
MACRO_ROOT  = _find_dir(DESK, "거시")

# 데이터셋 폴더 보장
for sub in ["01_admin_boundaries","02_power_infrastructure","03_transport",
            "04_waterways","08_flood_risk","09_landslide_risk",
            "10_farmland","11_conservation_forest","12_ecology_core"]:
    os.makedirs(os.path.join(DS, sub), exist_ok=True)

CRS = "EPSG:5179"

# ── 거시 결과에서 Top-N 자동 읽기 ───────────────────
def get_macro_topn(n=10):
    if MACRO_ROOT is None:
        print("[경고] 거시분석 폴더를 찾을 수 없음")
        return []
    for root, dirs, files in os.walk(MACRO_ROOT):
        for f in files:
            if unicodedata.normalize('NFC', f) == "최종_랭킹_144개.csv":
                df = pd.read_csv(os.path.join(root, f), encoding='utf-8-sig')
                df.columns = [c.lstrip('﻿') for c in df.columns]
                df = df.sort_values("종합_점수", ascending=False).head(n)
                regions = []
                for _, row in df.iterrows():
                    regions.append({
                        "name": row["시군구"],
                        "sido": row.get("시도", ""),
                        "score": row["종합_점수"]
                    })
                print(f"[거시 결과] Top-{n} 지역: {[r['name'] for r in regions]}")
                return regions
    print("[경고] 최종_랭킹_144개.csv 없음 — 거시분석을 먼저 실행하세요")
    return []

# ── Overpass API 쿼리 ────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def overpass_query(ql, timeout=120, retry=3):
    """Overpass QL 쿼리 실행, GeoDataFrame 반환"""
    for attempt in range(retry):
        try:
            r = requests.post(OVERPASS_URL,
                              data={"data": ql},
                              timeout=timeout,
                              headers={"Accept-Encoding": "gzip, deflate"})
            r.raise_for_status()
            data = r.json()
            features = []
            for el in data.get("elements", []):
                props = {k: v for k, v in el.items() if k not in ("lat","lon","nodes","members","geometry")}
                props.update(el.get("tags", {}))
                if el["type"] == "node" and "lat" in el:
                    geom = Point(el["lon"], el["lat"])
                elif el["type"] == "way" and "geometry" in el:
                    from shapely.geometry import LineString
                    coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
                    geom = LineString(coords) if len(coords) >= 2 else None
                else:
                    continue
                if geom:
                    features.append({"geometry": geom, **props})
            if not features:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
            return gdf
        except requests.exceptions.Timeout:
            print(f"  [Overpass] 타임아웃 (시도 {attempt+1}/{retry})")
            time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"  [Overpass] 오류: {e} (시도 {attempt+1}/{retry})")
            time.sleep(5)
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def get_region_bbox(region_name):
    """Nominatim으로 지역 bounding box 조회"""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{region_name} 대한민국", "format": "json",
                    "limit": 1, "addressdetails": 1},
            headers={"User-Agent": "AI-DataCenter-Location-Analysis/1.0"},
            timeout=15
        )
        data = r.json()
        if data:
            bb = data[0]["boundingbox"]
            # [south, north, west, east]
            s, n, w, e = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            print(f"  [Nominatim] {region_name}: ({s:.3f},{w:.3f}) ~ ({n:.3f},{e:.3f})")
            return s, n, w, e
    except Exception as e:
        print(f"  [Nominatim] 오류: {e}")
    return None

# ── V-World 행정경계 ─────────────────────────────────
VWORLD_KEY = os.environ.get("VWORLD_KEY", "")  # 환경변수로 설정

def fetch_admin_vworld(region_name, bbox):
    """V-World WFS API로 시군구 행정경계 조회"""
    if not VWORLD_KEY:
        print(f"  [V-World] API 키 없음 — Nominatim bbox로 대체")
        return None
    s, n, w, e = bbox
    url = "https://api.vworld.kr/req/wfs"
    params = {
        "service": "WFS", "version": "2.0.0", "request": "GetFeature",
        "typeName": "lt_c_adsigg", "srsName": "EPSG:4326",
        "bbox": f"{w},{s},{e},{n},EPSG:4326",
        "output": "application/json",
        "key": VWORLD_KEY, "domain": ""
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        gj = r.json()
        if "features" in gj and gj["features"]:
            gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
            # 지역명 필터
            for col in gdf.columns:
                mask = gdf[col].astype(str).str.contains(region_name, na=False)
                if mask.any():
                    return gdf[mask]
            return gdf
    except Exception as e:
        print(f"  [V-World] 오류: {e}")
    return None

# ── Overpass로 행정경계 대안 ─────────────────────────
def fetch_admin_overpass(region_name, bbox):
    """Overpass로 시군구 행정경계 (admin_level=5)"""
    s, n, w, e = bbox
    ql = f"""
[out:json][timeout:60];
(
  relation["admin_level"="5"]["name"~"{region_name}"]({s},{w},{n},{e});
  relation["admin_level"="6"]["name"~"{region_name}"]({s},{w},{n},{e});
);
out geom;
"""
    # Overpass 관계 geometry 직접 처리
    try:
        r = requests.post(OVERPASS_URL, data={"data": ql}, timeout=60,
                          headers={"Accept-Encoding": "gzip, deflate"})
        data = r.json()
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        polys = []
        for el in data.get("elements", []):
            if el["type"] != "relation": continue
            for member in el.get("members", []):
                if member.get("role") not in ("outer", "inner"): continue
                geom_pts = member.get("geometry", [])
                if len(geom_pts) >= 3:
                    coords = [(g["lon"], g["lat"]) for g in geom_pts]
                    try:
                        polys.append(Polygon(coords))
                    except: pass
        if polys:
            merged = unary_union(polys)
            gdf = gpd.GeoDataFrame({"name": [region_name]},
                                   geometry=[merged], crs="EPSG:4326")
            return gdf
    except Exception as e:
        print(f"  [Overpass 행정경계] 오류: {e}")
    return None

# ── 핵심 데이터 수집 함수들 ──────────────────────────
def fetch_power(region_name, bbox, existing_gdf=None):
    s, n, w, e = bbox
    print(f"  [Overpass] 전력 인프라 수집 중...")
    ql = f"""
[out:json][timeout:90];
(
  node["power"="substation"]({s},{w},{n},{e});
  node["power"="tower"]({s},{w},{n},{e});
  way["power"="line"]({s},{w},{n},{e});
);
out geom;
"""
    gdf = overpass_query(ql, timeout=90)
    print(f"    → {len(gdf)}개 피처 수집")
    return gdf

def fetch_transport(region_name, bbox, existing_gdf=None):
    s, n, w, e = bbox
    print(f"  [Overpass] IC / 고속도로 수집 중...")
    ql = f"""
[out:json][timeout:90];
(
  node["highway"="motorway_junction"]({s},{w},{n},{e});
  way["highway"~"motorway|trunk"]({s},{w},{n},{e});
);
out geom;
"""
    gdf = overpass_query(ql, timeout=90)
    print(f"    → {len(gdf)}개 피처 수집")
    return gdf

def fetch_waterways(region_name, bbox, existing_gdf=None):
    s, n, w, e = bbox
    print(f"  [Overpass] 하천 수집 중...")
    ql = f"""
[out:json][timeout:90];
(
  way["waterway"~"river|stream|canal"]({s},{w},{n},{e});
  node["natural"="water"]({s},{w},{n},{e});
);
out geom;
"""
    gdf = overpass_query(ql, timeout=90)
    print(f"    → {len(gdf)}개 피처 수집")
    return gdf

# ── 통합 데이터셋 업데이트 ──────────────────────────
def merge_and_save(new_gdf, existing_path, region_name, id_col=None):
    """기존 GeoJSON에 새 지역 데이터를 병합 저장"""
    if new_gdf is None or len(new_gdf) == 0:
        print(f"    [스킵] 수집된 데이터 없음")
        return

    if os.path.exists(existing_path):
        existing = gpd.read_file(existing_path)
        # 이미 해당 지역 데이터가 있으면 제거 후 재병합
        if "region" in existing.columns:
            existing = existing[existing["region"] != region_name]
        elif "name" in existing.columns:
            existing = existing[~existing["name"].astype(str).str.contains(region_name, na=False)]
        new_gdf["region"] = region_name
        combined = pd.concat([existing, new_gdf], ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=new_gdf.crs)
    else:
        new_gdf["region"] = region_name
        combined = new_gdf

    combined.to_file(existing_path, driver="GeoJSON")
    print(f"    저장: {existing_path} ({len(combined)}개 피처)")

# ── 지역별 전체 수집 파이프라인 ─────────────────────
def fetch_region(region_name, force=False):
    """
    region_name 지역의 모든 필요 데이터를 수집.
    force=False이면 기존 데이터가 있는 레이어는 스킵.
    """
    print(f"\n{'='*50}")
    print(f"지역 데이터 수집: {region_name}")
    print(f"{'='*50}")

    # bounding box 조회
    bbox = get_region_bbox(region_name)
    if bbox is None:
        print(f"  [오류] {region_name} bounding box 조회 실패 — 스킵")
        return False

    # 행정경계
    admin_path = os.path.join(DS, "01_admin_boundaries", "sigungu_top10.geojson")
    admin_exists = False
    if os.path.exists(admin_path):
        adm = gpd.read_file(admin_path)
        admin_exists = adm.apply(
            lambda r: region_name in " ".join(str(v) for v in r.values), axis=1
        ).any()

    if not admin_exists or force:
        print(f"  [행정경계] 수집 중...")
        adm_gdf = fetch_admin_overpass(region_name, bbox)
        if adm_gdf is None and VWORLD_KEY:
            adm_gdf = fetch_admin_vworld(region_name, bbox)
        if adm_gdf is not None:
            merge_and_save(adm_gdf, admin_path, region_name)
        else:
            # 최후 수단: bbox 폴리곤
            s, n, w, e = bbox
            from shapely.geometry import box as bbox_poly
            adm_gdf = gpd.GeoDataFrame(
                {"name": [region_name], "region": [region_name]},
                geometry=[bbox_poly(w, s, e, n)], crs="EPSG:4326"
            )
            merge_and_save(adm_gdf, admin_path, region_name)
            print(f"    [주의] bbox 폴리곤으로 대체 — 정확도 낮음")
    else:
        print(f"  [행정경계] 이미 존재 — 스킵")

    # 전력 인프라
    pow_path = os.path.join(DS, "02_power_infrastructure", "power_infrastructure_top10.geojson")
    pow_exists = False
    if os.path.exists(pow_path):
        p = gpd.read_file(pow_path)
        pow_exists = "region" in p.columns and (p["region"] == region_name).any()
    if not pow_exists or force:
        gdf = fetch_power(region_name, bbox)
        merge_and_save(gdf, pow_path, region_name)
    else:
        print(f"  [전력] 이미 존재 — 스킵")

    # 교통 (IC)
    tr_path = os.path.join(DS, "03_transport", "roads_highways_top10.geojson")
    tr_exists = False
    if os.path.exists(tr_path):
        p = gpd.read_file(tr_path)
        tr_exists = "region" in p.columns and (p["region"] == region_name).any()
    if not tr_exists or force:
        gdf = fetch_transport(region_name, bbox)
        merge_and_save(gdf, tr_path, region_name)
    else:
        print(f"  [교통] 이미 존재 — 스킵")

    # 하천
    riv_path = os.path.join(DS, "04_waterways", "waterways_top10.geojson")
    riv_exists = False
    if os.path.exists(riv_path):
        p = gpd.read_file(riv_path)
        riv_exists = "region" in p.columns and (p["region"] == region_name).any()
    if not riv_exists or force:
        gdf = fetch_waterways(region_name, bbox)
        merge_and_save(gdf, riv_path, region_name)
    else:
        print(f"  [하천] 이미 존재 — 스킵")

    # 농업진흥/보전산지/생태경관 — 국내 정부 API 필요
    for label, folder, fname in [
        ("농업진흥지역", "10_farmland", "farmland_top10.geojson"),
        ("보전산지",   "11_conservation_forest", "conservation_forest_top10.geojson"),
        ("생태경관핵심", "12_ecology_core", "ecology_core_top10.geojson"),
    ]:
        p = os.path.join(DS, folder, fname)
        if not os.path.exists(p):
            print(f"  [{label}] 데이터 없음 — 국토환경성평가지도 API 또는 산림청 API 필요")
            print(f"           수동 다운로드 후 {p} 에 저장하세요")

    print(f"\n  완료: {region_name}")
    return True

# ── 메인 ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미시분석 데이터 자동 수집")
    parser.add_argument("--topn", type=int, default=10, help="거시 Top-N 지역 수집 (기본 10)")
    parser.add_argument("--region", type=str, default=None, help="특정 지역만 수집")
    parser.add_argument("--force", action="store_true", help="기존 데이터 있어도 재수집")
    args = parser.parse_args()

    if args.region:
        regions = [{"name": args.region, "sido": "", "score": 0}]
    else:
        regions = get_macro_topn(args.topn)
        if not regions:
            print("수집할 지역 없음. --region 옵션으로 직접 지정하세요.")
            sys.exit(1)

    print(f"\n수집 대상 ({len(regions)}개): {[r['name'] for r in regions]}")
    print("이미 데이터가 있는 레이어는 자동 스킵됩니다 (--force로 강제 재수집)")

    failed = []
    for r in regions:
        ok = fetch_region(r["name"], force=args.force)
        if not ok:
            failed.append(r["name"])
        time.sleep(2)  # Overpass API 부하 방지

    print(f"\n{'='*50}")
    print(f"수집 완료: {len(regions) - len(failed)}/{len(regions)}개 지역")
    if failed:
        print(f"실패: {failed}")
    print(f"데이터 저장 위치: {DS}")
    print("\n이후 revised_analysis.py를 실행하면 자동으로 새 지역 데이터를 사용합니다.")
