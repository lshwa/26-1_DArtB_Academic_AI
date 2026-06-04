"""
OSM 데이터 수집 스크립트 (Overpass API)
Top 10 시군구 대상: 변전소, 송전선, 고속도로IC, 주요도로, 하천
"""
import requests
import json
import time
import os

BASE = os.path.dirname(os.path.abspath(__file__))

TOP10 = [
    {"시도": "전남", "시군구": "해남군"},
    {"시도": "전남", "시군구": "영광군"},
    {"시도": "전북", "시군구": "익산시"},
    {"시도": "강원", "시군구": "정선군"},
    {"시도": "충북", "시군구": "단양군"},
    {"시도": "강원", "시군구": "원주시"},
    {"시도": "전남", "시군구": "고흥군"},
    {"시도": "강원", "시군구": "영월군"},
    {"시도": "전남", "시군구": "영암군"},
    {"시도": "강원", "시군구": "횡성군"},
]

OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def overpass_query(query, retries=3):
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  재시도 {attempt+1}/{retries}: {e}")
            time.sleep(10)
    return None


def osm_to_geojson(data):
    """Overpass JSON -> GeoJSON FeatureCollection"""
    features = []
    node_map = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            node_map[el["id"]] = (el.get("lon", 0), el.get("lat", 0))
    for el in data.get("elements", []):
        props = el.get("tags", {})
        props["osm_id"] = el["id"]
        props["osm_type"] = el["type"]
        if el["type"] == "node" and "lat" in el:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]},
                "properties": props
            })
        elif el["type"] == "way":
            coords = [node_map[n] for n in el.get("nodes", []) if n in node_map]
            if len(coords) >= 2:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": props
                })
    return {"type": "FeatureCollection", "features": features}


def fetch_power_infrastructure():
    print("\n[전력 인프라] 변전소 + 송전선 수집 중...")
    all_features = []
    for region in TOP10:
        name = region["시군구"]
        print(f"  -> {name}")
        query = f"""
[out:json][timeout:90];
area["name"="{name}"]["boundary"="administrative"]->.searchArea;
(
  node["power"="substation"](area.searchArea);
  way["power"="substation"](area.searchArea);
  node["power"="tower"](area.searchArea);
  way["power"="line"](area.searchArea);
  way["power"="minor_line"](area.searchArea);
);
out body;
>;
out skel qt;
"""
        data = overpass_query(query)
        if data:
            gj = osm_to_geojson(data)
            for f in gj["features"]:
                f["properties"]["region"] = name
                f["properties"]["sido"] = region["시도"]
            all_features.extend(gj["features"])
            print(f"     {len(gj['features'])}개 요소")
        time.sleep(4)

    out = {"type": "FeatureCollection", "features": all_features}
    path = os.path.join(BASE, "02_power_infrastructure", "power_infrastructure_top10.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path} (총 {len(all_features)}개)")


def fetch_transport():
    print("\n[교통] 고속도로 IC + 주요도로 수집 중...")
    all_features = []
    for region in TOP10:
        name = region["시군구"]
        print(f"  -> {name}")
        query = f"""
[out:json][timeout:90];
area["name"="{name}"]["boundary"="administrative"]->.searchArea;
(
  node["highway"="motorway_junction"](area.searchArea);
  way["highway"~"motorway|trunk|primary|secondary"](area.searchArea);
);
out body;
>;
out skel qt;
"""
        data = overpass_query(query)
        if data:
            gj = osm_to_geojson(data)
            for f in gj["features"]:
                f["properties"]["region"] = name
                f["properties"]["sido"] = region["시도"]
            all_features.extend(gj["features"])
            print(f"     {len(gj['features'])}개 요소")
        time.sleep(4)

    out = {"type": "FeatureCollection", "features": all_features}
    path = os.path.join(BASE, "03_transport", "roads_highways_top10.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path} (총 {len(all_features)}개)")


def fetch_waterways():
    print("\n[수자원] 하천 + 저수지 수집 중...")
    all_features = []
    for region in TOP10:
        name = region["시군구"]
        print(f"  -> {name}")
        query = f"""
[out:json][timeout:90];
area["name"="{name}"]["boundary"="administrative"]->.searchArea;
(
  way["waterway"~"river|stream|canal"](area.searchArea);
  way["natural"="water"](area.searchArea);
  way["landuse"="reservoir"](area.searchArea);
  node["natural"="water"]["water"="reservoir"](area.searchArea);
);
out body;
>;
out skel qt;
"""
        data = overpass_query(query)
        if data:
            gj = osm_to_geojson(data)
            for f in gj["features"]:
                f["properties"]["region"] = name
                f["properties"]["sido"] = region["시도"]
            all_features.extend(gj["features"])
            print(f"     {len(gj['features'])}개 요소")
        time.sleep(4)

    out = {"type": "FeatureCollection", "features": all_features}
    path = os.path.join(BASE, "04_waterways", "waterways_top10.geojson")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  저장: {path} (총 {len(all_features)}개)")


def fetch_admin_boundaries():
    print("\n[행정경계] 전국 시군구 폴리곤 다운로드 중...")
    url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2015/json/skorea_municipalities_geo_simple.json"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        full_gj = r.json()

        target = {r["시군구"] for r in TOP10}
        filtered_feats = []
        for f in full_gj["features"]:
            pname = f["properties"].get("name", "")
            if pname in target:
                filtered_feats.append(f)

        top10_gj = {"type": "FeatureCollection", "features": filtered_feats}
        path_top10 = os.path.join(BASE, "01_admin_boundaries", "sigungu_top10.geojson")
        with open(path_top10, "w", encoding="utf-8") as f:
            json.dump(top10_gj, f, ensure_ascii=False, indent=2)
        print(f"  저장 (Top10 경계): {path_top10} ({len(filtered_feats)}개 폴리곤)")

        path_all = os.path.join(BASE, "01_admin_boundaries", "sigungu_all_korea.geojson")
        with open(path_all, "w", encoding="utf-8") as f:
            json.dump(full_gj, f, ensure_ascii=False, indent=2)
        print(f"  저장 (전국): {path_all} ({len(full_gj['features'])}개 폴리곤)")
    except Exception as e:
        print(f"  다운로드 실패: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("미시분석 OSM 데이터 수집")
    print("=" * 60)
    fetch_admin_boundaries()
    fetch_power_infrastructure()
    fetch_transport()
    fetch_waterways()
    print("\n완료!")
