"""
07_zoning: V-World WFS 용도지역 (전체 다운로드 후 시군구별 client-side 필터)
08_flood_risk: safemap.go.kr WMS GeoTIFF 침수위험
09_landslide_risk: sansatai.forest.go.kr gridValue API (EPSG:5181 좌표 변환 적용)
"""
import requests, json, os, time
from pyproj import Transformer

BASE = os.path.dirname(os.path.abspath(__file__))
VWORLD_KEY = '1A94D935-77C2-3F68-AE71-1BD3DD8EDA3A'

TOP10 = [
    {"name":"해남군","sido":"전남","rank":1,"bbox4326":[126.17,34.26,126.92,34.72],"bbox5186":[130292,88684,175637,140960]},
    {"name":"영광군","sido":"전남","rank":2,"bbox4326":[126.38,35.20,126.90,35.47],"bbox5186":[107726,185437,167701,215260]},
    {"name":"익산시","sido":"전북","rank":3,"bbox4326":[126.84,35.86,127.09,36.14],"bbox5186":[186588,265117,212589,295419]},
    {"name":"정선군","sido":"강원","rank":4,"bbox4326":[128.56,37.19,128.96,37.46],"bbox5186":[333342,407027,375982,456487]},
    {"name":"단양군","sido":"충북","rank":5,"bbox4326":[128.19,36.83,128.56,37.17],"bbox5186":[308288,367922,346935,407442]},
    {"name":"원주시","sido":"강원","rank":6,"bbox4326":[127.79,37.17,128.25,37.54],"bbox5186":[266052,405220,307939,445676]},
    {"name":"고흥군","sido":"전남","rank":7,"bbox4326":[127.12,34.40,127.52,34.83],"bbox5186":[207509,91403,250086,148038]},
    {"name":"영월군","sido":"강원","rank":8,"bbox4326":[128.31,37.04,128.97,37.37],"bbox5186":[298173,393540,369911,435132]},
    {"name":"영암군","sido":"전남","rank":9,"bbox4326":[126.50,34.69,126.93,34.94],"bbox5186":[141444,129231,187134,161178]},
    {"name":"횡성군","sido":"강원","rank":10,"bbox4326":[127.79,37.36,128.29,37.74],"bbox5186":[267157,422748,315098,465107]},
]


def fetch_flood_risk():
    print("\n[08_flood_risk] safemap WMS GeoTIFF 침수위험 수집...")
    out_dir = os.path.join(BASE, "08_flood_risk")
    WMS = "https://www.safemap.go.kr/geoserver_pos/wms"
    for region in TOP10:
        name = region["name"]
        bb = region["bbox5186"]
        # 이미 수집된 경우 건너뜀
        if os.path.exists(os.path.join(out_dir, f"flood_risk_{name}.tif")):
            print(f"  -> {name} (이미 존재, 건너뜀)")
            continue
        print(f"  -> {name}")
        for key, fname_prefix in [("2025_A2SM_FLOODFOVRRISK0","flood_risk"),("2025_A2SM_FLOODDAMAGE","flood_damage")]:
            params = {"SERVICE":"WMS","VERSION":"1.3.0","REQUEST":"GetMap",
                      "LAYERS":f"safemap:{key}","CRS":"EPSG:5186",
                      "BBOX":f"{bb[0]},{bb[1]},{bb[2]},{bb[3]}",
                      "WIDTH":"1024","HEIGHT":"1024","FORMAT":"image/geotiff","TRANSPARENT":"true"}
            r = requests.get(WMS, params=params, timeout=60, headers={"User-Agent":"Mozilla/5.0"})
            ct = r.headers.get("Content-Type","")
            if "tiff" in ct.lower() and len(r.content)>10000:
                path = os.path.join(out_dir, f"{fname_prefix}_{name}.tif")
                with open(path,"wb") as f: f.write(r.content)
                print(f"     {fname_prefix}: {len(r.content)//1024}KB 저장")
            else:
                print(f"     {key}: 데이터없음 (ct={ct[:40]})")
            time.sleep(1)


def fetch_zoning():
    """
    V-World WFS lt_c_uq111 (용도지역지구) 전체 다운로드 후 시군구별 client-side 필터링.
    CQL_FILTER 서버사이드 미적용 문제 우회: 페이지네이션으로 전국 데이터를 받아 sigg_name으로 분류.
    """
    print("\n[07_zoning] V-World WFS 용도지역 수집 (전국 페이지네이션 후 시군구 필터)...")
    out_dir = os.path.join(BASE, "07_zoning")
    WFS_URL = "https://api.vworld.kr/req/wfs"
    PAGE = 1000

    # 우선 전체 건수 확인
    r0 = requests.get(WFS_URL, params={
        "SERVICE":"WFS","VERSION":"2.0.0","REQUEST":"GetFeature",
        "KEY":VWORLD_KEY,"DOMAIN":"localhost","TYPENAME":"lt_c_uq111",
        "SRSNAME":"EPSG:4326","OUTPUT":"application/json","COUNT":"1","STARTINDEX":"0"
    }, timeout=30)
    total = int(r0.json().get("totalFeatures", 0))
    print(f"  전국 총 용도지역 피처 수: {total:,}개 → 페이지 {(total//PAGE)+1}회 요청 예정")

    # 시군구별 bucket
    buckets = {r["name"]: [] for r in TOP10}
    other_count = 0

    start = 0
    page_num = 0
    done = False
    while start <= total and not done:
        page_num += 1
        feats = []
        for attempt in range(4):  # 최대 4회 재시도
            try:
                r = requests.get(WFS_URL, params={
                    "SERVICE":"WFS","VERSION":"2.0.0","REQUEST":"GetFeature",
                    "KEY":VWORLD_KEY,"DOMAIN":"localhost","TYPENAME":"lt_c_uq111",
                    "SRSNAME":"EPSG:4326","OUTPUT":"application/json",
                    "COUNT":str(PAGE),"STARTINDEX":str(start)
                }, timeout=60)
                body = r.text.strip()
                if r.status_code != 200 or not body:
                    wait = 15 * (attempt + 1)
                    print(f"  페이지 {page_num} 실패(HTTP {r.status_code}, 본문={len(body)}), {wait}s 대기 후 재시도")
                    time.sleep(wait)
                    continue
                feats = r.json().get("features", [])
                break  # 성공
            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"  페이지 {page_num} 예외({attempt+1}): {e}, {wait}s 대기")
                time.sleep(wait)

        if not feats:
            print(f"  페이지 {page_num} 재시도 소진 → 종료")
            done = True
            break

        for f in feats:
            sname = f.get("properties", {}).get("sigg_name", "")
            if sname in buckets:
                buckets[sname].append(f)
            else:
                other_count += 1
        collected = sum(len(v) for v in buckets.values())
        print(f"  페이지 {page_num}: {start}~{start+len(feats)-1} "
              f"(수집대상={collected}개, 기타={other_count}개)")
        if len(feats) < PAGE:
            done = True
        start += PAGE
        time.sleep(2)

    # 시군구별 저장
    for region in TOP10:
        name = region["name"]
        feats = buckets[name]
        gj = {"type":"FeatureCollection","features":feats}
        path = os.path.join(out_dir, f"zoning_{name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gj, f, ensure_ascii=False)
        print(f"  저장: zoning_{name}.geojson ({len(feats)}개)")


def fetch_landslide():
    """
    sansatai.forest.go.kr SuperMap iServer gridValue API.
    bbox4326을 EPSG:5181로 변환 후 500m 격자 샘플링.
    """
    print("\n[09_landslide_risk] 산사태위험 격자 샘플링 (EPSG:5181, 500m 간격)...")
    out_dir = os.path.join(BASE, "09_landslide_risk")
    GRID_URL = ("https://sansatai.forest.go.kr/gis1/ictout/iserver"
                "/services/data-HazardParam2025/rest/data"
                "/datasources/HazardParam2025/datasets/hazard2025/gridValue.json")
    STEP = 500  # meters in EPSG:5181

    # WGS84 → EPSG:5181
    t = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)

    for region in TOP10:
        name = region["name"]
        lon0, lat0, lon1, lat1 = region["bbox4326"]
        x0, y0 = t.transform(lon0, lat0)
        x1, y1 = t.transform(lon1, lat1)

        xs = list(range(int(x0), int(x1), STEP))
        ys = list(range(int(y0), int(y1), STEP))
        total_pts = len(xs) * len(ys)
        print(f"  -> {name}: {len(xs)}×{len(ys)}={total_pts}점 (EPSG:5181 범위: x={int(x0)}~{int(x1)}, y={int(y0)}~{int(y1)})")

        features = []
        for xi, x in enumerate(xs):
            for y in ys:
                try:
                    r = requests.get(GRID_URL, params={"x": x, "y": y}, timeout=8,
                                     headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code == 200:
                        val = r.json().get("value", None)
                        try:
                            grade = int(val) if val is not None and int(val) in [1,2,3,4,5] else None
                        except (ValueError, TypeError):
                            grade = None
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [x, y]},
                            "properties": {
                                "x_5181": x, "y_5181": y,
                                "hazard_grade": grade,
                                "region": name
                            }
                        })
                except Exception:
                    pass

            if (xi + 1) % 5 == 0 or (xi + 1) == len(xs):
                valid = sum(1 for f in features if f["properties"]["hazard_grade"] is not None)
                print(f"     진행: {(xi+1)*len(ys)}/{total_pts}, 유효={valid}")
            time.sleep(0.02)

        out = {
            "type": "FeatureCollection",
            "features": features,
            "crs": {"type": "name", "properties": {"name": "EPSG:5181"}}
        }
        path = os.path.join(out_dir, f"landslide_{name}.geojson")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        valid_c = sum(1 for feat in features if feat["properties"]["hazard_grade"] is not None)
        print(f"     저장 완료: {len(features)}점, 유효등급={valid_c}개")


if __name__ == "__main__":
    print("=" * 60)
    print("데이터 재수집: 07_zoning + 09_landslide_risk (+ 08_flood_risk 보완)")
    print("=" * 60)
    fetch_zoning()
    fetch_landslide()
    fetch_flood_risk()
    print("\n모든 수집 완료!")
