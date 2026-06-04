"""
최종 결과물 생성 스크립트
- 기존 grids/clusters 데이터 활용
- V-World 용도지역 필터 적용
- 루트 폴더에 웹사이트 생성
- PPT용 시각화 생성
"""
import os, sys, json, time, csv, unicodedata, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import requests
from shapely.geometry import box, Point
from shapely.ops import unary_union
from pyproj import Transformer
from scipy.spatial import KDTree
from collections import deque

def nfc(s): return unicodedata.normalize("NFC", str(s))

# ── 경로 ──────────────────────────────────────
DESK = "/Users/lshwa/Desktop/학술제 "

def find_dir(kw1, kw2, exclude="회의"):
    for root, dirs, files in os.walk(DESK):
        r = nfc(root)
        if kw1 in r and kw2 in r and exclude not in r:
            return root
    return None

DS  = find_dir("미시", "datasets")
RES = find_dir("미시", "results")

GRID_DIR   = os.path.join(RES, "top15", "grids")
CLUST_DIR  = os.path.join(RES, "top15", "clusters")
ZONE_CACHE = os.path.join(DS,  "07_zoning")
VIZ_DIR    = os.path.join(RES, "top15", "viz_ppt")
OUT_WEB    = os.path.join(DESK, "datacenter_map.html")

for d in [VIZ_DIR]:
    os.makedirs(d, exist_ok=True)

VWORLD_KEY = "1A94D935-77C2-3F68-AE71-1BD3DD8EDA3A"
CRS     = "EPSG:5179"
GRID_M  = 250
W_MACRO = 0.30
W_MICRO = 0.70
BCOV    = 0.60
MIN_CL  = 4
t5179   = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
t4326   = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

# 개발불가 용도지역 키워드
NO_DEV = ["주거", "보전녹지", "생산녹지", "농림지역", "자연환경보전", "보전관리", "생산관리"]

# ── 거시 데이터 로드 ──────────────────────────
print("[1/6] 거시 데이터 로드")
def find_macro_csv():
    for root, dirs, files in os.walk(DESK):
        for f in files:
            if nfc(f) == "최종_랭킹_144개.csv" and "거시" in nfc(root):
                return os.path.join(root, f)
    return None

with open(find_macro_csv(), encoding="utf-8-sig") as f:
    all_rows = list(csv.DictReader(f))
all_rows.sort(key=lambda r: float(r["종합_점수"]), reverse=True)
top15_rows = all_rows[:15]

mn = min(float(r["종합_점수"]) for r in top15_rows)
mx = max(float(r["종합_점수"]) for r in top15_rows)
for r in top15_rows:
    v = float(r["종합_점수"])
    r["macro_norm"] = (v - mn) / (mx - mn) * 100 if mx > mn else 100.0

MACRO = {r["시군구"]: r for r in top15_rows}
TOP15 = [(r["시군구"], r["시도"], i+1, r["macro_norm"]) for i, r in enumerate(top15_rows)]
print(f"  Top15 로드 완료")

# ── 행정경계 로드 ─────────────────────────────
admin_path = os.path.join(DS, "01_admin_boundaries", "sigungu_all_korea.geojson")
admin_gdf = gpd.read_file(admin_path).to_crs(CRS)

def get_region_bounds(sigg_name, sido):
    reg = admin_gdf[admin_gdf["name"].apply(lambda x: sigg_name in str(x))]
    if len(reg) > 1 and sido:
        sido2 = sido[:2]
        reg2 = reg[reg.apply(lambda r: sido2 in " ".join(str(v) for v in r.values), axis=1)]
        if len(reg2) > 0:
            reg = reg2
    if len(reg) == 0:
        return None
    bounds = reg.to_crs("EPSG:4326").total_bounds
    return bounds

# ── V-World 용도지역 로드 함수 ────────────────
def fetch_zoning(region_key, bounds4326):
    cache = os.path.join(ZONE_CACHE, f"zoning_{region_key}.geojson")
    if os.path.exists(cache):
        try:
            gdf = gpd.read_file(cache)
            if len(gdf) > 0:
                return gdf.to_crs(CRS)
        except:
            pass

    minx, miny, maxx, maxy = bounds4326[0]-0.01, bounds4326[1]-0.01, bounds4326[2]+0.01, bounds4326[3]+0.01
    feats = []
    for attempt in range(3):
        try:
            r = requests.get(
                "https://api.vworld.kr/req/wfs",
                params={
                    "SERVICE":"WFS","VERSION":"2.0.0","REQUEST":"GetFeature",
                    "KEY":VWORLD_KEY,"DOMAIN":"localhost","TYPENAME":"lt_c_uq111",
                    "SRSNAME":"EPSG:4326","OUTPUT":"application/json",
                    "COUNT":"5000","STARTINDEX":"0",
                    "BBOX":f"{minx},{miny},{maxx},{maxy}",
                },
                timeout=60,
            )
            if r.status_code == 200 and r.text.strip().startswith("{"):
                feats = r.json().get("features", [])
                break
            time.sleep(15 * (attempt+1))
        except Exception as e:
            print(f"    API 오류({attempt+1}): {e}")
            time.sleep(15 * (attempt+1))

    gj = {"type": "FeatureCollection", "features": feats}
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)

    if feats:
        return gpd.read_file(cache).to_crs(CRS)
    return gpd.GeoDataFrame(geometry=[], crs=CRS)

def nodev_union(zone_gdf):
    if len(zone_gdf) == 0:
        return None
    for col in ["prpos_area1_nm","uname","zone_name","name"]:
        if col in zone_gdf.columns:
            geoms = []
            for _, row in zone_gdf.iterrows():
                z = str(row.get(col, ""))
                if any(kw in z for kw in NO_DEV):
                    if row.geometry and row.geometry.is_valid:
                        geoms.append(row.geometry)
            return unary_union(geoms) if geoms else None
    return None

# ── BFS 클러스터링 ────────────────────────────
def bfs_clusters(gx, gy, scores, grid_m=GRID_M, min_grids=MIN_CL):
    ix = np.round(gx / grid_m).astype(int)
    iy = np.round(gy / grid_m).astype(int)
    pos2idx = {(ix[i], iy[i]): i for i in range(len(gx))}
    visited = np.zeros(len(gx), dtype=bool)
    clusters = []
    for start in np.argsort(scores)[::-1]:
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
                nb = pos2idx.get((cx+dx, cy+dy))
                if nb is not None and not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        if len(comp) >= min_grids:
            carr = np.array(comp)
            clusters.append((carr, float(scores[carr].mean()), len(comp)))
    clusters.sort(key=lambda x: x[1] * math.sqrt(x[2]), reverse=True)
    return clusters

def grids_to_polygon_4326(gx_5179, gy_5179):
    half = GRID_M / 2
    squares = [box(x-half, y-half, x+half, y+half) for x, y in zip(gx_5179, gy_5179)]
    poly_5179 = unary_union(squares)
    gdf = gpd.GeoDataFrame(geometry=[poly_5179], crs=CRS).to_crs("EPSG:4326")
    return gdf.geometry[0]

# ── 메인 처리 ──────────────────────────────────
print("\n[2/6] 용도지역 필터 + 클러스터 재계산")
print(f"  (V-World API - 지역당 최대 60초 소요)")

# sigungu → grid key 매핑
# 동명 지역: 동구/서구/중구 등
def sigungu_to_key(sigg_name, sido):
    ambiguous = {"서구", "동구", "중구", "남구", "북구"}
    if sigg_name in ambiguous:
        return f"{sido[:2]}_{sigg_name}"
    return sigg_name

# 기존 grid 파일과 Top15 매핑
existing_files = {}
for f in os.listdir(GRID_DIR):
    if f.endswith(".geojson"):
        existing_files[nfc(f[6:-8])] = os.path.join(GRID_DIR, f)

region_results = {}
region_summaries = []

for sigg_name, sido, rank, macro_norm in TOP15:
    rkey = sigungu_to_key(sigg_name, sido)
    print(f"\n  [{rank:2d}위] {sido} {sigg_name} → key={rkey}")

    # grid 파일 찾기
    grid_path = existing_files.get(rkey) or existing_files.get(sigg_name)

    # 특수 케이스: 서구/동구/중구 파일이 ambiguous key로 없을 때 fallback
    if not grid_path:
        base = sigg_name
        grid_path = existing_files.get(base)

    if not grid_path:
        print(f"    grid 파일 없음, SKIP")
        continue

    # grids 로드 (EPSG:4326)
    gdf = gpd.read_file(grid_path)
    gdf_5179 = gdf.to_crs(CRS)
    # Point 또는 Polygon 모두 지원
    if gdf_5179.geometry.geom_type.iloc[0] == "Point":
        gx = gdf_5179.geometry.x.values
        gy = gdf_5179.geometry.y.values
    else:
        gx = gdf_5179.geometry.centroid.x.values
        gy = gdf_5179.geometry.centroid.y.values
    micro = gdf_5179["micro_score"].values if "micro_score" in gdf_5179.columns else np.full(len(gx), 70.0)
    print(f"    격자 {len(gx)}개 로드")

    # 용도지역 필터
    bounds4326 = get_region_bounds(sigg_name, sido)
    n_before = len(gx)
    if bounds4326 is not None:
        zone_gdf = fetch_zoning(rkey, bounds4326)
        nd_u = nodev_union(zone_gdf)
        if nd_u is not None:
            pp = gpd.GeoDataFrame({"i": np.arange(len(gx))},
                geometry=gpd.points_from_xy(gx, gy), crs=CRS)
            in_nd = pp.within(nd_u)
            keep = ~in_nd.values
            gx     = gx[keep]
            gy     = gy[keep]
            micro  = micro[keep]
            print(f"    용도지역 필터: {n_before} → {len(gx)} ({n_before-len(gx)} 제거)")
            time.sleep(8)  # rate limit
        else:
            print(f"    용도지역: 개발불가 영역 없거나 데이터 없음")
            time.sleep(4)
    else:
        print(f"    행정경계 없음")

    if len(gx) == 0:
        print("    격자 없음 SKIP")
        continue

    # final score
    final = W_MACRO * macro_norm + W_MICRO * micro

    # BFS 클러스터링
    clusters = bfs_clusters(gx, gy, final)
    print(f"    클러스터: {len(clusters)}개")

    clust_feats = []
    for c_rank, (comp, avg_s, n_g) in enumerate(clusters[:5], 1):
        cgx, cgy = gx[comp], gy[comp]
        buildable = n_g * GRID_M * GRID_M * BCOV / 10000
        poly4326 = grids_to_polygon_4326(cgx, cgy)
        cx, cy = poly4326.centroid.x, poly4326.centroid.y
        cscore = avg_s * math.sqrt(n_g)
        clust_feats.append({
            "type": "Feature",
            "geometry": poly4326.__geo_interface__,
            "properties": {
                "rank": c_rank, "n_grids": n_g,
                "buildable_ha": round(buildable, 2),
                "avg_final_score": round(avg_s, 2),
                "cluster_score": round(cscore, 2),
                "macro_norm": round(macro_norm, 2),
                "region": f"{sido} {sigg_name}",
                "region_rank": rank,
                "lat": round(cy, 5), "lon": round(cx, 5),
            }
        })

    # 클러스터 GeoJSON 저장 (업데이트)
    clust_key = rkey.replace("/","_")
    clust_path = os.path.join(CLUST_DIR, f"clusters_{clust_key}.geojson")
    with open(clust_path, "w", encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":clust_feats}, f, ensure_ascii=False)

    best = clust_feats[0]["properties"] if clust_feats else {}
    region_results[rkey] = clust_feats

    # grids 업데이트 (final_score 추가)
    top_idx = np.argsort(final)[::-1][:500]
    lons_t, lats_t = t4326.transform(gx[top_idx], gy[top_idx])
    updated = gpd.GeoDataFrame({
        "micro_score": micro[top_idx],
        "final_score": final[top_idx],
        "macro_norm":  macro_norm,
    }, geometry=gpd.points_from_xy(lons_t, lats_t), crs="EPSG:4326")
    updated.to_file(os.path.join(GRID_DIR, f"grids_{clust_key}.geojson"), driver="GeoJSON")

    region_summaries.append({
        "rank": rank, "sido": sido, "sigungu": sigg_name,
        "rkey": rkey,
        "macro_norm":  round(macro_norm, 2),
        "n_grids": len(gx),
        "best_buildable_ha": best.get("buildable_ha", 0),
        "cluster_score": best.get("cluster_score", 0),
        "final_score": best.get("avg_final_score", 0),
        "lat": best.get("lat", 0),
        "lon": best.get("lon", 0),
    })

print("\n  [완료]")

# ── Top 5 도출 ────────────────────────────────
print("\n[3/6] Top 5 최적 입지 도출")
all_clusters = []
for rkey, feats in region_results.items():
    for feat in feats:
        all_clusters.append(feat)
all_clusters.sort(key=lambda f: f["properties"]["cluster_score"], reverse=True)

top5 = []
seen = set()
for feat in all_clusters:
    rg = feat["properties"]["region"]
    if rg not in seen:
        top5.append(feat)
        seen.add(rg)
    if len(top5) == 5:
        break

print("Top 5:")
for i, feat in enumerate(top5, 1):
    p = feat["properties"]
    print(f"  {i}위: {p['region']} - {p['buildable_ha']:.0f}ha, score={p['avg_final_score']:.1f}")

# summary CSV
region_summaries.sort(key=lambda r: r["cluster_score"], reverse=True)
for i, r in enumerate(region_summaries):
    r["micro_rank"] = i + 1

sum_path = os.path.join(RES, "top15", "summary_final.csv")
if region_summaries:
    with open(sum_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in region_summaries[0].keys() if k != "rkey"])
        writer.writeheader()
        for r in region_summaries:
            writer.writerow({k:v for k,v in r.items() if k != "rkey"})

# ── 웹사이트 생성 ──────────────────────────────
print("\n[4/6] 웹사이트 생성")

# 그리드 데이터 준비
grid_js = {}
for clust_key_f in os.listdir(GRID_DIR):
    if not clust_key_f.endswith(".geojson"):
        continue
    try:
        gdf = gpd.read_file(os.path.join(GRID_DIR, clust_key_f))
        score_col = "final_score" if "final_score" in gdf.columns else "micro_score"
        gdf = gdf.sort_values(score_col, ascending=False).head(400)
        rk = nfc(clust_key_f[6:-8])
        pts = []
        for _, row in gdf.iterrows():
            pts.append({
                "lat": round(row.geometry.centroid.y if row.geometry.geom_type != "Point" else row.geometry.y, 5),
                "lon": round(row.geometry.centroid.x if row.geometry.geom_type != "Point" else row.geometry.x, 5),
                "s": round(float(row.get(score_col, 60)), 1),
            })
        grid_js[rk] = pts
    except:
        pass

# 클러스터 데이터 준비
clust_js = {}
for f in os.listdir(CLUST_DIR):
    if not f.endswith(".geojson"):
        continue
    try:
        with open(os.path.join(CLUST_DIR, f), encoding="utf-8") as fp:
            gj = json.load(fp)
        rk = nfc(f[9:-8])
        clust_js[rk] = gj["features"]
    except:
        pass

summary_js = region_summaries
top5_js = [f["properties"] for f in top5]

HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AI 하이퍼스케일 데이터센터 최적 입지 분석</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;background:#0a1628;color:#d0dae8;display:flex;height:100vh;overflow:hidden;}}
#sidebar{{width:320px;min-width:260px;background:#111d2e;display:flex;flex-direction:column;border-right:1px solid #1e3250;overflow:hidden;}}
#shead{{padding:16px 14px 10px;background:linear-gradient(135deg,#0d2547,#071630);border-bottom:1px solid #1e3250;}}
#shead h1{{font-size:13px;font-weight:700;color:#e8f4ff;line-height:1.5;}}
#shead p{{font-size:10px;color:#6a92c0;margin-top:3px;}}
#top5box{{padding:10px 12px;background:#0d1e33;border-bottom:1px solid #1e3250;}}
#top5box h2{{font-size:11px;color:#f4a429;font-weight:700;margin-bottom:6px;letter-spacing:.3px;}}
.t5item{{display:flex;align-items:center;gap:7px;padding:5px 8px;margin-bottom:3px;background:#0a1628;border-radius:5px;cursor:pointer;border:1px solid transparent;transition:.15s;}}
.t5item:hover{{border-color:#f4a429;background:#141f30;}}
.t5num{{width:20px;height:20px;border-radius:50%;background:#f4a429;color:#000;font-size:9px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;}}
.t5info{{flex:1;}}
.t5name{{font-size:11px;font-weight:600;color:#e8f4ff;}}
.t5sub{{font-size:9px;color:#5a7a9a;margin-top:1px;}}
#rlist{{flex:1;overflow-y:auto;padding:6px 10px;}}
#rlist h2{{font-size:10px;color:#5a7a9a;padding:5px 3px;letter-spacing:.3px;border-bottom:1px solid #1e3250;margin-bottom:4px;}}
.ritem{{display:flex;align-items:center;gap:7px;padding:7px 9px;margin-bottom:2px;background:#0a1628;border-radius:5px;cursor:pointer;border:1px solid transparent;transition:.15s;}}
.ritem:hover{{border-color:#2b7a4b;background:#0d1e33;}}
.ritem.active{{border-color:#2b7a4b;background:#0f2518;}}
.rnk{{width:24px;height:24px;border-radius:5px;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;color:#fff;}}
.rinfo{{flex:1;}}
.rname{{font-size:11px;font-weight:600;color:#e8f4ff;}}
.rmeta{{font-size:9px;color:#5a7a9a;margin-top:1px;}}
.sbar{{width:55px;height:4px;background:#1e3250;border-radius:2px;margin-top:3px;}}
.sfill{{height:100%;border-radius:2px;}}
#map{{flex:1;position:relative;}}
.leaflet-popup-content-wrapper{{background:#111d2e;color:#d0dae8;border:1px solid #2a4a6a;border-radius:8px;}}
.leaflet-popup-tip{{background:#111d2e;}}
.leaflet-popup-content{{font-size:12px;line-height:1.6;}}
.leaflet-popup-content b{{color:#f4a429;}}
.legend{{background:#111d2e;padding:10px 13px;border-radius:7px;border:1px solid #1e3250;font-size:10px;color:#d0dae8;}}
.lg-title{{font-weight:700;color:#e8f4ff;margin-bottom:5px;font-size:11px;}}
.lg-row{{display:flex;align-items:center;gap:5px;margin-bottom:3px;}}
.lg-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
</style>
</head>
<body>
<div id="sidebar">
  <div id="shead">
    <h1>AI 하이퍼스케일 데이터센터<br>최적 입지 분석 (비수도권)</h1>
    <p>거시 30% + 미시 70% | 250m 격자 | 건폐율 60%</p>
  </div>
  <div id="top5box">
    <h2>TOP 5 최적 입지</h2>
    <div id="t5list"></div>
  </div>
  <div id="rlist">
    <h2>TOP 15 전체 지역</h2>
    <div id="ritems"></div>
  </div>
</div>
<div id="map"></div>

<script>
const SUMMARY = {json.dumps(summary_js, ensure_ascii=False)};
const TOP5    = {json.dumps(top5_js,    ensure_ascii=False)};
const GRIDS   = {json.dumps(grid_js,    ensure_ascii=False)};
const CLUSTERS= {json.dumps(clust_js,   ensure_ascii=False)};

const SC = s => s>=80?'#e74c3c':s>=70?'#f39c12':s>=60?'#2ecc71':'#3498db';
const COLS = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c','#e91e63','#00bcd4','#8bc34a','#ff5722','#607d8b','#795548','#009688','#673ab7'];

const map = L.map('map',{{center:[36.5,127.8],zoom:7}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',{{
  attribution:'&copy; CartoDB &copy; OSM',maxZoom:18,subdomains:'abcd'
}}).addTo(map);

// 범례
const leg = L.control({{position:'bottomright'}});
leg.onAdd=()=>{{
  const d=L.DomUtil.create('div','legend');
  d.innerHTML=`<div class="lg-title">격자 최종점수</div>
    <div class="lg-row"><div class="lg-dot" style="background:#e74c3c"></div>80점 이상</div>
    <div class="lg-row"><div class="lg-dot" style="background:#f39c12"></div>70-80점</div>
    <div class="lg-row"><div class="lg-dot" style="background:#2ecc71"></div>60-70점</div>
    <div class="lg-row"><div class="lg-dot" style="background:#3498db"></div>60점 미만</div>
    <div style="margin-top:7px;border-top:1px solid #1e3250;padding-top:5px">
    <div class="lg-row"><div class="lg-dot" style="background:#f4a429;border-radius:0"></div>클러스터 영역</div>
    <div class="lg-row" style="color:#f4a429"><b>★</b> Top 5 입지</div>
    </div>`;
  return d;
}};
leg.addTo(map);

const clLayers = {{}};
const clustLayer = L.layerGroup().addTo(map);
const top5Layer  = L.layerGroup().addTo(map);

// 클러스터 폴리곤
Object.entries(CLUSTERS).forEach(([rk,feats])=>{{
  feats.forEach(feat=>{{
    const p=feat.properties;
    const isTop = TOP5.some(t=>t.region===p.region);
    const col = isTop?'#f4a429':'#2b7a9a';
    L.geoJSON(feat,{{
      style:{{color:col,weight:isTop?2.5:1.5,fillColor:col,fillOpacity:isTop?0.2:0.1,
              dashArray:p.rank>1?'4,4':null}},
      onEachFeature:(_,l)=>l.bindPopup(
        `<b>${{p.region}}</b><br>클러스터 #${{p.rank}}<br>`+
        `격자 수: <b>${{p.n_grids.toLocaleString()}}</b>개<br>`+
        `건폐 가능: <b>${{p.buildable_ha.toFixed(0)}} ha</b><br>`+
        `통합점수: <b>${{p.avg_final_score.toFixed(1)}}</b><br>`+
        `거시 정규화: ${{p.macro_norm.toFixed(1)}}`
      )
    }}).addTo(clustLayer);
  }});
}});

// Top5 마커
TOP5.forEach((t,i)=>{{
  if(!t.lat||!t.lon) return;
  const ico=L.divIcon({{className:'',iconAnchor:[50,18],
    html:`<div style="background:#f4a429;color:#111;font-weight:700;font-size:12px;padding:4px 9px;border-radius:16px;white-space:nowrap;box-shadow:0 3px 10px rgba(0,0,0,.5);border:2px solid #fff">Top${{i+1}} ${{t.region.split(' ').slice(1).join(' ')}}</div>`
  }});
  L.marker([t.lat,t.lon],{{icon:ico}})
    .addTo(top5Layer)
    .bindPopup(`<b>Top ${{i+1}} 최적 입지</b><br><b>${{t.region}}</b><br>`+
      `건폐 가능: <b>${{t.buildable_ha.toFixed(0)}} ha</b><br>`+
      `통합점수: <b>${{t.avg_final_score.toFixed(1)}}</b>`);
}});

// Top5 패널
const t5l=document.getElementById('t5list');
TOP5.forEach((t,i)=>{{
  const d=document.createElement('div');
  d.className='t5item';
  d.innerHTML=`<div class="t5num">${{i+1}}</div><div class="t5info">
    <div class="t5name">${{t.region}}</div>
    <div class="t5sub">${{t.buildable_ha.toFixed(0)}}ha 건폐 · 점수 ${{t.avg_final_score.toFixed(1)}}</div>
  </div>`;
  d.onclick=()=>map.setView([t.lat,t.lon],12);
  t5l.appendChild(d);
}});

// 지역 목록
const ri=document.getElementById('ritems');
const sorted=[...SUMMARY].sort((a,b)=>a.rank-b.rank);
const maxS=Math.max(...SUMMARY.map(s=>s.cluster_score));
let activeEl=null;

sorted.forEach((r,i)=>{{
  const col=COLS[i%COLS.length];
  const d=document.createElement('div');
  d.className='ritem';
  const pct=maxS>0?(r.cluster_score/maxS*100):0;
  d.innerHTML=`<div class="rnk" style="background:${{col}}">${{r.rank}}</div>
    <div class="rinfo">
      <div class="rname">${{r.sigungu}} <span style="color:#5a7a9a;font-size:9px">${{r.sido}}</span></div>
      <div class="rmeta">점수 ${{r.final_score.toFixed(1)}} · ${{r.best_buildable_ha.toFixed(0)}}ha</div>
      <div class="sbar"><div class="sfill" style="width:${{pct}}%;background:${{col}}"></div></div>
    </div>`;
  d.onclick=()=>{{
    if(activeEl) activeEl.classList.remove('active');
    d.classList.add('active'); activeEl=d;
    showRegion(r,col);
  }};
  ri.appendChild(d);
}});

function showRegion(r,col){{
  Object.values(clLayers).forEach(l=>map.removeLayer(l));
  const rk=r.rkey||r.sigungu;
  const pts=GRIDS[rk]||GRIDS[r.sigungu]||[];
  if(!clLayers[rk]){{
    const lg=L.layerGroup();
    pts.forEach(pt=>{{
      L.circleMarker([pt.lat,pt.lon],{{
        radius:4,color:SC(pt.s),fillColor:SC(pt.s),fillOpacity:.8,weight:.5
      }}).addTo(lg).bindPopup(`점수: ${{pt.s}}`);
    }});
    clLayers[rk]=lg;
  }}
  clLayers[rk].addTo(map);
  if(r.lat&&r.lon) map.setView([r.lat,r.lon],11);
  else if(pts.length>0) map.setView([pts[0].lat,pts[0].lon],11);
}}
</script>
</body>
</html>"""

with open(OUT_WEB, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"  웹사이트: {OUT_WEB}")

# ── PPT 시각화 생성 ────────────────────────────
print("\n[5/6] PPT 시각화 생성 (이모지 없음)")
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
plt.rcParams["figure.facecolor"] = "white"

PALETTE = ["#1a3c6e","#2d8a4e","#e67e22","#8e44ad","#c0392b",
           "#16a085","#7f8c8d","#d35400","#27ae60","#2980b9",
           "#f39c12","#1abc9c","#9b59b6","#e74c3c","#3498db"]
HIGHLIGHT = "#e74c3c"
BASE      = "#2c3e50"

sums = [r for r in region_summaries if r["final_score"] > 0]
top5_regions = {t["region"] for t in top5}

# viz1: 최종 순위 바차트 (가로)
fig, ax = plt.subplots(figsize=(14, 8))
sums_s = sorted(sums, key=lambda r: r["final_score"])
labels = [f"{r['sigungu']} ({r['sido']})" for r in sums_s]
vals   = [r["final_score"] for r in sums_s]
colors_bar = [HIGHLIGHT if f"{r['sido']} {r['sigungu']}" in top5_regions else BASE for r in sums_s]
bars = ax.barh(range(len(sums_s)), vals, color=colors_bar, edgecolor="white", linewidth=0.4, height=0.6)
ax.set_yticks(range(len(sums_s)))
ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlabel("통합 점수 (거시 30% + 미시 최고 클러스터 70%)", fontsize=11)
ax.set_title("AI 하이퍼스케일 데이터센터 최적 입지\n미시분석 최종 랭킹", fontsize=15, fontweight="bold", pad=12)
for bar, val in zip(bars, vals):
    ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9.5)
ax.set_xlim(0, max(vals)*1.12 if vals else 100)
ax.axvline(min(vals)*0.95 if vals else 0, color="gray", linestyle=":", alpha=0.4)
h1 = mpatches.Patch(color=HIGHLIGHT, label="Top 5 최적 입지")
h2 = mpatches.Patch(color=BASE, label="나머지 Top 10")
ax.legend(handles=[h1,h2], loc="lower right", fontsize=10)
ax.grid(axis="x", alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz1_final_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz1_final_ranking.png")

# viz2: 건폐 면적 vs 통합 점수 산점도
fig, ax = plt.subplots(figsize=(12, 8))
for i, r in enumerate(sums):
    is_top = f"{r['sido']} {r['sigungu']}" in top5_regions
    marker = "^" if is_top else "o"
    size   = 300 if is_top else 120
    color  = HIGHLIGHT if is_top else PALETTE[i % len(PALETTE)]
    ax.scatter(r["best_buildable_ha"], r["final_score"],
               s=size, color=color, marker=marker, zorder=5, edgecolors="white", linewidth=0.5)
    ax.annotate(r["sigungu"], (r["best_buildable_ha"], r["final_score"]),
                fontsize=8.5, ha="center", va="bottom",
                xytext=(0,9), textcoords="offset points",
                color=HIGHLIGHT if is_top else "#555")
ax.set_xlabel("최적 클러스터 건폐 가능 면적 (ha)", fontsize=12)
ax.set_ylabel("통합 점수 (거시 30% + 미시 70%)", fontsize=12)
ax.set_title("건폐 가능 면적 vs 입지 통합 점수\n삼각형 = Top 5 최적 입지", fontsize=13, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz2_area_vs_score.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz2_area_vs_score.png")

# viz3: 거시 vs 미시 산점도
fig, ax = plt.subplots(figsize=(12, 8))
for i, r in enumerate(sums):
    macro_raw = float(MACRO.get(r["sigungu"], {}).get("종합_점수", 65))
    micro_est = r["final_score"]
    is_top = f"{r['sido']} {r['sigungu']}" in top5_regions
    marker = "^" if is_top else "o"
    size   = 300 if is_top else 120
    color  = HIGHLIGHT if is_top else PALETTE[i % len(PALETTE)]
    ax.scatter(macro_raw, micro_est, s=size, color=color, marker=marker, zorder=5,
               edgecolors="white", linewidth=0.5)
    ax.annotate(r["sigungu"], (macro_raw, micro_est),
                fontsize=8.5, ha="center", va="bottom",
                xytext=(0,9), textcoords="offset points",
                color=HIGHLIGHT if is_top else "#555")
ax.set_xlabel("거시분석 종합 점수", fontsize=12)
ax.set_ylabel("미시분석 최종 통합 점수", fontsize=12)
ax.set_title("거시 점수 vs 미시 통합 점수 분포\n삼각형 = Top 5 최적 입지", fontsize=13, fontweight="bold")
ax.grid(alpha=0.25, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "viz3_macro_vs_micro.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz3_macro_vs_micro.png")

# viz4: Top 5 상세 바 차트 (점수 구성)
if top5:
    fig, ax = plt.subplots(figsize=(12, 6))
    top5names = [t["region"].split()[-1] for t in top5_js]
    final_scores = [t.get("avg_final_score",0) for t in top5_js]
    macro_contrib = [t.get("macro_norm",0) * W_MACRO for t in top5_js]
    micro_contrib = [(t.get("avg_final_score",0) - t.get("macro_norm",0) * W_MACRO) for t in top5_js]

    x = np.arange(len(top5names))
    w = 0.55
    b1 = ax.bar(x, macro_contrib, w, label=f"거시 기여 ({int(W_MACRO*100)}%)", color="#2980b9", edgecolor="white")
    b2 = ax.bar(x, micro_contrib, w, bottom=macro_contrib, label=f"미시 기여 ({int(W_MICRO*100)}%)", color="#27ae60", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top{i+1}\n{n}" for i,n in enumerate(top5names)], fontsize=11)
    ax.set_ylabel("통합 점수", fontsize=12)
    ax.set_title("Top 5 최적 입지 점수 구성 (거시 30% + 미시 70%)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, val in enumerate(final_scores):
        ax.text(i, val+0.5, f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "viz4_top5_detail.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  viz4_top5_detail.png")

# viz5: 건폐 면적 막대 (Top 15 전체)
if sums:
    sums_area = sorted(sums, key=lambda r: r["best_buildable_ha"], reverse=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    colors_a = [HIGHLIGHT if f"{r['sido']} {r['sigungu']}" in top5_regions else PALETTE[i%len(PALETTE)]
                for i, r in enumerate(sums_area)]
    bars = ax.bar(range(len(sums_area)), [r["best_buildable_ha"] for r in sums_area],
                  color=colors_a, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(sums_area)))
    ax.set_xticklabels([f"{r['sigungu']}\n({r['sido']})" for r in sums_area], fontsize=9, rotation=20)
    ax.set_ylabel("최적 클러스터 건폐 가능 면적 (ha)", fontsize=12)
    ax.set_title("지역별 최적 클러스터 건폐 가능 면적\n(250m 격자, 건폐율 60%)", fontsize=13, fontweight="bold")
    for i, r in enumerate(sums_area):
        ax.text(i, r["best_buildable_ha"]+30, f"{r['best_buildable_ha']:.0f}", ha="center", fontsize=8.5)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "viz5_buildable_area.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  viz5_buildable_area.png")

print(f"\n  시각화 저장: {nfc(VIZ_DIR)}")

# ── 방법론 MD 저장 ──────────────────────────────
print("\n[6/6] 방법론 문서 저장")
METHOD_MD = f"""# 미시분석 방법론 및 데이터 출처

## 분석 개요
비수도권 AI 하이퍼스케일 데이터센터 최적 입지 선정을 위한 250m 격자 기반 미시분석

## 격자 체계
- 격자 크기: 250m × 250m (EPSG:5179 Korean 2000 / Central Belt 2010)
- 건폐율: 60% (하이퍼스케일 DC 기준)
- 1격자 건폐 가능 면적: 250 × 250 × 0.60 = 37,500 m² (3.75 ha)

## 하드 필터 (개발 불가 격자 제거)
1. 경사도 > 15도 (SRTM DEM 기반, CGIAR-CSI)
2. 보전산지 (국가 SHP - 산림청)
3. 농업진흥지역 (국가 SHP - 농림축산식품부)
4. 생태경관핵심보전지역 (국가 SHP - 환경부)
5. 주거지역 / 군사시설 (OpenStreetMap)
6. 용도지역 개발불가 구역 (V-World WFS API - 국토부)
   - 주거지역, 보전녹지, 생산녹지, 농림지역, 자연환경보전지역, 보전관리, 생산관리

## 소프트 점수 (가중 합산)
| 항목 | 가중치 | 산정 기준 |
|------|--------|----------|
| 교통 접근성 | 35% | IC/TG 거리 기반 (0-100점) |
| 경사도 비용 | 30% | 5도 이하=100, 5-10도=70, 10-15도=30 |
| 냉각수 접근성 | 15% | 하천까지 거리 (2km 이내=100) |
| 산업단지 근접 | 15% | 산업단지까지 거리 (5km 이내=100) |
| 민원회피 | 5% | 주거지역과의 이격거리 |

## 클러스터링 방법론
- 알고리즘: BFS (너비 우선 탐색) 연결성분 탐색
- 인접 정의: 4-방향 (상하좌우) 250m 격자 인접
- 최소 규모: 4격자 이상 (100,000 m², 6만 m² 건폐 가능)
- 클러스터 점수 = 평균점수 × sqrt(격자수)  [규모와 품질의 조화]

## 최종 통합 점수
- 거시분석 30% + 미시분석 70%
- 거시점수 정규화: Top15 내 min-max 정규화 (0-100)
- 미시점수: 최적 클러스터 평균 최종점수

## 데이터 출처
| 데이터 | 출처 | 해상도/형식 |
|--------|------|------------|
| DEM (고도) | SRTM 30m (CGIAR-CSI) | 30m, GeoTIFF |
| 도로망/수계 | OpenStreetMap (Overpass API) | 실시간 |
| 보전산지 | 산림청 산지정보 시스템 | 1:25,000 SHP |
| 농업진흥지역 | 농림축산식품부 | 1:25,000 SHP |
| 생태경관핵심 | 환경부 국가생태정보 포털 | SHP |
| 용도지역지구 | 국토부 V-World WFS API | lt_c_uq111 |
| 산업단지 | 한국산업단지공단 | 좌표 지오코딩 |
| 행정경계 | KOSTAT / OpenStreetMap | GeoJSON |

## 최종 결과
"""

for r in sorted(region_summaries, key=lambda r: r.get("micro_rank", 99)):
    METHOD_MD += f"- {r.get('micro_rank','')}위: {r['sido']} {r['sigungu']} (건폐 {r['best_buildable_ha']:.0f}ha, 통합점수 {r['final_score']:.1f})\n"

md_path = os.path.join(RES, "top15", "미시분석_방법론_및_데이터출처.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(METHOD_MD)
print(f"  방법론 MD: {nfc(md_path)}")

print("\n" + "="*55)
print("최종 결과물 생성 완료")
print(f"  웹사이트: {nfc(OUT_WEB)}")
print(f"  시각화:  {nfc(VIZ_DIR)}")
print(f"  방법론:  {nfc(md_path)}")
print("="*55)
