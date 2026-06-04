"""
데이터센터 입지 심층 분석:
 - 거시 점수(종합) + 미시 Soft Score -> 통합 점수 (거시 30% + 미시 70%)
 - 격자별 Hard Filter 실패 이유 추적
 - 전국 TOP 10 격자 선정 (실제 AI DC 후보지)
"""
import subprocess, os, warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box, Point
from shapely.ops import unary_union
from scipy.spatial import KDTree
from pyproj import Transformer
import folium
from branca.colormap import LinearColormap

# == 경로 설정 ==
def find_ds():
    r = subprocess.run(
        ["find", "/Users/lshwa/Desktop/이승화/학교/동아리/DArt-B/학술제/26-1학기",
         "-maxdepth", "4", "-name", "09_landslide_risk", "-type", "d"],
        capture_output=True, text=True)
    return os.path.dirname(r.stdout.strip())

DS = find_ds()
NB_DIR    = os.path.dirname(os.path.abspath(__file__))
MICRO_ROOT = os.path.dirname(NB_DIR)
RES_DATA  = os.path.join(MICRO_ROOT, "results", "data")
RES_MAPS  = os.path.join(MICRO_ROOT, "results", "maps")
os.makedirs(RES_DATA, exist_ok=True)
os.makedirs(RES_MAPS, exist_ok=True)
print(f"DS: {DS}")

# == 거시 점수 로드 ==
def load_macro():
    r = subprocess.run(
        ["find", "/Users/lshwa/Desktop/이승화/학교/동아리/DArt-B/학술제/26-1학기",
         "-name", "최종_랭킹_144개.csv"], capture_output=True, text=True)
    paths = [p for p in r.stdout.strip().split("\n") if p.strip()]
    chosen = next((p for p in paths if "거시분석 시작" in p or "거시분석 시작" in p or "거시분석 시작" in p or "거시분석 시작" in p), paths[0])
    df = pd.read_csv(chosen)
    targets = ["해남군","영광군","익산시","정선군","단양군","원주시","고흥군","영월군","영암군","횡성군"]
    sub = df[df["시군구"].isin(targets)].copy()
    mn, mx = sub["종합_점수"].min(), sub["종합_점수"].max()
    sub["macro_norm"] = (sub["종합_점수"] - mn) / (mx - mn) * 100
    return sub.set_index("시군구").to_dict("index")

MACRO = load_macro()
print("\n[거시 점수 로드]")
for k in sorted(MACRO, key=lambda x: MACRO[x]["랭킹"]):
    v = MACRO[k]
    print(f"  {v['랭킹']}위 {k}({v['시도']}): 종합={v['종합_점수']:.4f} -> 정규화={v['macro_norm']:.1f}")

# == 상수 ==
CRS      = "EPSG:5179"
GRID_M   = 250
SLOPE_MAX = 5.0
SCORE_W  = {"power":0.35,"transport":0.25,"slope":0.20,"water":0.12,"periphery":0.08}
W_MACRO  = 0.30
W_MICRO  = 0.70

TOP10 = [
    ("해남군","전남",1),("영광군","전남",2),("익산시","전북",3),
    ("정선군","강원",4),("단양군","충북",5),("원주시","강원",6),
    ("고흥군","전남",7),("영월군","강원",8),("영암군","전남",9),
    ("횡성군","강원",10),
]
REJECT_LABELS = {
    "slope":    "경사도>5도",
    "landslide":"산사태1-2등급",
    "farmland": "농업진흥지역",
    "forest":   "보전산지",
    "ecology":  "생태경관핵심",
    "PASS":     "통과",
}

# == 공통 데이터 로드 ==
print("\n[공통 데이터 로드]")
admin       = gpd.read_file(os.path.join(DS,"01_admin_boundaries","sigungu_top10.geojson")).to_crs(CRS)
for col in ["name","NAME_KOR","name_kor","SIG_KOR_NM"]:
    if col in admin.columns:
        admin["name"] = admin[col].fillna("")
        break
power_gdf   = gpd.read_file(os.path.join(DS,"02_power_infrastructure","power_infrastructure_top10.geojson")).to_crs(CRS)
transport   = gpd.read_file(os.path.join(DS,"03_transport","roads_highways_top10.geojson")).to_crs(CRS)
waterways   = gpd.read_file(os.path.join(DS,"04_waterways","waterways_top10.geojson")).to_crs(CRS)
farmland    = gpd.read_file(os.path.join(DS,"10_farmland","farmland_top10.geojson")).to_crs(CRS)
cons_forest = gpd.read_file(os.path.join(DS,"11_conservation_forest","conservation_forest_top10.geojson")).to_crs(CRS)
ecology     = gpd.read_file(os.path.join(DS,"12_ecology_core","ecology_core_top10.geojson")).to_crs(CRS)

substations = power_gdf[power_gdf.get("power","") == "substation"].copy()
if len(substations)==0:
    substations = power_gdf[power_gdf.geometry.geom_type=="Point"].copy()
ic          = transport[transport.get("highway","").eq("motorway_junction")].copy() if "highway" in transport.columns else gpd.GeoDataFrame()
rivers      = waterways[waterways.get("waterway","").isin(["river","stream"]) |
                        waterways.get("natural","").eq("water")].copy() if "waterway" in waterways.columns else waterways

def pts_xy(gdf):
    cx = gdf.geometry.centroid.x if gdf.geometry.geom_type.iloc[0]!="Point" else gdf.geometry.x
    cy = gdf.geometry.centroid.y if gdf.geometry.geom_type.iloc[0]!="Point" else gdf.geometry.y
    return np.c_[cx.values, cy.values]

pow_tree = KDTree(pts_xy(substations)) if len(substations) else None
ic_tree  = KDTree(pts_xy(ic))          if len(ic)          else None
riv_tree = KDTree(pts_xy(rivers))      if len(rivers)       else None

# 전역 union (소규모 레이어만)
farm_union  = unary_union(farmland.geometry)  if len(farmland)  else None
eco_union   = unary_union(ecology.geometry)   if len(ecology)   else None
print(f"  변전소: {len(substations)}개 | IC: {len(ic)}개 | 하천: {len(rivers)}개")
print(f"  농업진흥: {len(farmland)}개 (union 완료) | 보전산지: {len(cons_forest)}개 (지역별 clip) | 생태경관: {len(ecology)}개 (union 완료)")

# DEM 파일
dem_paths = [p for p in [
    os.path.join(DS,"05_dem","srtm_62_05.tif"),
    os.path.join(DS,"05_dem","srtm_62_06.tif"),
] if os.path.exists(p)]

t4326_tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

def batch_slopes(gx, gy):
    """EPSG:5179 좌표 배열 -> 경사도(도) 배열. micro_analysis.py 동일 방식."""
    from rasterio.transform import rowcol as rrc
    lons, lats = t4326_tr.transform(gx, gy)
    slopes = np.full(len(gx), np.nan)
    for dem_p in dem_paths:
        if not os.path.exists(dem_p):
            continue
        with rasterio.open(dem_p) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            res_deg = abs(src.transform.a)
            res_m   = res_deg * 111320
            dy_arr, dx_arr = np.gradient(data, res_m, res_m)
            slope_arr = np.degrees(np.arctan(np.sqrt(dx_arr**2 + dy_arr**2)))
            for i, (lon, lat) in enumerate(zip(lons, lats)):
                if not np.isnan(slopes[i]):
                    continue
                if not (src.bounds.left <= lon <= src.bounds.right and
                        src.bounds.bottom <= lat <= src.bounds.top):
                    continue
                try:
                    row, col = rrc(src.transform, lon, lat)
                    if 0 <= row < slope_arr.shape[0] and 0 <= col < slope_arr.shape[1]:
                        slopes[i] = slope_arr[row, col]
                except Exception:
                    pass
    return slopes

def load_scored_map(name):
    path = os.path.join(RES_DATA, f"scored_{name}.geojson")
    if not os.path.exists(path): return {}
    with open(path) as f:
        gj = json.load(f)
    return {(round(ft["properties"]["x"],0), round(ft["properties"]["y"],0)):
             ft["properties"]["score_total"] for ft in gj["features"]}

def dist_score(d_m, breaks_km, scores):
    d_km = d_m/1000
    for i,b in enumerate(breaks_km):
        if d_km <= b: return scores[i]
    return scores[-1]

# == 지역별 분석 ==
print("\n[지역별 Hard Filter + 실패이유 추적]")
all_passing  = []
region_stats = {}

for name, sido, rank in TOP10:
    t0 = time.time()
    macro_info = MACRO[name]
    macro_norm = macro_info["macro_norm"]

    reg = admin[admin["name"]==name]
    if len(reg)==0:
        print(f"  {name}: 행정경계 없음 skip")
        region_stats[name] = {"sido":sido,"macro_rank":rank,"macro_norm":macro_norm,
                              "n_total":0,"n_pass":0,"pass_rate":0,"reasons":{},"elapsed":0}
        continue

    boundary   = reg.geometry.iloc[0]
    reg_union  = unary_union(reg.geometry)
    bounds     = reg_union.bounds

    # 격자 생성
    xs = np.arange(bounds[0]+GRID_M/2, bounds[2], GRID_M)
    ys = np.arange(bounds[1]+GRID_M/2, bounds[3], GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()

    # 행정경계 내부 필터
    pts_gdf = gpd.GeoDataFrame({"i":np.arange(len(gx))},
              geometry=gpd.points_from_xy(gx, gy), crs=CRS)
    in_b = pts_gdf.within(reg_union)
    pts_gdf = pts_gdf[in_b].copy()
    gx, gy  = gx[in_b.values], gy[in_b.values]
    n_total = len(gx)
    reason  = np.array(["PASS"]*n_total, dtype="<U20")

    print(f"\n  [{rank}위] {name}({sido}) — {n_total}개 격자")

    # ① 경사도
    slopes = batch_slopes(gx, gy)
    mask_bad = ~np.isnan(slopes) & (slopes > SLOPE_MAX)
    mask_nan = np.isnan(slopes)
    slopes[mask_nan] = 0.0  # NaN = 측정불가 = 평지 취급 (미시분석과 동일)
    reason[mask_bad] = "slope"
    n_sl = int(mask_bad.sum())
    print(f"     경사도>5도: {n_sl}개 제거 (NaN->평지취급: {int(mask_nan.sum())}개)")

    # ② 산사태 1-2등급
    ls_path = os.path.join(DS,"09_landslide_risk",f"landslide_{name}.geojson")
    n_ls = 0
    if os.path.exists(ls_path):
        ls_gdf = gpd.read_file(ls_path)
        if "hazard_grade" in ls_gdf.columns and ls_gdf["hazard_grade"].isin([1,2]).any():
            ls12 = ls_gdf[ls_gdf["hazard_grade"].isin([1,2])].to_crs(CRS)
            if len(ls12):
                ls_union = unary_union(ls12.geometry.buffer(400))
                pm = reason=="PASS"
                pp = gpd.GeoDataFrame({"i":np.where(pm)[0]},
                     geometry=gpd.points_from_xy(gx[pm], gy[pm]), crs=CRS)
                in_ls = pp.within(ls_union)
                reason[pp["i"].values[in_ls.values]] = "landslide"
                n_ls = int(in_ls.sum())
    if n_ls: print(f"     산사태1-2등급: {n_ls}개 제거")

    # ③ 농업진흥지역
    n_farm = 0
    if farm_union is not None:
        pm = reason=="PASS"
        pp = gpd.GeoDataFrame({"i":np.where(pm)[0]},
             geometry=gpd.points_from_xy(gx[pm], gy[pm]), crs=CRS)
        in_f = pp.within(farm_union)
        reason[pp["i"].values[in_f.values]] = "farmland"
        n_farm = int(in_f.sum())
    if n_farm: print(f"     농업진흥지역: {n_farm}개 제거")

    # ④ 보전산지 (지역 clip 후 union)
    n_cf = 0
    cf_clipped = cons_forest[cons_forest.intersects(boundary.buffer(1000))]
    if len(cf_clipped):
        cf_union = unary_union(cf_clipped.geometry)
        pm = reason=="PASS"
        pp = gpd.GeoDataFrame({"i":np.where(pm)[0]},
             geometry=gpd.points_from_xy(gx[pm], gy[pm]), crs=CRS)
        in_c = pp.within(cf_union)
        reason[pp["i"].values[in_c.values]] = "forest"
        n_cf = int(in_c.sum())
    if n_cf: print(f"     보전산지: {n_cf}개 제거")

    # ⑤ 생태경관핵심
    n_eco = 0
    if eco_union is not None:
        pm = reason=="PASS"
        pp = gpd.GeoDataFrame({"i":np.where(pm)[0]},
             geometry=gpd.points_from_xy(gx[pm], gy[pm]), crs=CRS)
        in_e = pp.within(eco_union)
        reason[pp["i"].values[in_e.values]] = "ecology"
        n_eco = int(in_e.sum())
    if n_eco: print(f"     생태경관핵심: {n_eco}개 제거")

    n_pass = int((reason=="PASS").sum())
    print(f"     통과: {n_pass}개 ({n_pass/n_total*100:.1f}%)")

    # Soft Score 매핑
    scored_map = load_scored_map(name)
    for i in np.where(reason=="PASS")[0]:
        x, y = float(gx[i]), float(gy[i])
        rk = (round(x,0), round(y,0))
        ms = scored_map.get(rk)
        if ms is None:
            for dx in [0,250,-250,125,-125,500,-500]:
                for dy in [0,250,-250,125,-125,500,-500]:
                    ms = scored_map.get((rk[0]+dx, rk[1]+dy))
                    if ms is not None: break
                if ms is not None: break
        if ms is None:
            # fallback 계산
            sl = float(slopes[i]) if not np.isnan(slopes[i]) else SLOPE_MAX
            s_sl  = max(0, 100-sl/SLOPE_MAX*100)
            s_pow = dist_score(pow_tree.query([x,y])[0],[5,15,30,999],[100,70,40,0]) if pow_tree else 50
            s_ic  = dist_score(ic_tree.query([x,y])[0], [5,15,30,999],[100,70,40,0]) if ic_tree  else 50
            s_wat = dist_score(riv_tree.query([x,y])[0],[2,5,10,999],[100,80,50,0])  if riv_tree else 50
            ms    = (s_pow*SCORE_W["power"]+s_ic*SCORE_W["transport"]+
                     s_sl*SCORE_W["slope"]+s_wat*SCORE_W["water"]+50*SCORE_W["periphery"])
        combined = W_MACRO*macro_norm + W_MICRO*ms
        lon, lat = t4326_tr.transform(x, y)
        all_passing.append({
            "region":name,"sido":sido,"macro_rank":rank,
            "macro_score":macro_info["종합_점수"],
            "macro_norm":macro_norm,"micro_score":ms,
            "combined_score":combined,
            "x5179":x,"y5179":y,"lat":lat,"lon":lon,
            "slope":float(slopes[i]) if not np.isnan(slopes[i]) else None,
        })

    # 통계
    rc = {REJECT_LABELS[lbl]:int((reason==lbl).sum())
          for lbl in ["slope","landslide","farmland","forest","ecology","PASS"]}
    region_stats[name] = {
        "sido":sido,"macro_rank":rank,"macro_score":macro_info["종합_점수"],
        "macro_norm":macro_norm,"n_total":n_total,"n_pass":n_pass,
        "pass_rate":n_pass/n_total*100,"reasons":rc,"elapsed":time.time()-t0,
    }

    # 격자 이유 GeoJSON 저장
    feats = []
    for i in range(n_total):
        ln, lt = t4326_tr.transform(float(gx[i]), float(gy[i]))
        feats.append({"type":"Feature",
                      "geometry":{"type":"Point","coordinates":[ln,lt]},
                      "properties":{"region":name,"reason":reason[i],
                                    "reason_label":REJECT_LABELS.get(reason[i],reason[i])}})
    with open(os.path.join(RES_DATA,f"grid_reasons_{name}.geojson"),"w",encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":feats},f,ensure_ascii=False)
    print(f"     -> grid_reasons_{name}.geojson ({time.time()-t0:.0f}s)")

# == 전국 TOP 10 선정 ==
print("\n[전국 TOP 10 격자 선정]")
df_all = pd.DataFrame(all_passing).sort_values("combined_score",ascending=False).reset_index(drop=True)
df_all["global_rank"] = range(1,len(df_all)+1)

top_list, seen = [], {}
for _, row in df_all.iterrows():
    if seen.get(row["region"],0) >= 2: continue
    top_list.append(row)
    seen[row["region"]] = seen.get(row["region"],0)+1
    if len(top_list)==10: break
if len(top_list)<10:
    top_list = list(df_all.iloc[:min(10,len(df_all))].itertuples())

top10_df = pd.DataFrame(top_list).reset_index(drop=True)
top10_df["site_rank"] = range(1,len(top10_df)+1)
top10_df.to_csv(os.path.join(RES_DATA,"top10_sites.csv"),index=False,encoding="utf-8-sig")
df_all.to_csv(os.path.join(RES_DATA,"all_passing_scored.csv"),index=False,encoding="utf-8-sig")

print("\n  최종 TOP 10:")
for _,row in top10_df.iterrows():
    print(f"  {int(row.site_rank):2d}위 | {row.region}({row.sido}) | "
          f"통합={row.combined_score:.1f} (거시={row.macro_norm:.1f}, 미시={row.micro_score:.1f}) | "
          f"{row.lat:.4f}N {row.lon:.4f}E")

# == 통합 지도 생성 ==
print("\n[통합 지도 생성]")
m = folium.Map(location=[36.5,127.5], zoom_start=7,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery")

score_cmap = LinearColormap(["#FFFFB2","#FD8D3C","#BD0026"], vmin=0, vmax=100)
score_cmap.caption = "미시 Soft Score (0-100)"
score_cmap.add_to(m)

FAIL_COLORS = {
    "경사도>5도":"#E74C3C","산사태1-2등급":"#E67E22",
    "농업진흥지역":"#27AE60","보전산지":"#8E44AD","생태경관핵심":"#2980B9"
}

# 통과 격자 레이어
t5179 = Transformer.from_crs("EPSG:5179","EPSG:4326",always_xy=True)
for name, sido, rank in TOP10:
    sp = os.path.join(RES_DATA,f"scored_{name}.geojson")
    if not os.path.exists(sp): continue
    with open(sp) as f: gj = json.load(f)
    fg = folium.FeatureGroup(name=f"{rank}위 {name} — 미시점수", show=(rank<=5))
    for ft in gj["features"]:
        p = ft["properties"]
        lon,lat = t5179.transform(p["x"],p["y"])
        folium.Rectangle(
            bounds=[[lat-0.001,lon-0.001],[lat+0.001,lon+0.001]],
            color=None,fill=True,fill_opacity=0.7,
            fill_color=score_cmap(p["score_total"]),
            tooltip=f"{name} 미시={p['score_total']:.1f}"
        ).add_to(fg)
    fg.add_to(m)

# TOP10 마커
top_fg = folium.FeatureGroup(name="★ TOP 10 AI DC 후보지 (통합점수)", show=True)
for _,row in top10_df.iterrows():
    sr = int(row.site_rank)
    popup_html = (
        f"<div style='font-family:sans-serif;width:290px;font-size:13px'>"
        f"<h4 style='color:#C0392B;margin:4px 0'>★ {sr}위 AI DC 후보지</h4><hr style='margin:4px 0'>"
        f"<b>지역:</b> {row.region} ({row.sido})<br>"
        f"<b>좌표:</b> {row.lat:.4f}N, {row.lon:.4f}E<br><hr style='margin:4px 0'>"
        f"<table style='width:100%'>"
        f"<tr><td><b>통합 점수</b></td>"
        f"<td style='color:#E74C3C;font-size:18px;font-weight:bold'>{row.combined_score:.1f}점</td></tr>"
        f"<tr><td>거시 점수 (30%)</td><td>{row.macro_norm:.1f}점</td></tr>"
        f"<tr><td>미시 점수 (70%)</td><td>{row.micro_score:.1f}점</td></tr>"
        f"</table><hr style='margin:4px 0'>"
        f"<small>경사도: {str(round(row.slope,1))+'도' if row.slope else '미계산'} | 거시순위: {int(row.macro_rank)}위</small></div>"
    )
    ic_color = "red" if sr<=3 else ("orange" if sr<=6 else "blue")
    folium.Marker(location=[row.lat,row.lon],
        popup=folium.Popup(popup_html,max_width=310),
        tooltip=f"★{sr}위 {row.region} | 통합={row.combined_score:.1f}점",
        icon=folium.Icon(color=ic_color,icon="star",prefix="fa")).add_to(top_fg)
    folium.Marker(location=[row.lat+0.015,row.lon],
        icon=folium.DivIcon(
            html=(f'<div style="font-size:13px;font-weight:bold;color:white;'
                  f'background:{"#C0392B" if sr<=3 else "#E67E22" if sr<=6 else "#2980B9"};'
                  f'border-radius:50%;width:26px;height:26px;line-height:26px;'
                  f'text-align:center;border:2px solid white">{sr}</div>'),
            icon_size=(26,26),icon_anchor=(13,13))).add_to(top_fg)
top_fg.add_to(m)

legend = ("<div style='position:fixed;bottom:40px;right:40px;z-index:9999;"
          "background:rgba(0,0,0,0.82);border-radius:10px;padding:14px;"
          "color:white;font-size:12px;font-family:sans-serif;line-height:1.7'>"
          "<b>통합 = 거시30% + 미시70%</b><br>"
          "<span style='color:#C0392B'>★</span> 1~3위 &nbsp;"
          "<span style='color:#E67E22'>★</span> 4~6위 &nbsp;"
          "<span style='color:#2980B9'>★</span> 7~10위<br>"
          "<hr style='margin:5px 0;border-color:#666'><b>미시 Soft Score</b><br>"
          "<span style='color:#FFFFB2'>■</span>낮음 "
          "<span style='color:#FD8D3C'>■</span>중간 "
          "<span style='color:#BD0026'>■</span>높음</div>")
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl().add_to(m)
m.save(os.path.join(RES_MAPS,"top10_integrated_map.html"))
print(f"  -> top10_integrated_map.html")

# == 지역별 실패이유 지도 ==
print("\n[지역별 실패이유 지도]")
for name, sido, rank in TOP10:
    gj_p = os.path.join(RES_DATA,f"grid_reasons_{name}.geojson")
    if not os.path.exists(gj_p): continue
    with open(gj_p) as f: gj = json.load(f)
    feats = gj["features"]
    lats = [ft["geometry"]["coordinates"][1] for ft in feats[:300]]
    lons = [ft["geometry"]["coordinates"][0] for ft in feats[:300]]
    center = [np.mean(lats), np.mean(lons)]
    rm = folium.Map(location=center, zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery")
    sample = feats[:12000]
    for ft in sample:
        lon,lat = ft["geometry"]["coordinates"]
        rl = ft["properties"].get("reason_label","통과")
        color = FAIL_COLORS.get(rl,"#2ECC71")
        folium.Rectangle(bounds=[[lat-0.001,lon-0.001],[lat+0.001,lon+0.001]],
            color=None,fill=True,fill_color=color,
            fill_opacity=0.45 if rl!="통과" else 0.75,tooltip=rl).add_to(rm)
    st = region_stats.get(name,{}); rc = st.get("reasons",{})
    for _,row in top10_df[top10_df["region"]==name].iterrows():
        folium.Marker(location=[row.lat,row.lon],
            popup=f"★{int(row.site_rank)}위 통합={row.combined_score:.1f}",
            icon=folium.Icon(color="red",icon="star",prefix="fa")).add_to(rm)
    parts=[f"<b>{name}({sido})</b> 거시{rank}위<br>",
           f"통과율: {st.get('pass_rate',0):.1f}% ({rc.get('통과',0)}/{st.get('n_total',0)})<br>",
           "<hr style='border-color:#555;margin:3px 0'>"]
    for lb,cl in FAIL_COLORS.items():
        cnt=rc.get(lb,0)
        if cnt>0:
            parts.append(f"<span style='color:{cl}'>■</span> {lb}: {cnt}개 ({cnt/max(st.get('n_total',1),1)*100:.1f}%)<br>")
    rm.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:40px;right:40px;z-index:9999;"
        "background:rgba(0,0,0,0.82);border-radius:8px;padding:12px;"
        "color:white;font-size:12px'>"+"".join(parts)+"</div>"))
    rm.save(os.path.join(RES_MAPS,f"reasons_{name}.html"))
    print(f"  -> reasons_{name}.html")

# == 최종 리포트 ==
print("\n"+"="*65)
print("최종 결과: 거시+미시 통합 TOP 10 AI DC 후보지")
print("="*65)
print(f"{'순위':<4}{'지역':<8}{'거시순위':<7}{'거시정규화':<10}{'미시점수':<10}{'통합점수':<10} 좌표")
for _,row in top10_df.iterrows():
    print(f"{int(row.site_rank):<4}{row.region:<8}{int(row.macro_rank):<7}"
          f"{row.macro_norm:<10.1f}{row.micro_score:<10.1f}{row.combined_score:<10.1f}"
          f"{row.lat:.4f}N {row.lon:.4f}E")

print("\n지역별 탈락 이유:")
for name,sido,rank in TOP10:
    st=region_stats.get(name,{})
    if not st: continue
    rc=st.get("reasons",{})
    reasons_str=" | ".join(
        f"{lb}:{cnt}({cnt/max(st['n_total'],1)*100:.0f}%)"
        for lb,cnt in sorted(rc.items(),key=lambda x:-x[1]) if cnt>0 and lb!="통과")
    verdict="적합" if st["pass_rate"]>40 else ("제한적" if st["pass_rate"]>10 else "부적합")
    print(f"  {rank}위 {name}({sido}) [{verdict}] 통과율={st['pass_rate']:.1f}% | {reasons_str}")

print("\n분석 완료!")
