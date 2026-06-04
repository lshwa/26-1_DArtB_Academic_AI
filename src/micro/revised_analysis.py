"""
개정 미시분석 파이프라인
─────────────────────────────────────────────────
변경사항:
  Hard Filter  : 경사도 임계값 5° → 15° (산업입지법 공장 기준)
  Soft Score   : 전력(변전소거리) 제거 (거시 전력 피처와 중복)
                 경사도 3단계 점수화
                   0~5°  : 100점
                   5~10° :  60점
                   10~15°:  30점
                 교통(IC) 45% / 경사도 35% / 하천거리 15% / 민원회피 5%
"""
import os, json, warnings, unicodedata
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import KDTree
from pyproj import Transformer

# ── 경로 ────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(BASE)
DESK  = os.path.dirname(ROOT)

# datasets may be in a sibling folder (Korean vs Hanja 석 folder)
def _find_datasets():
    for d in os.listdir(DESK):
        candidate = os.path.join(DESK, d)
        ds_path = os.path.join(candidate, "datasets")
        if "미시" in unicodedata.normalize('NFC', d) and os.path.isdir(ds_path):
            return ds_path
    return os.path.join(ROOT, "datasets")

DS    = _find_datasets()
RES_DATA = os.path.join(ROOT, "results", "data")
RES_MAPS = os.path.join(ROOT, "results", "maps")
os.makedirs(RES_DATA, exist_ok=True)
print(f"datasets 경로: {DS}")

# ── 상수 ────────────────────────────────────────────
CRS       = "EPSG:5179"
GRID_M    = 250
SLOPE_MAX = 15.0          # Hard Filter 임계값 변경: 5° → 15° (산업입지법)

# 경사도 구간별 점수 (새 Soft Score)
def slope_score(deg):
    if deg <= 5.0:  return 100
    if deg <= 10.0: return 60
    return 30                 # 10~15° 통과 구간

# Soft Score 가중치 (전력 제거, 거시 비중복)
SCORE_W = {"transport": 0.45, "slope": 0.35, "water": 0.15, "periphery": 0.05}

W_MACRO = 0.30
W_MICRO = 0.70
TOP_N   = 10          # ← 여기만 바꾸면 Top-N 지역으로 분석 가능

# ── 거시 점수 로드 & Top-N 자동 결정 ────────────────
def load_macro(top_n=TOP_N):
    """
    최종_랭킹_144개.csv를 읽어 상위 top_n 지역을 자동으로 선택.
    거시분석 가중치가 바뀌어 CSV가 재생성되면 이 파이프라인도 자동 반영.
    """
    DESK = "/Users/lshwa/Desktop/학술제 /"
    TARGET = "최종_랭킹_144개.csv"
    for root, dirs, files in os.walk(DESK):
        for f in files:
            f_nfc    = unicodedata.normalize('NFC', f)
            root_nfc = unicodedata.normalize('NFC', root)
            if f_nfc == TARGET and "거시" in root_nfc:
                df = pd.read_csv(os.path.join(root, f), encoding='utf-8-sig')
                df.columns = [c.lstrip('﻿') for c in df.columns]
                df = df.sort_values("종합_점수", ascending=False).reset_index(drop=True)
                top_df = df.head(top_n).copy()
                # 정규화: Top-N 범위 내에서 0~100
                mn, mx = top_df["종합_점수"].min(), top_df["종합_점수"].max()
                top_df["macro_norm"] = (
                    (top_df["종합_점수"] - mn) / (mx - mn) * 100
                    if mx > mn else pd.Series([100.0]*len(top_df))
                )
                # (지역명, 시도, 거시순위) 튜플 리스트
                top_list = [
                    (row["시군구"], row.get("시도", ""), i+1)
                    for i, row in top_df.iterrows()
                ]
                # 랭킹은 정렬 순서대로 재할당
                top_list = [(nm, sido, rank)
                            for rank, (nm, sido, _) in enumerate(top_list, 1)]
                score_map = top_df.set_index("시군구").to_dict("index")
                return top_list, score_map
    raise FileNotFoundError("최종_랭킹_144개.csv 없음")

TOP10, MACRO = load_macro()
print(f"거시 점수 로드 완료 (Top {TOP_N})")
print(f"  ※ 거시분석 가중치 변경 → CSV 재생성 → 이 파이프라인 재실행 시 자동 반영")
for name, sido, rank in TOP10:
    mn = MACRO[name]["macro_norm"]
    print(f"  {rank:>2}위 {name}({sido}): 거시 정규화 = {mn:.1f}")

# ── 공통 데이터 로드 ─────────────────────────────────
print("\n[데이터 로드]")
admin       = gpd.read_file(os.path.join(DS,"01_admin_boundaries","sigungu_top10.geojson")).to_crs(CRS)
power_gdf   = gpd.read_file(os.path.join(DS,"02_power_infrastructure","power_infrastructure_top10.geojson")).to_crs(CRS)
transport   = gpd.read_file(os.path.join(DS,"03_transport","roads_highways_top10.geojson")).to_crs(CRS)
waterways   = gpd.read_file(os.path.join(DS,"04_waterways","waterways_top10.geojson")).to_crs(CRS)
farmland    = gpd.read_file(os.path.join(DS,"10_farmland","farmland_top10.geojson")).to_crs(CRS)
cons_forest = gpd.read_file(os.path.join(DS,"11_conservation_forest","conservation_forest_top10.geojson")).to_crs(CRS)
ecology     = gpd.read_file(os.path.join(DS,"12_ecology_core","ecology_core_top10.geojson")).to_crs(CRS)

# 인프라 KDTree
substations = power_gdf[power_gdf.get("power","") == "substation"].copy()
if len(substations) == 0:
    substations = power_gdf[power_gdf.geometry.geom_type == "Point"].copy()
ic      = transport[transport.get("highway","").eq("motorway_junction")].copy() if "highway" in transport.columns else gpd.GeoDataFrame()
rivers  = waterways[waterways.get("waterway","").isin(["river","stream"]) | waterways.get("natural","").eq("water")].copy() if "waterway" in waterways.columns else waterways

def pts_xy(gdf):
    geom = gdf.geometry
    if len(gdf) == 0: return None
    x = geom.centroid.x if geom.iloc[0].geom_type != "Point" else geom.x
    y = geom.centroid.y if geom.iloc[0].geom_type != "Point" else geom.y
    return np.c_[x.values, y.values]

pow_tree = KDTree(pts_xy(substations)) if len(substations) > 0 else None
ic_tree  = KDTree(pts_xy(ic))          if len(ic) > 0          else None
riv_tree = KDTree(pts_xy(rivers))      if len(rivers) > 0      else None

farm_union = unary_union(farmland.geometry)  if len(farmland) > 0  else None
eco_union  = unary_union(ecology.geometry)   if len(ecology) > 0   else None

print(f"  변전소: {len(substations)}개 | IC: {len(ic)}개 | 하천: {len(rivers)}개")
print(f"  농업진흥: {len(farmland)}개 | 보전산지: {len(cons_forest)}개 | 생태경관: {len(ecology)}개")

# DEM 경로
DEM_PATHS = [p for p in [
    os.path.join(DS,"05_dem","srtm_62_05.tif"),
    os.path.join(DS,"05_dem","srtm_62_06.tif"),
] if os.path.exists(p)]
print(f"  DEM 파일: {len(DEM_PATHS)}개")

t4326 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
t5186 = Transformer.from_crs(CRS, "EPSG:5186", always_xy=True)

# ── DEM 경사도 계산 ──────────────────────────────────
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
                if not np.isnan(slopes[i]): continue
                if not (src.bounds.left <= lon <= src.bounds.right and
                        src.bounds.bottom <= lat <= src.bounds.top): continue
                try:
                    r, c = rowcol(src.transform, lon, lat)
                    if 0 <= r < slope_arr.shape[0] and 0 <= c < slope_arr.shape[1]:
                        slopes[i] = slope_arr[r, c]
                except: pass
    slopes[np.isnan(slopes)] = 0.0  # NaN → 0도 (평지 취급)
    return slopes

def flood_risk(xs_5179, ys_5179, region_name):
    tif = os.path.join(DS,"08_flood_risk",f"flood_risk_{region_name}.tif")
    if not os.path.exists(tif): return np.zeros(len(xs_5179))
    risks = np.zeros(len(xs_5179))
    xs_5186, ys_5186 = t5186.transform(xs_5179, ys_5179)
    with rasterio.open(tif) as src:
        for i,(x,y) in enumerate(zip(xs_5186, ys_5186)):
            if not (src.bounds.left <= x <= src.bounds.right and
                    src.bounds.bottom <= y <= src.bounds.top): continue
            r, c = rowcol(src.transform, x, y)
            if 0 <= r < src.height and 0 <= c < src.width:
                val = src.read(1)[r, c]
                risks[i] = float(val) if val != src.nodata else 0.0
    return risks

def dist_score(d_m, breaks_km, scores):
    d_km = d_m / 1000
    for b, s in zip(breaks_km, scores):
        if d_km <= b: return s
    return scores[-1]

# ── 메인 루프 ────────────────────────────────────────
print("\n[격자 분석 시작]")
print(f"Hard Filter 임계값: 경사도 > {SLOPE_MAX}° (산업입지법 공장 기준)")
print(f"Soft Score: 교통 {SCORE_W['transport']*100:.0f}% / "
      f"경사도 {SCORE_W['slope']*100:.0f}% / "
      f"하천 {SCORE_W['water']*100:.0f}% / "
      f"민원 {SCORE_W['periphery']*100:.0f}%")
print("="*60)

all_passing = []
region_summaries = []

for sigg_name, sido, rank in TOP10:
    print(f"\n  [{rank}위] {sigg_name}({sido})")
    macro_norm = MACRO[sigg_name]["macro_norm"]

    # 행정 경계
    reg = admin[admin.apply(lambda r: sigg_name in " ".join(str(v) for v in r.values), axis=1)]
    if len(reg) == 0:
        print(f"    ⚠ 경계 없음 SKIP")
        continue
    boundary = reg.geometry.iloc[0]
    reg_union = unary_union(reg.geometry)
    bounds = reg_union.bounds

    # 250m 격자 생성
    xs = np.arange(bounds[0]+GRID_M/2, bounds[2], GRID_M)
    ys = np.arange(bounds[1]+GRID_M/2, bounds[3], GRID_M)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()

    pts_gdf = gpd.GeoDataFrame({"i": np.arange(len(gx))},
                geometry=gpd.points_from_xy(gx, gy), crs=CRS)
    in_b = pts_gdf.within(reg_union)
    pts_gdf = pts_gdf[in_b].copy()
    gx, gy  = gx[in_b.values], gy[in_b.values]
    n_total = len(gx)
    print(f"    격자: {n_total}개")

    if n_total == 0: continue

    # 경사도 계산 (전체 격자)
    slopes = batch_slopes(gx, gy)

    # ── Hard Filter ────────────────────────────────
    keep = np.ones(n_total, dtype=bool)
    filter_log = {}

    # ① 경사도 > 15° (산업입지법 공장 기준)
    sl_mask = slopes > SLOPE_MAX
    filter_log["경사도>15°"] = int(sl_mask.sum())
    keep &= ~sl_mask

    # ② 침수위험
    flood = flood_risk(gx, gy, sigg_name)
    fl_mask = flood > 0
    filter_log["침수위험"] = int(fl_mask.sum())
    keep &= ~fl_mask

    # ③ 농업진흥지역
    if farm_union is not None:
        pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
             geometry=gpd.points_from_xy(gx[keep], gy[keep]), crs=CRS)
        in_f = pp.within(farm_union)
        filter_log["농업진흥지역"] = int(in_f.sum())
        keep[pp["i"].values[in_f.values]] = False

    # ④ 보전산지
    cf_clip = cons_forest[cons_forest.intersects(boundary.buffer(1000))]
    if len(cf_clip) > 0:
        cf_union = unary_union(cf_clip.geometry)
        pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
             geometry=gpd.points_from_xy(gx[keep], gy[keep]), crs=CRS)
        in_c = pp.within(cf_union)
        filter_log["보전산지"] = int(in_c.sum())
        keep[pp["i"].values[in_c.values]] = False

    # ⑤ 생태경관핵심
    if eco_union is not None:
        pp = gpd.GeoDataFrame({"i": np.where(keep)[0]},
             geometry=gpd.points_from_xy(gx[keep], gy[keep]), crs=CRS)
        in_e = pp.within(eco_union)
        filter_log["생태경관핵심"] = int(in_e.sum())
        keep[pp["i"].values[in_e.values]] = False

    # ⑥ 산사태 1-2등급 500m 버퍼
    ls_path = os.path.join(DS,"09_landslide_risk",f"landslide_{sigg_name}.geojson")
    if os.path.exists(ls_path):
        ls_gdf = gpd.read_file(ls_path)
        if "hazard_grade" in ls_gdf.columns:
            ls12 = ls_gdf[ls_gdf["hazard_grade"].isin([1,2])].copy()
            if len(ls12) > 0:
                t5181 = Transformer.from_crs("EPSG:5181", CRS, always_xy=True)
                x5179, y5179 = t5181.transform(ls12["x_5181"].values, ls12["y_5181"].values)
                ls_tree = KDTree(np.c_[x5179, y5179])
                pp_idx = np.where(keep)[0]
                if len(pp_idx) > 0:
                    d_ls, _ = ls_tree.query(np.c_[gx[pp_idx], gy[pp_idx]])
                    ls_mask = d_ls < 500
                    filter_log["산사태1-2등급"] = int(ls_mask.sum())
                    keep[pp_idx[ls_mask]] = False

    n_pass = int(keep.sum())
    pass_rate = n_pass / n_total * 100
    print(f"    Hard Filter: {n_total} → {n_pass}개 ({pass_rate:.1f}% 통과)")
    for k, v in filter_log.items():
        if v > 0: print(f"      제거 {k}: {v}개")

    if n_pass == 0:
        region_summaries.append({
            "거시순위":rank,"지역":sigg_name,"시도":sido,
            "전체격자":n_total,"통과격자":0,"통과율":0,
            "거시정규화":round(macro_norm,1),
            "최고Soft Score":None,"최고통합점수":None,
            "위도":None,"경도":None
        })
        continue

    # ── Soft Score ─────────────────────────────────
    sv_idx = np.where(keep)[0]
    sv_x   = gx[sv_idx]
    sv_y   = gy[sv_idx]
    sv_sl  = slopes[sv_idx]
    sv_xy  = np.c_[sv_x, sv_y]

    # 교통 (IC거리) 45%
    if ic_tree:
        d_ic, _ = ic_tree.query(sv_xy)
        s_transport = np.array([dist_score(d, [5,15,30,999], [100,70,40,0]) for d in d_ic/1000])
    else:
        s_transport = np.full(n_pass, 50.0)

    # 경사도 세부 35% — 3단계 점수
    s_slope = np.array([slope_score(sl) for sl in sv_sl])

    # 하천거리 15%
    if riv_tree:
        d_riv, _ = riv_tree.query(sv_xy)
        s_water = np.array([dist_score(d, [1,3,5,999], [100,70,40,10]) for d in d_riv/1000])
    else:
        s_water = np.full(n_pass, 30.0)

    # 민원회피 5% (시군구 중심에서 멀수록)
    cx, cy = boundary.centroid.x, boundary.centroid.y
    d_ctr  = np.sqrt((sv_x - cx)**2 + (sv_y - cy)**2)
    max_d  = d_ctr.max() if d_ctr.max() > 0 else 1
    s_peri = (d_ctr / max_d) * 100

    soft_score = (s_transport * SCORE_W["transport"] +
                  s_slope     * SCORE_W["slope"] +
                  s_water     * SCORE_W["water"] +
                  s_peri      * SCORE_W["periphery"])

    combined = W_MACRO * macro_norm + W_MICRO * soft_score

    # 결과 정리
    lons, lats = t4326.transform(sv_x, sv_y)
    for i in range(n_pass):
        all_passing.append({
            "region": sigg_name, "sido": sido, "macro_rank": rank,
            "macro_norm": macro_norm,
            "soft_score": soft_score[i], "combined": combined[i],
            "slope_deg": sv_sl[i],
            "s_transport": s_transport[i], "s_slope": s_slope[i],
            "s_water": s_water[i], "s_periphery": s_peri[i],
            "x": sv_x[i], "y": sv_y[i], "lat": lats[i], "lon": lons[i]
        })

    best_i = combined.argmax()
    print(f"    최고 Soft Score: {soft_score.max():.1f} | 통합: {combined.max():.1f}")
    print(f"    최적 부지: {lats[best_i]:.4f}N {lons[best_i]:.4f}E (경사도 {sv_sl[best_i]:.1f}°)")

    region_summaries.append({
        "거시순위": rank, "지역": sigg_name, "시도": sido,
        "전체격자": n_total, "통과격자": n_pass,
        "통과율": round(pass_rate, 1),
        "거시정규화": round(macro_norm, 1),
        "최고Soft Score": round(soft_score.max(), 1),
        "최고통합점수": round(combined.max(), 1),
        "위도": round(lats[best_i], 4),
        "경도": round(lons[best_i], 4)
    })

# ── 최종 결과 ────────────────────────────────────────
print("\n" + "="*60)
print("개정 미시분석 최종 결과")
print("Hard Filter: 경사도>15° / Soft Score: 교통45%+경사도35%+하천15%+민원5%")
print("="*60)

df_all = pd.DataFrame(all_passing).sort_values("combined", ascending=False)
top_list, seen = [], set()
for _, row in df_all.iterrows():
    if row["region"] not in seen:
        top_list.append(row)
        seen.add(row["region"])
    if len(top_list) == 10: break

print(f"\n{'순위':<4}{'지역':<8}{'시도':<6}{'거시정규화':<12}{'Soft Score':<12}{'통합점수':<10}{'경사도':<8} 좌표")
print("-"*70)
for i, row in enumerate(top_list, 1):
    print(f"{i:<4}{row['region']:<8}{row['sido']:<6}"
          f"{row['macro_norm']:<12.1f}{row['soft_score']:<12.1f}{row['combined']:<10.1f}"
          f"{row['slope_deg']:<8.1f}{row['lat']:.4f}N {row['lon']:.4f}E")

print("\n지역별 요약:")
df_sum = pd.DataFrame(region_summaries).sort_values("최고통합점수", ascending=False, na_position="last")
print(df_sum.to_string(index=False))

# 저장
df_all.to_csv(os.path.join(RES_DATA,"revised_all_passing.csv"), index=False, encoding="utf-8-sig")
df_sum.to_csv(os.path.join(RES_DATA,"revised_summary.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(top_list).to_csv(os.path.join(RES_DATA,"revised_top10.csv"), index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {RES_DATA}/revised_*.csv")
