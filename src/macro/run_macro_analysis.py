"""
거시분석 파이프라인 v4 — 순위 기반 정규화판
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
주요 변경:
  - Min-Max → 순위 기반 정규화(rank_norm): score = rank/(n+1)×100
    → 최솟값도 0점 아님, 분포 왜곡 무관, 이상치 영향 없음
  - log1p 전처리 제거 (rank_norm은 단조변환에 불변이므로 불필요)
  - 나머지 로직 동일 유지
"""
import os, warnings, unicodedata
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 경로 설정 ─────────────────────────────────────────
def nfc(s): return unicodedata.normalize("NFC", s)
DESK = "/Users/lshwa/Desktop/학술제 "

_FILE_MAP = {}
_RES_DIR = None

for _root, _dirs, _files in os.walk(DESK):
    _rn = nfc(_root)
    for _f in _files:
        _fn = nfc(_f)
        if _fn in {
            "사전필터_통과_시군구_144개.csv", "전력피처_시군구별.csv",
            "재해_피처_수정완료.csv", "냉각_피처_최종.csv",
            "신재생_피처_수정완료.csv", "산림청_산불통계데이터_20250911.csv",
            "변전소_송전망여유용량.csv",
        }:
            _FILE_MAP[_fn] = os.path.join(_root, _f)
        # 네트워크 원본 (회의록에도 있음)
        if _fn == "네트워크_원본.csv":
            _FILE_MAP[_fn] = os.path.join(_root, _f)

# 결과 폴더: PNG/CSV 있는 곳
for _root, _dirs, _files in os.walk(DESK):
    _rn = nfc(_root)
    if "회의록" in _rn:
        continue
    _fnlist = [nfc(_f) for _f in _files]
    if "viz1_top15_bar.png" in _fnlist or "피처별_점수_144개.csv" in _fnlist:
        _RES_DIR = _root
        break

if _RES_DIR is None:
    for _root, _dirs, _files in os.walk(DESK):
        _rn = nfc(_root)
        if "거시" in _rn and "결과" in _rn and "코드" not in _rn and "회의" not in _rn:
            _RES_DIR = _root
            break

RES_DIR = _RES_DIR
os.makedirs(RES_DIR, exist_ok=True)
print(f"  결과 저장 위치: {nfc(RES_DIR)}")

def ds(fname):
    fn = nfc(fname)
    if fn in _FILE_MAP:
        return _FILE_MAP[fn]
    raise FileNotFoundError(f"{fname}")

# ── 한글 폰트 ─────────────────────────────────────────
def set_korean_font():
    for path in [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ── 시도 이름 정규화 (전력 CSV용) ────────────────────────
SIDO_SHORT = {
    "강원특별자치도": "강원", "경기도": "경기", "경상남도": "경남",
    "경상북도": "경북", "광주광역시": "광주", "대구광역시": "대구",
    "대전광역시": "대전", "부산광역시": "부산", "세종특별자치시": "세종",
    "울산광역시": "울산", "인천광역시": "인천", "전라남도": "전남",
    "전북특별자치도": "전북", "제주특별자치도": "제주",
    "충청남도": "충남", "충청북도": "충북", "서울특별시": "서울",
}
def normalize_sido(s): return SIDO_SHORT.get(str(s).strip(), str(s).strip())

# ── 시군구 이름 정규화 ────────────────────────────────
def normalize_sigg(name, ref_set):
    n = str(name).strip()
    if n in ref_set: return n
    for suf in ["시", "군", "구"]:
        if (n + suf) in ref_set: return n + suf
    for ref in ref_set:
        base = ref
        for suf in ["시", "군", "구", "특별자치시", "특별자치도"]:
            if base.endswith(suf): base = base[:-len(suf)]; break
        if base == n: return ref
    return n

# ── 스케일링 함수 ─────────────────────────────────────
from scipy.stats import rankdata as _rankdata

def rank_norm(s, invert=False):
    """순위 기반 정규화 — 0/1 극단값 없음, 분포 무관.
    score = rank / (n+1)  ∈ (0, 1)
    ties → average rank (동점 처리 자동)
    invert=True: 작은 값일수록 높은 점수 (재해 등 역방향)
    """
    arr = np.array(s, dtype=float)
    valid = ~np.isnan(arr)
    result = np.full(len(arr), 0.5)
    if valid.sum() > 0:
        vals = arr[valid]
        if invert:
            vals = -vals
        r = _rankdata(vals, method='average')
        n = int(valid.sum())
        result[valid] = r / (n + 1)
    idx = s.index if isinstance(s, pd.Series) else None
    return pd.Series(result, index=idx)

# ── 안전 조인 (144행 보장) ───────────────────────────
def safe_join(base, src, cols, key="시군구"):
    """
    (시도, 시군구) 기준 left join.
    시도 없는 경우 시군구만으로 fallback.
    결과가 항상 len(base) 행임을 보장.
    """
    has_sido = "시도" in src.columns
    if has_sido:
        src_dedup = src[["시도","시군구"]+cols].drop_duplicates(["시도","시군구"])
        result = base.merge(src_dedup, on=["시도","시군구"], how="left")
    else:
        src_dedup = src[["시군구"]+cols].drop_duplicates("시군구", keep="first")
        result = base.merge(src_dedup, on="시군구", how="left")

    # 미매칭 → 시군구 단독 fallback
    for col in cols:
        missing = result[col].isna()
        if missing.any():
            fb = src.drop_duplicates("시군구", keep="first")[["시군구", col]]
            filled = base[missing][["시군구"]].merge(fb, on="시군구", how="left")[col]
            result.loc[missing, col] = filled.values

    assert len(result) == len(base), f"조인 행 수 오류: {len(result)} != {len(base)}"
    return result

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[1] 데이터 로드")

def load_csv(fname, enc="utf-8-sig"):
    path = ds(fname)
    try:
        df = pd.read_csv(path, encoding=enc)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    df.columns = [nfc(c.lstrip("﻿")) for c in df.columns]
    return df

df_filter   = load_csv("사전필터_통과_시군구_144개.csv")
df_power    = load_csv("전력피처_시군구별.csv")
df_trans    = load_csv("변전소_송전망여유용량.csv")
df_disaster = load_csv("재해_피처_수정완료.csv")
df_cool     = load_csv("냉각_피처_최종.csv")
df_renew    = load_csv("신재생_피처_수정완료.csv")
df_net_raw  = load_csv("네트워크_원본.csv")

try:
    df_fire = load_csv("산림청_산불통계데이터_20250911.csv", enc="cp949")
except:
    df_fire = load_csv("산림청_산불통계데이터_20250911.csv")

BASE = df_filter[df_filter["사전필터"] == "통과"][["시도", "시군구"]].copy().reset_index(drop=True)
print(f"  대상 시군구: {len(BASE)}개")
assert len(BASE) == 144

SIGG_SET = set(BASE["시군구"].tolist())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 피처별 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] 피처 계산")

# ── 2-1. 전력 ─────────────────────────────────────
# 신규 공식: rank_norm(전력공급여유량_2032 × 합산_송전망_2032)
# - 전력공급여유량_2032: 계통 공급 여유 (공급 측)
# - 합산_송전망_2032   : 시군구 내 전체 변전소 송전 여유 합산 (선로 측)
# - 곱셈(AND 조건): 공급·송전 둘 다 있어야 전력 연결 가능
# - 기존 대표변전소 1개 → 전체 변전소 합산으로 교체 (데이터 정확도 개선)
pwr_raw = df_power.copy()
pwr_raw["시도"] = pwr_raw["시도"].apply(normalize_sido)

# 변전소 합산: 시군구별 여유용량_2032_MW 전체 합산
trans_sum = df_trans.groupby("시군구")["여유용량_2032_MW"].sum().reset_index()
trans_sum.columns = ["시군구", "합산_송전망_2032"]

# 전력공급여유량_2032 조인
pwr_base = pwr_raw[["시도","시군구","전력공급여유량_2032"]].drop_duplicates(["시도","시군구"])
pwr_base = pwr_base.merge(trans_sum, on="시군구", how="left")
pwr_base["합산_송전망_2032"] = pwr_base["합산_송전망_2032"].fillna(0)

# 신규 전력가용량 = 공급여유 × 합산 송전망
pwr_base["전력가용량_신규"] = pwr_base["전력공급여유량_2032"] * pwr_base["합산_송전망_2032"]

pwr = safe_join(BASE, pwr_base, ["전력가용량_신규"])

n_missing_pwr = pwr["전력가용량_신규"].isna().sum()
print(f"  전력 — 144개 중 미매칭: {n_missing_pwr}개")
median_pwr = pwr["전력가용량_신규"].dropna().median()
pwr["전력가용량_신규"] = pwr["전력가용량_신규"].fillna(median_pwr)

# rank_norm → 전력_점수 (큰 값 = 높은 점수)
pwr["N_전력가용량"] = rank_norm(pwr["전력가용량_신규"])
pwr["전력_점수"]   = pwr["N_전력가용량"]

n_zero_pwr = (pwr["전력가용량_신규"] == 0).sum()
print(f"  전력가용량_신규 — min: {pwr['전력가용량_신규'].min():.0f} / max: {pwr['전력가용량_신규'].max():.0f}")
print(f"  0값 지역 수: {n_zero_pwr}개 (공급·송전 중 하나 포화) → rank_norm 최하위 처리")
print(f"  전력_점수 — min: {pwr['전력_점수'].min():.3f} / max: {pwr['전력_점수'].max():.3f}")

assert len(pwr) == 144

# ── 2-2. 재해 ─────────────────────────────────────
dis_cols = ["지진위험지수","재해위험_가중합","산사태_건수","해안위험_등급"]
dis = safe_join(BASE, df_disaster, dis_cols)

for col in dis_cols:
    n_miss = dis[col].isna().sum()
    if n_miss > 0:
        print(f"  재해 {col} 미매칭: {n_miss}개 → 중앙값 대체")
    med = dis[col].dropna().median()
    dis[col] = dis[col].fillna(med)

# 산불: 시군구 정규화 후 집계
sido_col = next(c for c in df_fire.columns if "시도" in c and "장소" in c)
sigg_col = next(c for c in df_fire.columns if "시군구" in c and "장소" in c)
area_col = next(c for c in df_fire.columns if "피해면적" in c)

df_fire_cp = df_fire.copy()
df_fire_cp[sigg_col] = df_fire_cp[sigg_col].apply(lambda x: normalize_sigg(x, SIGG_SET))
fire_sum = df_fire_cp.groupby(sigg_col)[area_col].sum().reset_index()
fire_sum.columns = ["시군구","산불피해면적"]

dis = dis.merge(fire_sum, on="시군구", how="left")
dis["산불피해면적"] = dis["산불피해면적"].fillna(0).clip(lower=0)
fire_covered = (dis["산불피해면적"] > 0).sum()
print(f"  산불 — 데이터 있는 시군구: {fire_covered}개")

# 순위 기반 역방향 정규화 (위험 클수록 낮은 점수)
dis["N_지진"]   = rank_norm(dis["지진위험지수"], invert=True)
dis["N_홍수"]   = rank_norm(dis["재해위험_가중합"], invert=True)
dis["N_산사태"] = rank_norm(dis["산사태_건수"], invert=True)
dis["N_해안"]   = rank_norm(dis["해안위험_등급"], invert=True)
dis["N_산불"]   = rank_norm(dis["산불피해면적"], invert=True)

dis["재해_점수"] = (
    0.30 * dis["N_지진"] +
    0.26 * dis["N_홍수"] +
    0.20 * dis["N_산사태"] +
    0.12 * dis["N_해안"] +
    0.12 * dis["N_산불"]
)
print(f"  재해_점수 — min: {dis['재해_점수'].min():.3f} / max: {dis['재해_점수'].max():.3f}")
assert len(dis) == 144

# ── 2-3. 냉각 ─────────────────────────────────────
cool = safe_join(BASE, df_cool, ["취수여유율_%","연간_15도이하_일수"])

for col in ["취수여유율_%","연간_15도이하_일수"]:
    n_miss = cool[col].isna().sum()
    if n_miss > 0:
        print(f"  냉각 {col} 미매칭: {n_miss}개 → 중앙값 대체")
    cool[col] = cool[col].fillna(cool[col].dropna().median())

cool["N_취수여유율"] = rank_norm(cool["취수여유율_%"])
cool["N_냉각일수"]   = rank_norm(cool["연간_15도이하_일수"])
cool["냉각_점수"]    = 0.60 * cool["N_취수여유율"] + 0.40 * cool["N_냉각일수"]
print(f"  냉각 — 취수여유율 평균: {cool['취수여유율_%'].mean():.1f}%")
print(f"  냉각_점수 — min: {cool['냉각_점수'].min():.3f} / max: {cool['냉각_점수'].max():.3f}")
assert len(cool) == 144

# ── 2-4. 신재생 ───────────────────────────────────
ren_cols_needed = ["신재생발전량_GWh","신재생접근성_유효"]
ren_src = df_renew[
    (["시도"] if "시도" in df_renew.columns else []) + ["시군구"] + ren_cols_needed
].copy()

ren = safe_join(BASE, ren_src, ren_cols_needed)

for col in ren_cols_needed:
    n_miss = ren[col].isna().sum()
    if n_miss > 0:
        print(f"  신재생 {col} 미매칭: {n_miss}개 → 0 대체")
    ren[col] = ren[col].fillna(0)

ren["N_발전량"] = rank_norm(ren["신재생발전량_GWh"])
ren["N_접근성"] = rank_norm(ren["신재생접근성_유효"])
ren["신재생_점수"] = 0.80 * ren["N_발전량"] + 0.20 * ren["N_접근성"]

top5_ren = ren.nlargest(5,"신재생발전량_GWh")[["시군구","신재생발전량_GWh"]].values.tolist()
print(f"  신재생 — 발전량 상위 5: {top5_ren}")
print(f"  신재생_점수 — min: {ren['신재생_점수'].min():.3f} / max: {ren['신재생_점수'].max():.3f}")
assert len(ren) == 144

# ── 2-5. 네트워크 ─────────────────────────────────
net_raw = df_net_raw.copy()
net_raw["시도"] = net_raw["시도"].apply(normalize_sido)
# 서울거리_km 역순위 정규화 (가까울수록 높은 점수)
net_raw["네트워크_접근성"] = rank_norm(net_raw["서울거리_km"], invert=True)

net = safe_join(BASE, net_raw[["시도","시군구","네트워크_접근성"]], ["네트워크_접근성"])

n_miss_net = net["네트워크_접근성"].isna().sum()
if n_miss_net > 0:
    print(f"  네트워크 미매칭: {n_miss_net}개 → 중앙값 대체")
net["네트워크_접근성"] = net["네트워크_접근성"].fillna(net["네트워크_접근성"].dropna().median())
net["네트워크_점수"] = net["네트워크_접근성"]
print(f"  네트워크_점수 — 평균: {net['네트워크_점수'].mean():.3f} / min: {net['네트워크_점수'].min():.3f} / max: {net['네트워크_점수'].max():.3f}")
assert len(net) == 144

# 네트워크 CSV 저장 (미시분석에서도 사용)
for _root, _dirs, _files in os.walk(DESK):
    _rn = nfc(_root)
    if "네트워크" in _rn and "거시" in _rn and "데이터셋" in _rn and "회의" not in _rn:
        net_save = os.path.join(_root, "네트워크_인구가중거리_피처.csv")
        net[["시도","시군구","네트워크_접근성"]].to_csv(net_save, index=False, encoding="utf-8-sig")
        print(f"  네트워크 CSV 저장: {nfc(net_save)}")
        break

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 합치기 (144행 보장)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] 합치기")
df = BASE.copy()
for src, cols in [
    (pwr,  ["N_전력가용량","전력_점수","전력가용량_신규","전력가용량_신규"]),
    (dis,  ["N_지진","N_홍수","N_산사태","N_해안","N_산불","재해_점수"]),
    (cool, ["N_취수여유율","N_냉각일수","냉각_점수"]),
    (ren,  ["N_발전량","N_접근성","신재생_점수"]),
    (net,  ["네트워크_점수"]),
]:
    df = df.merge(src[["시도","시군구"]+cols].drop_duplicates(["시도","시군구"]),
                  on=["시도","시군구"], how="left")

assert len(df) == 144, f"최종 행 수 오류: {len(df)}"
dup = df.duplicated(["시도","시군구"]).sum()
assert dup == 0, f"중복 (시도+시군구): {dup}개"
print(f"  완료 — shape: {df.shape}, 중복: {dup}개")

# 이상치 검증
print("\n[이상치 검증]")
for col in ["전력_점수","재해_점수","냉각_점수","신재생_점수","네트워크_점수"]:
    vals = df[col]
    n_zero = (vals == 0).sum()
    n_one  = (vals == 1.0).sum()
    print(f"  {col}: range=[{vals.min():.3f}, {vals.max():.3f}] | 0점={n_zero}개 | 1점={n_one}개")
    if n_zero > 5:
        print(f"    ⚠ 0점 지역: {df[vals==0]['시군구'].tolist()[:10]}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 민감도 분석 (One-way + Monte Carlo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLUE  = "#1a3c6e"
GREEN = "#2d8a4e"
COLORS = ["#1a3c6e","#2d8a4e","#c0392b","#e67e22","#8e44ad"]

print("\n[4] 민감도 분析 (One-way + Monte Carlo)")

BASE_W = {"전력": 0.35, "재해": 0.20, "냉각": 0.20, "신재생": 0.20, "네트워크": 0.05}
FEAT_COLS = {
    "전력": "전력_점수", "재해": "재해_점수", "냉각": "냉각_점수",
    "신재생": "신재생_점수", "네트워크": "네트워크_점수"
}
KEYS = list(BASE_W.keys())

def get_score(weights):
    return sum(weights[k] * df[FEAT_COLS[k]] for k in KEYS)

def get_top10_id(weights):
    score = get_score(weights)
    top = df.assign(s=score).nlargest(10, "s")
    return set(top["시도"].astype(str) + "_" + top["시군구"].astype(str))

base_top10 = get_top10_id(BASE_W)
base_score = get_score(BASE_W)
print(f"  기준 가중치: { {k: f'{v*100:.0f}%' for k,v in BASE_W.items()} }")
print(f"  기준 Top10: {sorted(t.split('_')[1] for t in base_top10)}")

# ── One-way 민감도 분석 ─────────────────────────────
# 한 피처 가중치를 변화시키고 나머지는 비례 재조정 → Top10 안정성 측정
ow_results = []
delta_range = np.arange(-0.15, 0.20, 0.05)

for key in KEYS:
    for delta in delta_range:
        nw = dict(BASE_W)
        nw[key] = max(0.01, round(BASE_W[key] + delta, 4))
        rest_orig = sum(v for k, v in BASE_W.items() if k != key)
        rest_new  = 1.0 - nw[key]
        for k in KEYS:
            if k != key:
                nw[k] = BASE_W[k] / rest_orig * rest_new
        top10_new = get_top10_id(nw)
        overlap   = len(top10_new & base_top10)
        ow_results.append({
            "피처": key,
            "기준가중치": round(BASE_W[key]*100, 1),
            "변경가중치": round(nw[key]*100, 1),
            "가중치변화": round(delta*100, 1),
            "Top10겹침": overlap,
            "안정성": f"{overlap}/10"
        })

df_ow = pd.DataFrame(ow_results)

print("\n  [One-way] 피처별 가중치 변화 → Top10 겹침 수")
print(f"  {'피처':<8} {'기준%':>6} {'변화범위':>12} {'겹침 min→max':>14} {'판단'}")
print("  " + "-"*55)
for key in KEYS:
    sub = df_ow[df_ow["피처"]==key]
    mn, mx = sub["Top10겹침"].min(), sub["Top10겹침"].max()
    stable = "✓ 안정" if mn >= 7 else ("△ 보통" if mn >= 5 else "✗ 불안정")
    print(f"  {key:<8} {BASE_W[key]*100:>5.0f}%  ±15%p 범위  {mn}→{mx}개     {stable}")

# One-way 시각화
fig, axes = plt.subplots(1, len(KEYS), figsize=(16, 4), sharey=True)
for ax, key in zip(axes, KEYS):
    sub = df_ow[df_ow["피처"]==key].sort_values("변경가중치")
    ax.plot(sub["변경가중치"], sub["Top10겹침"], "o-", color=BLUE, lw=2)
    ax.axhline(8, color="red", ls="--", lw=1, alpha=0.7, label="8개 기준선")
    ax.axvline(BASE_W[key]*100, color="gray", ls=":", lw=1.5, alpha=0.8)
    ax.set_title(f"{key} 가중치", fontsize=10, fontweight="bold")
    ax.set_xlabel("가중치 (%)")
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 12, 2))
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Top10 겹침 수")
axes[0].legend(fontsize=8)
plt.suptitle("One-way 민감도 분석: 가중치 변화 → Top10 안정성", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz_sensitivity_oneway.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz_sensitivity_oneway.png 저장")

# One-way 결과 CSV
df_ow.to_csv(os.path.join(RES_DIR, "민감도_oneway.csv"), index=False, encoding="utf-8-sig")

# ── Monte Carlo 민감도 분석 ─────────────────────────
# 디리클레 분포로 가중치 랜덤 샘플링 → Top10 안정성 및 각 지역 선택 빈도
print("\n  [Monte Carlo] N=5000 시뮬레이션")
np.random.seed(42)
N_ITER = 5000
# 집중도 파라미터: alpha 클수록 기준 가중치 근처에서 샘플링
alpha = np.array([BASE_W[k] for k in KEYS]) * 20
samples = np.random.dirichlet(alpha, N_ITER)

mc_overlaps   = []
mc_top10_freq = {}  # 각 시군구가 Top10에 포함되는 빈도

for s in samples:
    w = dict(zip(KEYS, s))
    top10_new = get_top10_id(w)
    mc_overlaps.append(len(top10_new & base_top10))
    for sid in top10_new:
        mc_top10_freq[sid] = mc_top10_freq.get(sid, 0) + 1

mc_overlaps = np.array(mc_overlaps)
mean_overlap = mc_overlaps.mean()
p80_overlap  = (mc_overlaps >= 8).mean() * 100
p10_overlap  = (mc_overlaps == 10).mean() * 100

# MC 평균 가중치 (기준과의 차이 확인)
mc_mean_w = {k: float(np.mean(samples[:,i])) for i,k in enumerate(KEYS)}

print(f"  평균 Top10 겹침: {mean_overlap:.2f}/10")
print(f"  Top10 완전일치(10/10): {p10_overlap:.1f}%")
print(f"  Top10 8개 이상 유지: {p80_overlap:.1f}%")

# 지역별 Top10 선택 빈도
freq_rows = []
for sid, cnt in sorted(mc_top10_freq.items(), key=lambda x: -x[1]):
    sido, sigg = sid.split("_", 1)
    freq_rows.append({"시도": sido, "시군구": sigg, "선택횟수": cnt, "선택비율_%": round(cnt/N_ITER*100,1)})
df_freq = pd.DataFrame(freq_rows)
df_freq.to_csv(os.path.join(RES_DIR, "민감도_MC_선택빈도.csv"), index=False, encoding="utf-8-sig")

print("\n  Monte Carlo Top15 선택 빈도:")
print(df_freq.head(15)[["시군구","시도","선택비율_%"]].to_string(index=False))

# MC 시각화 (히스토그램 + 빈도 바차트)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 히스토그램: Top10 겹침 분포
ax1.hist(mc_overlaps, bins=range(0,12), align="left", color=BLUE, alpha=0.8, edgecolor="white")
ax1.axvline(mean_overlap, color="red", ls="--", lw=2, label=f"평균: {mean_overlap:.2f}")
ax1.axvline(8, color="orange", ls=":", lw=2, label=f"8개 기준: {p80_overlap:.1f}%")
ax1.set_xlabel("기준 Top10과의 겹침 수")
ax1.set_ylabel("시뮬레이션 횟수")
ax1.set_title(f"Monte Carlo Top10 안정성\n(N={N_ITER:,}, 평균 {mean_overlap:.2f}/10)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 바차트: 지역별 Top10 선택 빈도
top20_freq = df_freq.head(20)
colors_bar = [GREEN if row["시군구"] in [t.split("_")[1] for t in base_top10] else BLUE
              for _, row in top20_freq.iterrows()]
ax2.barh(range(len(top20_freq)), top20_freq["선택비율_%"].values[::-1], color=colors_bar[::-1])
ax2.set_yticks(range(len(top20_freq)))
ax2.set_yticklabels([f"{r['시군구']}({r['시도']})" for _, r in top20_freq.iloc[::-1].iterrows()], fontsize=9)
ax2.set_xlabel("Top10 선택 비율 (%)")
ax2.set_title("지역별 Monte Carlo Top10 선택 빈도\n(초록=기준 Top10 포함)")
ax2.axvline(50, color="gray", ls="--", alpha=0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz_sensitivity_montecarlo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  viz_sensitivity_montecarlo.png 저장")

# ── 가중치 결정 근거 출력 ──────────────────────────
print("\n  [가중치 결정 근거]")
print(f"  {'피처':<8} {'기준가중치':>10} {'One-way 안정성':>14} {'MC 기여도':>12}")
print("  " + "-"*50)
for key in KEYS:
    sub = df_ow[df_ow["피처"]==key]
    min_ov = sub["Top10겹침"].min()
    stability = "✓ 안정" if min_ov >= 7 else ("△ 보통" if min_ov >= 5 else "✗ 불안정")
    print(f"  {key:<8} {BASE_W[key]*100:>9.0f}%  {stability:>14}  {mc_mean_w[key]*100:>10.1f}%")

FINAL_W = BASE_W

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 최종 점수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6] 최종 랭킹 계산")

df["종합_점수"] = sum(FINAL_W[k] * df[FEAT_COLS[k]] for k in KEYS)
df["종합_점수"] = (df["종합_점수"] * 100).round(3)

# 피처 점수 0-100 스케일
for col in ["전력_점수","재해_점수","냉각_점수","신재생_점수","네트워크_점수"]:
    df[col] = (df[col] * 100).round(2)

df = df.sort_values("종합_점수", ascending=False).reset_index(drop=True)
df["랭킹"] = df.index + 1

cols_show = ["랭킹","시도","시군구","종합_점수","전력_점수","재해_점수","냉각_점수","신재생_점수","네트워크_점수"]
print("  최종 Top 15:")
print(df[cols_show].head(15).to_string(index=False))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
feat_out = os.path.join(RES_DIR, "피처별_점수_144개.csv")
rank_out = os.path.join(RES_DIR, "최종_랭킹_144개.csv")

feat_cols_out = ["랭킹","시도","시군구",
                 "N_전력가용량","전력_점수","전력가용량_신규",
                 "N_지진","N_홍수","N_산사태","N_해안","N_산불","재해_점수",
                 "N_취수여유율","N_냉각일수","냉각_점수",
                 "N_발전량","N_접근성","신재생_점수",
                 "네트워크_점수","종합_점수"]
df[feat_cols_out].to_csv(feat_out, index=False, encoding="utf-8-sig")
df[cols_show].to_csv(rank_out, index=False, encoding="utf-8-sig")

# 저장 검증
for path in [feat_out, rank_out]:
    check = pd.read_csv(path, encoding="utf-8-sig")
    assert len(check) == 144, f"저장 오류: {nfc(path)} → {len(check)}행"
    print(f"  저장 완료: {nfc(path)} ({len(check)}행)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 시각화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7] 시각화 생성")
BLUE  = "#1a3c6e"
GREEN = "#2d8a4e"
COLORS = ["#1a3c6e","#2d8a4e","#c0392b","#e67e22","#8e44ad"]

top15 = df.head(15).copy()
score_cols = ["전력_점수","재해_점수","냉각_점수","신재생_점수","네트워크_점수"]
feat_names = {"전력_점수":"전력","재해_점수":"재해","냉각_점수":"냉각",
              "신재생_점수":"신재생","네트워크_점수":"네트워크"}
w_arr = [FINAL_W["전력"],FINAL_W["재해"],FINAL_W["냉각"],FINAL_W["신재생"],FINAL_W["네트워크"]]

# viz1: Top15 종합점수 바차트
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(15), top15["종합_점수"][::-1].values,
               color=[BLUE if i >= 10 else GREEN for i in range(15)])
ax.set_yticks(range(15))
ax.set_yticklabels([f"{r['랭킹']}위 {r['시군구']} ({r['시도']})"
                    for _, r in top15.iloc[::-1].iterrows()], fontsize=10)
ax.set_xlabel("종합 점수 (0~100)")
ax.set_title("거시분석 Top 15 시군구 종합 점수", fontsize=14, fontweight="bold")
ax.axvline(top15["종합_점수"].iloc[9], color="gray", ls="--", alpha=0.5, label="Top10 경계")
for bar, val in zip(bars, top15["종합_점수"][::-1]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", fontsize=9)
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz1_top15_bar.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz1 저장")

# viz2: 누적 바차트
fig, ax = plt.subplots(figsize=(13, 7))
bottoms = np.zeros(15)
for i, (col, color) in enumerate(zip(score_cols, COLORS)):
    vals = (top15[col] * w_arr[i]).values
    ax.bar(range(15), vals, bottom=bottoms, color=color,
           label=f"{list(feat_names.values())[i]} ({w_arr[i]*100:.0f}%)", alpha=0.9)
    bottoms += vals
ax.set_xticks(range(15))
ax.set_xticklabels([f"{r['랭킹']}위\n{r['시군구']}" for _, r in top15.iterrows()],
                   fontsize=9, rotation=30)
ax.set_ylabel("가중 점수"); ax.set_title("Top 15 피처별 기여도", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz2_top15_stacked.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz2 저장")

# viz3: Top5 레이더
from math import pi
top5 = df.head(5)
cats = list(feat_names.values()); N = len(cats)
angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for i, (_, row) in enumerate(top5.iterrows()):
    vals = [row[c] for c in score_cols] + [row[score_cols[0]]]
    ax.plot(angles, vals, "o-", lw=2, label=f"{row['랭킹']}위 {row['시군구']}", color=COLORS[i])
    ax.fill(angles, vals, alpha=0.08, color=COLORS[i])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=11); ax.set_ylim(0, 100)
ax.set_title("Top 5 피처별 점수 레이더", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9); plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz3_top5_radar.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz3 저장")

# viz4: 시도별 박스플롯
fig, ax = plt.subplots(figsize=(14, 6))
sido_order = df.groupby("시도")["종합_점수"].median().sort_values(ascending=False).index
groups = [df[df["시도"]==s]["종합_점수"].values for s in sido_order]
bp = ax.boxplot(groups, patch_artist=True)
for p in bp["boxes"]: p.set_facecolor("#d4e6f1"); p.set_edgecolor(BLUE)
ax.set_xticks(range(1, len(sido_order)+1)); ax.set_xticklabels(sido_order, rotation=30, fontsize=9)
ax.set_ylabel("종합 점수"); ax.set_title("시도별 종합 점수 분포", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz4_sido_boxplot.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz4 저장")

# viz5: 전력 vs 신재생 산점도
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(df["전력_점수"], df["신재생_점수"], c=df["종합_점수"],
                cmap="YlOrRd", s=60, alpha=0.7, edgecolors="gray", lw=0.3)
plt.colorbar(sc, ax=ax, label="종합 점수")
for _, row in df.head(10).iterrows():
    ax.annotate(row["시군구"], (row["전력_점수"], row["신재생_점수"]),
                fontsize=8, ha="center", va="bottom", xytext=(0,5), textcoords="offset points")
ax.set_xlabel("전력 점수"); ax.set_ylabel("신재생 점수")
ax.set_title("전력 vs 신재생 점수 (색=종합점수)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz5_scatter_power_renew.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz5 저장")

# viz6: Top20 히트맵
top20 = df.head(20).copy()
hmap_data = top20[score_cols].T.values.astype(float)
fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(hmap_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label="점수")
ax.set_xticks(range(20))
ax.set_xticklabels([f"{r['랭킹']}위\n{r['시군구']}" for _, r in top20.iterrows()], fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(list(feat_names.values()), fontsize=10)
ax.set_title("Top 20 피처별 점수 히트맵", fontsize=13, fontweight="bold")
for i in range(5):
    for j in range(20):
        ax.text(j, i, f"{hmap_data[i,j]:.0f}", ha="center", va="center", fontsize=7, color="black")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz6_heatmap_top20.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz6 저장")

# viz7: One-way 민감도
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
for ax, key in zip(axes, KEYS):
    sub = df_ow[df_ow["피처"]==key].sort_values("변경가중치")
    ax.plot(sub["변경가중치"], sub["Top10겹침"], "o-", color=BLUE, lw=2, ms=6)
    ax.axhline(10, color="green", ls="--", alpha=0.4, label="완전 일치")
    ax.axvline(BASE_W[key]*100, color="red", ls="--", alpha=0.5, label="기준 가중치")
    ax.set_xlabel(f"{key} 가중치(%)"); ax.set_title(key, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11); ax.grid(True, alpha=0.3)
    if ax == axes[0]: ax.set_ylabel("Top10 겹침 수")
axes[0].legend(fontsize=7)
fig.suptitle("One-way 민감도 분석: 가중치 변화에 따른 Top10 안정성", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz7_oneway_sensitivity.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz7 저장")

# viz8: Monte Carlo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
w_data = [samples[:,i]*100 for i in range(5)]
vp = axes[0].violinplot(w_data, positions=range(5), showmedians=True)
for pc, c in zip(vp["bodies"], COLORS): pc.set_facecolor(c); pc.set_alpha(0.7)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(KEYS, fontsize=10)
axes[0].set_ylabel("가중치 (%)"); axes[0].set_title("Monte Carlo 가중치 분포 (N=5,000)", fontsize=11, fontweight="bold")
for i, k in enumerate(KEYS):
    axes[0].axhline(BASE_W[k]*100, color=COLORS[i], ls="--", alpha=0.5, lw=1)

ov_cnt = pd.Series(mc_overlaps).value_counts().sort_index()
axes[1].bar(ov_cnt.index, ov_cnt.values/N_ITER*100, color=BLUE, alpha=0.8, edgecolor="white")
axes[1].set_xlabel("기준 Top10과 겹치는 수"); axes[1].set_ylabel("비율 (%)")
axes[1].set_title(f"Top10 안정성 (평균 {mean_overlap:.1f}개 일치)", fontsize=11, fontweight="bold")
axes[1].axvline(mean_overlap, color="red", ls="--", label=f"평균 {mean_overlap:.1f}"); axes[1].legend()
plt.suptitle(f"Monte Carlo 민감도 분석 (N={N_ITER:,})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "viz8_montecarlo.png"), dpi=150, bbox_inches="tight"); plt.close()
print("  viz8 저장")

# Folium 지도
try:
    import folium
    m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles="CartoDB positron")
    for _, row in df.head(10).iterrows():
        coord_row = df_filter[df_filter["시군구"]==row["시군구"]]
        if len(coord_row) == 0: continue
        lat, lon = float(coord_row["위도"].iloc[0]), float(coord_row["경도"].iloc[0])
        folium.CircleMarker([lat,lon], radius=12, color=BLUE, fill=True,
                            fill_color=BLUE, fill_opacity=0.7,
                            popup=f"{row['랭킹']}위 {row['시군구']}\n{row['종합_점수']:.1f}점").add_to(m)
        folium.Marker([lat,lon],
                      icon=folium.DivIcon(html=f'<div style="font-size:9px;font-weight:bold;color:white;'
                                              f'background:{BLUE};padding:2px 4px;border-radius:3px">'
                                              f'{row["랭킹"]}위</div>')).add_to(m)
    map_path = os.path.join(RES_DIR, "viz7_map.html")
    m.save(map_path)
    print("  viz_map 저장")
except ImportError:
    print("  folium 없음 — 지도 스킵")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 최종 출력 & 해남군 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("거시분석 최종 Top 10")
print(f"가중치: 전력{FINAL_W['전력']*100:.0f}% / 재해{FINAL_W['재해']*100:.0f}% / "
      f"냉각{FINAL_W['냉각']*100:.0f}% / 신재생{FINAL_W['신재생']*100:.0f}% / "
      f"네트워크{FINAL_W['네트워크']*100:.0f}%")
print(f"MC 안정성: Top10 중 {mean_overlap:.1f}개 평균 유지 / 8개 이상 {p80_overlap:.1f}%")
print("="*60)
print(df[cols_show].head(10).to_string(index=False))

# 해남군 등 이전 상위 지역 확인
print("\n[이전 상위권 지역 점수 확인]")
check_regions = ["해남군", "영광군", "익산시", "태안군", "서산시"]
for r in check_regions:
    row = df[df["시군구"]==r]
    if len(row) > 0:
        r_data = row.iloc[0]
        print(f"  {r_data['시군구']} (랭킹 {int(r_data['랭킹'])}위): "
              f"종합{r_data['종합_점수']:.1f} / 전력{r_data['전력_점수']:.1f} / "
              f"재해{r_data['재해_점수']:.1f} / 냉각{r_data['냉각_점수']:.1f} / "
              f"신재생{r_data['신재생_점수']:.1f}")

print("\n결과 저장 위치:", nfc(RES_DIR))
