"""
네트워크 피처 - 인구 가중 평균 거리 접근성 지수

계산 방법:
  candidate c 에 대해:
    weighted_dist(c) = sum_j [세대수(j) * haversine(c, j)] / sum_j 세대수(j)
    (j = 비수도권 모든 시군구)
  Min-Max 정규화 후 반전 (1 - normalized, 낮을수록 접근 좋음)

출력:
  거시분석 시작(한글)/데이터셋/네트워크 피처 데이터셋/네트워크_인구가중거리_피처.csv
"""

import os
import math
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_folder(parent, suffix_hex):
    """parent 하위에서 UTF-8 hex suffix로 끝나는 폴더 반환"""
    suffix = bytes.fromhex(suffix_hex)
    for item in os.listdir(parent):
        if item.encode('utf-8').endswith(suffix):
            return os.path.join(parent, item)
    return None


def find_nested(parent, middle_keyword, suffix_hex):
    """parent/*/middle_keyword* & suffix_hex 끝 폴더"""
    suffix = bytes.fromhex(suffix_hex)
    for item in os.listdir(parent):
        subdir = os.path.join(parent, item)
        if not os.path.isdir(subdir):
            continue
        for sub in os.listdir(subdir):
            if middle_keyword in sub and sub.encode('utf-8').endswith(suffix):
                return os.path.join(subdir, sub)
    return None


# 거시분석 시작 (Korean 석 = ec849d) = 데이터 폴더
DATA_BASE = find_folder(ROOT, 'eab1b0ec8b9cebb684ec849d20ec8b9cec9e91')

# 마지막 분석 (Korean 석) 안의 데이터셋 폴더
MASTER_DIR = find_nested(ROOT, '마지막', 'ec849d')  # ...분석 with Korean 석

if DATA_BASE is None:
    raise RuntimeError("거시분석 시작 (Korean) 폴더를 찾을 수 없음")
if MASTER_DIR is None:
    raise RuntimeError("마지막 분석 (Korean) 폴더를 찾을 수 없음")

PATH_144 = os.path.join(DATA_BASE, '데이터셋', '사전필터_통과_시군구_144개.csv')
PATH_HH  = os.path.join(MASTER_DIR, '데이터셋', '행정구역_시군구_별_주민등록세대수.csv')
PATH_MAST = os.path.join(MASTER_DIR, '데이터셋', '최종_피처_마스터.csv')
OUT_DIR  = os.path.join(DATA_BASE, '데이터셋', '네트워크 피처 데이터셋')
OUT_PATH = os.path.join(OUT_DIR, '네트워크_인구가중거리_피처.csv')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Haversine 거리 (km) ──────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ── 행정구역 CSV 파싱: 비수도권 시군구 + 세대수 ─────────────────
def parse_household():
    df = pd.read_csv(PATH_HH, encoding='cp949')
    df.columns = ['행정구역', '세대수']
    df['세대수'] = pd.to_numeric(df['세대수'], errors='coerce').fillna(0).astype(int)

    SIDO_ALL = {
        '전국', '서울특별시', '부산광역시', '대구광역시', '인천광역시',
        '광주광역시', '대전광역시', '울산광역시', '세종특별자치시',
        '경기도', '강원특별자치도', '충청북도', '충청남도',
        '전북특별자치도', '전라남도', '경상북도', '경상남도', '제주특별자치도'
    }
    SUDOBON = {'서울특별시', '인천광역시', '경기도'}
    METRO   = {'부산광역시', '대구광역시', '광주광역시', '대전광역시', '울산광역시'}
    DO      = {'강원특별자치도', '충청북도', '충청남도',
               '전북특별자치도', '전라남도', '경상북도', '경상남도'}

    SIDO_SHORT = {
        '부산광역시': '부산', '대구광역시': '대구', '광주광역시': '광주',
        '대전광역시': '대전', '울산광역시': '울산', '세종특별자치시': '세종',
        '강원특별자치도': '강원', '충청북도': '충북', '충청남도': '충남',
        '전북특별자치도': '전북', '전라남도': '전남', '경상북도': '경북',
        '경상남도': '경남', '제주특별자치도': '제주',
    }

    result = []
    current_sido = None
    prev_was_si  = False
    in_do = in_metro = False

    for _, row in df.iterrows():
        nm = str(row['행정구역']).strip()
        hh = int(row['세대수'])

        if nm in SIDO_ALL:
            current_sido = nm
            prev_was_si  = False
            in_do        = nm in DO
            in_metro     = nm in METRO
            continue

        if current_sido in SUDOBON:
            prev_was_si = False
            continue

        if '출장소' in nm or hh == 0:
            continue

        sido_s = SIDO_SHORT.get(current_sido, current_sido)

        if current_sido == '세종특별자치시':
            if nm == '세종시':
                result.append({'시도': '세종', '시군구': '세종시', '세대수': hh})
            continue

        if current_sido == '제주특별자치도':
            result.append({'시도': '제주', '시군구': nm, '세대수': hh})
            continue

        if in_metro:
            result.append({'시도': sido_s, '시군구': nm, '세대수': hh})
            prev_was_si = False
            continue

        if in_do:
            if nm.endswith('시') or nm.endswith('군'):
                result.append({'시도': sido_s, '시군구': nm, '세대수': hh})
                prev_was_si = nm.endswith('시')
            elif nm.endswith('구') and prev_was_si:
                pass  # 행정구 (수원시 내 장안구 등) 스킵
            else:
                prev_was_si = False
            continue

    return pd.DataFrame(result)


# ── 좌표 DB 구축 ─────────────────────────────────────────────
def build_coord_db():
    SIDO_MAP = {
        '강원특별자치도': '강원', '경기도': '경기',
        '경상남도': '경남', '경상북도': '경북',
        '광주광역시': '광주', '대구광역시': '대구',
        '대전광역시': '대전', '부산광역시': '부산',
        '서울특별시': '서울', '세종특별자치시': '세종',
        '울산광역시': '울산', '인천광역시': '인천',
        '전라남도': '전남', '전북특별자치도': '전북',
        '제주특별자치도': '제주', '충청남도': '충남',
        '충청북도': '충북',
    }

    coords = {}

    # 1) 144개 CSV
    df144 = pd.read_csv(PATH_144)
    for _, r in df144.iterrows():
        key = (str(r['시도']).strip(), str(r['시군구']).strip())
        coords[key] = (float(r['위도']), float(r['경도']))

    # 2) 최종_피처_마스터
    dfm = pd.read_csv(PATH_MAST)
    for _, r in dfm.iterrows():
        sido_s = SIDO_MAP.get(str(r['시도']).strip(), str(r['시도']).strip())
        key = (sido_s, str(r['시군구']).strip())
        if key not in coords:
            coords[key] = (float(r['위도']), float(r['경도']))

    # 3) 하드코딩 보완 (누락 자치구/시)
    EXTRA = {
        ('부산', '중구'): (35.1050, 129.0328),
        ('부산', '동구'): (35.1364, 129.0505),
        ('부산', '연제구'): (35.1814, 129.0818),
        ('부산', '수영구'): (35.1429, 129.1132),
        ('대구', '중구'): (35.8706, 128.5940),
        ('대구', '남구'): (35.8470, 128.5987),
        ('대구', '수성구'): (35.8582, 128.6320),
        ('대구', '달서구'): (35.8416, 128.5337),
        ('대전', '유성구'): (36.3626, 127.3568),
        ('울산', '중구'): (35.5698, 129.3360),
        ('세종', '세종시'): (36.4800, 127.2890),
        ('제주', '제주시'): (33.4897, 126.4983),
        ('제주', '서귀포시'): (33.2534, 126.5596),
        ('강원', '태백시'): (37.1640, 128.9856),
        ('경북', '포항시'): (36.0190, 129.3435),
        ('경북', '구미시'): (36.1195, 128.3446),
        ('경남', '창원시'): (35.2278, 128.6810),
        ('경남', '진주시'): (35.1799, 128.1076),
        ('충북', '청주시'): (36.6421, 127.4890),
        ('충남', '천안시'): (36.8065, 127.1520),
        ('전북', '전주시'): (35.8241, 127.1480),
        ('충남', '금산군'): (36.1082, 127.4882),
        ('전남', '목포시'): (34.8118, 126.3922),
    }
    for key, val in EXTRA.items():
        if key not in coords:
            coords[key] = val

    return coords


# ── 메인 ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("네트워크 피처 - 인구 가중 평균 거리 접근성")
    print(f"데이터 폴더: {DATA_BASE}")
    print("=" * 60)

    df_hh = parse_household()
    print(f"\n비수도권 시군구: {len(df_hh)}개 / 총 세대수: {df_hh['세대수'].sum():,}")

    coords = build_coord_db()
    print(f"좌표 DB: {len(coords)}개 시군구")

    def get_coord(sido, sigg):
        key = (sido, sigg)
        if key in coords:
            return coords[key]
        for (s, g), v in coords.items():
            if g == sigg:
                return v
        return None

    df_hh['위도'] = df_hh.apply(
        lambda r: get_coord(r['시도'], r['시군구'])[0] if get_coord(r['시도'], r['시군구']) else None, axis=1)
    df_hh['경도'] = df_hh.apply(
        lambda r: get_coord(r['시도'], r['시군구'])[1] if get_coord(r['시도'], r['시군구']) else None, axis=1)

    missing = df_hh[df_hh['위도'].isna()]
    if len(missing):
        print(f"\n좌표 없는 시군구 {len(missing)}개 (제외됨):")
        print(missing[['시도', '시군구']].to_string())

    df_hh = df_hh.dropna(subset=['위도', '경도']).reset_index(drop=True)
    total_hh = df_hh['세대수'].sum()
    print(f"\n최종 비수도권 시군구: {len(df_hh)}개 / 총 세대수: {total_hh:,}\n")

    # 벡터화 계산 (numpy 사용)
    df144 = pd.read_csv(PATH_144)
    lats_dest = df_hh['위도'].values
    lons_dest = df_hh['경도'].values
    hhs_dest  = df_hh['세대수'].values.astype(float)

    R = 6371.0
    p2 = np.radians(lats_dest)
    l2 = np.radians(lons_dest)

    weighted_dists = []
    for idx, cand in df144.iterrows():
        p1 = math.radians(float(cand['위도']))
        l1 = math.radians(float(cand['경도']))

        dp = p2 - p1
        dl = l2 - l1
        a  = np.sin(dp/2)**2 + math.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        dists = 2 * R * np.arcsin(np.sqrt(a))

        wdist = np.sum(hhs_dest * dists) / np.sum(hhs_dest)
        weighted_dists.append(wdist)

        if idx % 30 == 0:
            print(f"  [{idx+1:03d}/{len(df144)}] {cand['시도']} {cand['시군구']} -> {wdist:.1f} km")

    df144 = df144.copy()
    df144['인구가중평균거리_km'] = weighted_dists

    vmin = df144['인구가중평균거리_km'].min()
    vmax = df144['인구가중평균거리_km'].max()
    df144['정규화_거리'] = (df144['인구가중평균거리_km'] - vmin) / (vmax - vmin)
    df144['네트워크_접근성'] = 1 - df144['정규화_거리']

    out_cols = ['시도', '시군구', '위도', '경도', '인구가중평균거리_km', '네트워크_접근성']
    df_out = df144[out_cols].sort_values('네트워크_접근성', ascending=False).reset_index(drop=True)
    df_out.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')

    print(f"\n{'='*60}")
    print(f"완료! {OUT_PATH}")
    print(f"총 {len(df_out)}개 후보 시군구\n")
    print("상위 20개 (접근성 높은 순):")
    print(df_out.head(20)[['시도', '시군구', '인구가중평균거리_km', '네트워크_접근성']].to_string(index=False))
    print("\n하위 10개 (접근성 낮은 순):")
    print(df_out.tail(10)[['시도', '시군구', '인구가중평균거리_km', '네트워크_접근성']].to_string(index=False))


if __name__ == '__main__':
    main()
