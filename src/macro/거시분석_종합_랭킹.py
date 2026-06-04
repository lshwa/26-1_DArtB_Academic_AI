"""
AI 데이터센터 최적 입지 - 거시분석 종합 랭킹 산출
DArt-B 학술제 26-1학기

5개 피처 카테고리 가중치 (근거: 가중치_산정_근거.md 참조):
  1. 전력     35% — 안정적 전력 공급 가능성 (핵심 인프라)
  2. 냉각     20% — PUE·수냉 환경 (운영비 직결)
  3. 신재생   20% — RE100·ESG 조달 가능성
  4. 네트워크 15% — 비수도권 인구 접근성 (추론 서비스 latency)
  5. 재해     10% — 자연재해 위험도 (시설 안전성)

피처 내 세부 변수 가중치는 함수 내 주석 참조.
"""

import os
import pandas as pd
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE = '/Users/lshwa/Desktop/이승화/학교/동아리/DArt-B/학술제/26-1학기'

def find_korean_folder():
    """Korean 석 (ec849d) 거시분석 시작 폴더 반환"""
    for item in os.listdir(BASE):
        h = item.encode('utf-8').hex()
        if 'ec849d' in h and h.endswith('ec8b9cec9e91'):
            return os.path.join(BASE, item)
    raise RuntimeError("Korean 석 폴더를 찾을 수 없음")

FOLDER = find_korean_folder()
DS = os.path.join(FOLDER, '데이터셋')

PATH_144    = os.path.join(DS, '사전필터_통과_시군구_144개.csv')
PATH_전력   = os.path.join(DS, '전력 피처 데이터셋', '전력피처_시군구별.csv')
PATH_신재생 = os.path.join(DS, '신재생 피처 데이터셋', '신재생_피처_수정완료.csv')
PATH_네트워크 = os.path.join(DS, '네트워크 피처 데이터셋', '네트워크_인구가중거리_피처.csv')
PATH_냉각   = os.path.join(DS, '냉각 피처 데이터셋', '냉각_피처_최종.csv')
PATH_재해   = os.path.join(DS, '재해 피처 데이터셋', '재해_피처_수정완료.csv')

def find_result_dir():
    """거시분석 결과 또는 거시분석 폴더 내 결과 폴더 탐색"""
    import subprocess
    result = subprocess.run(
        ['find', FOLDER, '-maxdepth', '2', '-name', '결과', '-type', 'd'],
        capture_output=True, text=True
    )
    dirs = [p for p in result.stdout.strip().split('\n') if p]
    if dirs:
        return dirs[0]
    # 없으면 생성
    d = os.path.join(FOLDER, '거시분석_결과', '결과')
    os.makedirs(d, exist_ok=True)
    return d

OUT_DIR    = find_result_dir()
OUT_랭킹   = os.path.join(OUT_DIR, '최종_랭킹_144개.csv')
OUT_피처별 = os.path.join(OUT_DIR, '피처별_점수_144개.csv')

# 전국 시도명 → 단축 매핑
SIDO_MAP = {
    '강원특별자치도': '강원', '경상남도': '경남', '경상북도': '경북',
    '광주광역시': '광주', '대구광역시': '대구', '대전광역시': '대전',
    '부산광역시': '부산', '세종특별자치시': '세종', '울산광역시': '울산',
    '전라남도': '전남', '전북특별자치도': '전북', '제주특별자치도': '제주',
    '충청남도': '충남', '충청북도': '충북',
}
수도권 = {'서울특별시', '인천광역시', '경기도'}


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-Max 정규화, invert=True 이면 낮을수록 높은 점수"""
    vmin, vmax = series.min(), series.max()
    if vmax == vmin:
        return pd.Series(0.5, index=series.index)
    norm = (series - vmin) / (vmax - vmin)
    return (1 - norm) if invert else norm


def join_144(df: pd.DataFrame, keys144: set) -> pd.DataFrame:
    """(시도, 시군구) 기준 144개 필터링"""
    mask = df.apply(lambda r: (r['시도'], r['시군구']) in keys144, axis=1)
    return df[mask].copy().reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
def process_전력(df144):
    """
    전력 피처 처리 및 점수화

    [피처 수정 사유]
    기존 전력가용량_MAX = 전력공급여유량 × 송전망여유량 (곱셈) 방식은
    두 값이 모두 높은 강원내륙 변전소 권역(정선·원주·영월·횡성·단양 등)이
    Min-Max 정규화 후 1.0을 독점하는 왜곡을 발생시켰음.
    → 전력공급여유량_2032(실제 DC 접속 가능 공급 여유, 단위: MW)만 사용.

    전력공급여유량 = 0 인 시군구 처리:
      나주시·아산시 등은 기존 산업 수요로 변전소 여유가 현재 0이나,
      에너지밸리·반도체특구 등 신규 인프라 계획이 있어
      시도 내 비포화 시군구 중앙값의 50%를 부여 (보수적 보완).
    """
    print("\n[1] 전력 피처 처리...")
    df = pd.read_csv(PATH_전력)

    # 수도권 제외 + 시도명 단축
    df = df[~df['시도'].isin(수도권)].copy()
    df['시도'] = df['시도'].map(SIDO_MAP).fillna(df['시도'])

    keys144 = set(zip(df144['시도'], df144['시군구']))
    VAL_COLS = ['송전망여유량_2032', '전력공급여유량_2032', '전력가용량_MAX']

    # ── 특수 매핑: 구 단위 분리 도시 → 144개 시군구명으로 통합 ──────────
    MERGE_RULES = [
        ('경남', '창원시', '창원시'),
        ('전북', '전주시', '전주시'),
        ('충남', '천안시', '천안시'),
        ('경북', '포항시', '포항시'),
    ]
    extra_rows = []
    for sido, prefix, target in MERGE_RULES:
        sub = df[(df['시도'] == sido) & df['시군구'].str.startswith(prefix)]
        if len(sub):
            merged = {c: sub[c].max() for c in VAL_COLS}
            extra_rows.append({'시도': sido, '시군구': target, **merged})

    # 세종시 → 4개 읍면동 각각 복사
    sejong_row = df[(df['시도'] == '세종') & (df['시군구'] == '세종시')]
    if len(sejong_row):
        for dong in ['전의면', '연서면', '금남면', '고운동']:
            r = {c: sejong_row.iloc[0][c] for c in VAL_COLS}
            extra_rows.append({'시도': '세종', '시군구': dong, **r})

    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    df = join_144(df, keys144)

    # 복수 행(같은 시군구, 다른 변전소) → 최대값 집계
    df = df.groupby(['시도', '시군구'], as_index=False).agg({
        '송전망여유량_2032':   'max',
        '전력공급여유량_2032': 'max',
        '전력가용량_MAX':      'max',
    })

    matched = len(df)
    missing_keys = keys144 - set(zip(df['시도'], df['시군구']))
    print(f"  매칭: {matched}개 / 미매칭 {len(missing_keys)}개")
    if missing_keys:
        print(f"  미매칭: {sorted(missing_keys)[:10]}")

    # 전력공급여유량_2032 사용 (곱셈 왜곡 제거)
    # 0값(현재 포화) → 시도 내 비포화 시군구 중앙값의 50%로 보수적 보완
    power_col = df['전력공급여유량_2032'].copy().astype(float)
    power_col = power_col.replace(0, np.nan)

    sido_medians = {}
    for sido, grp in df.groupby('시도'):
        vals = grp['전력공급여유량_2032'].replace(0, np.nan).dropna()
        sido_medians[sido] = vals.median() if len(vals) else np.nan

    global_median = power_col.dropna().median()
    for i, row in df.iterrows():
        if pd.isna(power_col.iloc[i]):
            med = sido_medians.get(row['시도'], global_median)
            if pd.isna(med):
                med = global_median
            power_col.iloc[i] = med * 0.5

    df['전력공급여유량_보완'] = power_col
    df['N_전력가용량'] = minmax(df['전력공급여유량_보완'])
    df['전력_점수']    = df['N_전력가용량']

    zero_cities = df[df['전력공급여유량_2032'] == 0][['시도', '시군구', '전력공급여유량_보완']].to_string(index=False)
    print(f"  전력=0 보완 시군구:\n{zero_cities}")
    print(f"  전력_점수: mean={df['전력_점수'].mean():.3f}, "
          f"max={df['전력_점수'].max():.3f}, min={df['전력_점수'].min():.3f}")
    return df[['시도', '시군구', '전력_점수', 'N_전력가용량']]


# ══════════════════════════════════════════════════════════════════════════════
def process_신재생(df144):
    """
    신재생 피처 처리 및 점수화
    변수 가중치 (합=1.0):
      PPA가능용량_kW      0.35  – 직접 계약 가능 용량 (RE100 가장 즉각적 수단)
      신재생발전량_GWh    0.30  – 지역 공급 규모 (공급 안정성)
      신재생접근성_유효   0.20  – 인근 사업자 수 기반 접근성 (계약 다양성)
      시군구자급률        0.15  – 지역 에너지 자립도 (장기 독립성)

    NaN PPA → 시도 중앙값 대체 (도시 지역 PPA 제약 반영)
    """
    print("\n[2] 신재생 피처 처리...")
    df = pd.read_csv(PATH_신재생)

    # PPA NaN → 시도 중앙값
    df['PPA가능용량_kW'] = df.groupby('시도')['PPA가능용량_kW'].transform(
        lambda x: x.fillna(x.median())
    )
    df['PPA가능용량_kW'] = df['PPA가능용량_kW'].fillna(0)  # 시도 전체 NaN 이면 0

    keys144 = set(zip(df144['시도'], df144['시군구']))
    df = join_144(df, keys144)

    matched = len(df)
    missing_keys = keys144 - set(zip(df['시도'], df['시군구']))
    print(f"  매칭: {matched}개 / 미매칭 {len(missing_keys)}개")

    df['N_PPA용량']    = minmax(df['PPA가능용량_kW'])
    df['N_발전량']     = minmax(df['신재생발전량_GWh'])
    df['N_신재생접근성'] = minmax(df['신재생접근성_유효'])
    df['N_자급률']     = minmax(df['시군구자급률'])

    df['신재생_점수'] = (
        df['N_PPA용량']      * 0.35 +
        df['N_발전량']       * 0.30 +
        df['N_신재생접근성'] * 0.20 +
        df['N_자급률']       * 0.15
    )

    print(f"  신재생_점수: mean={df['신재생_점수'].mean():.3f}, "
          f"max={df['신재생_점수'].max():.3f}, min={df['신재생_점수'].min():.3f}")
    return df[['시도', '시군구', '신재생_점수',
               'N_PPA용량', 'N_발전량', 'N_신재생접근성', 'N_자급률']]


# ══════════════════════════════════════════════════════════════════════════════
def process_네트워크():
    """
    네트워크 피처 — 이미 정규화된 '네트워크_접근성' 사용
    추론(Inference) 서비스 위주 AI DC 가정:
      - 추론 비중 ~70%, 학습 비중 ~30% (NVIDIA AI Factory 2023 기준)
      - 추론은 latency 민감 → 인구 근접성이 직접적 서비스 품질 결정
      - 따라서 네트워크 접근성 피처는 추론 워크로드 반영 가중치 적용
    단일 변수(인구가중평균거리)이므로 별도 내부 가중치 없음.
    """
    print("\n[3] 네트워크 피처 처리...")
    df = pd.read_csv(PATH_네트워크)
    df = df.rename(columns={'네트워크_접근성': '네트워크_점수'})
    print(f"  네트워크_점수: mean={df['네트워크_점수'].mean():.3f}, "
          f"max={df['네트워크_점수'].max():.3f}, min={df['네트워크_점수'].min():.3f}")
    return df[['시도', '시군구', '네트워크_점수']]


# ══════════════════════════════════════════════════════════════════════════════
def process_냉각():
    """
    냉각 피처 — 이미 냉각_종합 점수 산출 완료
    내부 가중치 (냉각_피처_최종.csv 에서 적용됨):
      N_기온      0.30  – 외기 냉각(free cooling) 기준 온도
      N_취수여유율 0.40 – 수냉식 냉각 수원 가용성 (전체 냉각 에너지의 1/3)
      N_냉각일수  0.20  – 연간 free cooling 가능 일수
      N_풍속      0.10  – 공기 냉각 보조 (열 분산)
    """
    print("\n[4] 냉각 피처 처리...")
    df = pd.read_csv(PATH_냉각)
    df = df.rename(columns={'냉각_종합': '냉각_점수'})
    print(f"  냉각_점수: mean={df['냉각_점수'].mean():.3f}, "
          f"max={df['냉각_점수'].max():.3f}, min={df['냉각_점수'].min():.3f}")
    return df[['시도', '시군구', '냉각_점수',
               'N_기온', 'N_풍속', 'N_냉각일수', 'N_취수여유율']]


# ══════════════════════════════════════════════════════════════════════════════
def process_재해(df144):
    """
    재해 피처 처리 및 점수화
    변수 가중치 (합=1.0):
      지진위험지수  0.40  – 구조 안전 (2016 경주 5.8, 2017 포항 5.4 사례)
      재해위험_가중합 0.30 – 종합 재해 발생·규모 (행안부 재해연보 기반)
      산사태_건수   0.20  – 산악 지형 인프라 리스크 (DC는 지형 안정 필요)
      해안위험_등급 0.10  – 폭풍해일·침수 위험 (0=내륙·안전, 2=고위험)
    모두 낮을수록 안전 → 반전 정규화
    """
    print("\n[5] 재해 피처 처리...")
    df = pd.read_csv(PATH_재해)
    keys144 = set(zip(df144['시도'], df144['시군구']))
    df = join_144(df, keys144)

    matched = len(df)
    missing_keys = keys144 - set(zip(df['시도'], df['시군구']))
    print(f"  매칭: {matched}개 / 미매칭 {len(missing_keys)}개")

    # 반전 정규화 (낮을수록 좋음)
    df['N_지진']     = minmax(df['지진위험지수'],    invert=True)
    df['N_재해가중'] = minmax(df['재해위험_가중합'], invert=True)
    df['N_산사태']   = minmax(df['산사태_건수'],     invert=True)
    df['N_해안']     = minmax(df['해안위험_등급'],   invert=True)

    df['재해_점수'] = (
        df['N_지진']     * 0.40 +
        df['N_재해가중'] * 0.30 +
        df['N_산사태']   * 0.20 +
        df['N_해안']     * 0.10
    )

    print(f"  재해_점수: mean={df['재해_점수'].mean():.3f}, "
          f"max={df['재해_점수'].max():.3f}, min={df['재해_점수'].min():.3f}")
    return df[['시도', '시군구', '재해_점수',
               'N_지진', 'N_재해가중', 'N_산사태', 'N_해안']]


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("AI 데이터센터 최적 입지 — 거시분석 종합 랭킹 산출")
    print("=" * 70)

    df144 = pd.read_csv(PATH_144)
    keys144 = set(zip(df144['시도'], df144['시군구']))
    print(f"\n사전필터 통과 시군구: {len(df144)}개")

    # ── 5개 피처 처리 ──
    df_전력   = process_전력(df144)
    df_신재생 = process_신재생(df144)
    df_네트워크 = process_네트워크()
    df_냉각   = process_냉각()
    df_재해   = process_재해(df144)

    # ── 병합 ──────────────────────────────────────────────────────────────
    print("\n[6] 피처 병합 중...")
    base_df = df144[['시도', '시군구']].copy()

    for df_, name in [
        (df_전력,   '전력'),
        (df_신재생, '신재생'),
        (df_네트워크, '네트워크'),
        (df_냉각,   '냉각'),
        (df_재해,   '재해'),
    ]:
        base_df = base_df.merge(df_, on=['시도', '시군구'], how='left')
        nan_cnt = base_df[f'{name}_점수'].isna().sum()
        if nan_cnt:
            print(f"  경고: {name} 미매칭 {nan_cnt}개 → 시도 중앙값 대체")
            base_df[f'{name}_점수'] = base_df.groupby('시도')[f'{name}_점수'].transform(
                lambda x: x.fillna(x.median())
            )
            base_df[f'{name}_점수'] = base_df[f'{name}_점수'].fillna(
                base_df[f'{name}_점수'].median()
            )

    # ── 최종 종합 점수 ────────────────────────────────────────────────────
    # 가중치 (합=1.0):
    # 가중치 (합=1.0):
    # 전력35% | 냉각20% | 신재생20% | 네트워크5% | 재해20%
    W = {'전력': 0.35, '냉각': 0.20, '신재생': 0.20, '네트워크': 0.05, '재해': 0.20}

    base_df['종합_점수'] = (
        base_df['전력_점수']   * W['전력']   +
        base_df['냉각_점수']   * W['냉각']   +
        base_df['신재생_점수'] * W['신재생'] +
        base_df['네트워크_점수'] * W['네트워크'] +
        base_df['재해_점수']   * W['재해']
    )

    base_df = base_df.sort_values('종합_점수', ascending=False).reset_index(drop=True)
    base_df.insert(0, '랭킹', range(1, len(base_df) + 1))

    # ── 저장 ─────────────────────────────────────────────────────────────
    # 랭킹 + 주요 점수만
    rank_cols = ['랭킹', '시도', '시군구', '종합_점수',
                 '전력_점수', '냉각_점수', '신재생_점수', '네트워크_점수', '재해_점수']
    base_df[rank_cols].to_csv(OUT_랭킹, index=False, encoding='utf-8-sig')

    # 세부 N_* 컬럼까지 모두 저장
    base_df.to_csv(OUT_피처별, index=False, encoding='utf-8-sig')

    # ── 결과 출력 ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"완료! 최종 랭킹 → {OUT_랭킹}")
    print(f"피처별  → {OUT_피처별}")
    print(f"\n[가중치] 전력 {W['전력']*100:.0f}% | 냉각 {W['냉각']*100:.0f}% | "
          f"신재생 {W['신재생']*100:.0f}% | 네트워크 {W['네트워크']*100:.0f}% | "
          f"재해 {W['재해']*100:.0f}%")

    print(f"\n{'─'*70}")
    print("★ TOP 15 — AI 데이터센터 최적 입지 후보")
    print(f"{'─'*70}")
    top15 = base_df.head(15)[['랭킹', '시도', '시군구', '종합_점수',
                               '전력_점수', '냉각_점수', '신재생_점수',
                               '네트워크_점수', '재해_점수']]
    print(top15.to_string(index=False, float_format='%.3f'))

    print(f"\n{'─'*70}")
    print("시도별 평균 종합 점수 (상위 순)")
    print(f"{'─'*70}")
    print(base_df.groupby('시도')['종합_점수'].mean().sort_values(
        ascending=False).round(3).to_string())

    print(f"\n{'─'*70}")
    print("하위 10개 (낮은 적합도)")
    print(f"{'─'*70}")
    print(base_df.tail(10)[['랭킹', '시도', '시군구', '종합_점수']].to_string(
        index=False, float_format='%.3f'))


if __name__ == '__main__':
    main()
