"""
KEPCO 전력공급 여유용량(EWM104D04) + 차단기 여유 Bay(EWM100D00) 크롤러
출처: https://online.kepco.co.kr

출력:
  거시분석 시작/데이터셋/전력공급_여유용량.csv        — EWM104D04 154kV
  거시분석 시작/데이터셋/차단기_여유Bay.csv            — EWM100D00 154kV (사용예정 포함)
  거시분석 시작/데이터셋/재생e_전력공급_차단기_통합.csv — 통합 머지
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime

BASE = "https://online.kepco.co.kr"
SAVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "데이터셋"
)
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 재생e 연계 여유용량 원본 위치 탐색 ──────────────────────────────
def find_renw_csv():
    base = "/Users/lshwa/Desktop/이승화/학교/동아리/DArt-B/학술제/26-1학기"
    for item in os.listdir(base):
        if "기타" in item:
            base2 = os.path.join(base, item)
            for sub in os.listdir(base2):
                if "마지막" in sub and sub.encode("utf-8").endswith(b"\xec\x84\x9d"):
                    for gdir in os.listdir(os.path.join(base2, sub)):
                        if "거시" in gdir:
                            p = os.path.join(base2, sub, gdir, "재생e연계_송전망여유용량.csv")
                            if os.path.exists(p):
                                return p
    return None


H_RENW = {
    "Content-Type": 'application/json; charset="UTF-8"',
    "Referer": f"{BASE}/EWM104D04",
    "User-Agent": "Mozilla/5.0",
}
H_CBR = {
    "Content-Type": 'application/json; charset="UTF-8"',
    "Referer": f"{BASE}/EWM100D00",
    "User-Agent": "Mozilla/5.0",
}


def api_post(url, payload, headers, retry=3):
    for attempt in range(retry):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
    return None


# ── EWM104D04: 전력공급 여유용량 ───────────────────────────────────
def crawl_power_supply():
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] 전력공급 여유용량 크롤링 시작 (EWM104D04)")
    print("=" * 60)

    final_path = os.path.join(SAVE_DIR, "전력공급_여유용량.csv")
    interim_path = os.path.join(SAVE_DIR, "_interim_power.csv")

    d = api_post(f"{BASE}/ew/api/energy/selectDo", {}, H_RENW)
    sido_list = d.get("dma_Dolist", []) if d else []
    print(f"시도: {len(sido_list)}개")

    all_rows = []
    done_gu = set()
    if os.path.exists(interim_path):
        prev = pd.read_csv(interim_path)
        all_rows = prev.to_dict("records")
        done_gu = set(zip(prev["_sido_code"].astype(str), prev["_sigg_code"].astype(str)))
        print(f"이전 저장 로드: {len(prev)}행")

    done_count = 0
    for si, sido in enumerate(sido_list, 1):
        sido_code = sido["NSDIP_ALL_ADDR_CD"]
        sido_nm = sido["ADDR_NM"]

        d2 = api_post(f"{BASE}/ew/api/energy/selectGu",
                      {"dma_viewMap": {"Do": sido_code}}, H_RENW)
        gu_list = d2.get("dma_Gulist", []) if d2 else []
        time.sleep(0.3)

        print(f"[{si:02d}/{len(sido_list)}] {sido_nm} — {len(gu_list)}개 시군구")

        for gi, gu in enumerate(gu_list, 1):
            sigg_code_5 = gu["NSDIP_ALL_ADDR_CD"]
            sigg_nm = gu["ADDR_NM"]
            sigg_code_3 = sigg_code_5[2:] if len(sigg_code_5) >= 5 else sigg_code_5

            if (sido_code, sigg_code_5) in done_gu:
                continue

            d3 = api_post(f"{BASE}/ew/api/energy/subSt154",
                          {"dma_subSt154": {
                              "sidoCode": sido_code,
                              "siggCode": sigg_code_3,
                              "emdCode": "",
                              "year": "2026",
                          }}, H_RENW)
            rows = d3.get("dma_subSt154list", []) if d3 else []
            time.sleep(0.35)

            for row in rows:
                all_rows.append({
                    "시도": row.get("SIDO_NM", sido_nm),
                    "시군구": row.get("SGG_NM", sigg_nm),
                    "읍면동": row.get("EMD_NM", ""),
                    "변전소": row.get("PSPWP_NM", row.get("PSPWPNM", "")),
                    "전력공급여유_2026": row.get("THIS_YY", ""),
                    "전력공급여유_2027": row.get("ONE_YY", ""),
                    "전력공급여유_2028": row.get("TWO_YY", ""),
                    "전력공급여유_2029": row.get("THR_YY", ""),
                    "전력공급여유_2030": row.get("FOR_YY", ""),
                    "전력공급여유_2031": row.get("FIV_YY", ""),
                    "전력공급여유_2032": row.get("SIX_YY", ""),
                    "_sido_code": sido_code,
                    "_sigg_code": sigg_code_5,
                })

            done_gu.add((sido_code, sigg_code_5))
            done_count += 1

            if rows:
                print(f"  [{gi:02d}/{len(gu_list)}] {sigg_nm} — {len(rows)}개 ✓")
            else:
                print(f"  [{gi:02d}/{len(gu_list)}] {sigg_nm} — 없음")

            if done_count % 15 == 0:
                pd.DataFrame(all_rows).to_csv(interim_path, index=False, encoding="utf-8-sig")
                print(f"  >>> 중간저장: {len(all_rows)}행")

    if all_rows:
        df = pd.DataFrame(all_rows).drop(columns=["_sido_code", "_sigg_code"], errors="ignore")
        # 숫자형 변환
        year_cols = [c for c in df.columns if "전력공급여유_" in c]
        for col in year_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # 모든 값이 빈 행 제거
        df = df.dropna(subset=year_cols, how="all")
        df.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\n완료! {final_path}")
        print(f"총 {len(df)}행 / 변전소 {df['변전소'].nunique()}개")
    else:
        print("수집된 데이터 없음")

    if os.path.exists(interim_path):
        os.remove(interim_path)

    return final_path if all_rows else None


# ── EWM100D00: 차단기 여유 Bay ─────────────────────────────────────
def crawl_cbr():
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] 차단기 여유 Bay 크롤링 시작 (EWM100D00)")
    print("=" * 60)

    final_path = os.path.join(SAVE_DIR, "차단기_여유Bay.csv")
    interim_path = os.path.join(SAVE_DIR, "_interim_cbr.csv")

    d = api_post(f"{BASE}/ew/cpct/selectChangeDoMapJson", {}, H_CBR)
    sido_list = d.get("dlt_sido", []) if d else []
    print(f"시도: {len(sido_list)}개")

    all_rows = []
    done_gu = set()
    if os.path.exists(interim_path):
        prev = pd.read_csv(interim_path)
        all_rows = prev.to_dict("records")
        done_gu = set(zip(prev["_sido_code"].astype(str), prev["_sigg_code"].astype(str)))
        print(f"이전 저장 로드: {len(prev)}행")

    done_count = 0
    for si, sido in enumerate(sido_list, 1):
        sido_code = sido["NSDIP_ALL_ADDR_CD"]
        sido_nm = sido["ADDR_NM"]

        d2 = api_post(f"{BASE}/ew/cpct/selectChangeGuMapJson",
                      {"dma_reqParam": {"sido_code": sido_code, "sigg_code": "", "emd_code": ""}},
                      H_CBR)
        gu_list = d2.get("dlt_gu", []) if d2 else []
        time.sleep(0.3)

        print(f"[{si:02d}/{len(sido_list)}] {sido_nm} — {len(gu_list)}개 시군구")

        for gi, gu in enumerate(gu_list, 1):
            sigg_code = gu["NSDIP_ALL_ADDR_CD"]
            sigg_nm = gu["ADDR_NM"]

            if (sido_code, sigg_code) in done_gu:
                continue

            d3 = api_post(f"{BASE}/ew/cpct/retrieveCbrOvplsInfo",
                          {"dma_reqParam": {"sido_code": sido_code, "sigg_code": sigg_code, "emd_code": ""}},
                          H_CBR)
            rows = d3.get("dlt_resultList", []) if d3 else []
            time.sleep(0.35)

            for row in rows:
                all_rows.append({
                    "시도": sido_nm,
                    "시군구": sigg_nm,
                    "변전소": row.get("S_NM", row.get("SNM", "")),
                    "변전소코드": row.get("S_CD", ""),
                    "차단기_154kV_전체대수": row.get("B005_TOTAL_BAY", 0),
                    "차단기_154kV_사용중": row.get("B005_USE_BAY_TO", 0),
                    "차단기_154kV_사용예정": row.get("B005_SUBS_BAY", 0),
                    "차단기_154kV_여유대수": row.get("B005_FUTU_BAY", 0),
                    "_sido_code": sido_code,
                    "_sigg_code": sigg_code,
                })

            done_gu.add((sido_code, sigg_code))
            done_count += 1

            if rows:
                print(f"  [{gi:02d}/{len(gu_list)}] {sigg_nm} — {len(rows)}개 ✓")
            else:
                print(f"  [{gi:02d}/{len(gu_list)}] {sigg_nm} — 없음")

            if done_count % 15 == 0:
                pd.DataFrame(all_rows).to_csv(interim_path, index=False, encoding="utf-8-sig")
                print(f"  >>> 중간저장: {len(all_rows)}행")

    if all_rows:
        df = pd.DataFrame(all_rows).drop(columns=["_sido_code", "_sigg_code"], errors="ignore")
        # 숫자형 변환
        bay_cols = [c for c in df.columns if "차단기_" in c]
        for col in bay_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # 모든 bay 값이 0인 행 제거 (빈 변전소)
        df = df[df[bay_cols].sum(axis=1) > 0].reset_index(drop=True)
        df.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"\n완료! {final_path}")
        print(f"총 {len(df)}행 / 변전소 {df['변전소'].nunique()}개")
    else:
        print("수집된 데이터 없음")

    if os.path.exists(interim_path):
        os.remove(interim_path)

    return final_path if all_rows else None


# ── 통합 머지 ──────────────────────────────────────────────────────
def merge_all():
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%H:%M:%S}] 데이터셋 통합 머지")
    print("=" * 60)

    # 재생e 연계 원본 로드
    renw_path = find_renw_csv()
    if renw_path is None:
        print("재생e연계 CSV를 찾을 수 없습니다.")
        return

    df_renw = pd.read_csv(renw_path)
    print(f"재생e연계: {len(df_renw)}행, 변전소 {df_renw['변전소'].nunique()}개")

    # 차단기 로드
    cbr_path = os.path.join(SAVE_DIR, "차단기_여유Bay.csv")
    if not os.path.exists(cbr_path):
        print("차단기 CSV 없음 — 재생e만 저장")
        df_renw.to_csv(os.path.join(SAVE_DIR, "재생e_차단기_통합.csv"),
                       index=False, encoding="utf-8-sig")
        return

    df_cbr = pd.read_csv(cbr_path)
    print(f"차단기: {len(df_cbr)}행, 변전소 {df_cbr['변전소'].nunique()}개")

    # 차단기 — 변전소별 최대값으로 집계 (같은 변전소가 여러 읍면동에 중복 가능)
    bay_cols = ["차단기_154kV_전체대수", "차단기_154kV_사용중", "차단기_154kV_사용예정", "차단기_154kV_여유대수"]
    df_cbr_agg = df_cbr.groupby("변전소")[bay_cols].max().reset_index()

    # 재생e에 차단기 조인 (변전소명 기준)
    df_merged = df_renw.merge(df_cbr_agg, on="변전소", how="left")

    # 아무것도 없는 행 제거 (연도별 값 모두 NaN)
    year_cols = ["2026년", "2027년", "2028년", "2029년", "2030년", "2031년", "2032년"]
    existing_year = [c for c in year_cols if c in df_merged.columns]
    if existing_year:
        df_merged = df_merged.dropna(subset=existing_year, how="all")

    out_path = os.path.join(SAVE_DIR, "재생e_차단기_통합.csv")
    df_merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n통합 완료! {out_path}")
    print(f"총 {len(df_merged)}행")
    matched = df_merged[bay_cols[0]].notna().sum()
    print(f"차단기 매칭된 변전소: {matched}개 / 전체 {len(df_merged)}개")

    return out_path


if __name__ == "__main__":
    crawl_power_supply()
    crawl_cbr()
    merge_all()
