# MDPI Drones 논문 직접 수정 가이드

이 디렉토리는 2026-09-05의 **MDPI Drones 작업물**을 복사한 편집용 스냅샷이다. 원본 PDF는 `originals/`에 보존되어 있고, 색상 표시본은 `FusedExactSwitchGLOBE_MDPI_Editing_Guide.pdf`, 편집할 TeX는 `source/main.tex`이다. 원문 비교용 무색 소스는 `source/main_clean_snapshot.tex`이다.

## 색상 규칙

- **빨간색 (`\MustEdit{...}`)**: 저자·소속·이메일·기여·연구비·데이터 공개·감사·이해상충처럼 제출 전에 확정해야 하는 항목이다.
- **파란색 (`\VerifyEdit{...}`)**: 모델명, 실험 설계, 임계값, 성능 주장처럼 연구 내용 변경 시 재검증해야 하는 항목이다.
- 이 PDF는 안내용이며 그대로 제출하면 안 된다. 최종본에서는 색상 래퍼와 안내 상자를 제거한다.

## 최우선 수정 항목

|우선순위|파일·위치|수정 내용|검증 기준|
|---|---|---|---|
|필수|`source/main.tex` 29행 부근|제목·제안기법 이름|현재 MDPI 스냅샷은 `Fused Exact SwitchGLOBE` 명칭이다. IEEE 최신본의 `DiSwitch`로 통일할지 먼저 결정하고, 통일 시 매크로·본문·그림·보충자료를 전부 함께 변경|
|필수|31--34행|저자, AuthorNames, 공식 소속, 교신 이메일|저널 템플릿의 저자 번호·소속 번호·이메일 형식까지 반영|
|필수|36행|초록 수치와 novelty 문장|최종 메인 결과표, latency 표, CI와 한 글자도 다르지 않게 대조|
|필수|494행 부근|Author Contributions|최종 CRediT 역할을 저자별로 명시|
|필수|495행 부근|Funding|과제명·기관·grant number·수혜자를 정확히 기재하거나 승인된 no funding 문구 사용|
|필수|496--497행|IRB / Informed Consent|시뮬레이션 연구에 `Not applicable`이 적절한지 기관·저널 기준으로 확인|
|필수|498행 부근|Data Availability|공개 URL·DOI 또는 제한 사유와 접근 절차 기재|
|필수|499--500행|Acknowledgments / Conflicts|감사 대상 및 두 저자의 이해상충을 최종 확인|

## 과학 내용 변경 시 연동 수정표

|변경 원인|함께 수정할 섹션|함께 갱신할 파일|
|---|---|---|
|DiSwitch로 리브랜딩|제목, `\SG`/`\FSG`/`\FastSG` 매크로, 초록, 모든 절·표·알고리즘·캡션|메인/보충 그림의 텍스트, cover letter, README, ZIP 파일명|
|risk feature·threshold 변경|Predictive Local Branch, Exact Risk Switch, Algorithm|gate calibration, ablation, checkpoint config|
|실험 규모 변경|Experimental Methodology, Statistical Analysis, Abstract, Conclusion|protocol 그림, 모든 CI·표·보충자료|
|baseline 변경|Related Work, claim-gap matrix, Results|성능표, heatmap, Pareto, literature CSV|
|PDR·DDR·성공 delay 변경|Abstract, Overall Results, Discussion, Conclusion|figure data, summary table, supplementary figures|
|decision latency 재측정|Benchmark scope, latency results, CPU-vs-CUDA discussion|A100 manifest, operator profile, latency plots|
|실제 E2E 또는 testbed 결과 추가|Metrics, E2E-vs-decision discussion, limitations|새 protocol·testbed 그림과 raw data provenance|

## 섹션별 점검 리스트

1. **MDPI 메타데이터**: `\pubvolume`, `\issuenum`, `\articlenumber`, `\datereceived` 등은 출판사 처리 필드인지 확인하고 임의 값을 만들지 않는다.
2. **Title/Abstract/Keywords**: IEEE 버전과 동일 연구를 설명한다면 명칭·수치·주요 결론을 동기화한다.
3. **Introduction/Related Work**: 최신 Drones, IEEE Access, IEEE TMC/TWC/IoTJ 계열 관련 연구를 확인하고 novelty 범위를 보수적으로 유지한다.
4. **Method**: 모든 기호, threshold, mask, DROP semantics, distillation loss를 코드·설정과 대조한다.
5. **Experimental Methodology**: seed가 inferential unit이라는 설명, scenario-macro 집계, CI 계산, timing scope를 재현 자료와 맞춘다.
6. **Results**: 메인 표, heatmap, ablation, latency, Pareto, negative result의 숫자를 동일 CSV에서 재생성한다.
7. **Discussion/Limitations**: adapted baseline과 simulator-only 검증의 한계를 삭제하지 말고 새 증거가 생길 때만 완화한다.
8. **Back Matter**: Author Contributions, Funding, IRB, Consent, Data Availability, Acknowledgments, Conflicts를 제출 직전 다시 확인한다.
9. **References**: `source/references.bib`의 DOI와 bibliographic metadata를 검증한다.
10. **Supplementary/Cover Letter**: `source/supplementary.tex`, `source/cover_letter.tex`의 모델명·수치·저자·저널명을 메인 원고와 동기화한다.

## 최종 무색본 만드는 방법

1. `source/main.tex`의 안내 상자를 삭제한다.
2. 모든 `\MustEdit{...}`와 `\VerifyEdit{...}`에서 래퍼만 제거한다.
3. `source/main_clean_snapshot.tex`는 비교 기준으로 보존한다.
4. MDPI class와 bibliography를 포함해 재컴파일하고 전 페이지 PNG 렌더링으로 표·그림·back matter를 확인한다.

