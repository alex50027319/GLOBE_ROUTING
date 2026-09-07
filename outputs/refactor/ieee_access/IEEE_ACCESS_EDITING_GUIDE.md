# IEEE Access 논문 직접 수정 가이드

이 디렉토리는 2026-09-07의 **DiSwitch IEEE Access 작업물**을 복사한 편집용 스냅샷이다. 원본 제출 후보 PDF는 `originals/`에 보존되어 있고, 색상 표시본은 `DiSwitch_IEEEAccess_Editing_Guide.pdf`, 편집할 TeX는 `source/main.tex`이다. 원문 비교용 무색 소스는 `source/main_clean_snapshot.tex`이다.

## 색상 규칙

- **빨간색 (`\MustEdit{...}`)**: 제출 전에 저자가 직접 확인하거나 실제 값으로 바꿔야 하는 항목이다.
- **파란색 (`\VerifyEdit{...}`)**: 실험, 수치, 모델 정의, 논문 포지셔닝이 바뀌면 함께 재검증해야 하는 항목이다.
- 색상본은 편집 안내용이며 제출본이 아니다. 최종 제출 전 `\MustEdit`/`\VerifyEdit` 래퍼와 안내 상자를 제거하고 다시 컴파일한다.

## 가장 먼저 수정할 항목

|우선순위|파일·위치|수정 내용|검증 기준|
|---|---|---|---|
|필수|`source/main.tex` 43행 부근|논문 제목과 `DiSwitch` 명칭|제목, 본문, 그림, 보충자료, 코드 릴리스의 표기가 완전히 같은지 확인|
|필수|44--47행|저자 순서, 영문명, 소속, 교신저자 이메일|Kyung Eun Kim / Howon Lee 표기, Ajou University 공식 영문 소속, 실제 기관 이메일 확인|
|필수|49--51행|초록의 모든 결과 수치|최종 CSV·통계표와 PDR, DDR, CI, A100/CPU p95 값 대조|
|필수|539행 부근|Author Contributions|두 저자가 승인한 최종 CRediT 역할만 기재|
|필수|542행 부근|Data Availability|공개 저장소 URL, 릴리스 태그, Zenodo/OSF DOI 또는 승인된 접근 조건 삽입|
|필수|545행 부근|Conflicts of Interest|두 저자가 제출 직전에 재확인|
|필수|548행 부근|AI 사용 고지|IEEE Access의 제출 시점 정책과 투고 시스템 문항에 맞게 유지·이동·수정|

## 과학 내용이 바뀔 때 연동 수정할 항목

|변경 원인|함께 수정할 본문|함께 교체할 자료|
|---|---|---|
|모델 이름·구조 변경|제목, 초록, Introduction 기여 목록, Proposed Method, Algorithm, Conclusion|`figures/figure_01_training.*`, `figure_02_deployment.*`, `figure_02_fusion.*`, 보충그림의 범례|
|risk switch 임계값 변경|Predictive Local Branch와 Risk Switch의 식 및 설명(217--240행 부근)|게이트 calibration 그림, ablation CSV, 체크포인트/설정 파일|
|seed·scenario·episode 수 변경|Experimental Methodology(293행 이후), 통계 분석, 초록, 결론|protocol 그림, 모든 표의 CI, supplementary 전체|
|baseline 추가·재구현|Related Work, novelty 표, Overall Routing Performance|메인 성능표, heatmap, Pareto plot, `literature_comparison.csv`|
|PDR·DDR·delay 결과 변경|초록, Results 330행 이후, Discussion, Conclusion|`figure_data/`, `supplementary_figures/`, 표의 평균·CI·효과크기|
|CPU/GPU 재벤치마크|Decision-Latency Benchmark, Results의 latency 절, CUDA 원인 분석|latency summary/forest/device comparison, A100 manifest와 operator profile|
|Fast-DiSwitch가 gate 통과|연구 질문 4, acceptance gate 해석, Fast 비교 절, conclusion|gate calibration, failed-candidate tradeoff, latency-quality 표|
|E2E delay 또는 실제 무선시험 추가|Metrics and Scope, End-to-End Delay 절, Limitations|새 실험 설정, HIL/testbed 도식, latency scope 표|

## 섹션별 점검 리스트

1. **Title/Abstract/Keywords**: 제목의 novelty가 본문 검증 범위를 넘지 않는지, 초록 수치가 최종 표와 일치하는지, 키워드가 IEEE taxonomy와 부합하는지 확인한다.
2. **Introduction**: 문제 정의, 연구 질문 4개, 기여 5개가 최종 실험으로 각각 뒷받침되는지 확인한다.
3. **Related Work**: 투고 직전 최신 UAV/FANET routing·policy distillation·inference latency 논문을 보강하고 모든 비교 주장의 인용을 확인한다.
4. **System Model**: 상태·행동·DROP·mask·deployment contract가 실제 코드와 일치해야 한다.
5. **Proposed Method**: PPO/KD 손실, normal/predictive logits, danger와 safety gain, switch 식, 알고리즘의 변수 정의와 코드 구현을 대조한다.
6. **Experimental Methodology**: simulator 버전, seed, scenario, episode, baseline contract, hardware, warm-up, repeat 수를 재현 manifest와 맞춘다.
7. **Results**: 표와 그림을 생성한 CSV를 단일 진실 원천으로 삼고 수기 숫자 변경을 피한다.
8. **Discussion/Limitations**: adapted baseline, 5-seed power, simulator-only evidence, conditional delay, CPU/GPU scope를 과장 없이 유지한다.
9. **References**: `source/references.bib`의 DOI, 연도, 권·호·쪽, online-first 상태를 투고 직전에 확인한다.
10. **Supplementary**: `source/supplementary.tex`의 수치·그림·캡션을 메인 원고와 동시에 갱신한다.

## IEEE Access 고유 주의사항

- `\history{...}`와 `\doi{...}`는 일반적으로 출판 과정에서 채워지는 영역이므로 임의의 실제값을 만들지 않는다.
- `\vol`, `\year`, running head(`\markboth`)는 최종 제출 단계와 편집부 지시에 맞춰 확인한다.
- 두 열 표·그림은 PDF에서 글자 크기와 잘림을 확인하고, 필요하면 `figure*`/`table*` 배치를 조정한다.
- 최종 PDF의 모든 폰트가 embedded인지, 참고문헌·그림·식 참조가 unresolved가 아닌지 검사한다.

## 최종 무색본 만드는 방법

1. `source/main.tex`의 안내 상자를 삭제한다.
2. `\MustEdit{내용}`과 `\VerifyEdit{내용}`을 각각 `내용`만 남기도록 제거한다.
3. `main_clean_snapshot.tex`에 덮어쓰지 말고 별도 버전으로 저장한다.
4. 참고문헌 포함 전체 컴파일 후 PDF 전 페이지를 렌더링해 겹침·잘림·색상 잔존 여부를 확인한다.

