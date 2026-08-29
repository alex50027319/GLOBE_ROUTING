# Current Phase

## Phase 13

Phase 13은 Risk-Switch Lite-GLOBE-P+의 calibration, full evaluation,
ablation 및 seed 단위 복구 가능한 실행을 다룬다.

정적 설정 기준:

- training seeds: `42, 77, 123, 314, 2718`
- evaluation episodes: scenario별·method별 seed당 `200`
- 기본 device: `auto`
- Colab queue 기본 GPU: `A100`
- seed 결과는 개별 ZIP으로 저장한 뒤 병합할 수 있다.

## 상태 판단 규칙

- 이 문서는 full run 완료를 주장하지 않는다.
- 현재 실행 여부는 로컬 프로세스, 원격 세션, 로그의 최신 수정 시각을 함께 확인한다.
- 오래된 마지막 로그 한 줄만으로 세션이 살아 있다고 판단하지 않는다.
- 완료 여부는 seed별 result ZIP과 병합 manifest의 seed 집합으로 검증한다.
