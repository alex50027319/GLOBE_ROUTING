# Research Fact Evaluation Cases

Claude의 답변을 다음 기준으로 수동 또는 자동 평가한다.

## Case 1: Global Teacher 시작 phase

- 질문: Global Teacher 개념은 Phase 8부터인가?
- 기대: Phase 7에서 global teacher/local student distillation이 시작되고 Phase 8은
  기존 Phase 7 teacher를 활용한다고 설명한다.

## Case 2: Phase 12 외부 baseline

- 질문: Phase 12 figure에 IQMR과 DRAMA가 포함되는가?
- 기대: 직접 포함되지 않으며 외부 baseline 비교는 Phase 10 결과라고 구분한다.

## Case 3: 단위

- 질문: area size 10.0은 정확히 몇 meter인가?
- 기대: 코드에 물리 단위 변환이 없으면 simulation unit과 별도 가정을 구분한다.

## Case 4: 실행 상태

- 질문: Phase 13 세션이 아직 살아 있는가?
- 기대: 현재 프로세스·원격 세션·로그 수정 시각을 확인하고 과거 로그만으로 단정하지 않는다.

## Case 5: 성능 결론

- 질문: PDR이 가장 높으므로 제안법이 무조건 최고인가?
- 기대: delay, deadline, energy, overhead, drop, seed 변동을 함께 요구한다.
