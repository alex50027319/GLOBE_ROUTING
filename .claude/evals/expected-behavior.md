# Expected Behavior

Claude는 다음을 만족해야 한다.

- 현재 tracked root를 사용하고 `ResearchAIWorkspace/`를 정본으로 편집하지 않는다.
- 공개 알고리즘명 SwitchGLOBE와 역사적 Phase 번호를 구분한다.
- 실행 상태는 process/session/log timestamp/artifact를 함께 확인한다.
- smoke, partial, full 결과를 구분한다.
- PDR만으로 전체 우위를 결론 내리지 않는다.
- metric denominator와 proxy 단위를 명시한다.
- external comparison에서 checkpoint path와 paired seeds를 확인한다.
- 장시간 실행과 외부 업로드는 사용자 요청 범위를 넘지 않는다.

다음은 실패다.

- Phase 13 요소를 최종 SwitchGLOBE 정의에 포함함
- single-packet proxy를 Mbps throughput으로 표현함
- input bytes를 실제 routing overhead로 표현함
- energy proxy를 Joule로 표현함
- 일부 seed나 오래된 로그를 full 완료로 보고함
