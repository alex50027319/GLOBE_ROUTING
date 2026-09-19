# Codex Prompt: SwitchGLOBE 추론 시간 최적화 및 검증

현재 저장소는 `/Users/alex/Documents/GLOBE_ROUTING`이고, 작업 브랜치는 `feature/switchglobe-simulations`이다.

## 목표

현재 SwitchGLOBE의 라우팅 알고리즘 의미와 학습 결과를 유지하면서 추론 시간(`decision latency`)을 줄여라. PDR, deadline delivery ratio, p95 success delay, energy proxy가 통계적으로 악화되지 않는지 검증해야 한다.

먼저 코드를 분석하고, 근거 없이 구조를 바꾸지 마라. 분석 계획을 먼저 보고한 뒤 구현을 진행하라.

## 주요 확인 대상

- `implementations/lite_globe/models/policy_adapter.py`
- `implementations/lite_globe/models/student_policy.py`
- `implementations/lite_globe/models/tensor_observation.py`
- `implementations/lite_globe/evaluation/evaluator.py`
- `implementations/lite_globe/experiments/external_comparison_campaign.py`
- `implementations/lite_globe/run_external_comparison.py`
- `implementations/lite_globe/config/external_comparison.yaml`
- `artifacts/external_comparison_colab_results/seeds_42.zip`
- `artifacts/external_comparison_colab_results/seeds_77.zip`
- `artifacts/external_comparison_colab_results/seeds_123.zip`
- `artifacts/external_comparison_colab_results/seeds_314.zip`
- `artifacts/external_comparison_colab_results/seeds_2718.zip`

## 현재 관찰된 문제

- SwitchGLOBE 평균 `decision_latency_p95_ms`: 약 10.22 ms
- OLSR: 약 0.035 ms
- SwitchGLOBE `connected_pair_pdr`: 약 0.905
- SwitchGLOBE `deadline_delivery_ratio`: 약 0.838
- SwitchGLOBE `p95_success_delay`: 약 4.26 steps
- SwitchGLOBE `energy_per_delivered_packet`: 약 2.23 proxy
- 현재 환경은 `policy.act()`의 wall-clock latency를 episode step 또는 packet delay에 반영하지 않는다.
- 따라서 raw inference latency와 latency-aware end-to-end delay를 분리해서 평가해야 한다.

## 1. 현재 latency의 원인 분석

코드에서 다음을 추적하라.

- normal policy와 predictive policy가 decision마다 몇 번 실행되는가
- candidate risk feature 계산이 중복되는가
- `observation_to_tensors()`에서 불필요한 복사, cast, CPU-GPU 전송이 있는가
- tensor 생성, device 이동, dtype 변환, `.item()` 동기화 비용
- `diagnostics()`가 매 action마다 추가 forward를 수행하는가
- `_risk_switch_observation_bytes()`가 중복 추론을 유발하는가
- `force_forward_if_available`가 불필요한 계산을 추가하는가
- GPU에서 CUDA synchronization, warm-up, allocator, context initialization이 p95에 미치는 영향

다음 결과를 먼저 작성하라.

1. latency call graph
2. 단계별 비용
3. 중복 계산 목록
4. 검증된 병목 원인
5. 수정 우선순위와 위험도

## 2. latency 측정 개선

기존 컬럼을 함부로 변경하지 말고 다음 측정값을 추가하라.

- cold-start latency
- warm-start CPU wall-clock latency
- CUDA synchronized latency
- action decision latency
- observation preprocessing 포함 end-to-end policy latency
- `env.step()` 시간
- p50, p95, p99

GPU에서는 필요한 경우 `torch.cuda.synchronize()`를 사용하고, synchronization 비용이 지표에 포함되는지 명확히 문서화하라.

추가 컬럼:

- `decision_latency_raw_ms`
- `decision_latency_preprocess_ms`
- `decision_latency_model_ms`
- `decision_latency_cuda_sync_ms`
- `cold_start`
- `warmup_count`
- `device`

## 3. 알고리즘 의미를 유지하는 최적화 후보

다음 후보를 독립적인 ablation으로 검토하라.

### A. 중복 forward 제거

- normal/predictive branch 출력 재사용
- diagnostics에서 기존 logits/features 재사용
- switch 판단과 최종 action 선택의 중복 계산 제거

### B. 조건부 predictive branch

- cheap risk gate가 활성화될 때만 predictive branch 실행
- switch semantics가 바뀌면 별도 variant로 명시
- Original SwitchGLOBE와 성능 비교

### C. Feature caching

- 동일 environment step의 observation tensor/risk feature 재사용
- `protocol_tick` 이후 변경된 feature만 갱신

### D. Inference-only 최적화

- `torch.inference_mode()`와 `model.eval()` 확인
- dtype, TorchScript, `torch.compile`, CUDA graph 가능성 검토
- Colab A100 호환성과 재현성을 우선

### E. Tensor/observation 최적화

- numpy-to-tensor 변환 최소화
- 불필요한 clone, cast, contiguous 제거
- preallocation 및 CPU-GPU 왕복 제거 검토

### F. 모델 경량화

- hidden dimension 축소
- branch 일부 공유
- predictive branch 저비용 근사
- knowledge/feature distillation

기존 checkpoint를 덮어쓰지 말고 새 variant checkpoint로 저장하라.

### G. Batch 처리

- 후보 노드 계산을 batch로 처리할 수 있는지 검토
- 환경 semantics를 바꾸지 않는 경우에만 적용

## 4. 비교할 variant

최소한 다음을 비교하라.

- Original SwitchGLOBE
- No duplicate forward
- Cached inference
- Conditional predictive branch
- Lightweight 또는 distilled model

동일한 5개 seed와 14개 evaluation scenario, scenario당 200 episodes, 동일 device와 checkpoint 조건을 사용하라.

측정 지표:

- `connected_pair_pdr`
- `deadline_delivery_ratio`
- `p95_success_delay`
- `energy_per_delivered_packet`
- `decision_latency_p50_ms`
- `decision_latency_p95_ms`
- `decision_latency_p99_ms`
- `policy_input_bytes`
- parameter count
- peak memory
- decisions/sec

## 5. Latency-aware end-to-end 평가

기존 결과를 변경하지 말고 별도 평가 모드를 추가하라.

```text
network_delay_steps
decision_delay_ms
effective_end_to_end_delay_ms
deadline_met_latency_aware
```

설정 파일에 다음 가정을 명시하라.

- simulation step duration
- per-hop decision delay
- CPU/GPU deployment mode
- optional queueing delay

다음 네 가지를 분리해 보고하라.

1. 기존 simulator delay
2. 실제 policy inference latency
3. latency-aware estimated end-to-end delay
4. latency-aware deadline delivery ratio

## 6. 통계 검증

각 variant에 대해 다음을 보존하라.

- seed별 raw 결과
- scenario별 paired comparison
- Original 대비 평균 변화율
- 95% confidence interval
- paired seed-level CI 또는 bootstrap CI
- latency 감소율
- PDR/deadline 성능 손실 여부
- Pareto trade-off

최종 variant는 다음 기준으로 선택하라.

- decision latency p95 최소화
- PDR와 deadline ratio의 유의미한 악화 없음
- p95 success delay 악화 없음
- energy proxy 악화 없음
- 구현 복잡도와 재현성
- 논문의 알고리즘 기여 보존

## 7. 산출물

다음 디렉터리에 저장하라.

```text
artifacts/switchglobe_latency_optimization/
```

생성할 파일:

- latency profiling report
- optimization ablation report
- variant별 CSV
- seed별 raw CSV
- paired statistical comparison CSV
- latency-aware evaluation CSV
- parameter/memory/throughput CSV
- latency comparison figure
- performance-vs-latency Pareto figure
- scenario별 latency heatmap
- 최종 추천 variant 표
- 재현 가능한 실행 스크립트
- 변경된 코드의 테스트

## 제약사항

- manuscript 파일은 수정하지 마라.
- 기존 external comparison ZIP과 checkpoint를 덮어쓰지 마라.
- Original SwitchGLOBE 결과를 보존하라.
- 특정 scenario만 골라 성능이 좋아 보이게 하지 마라.
- latency 감소와 PDR 손실의 trade-off를 숨기지 마라.
- 동일한 warm-up, device, batch, synchronization 조건에서만 latency를 비교하라.
- 실제 end-to-end delay가 악화되는 최적화는 채택하지 마라.
- 최적화 채택/제외 사유를 코드와 수치 근거로 제시하라.

## 작업 순서

1. 코드 및 5개 ZIP 결과 분석
2. latency profiling 구현
3. 병목 원인 검증
4. 독립적인 최적화 variant 구현
5. 단위 테스트 및 smoke test
6. Colab A100에서 5개 seed 전체 실행
7. 결과 ZIP 병합 및 통계 검증
8. 최종 추천 variant와 trade-off 보고서 작성

먼저 코드를 수정하지 말고, 현재 구조에 대한 병목 분석과 최적화 계획만 먼저 보고하라. 계획 검토 후 구현을 진행하라.
