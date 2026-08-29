# SwitchGLOBE 구현 지도

이 패키지의 최종 공개 진입점은 `run_switchglobe.py`다. 내부의 Phase 번호는 개발
계보와 기존 checkpoint 호환성을 위해 필요한 경우에만 유지한다.

## 학습 계보

| 단계 | 실행 모듈 | 필요한 입력 | 주요 출력 |
| --- | --- | --- | --- |
| Teacher foundation | `run_phase7` | 없음 | `global_teacher.pt`, `kd_only_student.pt` |
| Geo-Residual | `run_phase8` | Teacher foundation checkpoint | `geo_residual_kd.pt` |
| Predictive Student | `run_phase11` | Teacher + Geo-Residual checkpoint | `lite_globe_p.pt` |
| SwitchGLOBE | `run_switchglobe` | Geo-Residual + Predictive checkpoint | `switchglobe.pt`, raw/statistics/tables/figures |

전체 연결 실행은 저장소 루트에서 다음 명령을 사용한다.

```bash
python scripts/train_switchglobe_pipeline.py --device auto --resume
```

## 최종 실행 정책

- normal branch: `GeographicResidualStudentPolicy`
- predictive branch: `LiteGlobePStudentPolicy`의 predictive prior-only 모드
- final policy: `SwitchGlobePolicy`
- switch features: link margin, predicted current-link lifetime, predicted onward lifetime
- calibration constraint: normal 및 structural-hole PDR이 기준보다 0.005 이상 하락하는 후보는 제외

Teacher의 global graph는 offline PPO/KD에만 사용한다. 최종 정책은 self, neighbor,
edge, packet, forwardability, candidate risk feature로 구성된 1-hop 관측만 사용한다.

`run_phase12.py`, `Phase12Config`, `RiskSwitchLiteGlobePStudentPolicy`는 기존 결과와
checkpoint를 감사할 수 있도록 역사적 호환 이름으로 남겨 둔다. 새 코드와 문서에서는
각각 `run_switchglobe.py`, `SwitchGlobeConfig`, `SwitchGlobePolicy`를 사용한다.

## 외부 baseline suite

`run_baselines`는 최종 `switchglobe.pt`와 GPSR, Predictive Geographic, Evo-QGeo,
IQMR Q(lambda), DRAMA를 같은 seed와 scenario에서 평가한다. baseline 학습 checkpoint와
결과는 `artifacts/baselines/`에 분리해 저장하며 SwitchGLOBE checkpoint를 수정하지 않는다.
