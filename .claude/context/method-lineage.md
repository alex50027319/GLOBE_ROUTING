# Reproducible Method Lineage

```text
PPO Global Teacher + foundation KD (historical Phase 7)
        |
        +--> Geo-Residual Student (Phase 8, normal branch)
        |
        +--> Predictive Student (Phase 11, Phase 8 initialization)
                         |
Geo-Residual ------------+--> SwitchGLOBE calibration (Phase 12)
```

- Phase 번호는 checkpoint·재현 이력을 위해 코드에 남아 있다.
- 공개 표, checkpoint, 논문에서는 `SwitchGLOBE`를 최종 이름으로 사용한다.
- Phase 13 결과는 SwitchGLOBE 결과 또는 ablation에 병합하지 않는다.
- local PPO fine-tuning 공용 구현이 남아 있어도 최종 SwitchGLOBE 핵심 단계로 주장하지 않는다.
