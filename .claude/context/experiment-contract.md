# Experiment Contract

- full training seeds: `42, 77, 123, 314, 2718`
- full evaluation episodes: scenario-method-seed cell당 `200`
- smoke는 축소 검증이며 논문 수치로 사용하지 않는다.
- primary statistical unit은 training seed다.
- 모든 method는 동일 evaluation seed와 reset options를 사용한다.
- SwitchGLOBE ablation과 external comparison은 별도 manifest·결과 디렉터리를 사용한다.
- calibration data와 evaluation data를 겹치지 않는다.
- 완료 판단에는 `complete`, config, seed 집합, raw row 수를 함께 사용한다.

정확한 값은 `config/switchglobe.yaml`, `config/external_comparison.yaml`, campaign 코드,
manifest 순서로 재확인한다.
