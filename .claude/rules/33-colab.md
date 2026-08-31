---
paths:
  - "ResearchAIWorkspace/scripts/colab*.py"
  - "ResearchAIWorkspace/scripts/run_phase13_seed_queue.py"
  - "ResearchAIWorkspace/implementations/lite_globe/colab/**"
---

# Colab Rules

- 원격 GPU 실행은 비용과 세션 상태를 바꾸므로 사용자 요청 없이 시작하지 않는다.
- Phase 13 full run은 seed별 독립 세션·ZIP·로그를 사용한다.
- 세션 이름에 phase와 seed를 포함한다.
- 네트워크 단절 뒤에는 로컬 프로세스, 원격 session, 결과 ZIP을 각각 확인한다.
- stale process를 종료하기 전에 PID와 command line을 정확히 확인한다.
- 완료된 seed ZIP은 `--force`가 없는 한 재실행하지 않는다.
