# SwitchGLOBE Claude Harness

이 디렉터리는 `feature/switchglobe-simulations` 구조에 맞춘 공유 Claude Code 설정이다.

- `settings.json`: 공유 권한과 수정 후 Python syntax hook
- `context/`: 프로젝트에서 자주 혼동되는 연구 사실
- `rules/`: 코드 경로별 작업 계약
- `skills/`: 반복 가능한 실험·감사·문서 워크플로
- `agents/`: 독립 검증 역할
- `evals/`: Claude 답변의 사실성 점검 사례
- `templates/`: 실험 및 주장 기록 양식

개인 승인과 모델 선택은 `.claude/settings.local.json`에 두고 Git에 올리지 않는다.
Claude Code를 저장소 루트에서 시작한 뒤 `/status`, `/context`, `/agents`로 로드를
확인한다.
