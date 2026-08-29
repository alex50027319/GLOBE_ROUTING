# GLOBE_ROUTING Claude Harness

이 디렉터리는 Lite-GLOBE/GLOBE++ 연구를 위한 Claude Code 하네스다.

## 구성 원칙

- `CLAUDE.md`: 모든 작업에 적용되는 짧은 프로젝트 지침
- `context/`: 코드에서 확인한 연구 개념과 실험 계약
- `rules/`: 파일 경로별로 적용되는 작업 규칙
- `skills/`: 반복 가능한 연구·실험 워크플로
- `agents/`: 독립적으로 조사하거나 검증할 전문 역할
- `hooks/`: Claude의 파일 수정 직후 실행되는 결정론적 검사
- `evals/`: Claude가 프로젝트 사실을 정확히 설명하는지 점검하는 사례
- `templates/`: 실험 계획, 결과 감사, 논문 주장 작성 형식

## 정본 우선순위

1. 실행 코드와 YAML 설정
2. 원시 실험 CSV와 manifest
3. 자동 생성된 요약·표·그림
4. README와 연구 노트
5. 논문 및 발표자료

문서와 코드가 충돌하면 추측하지 말고 충돌을 보고한다.

## 시작 점검

Claude Code를 저장소 루트에서 실행하고 `/status`, `/context`, `/agents`로
설정과 지침의 로드 여부를 확인한다. 새 skill 또는 agent 디렉터리를 세션 중
처음 만들었다면 Claude Code를 재시작한다.
