# Safety

- `.env`, token, credential, cookie, private key를 읽거나 출력하지 않는다.
- `ResearchAIWorkspace/`를 현재 브랜치의 소스처럼 수정하지 않는다.
- 기존 checkpoint, full-run artifact, manuscript build를 수동 편집하거나 덮어쓰지 않는다.
- 장시간 CPU/GPU run 전에 smoke, dependency, output path를 확인한다.
- Colab/GPU full run과 외부 업로드는 사용자 요청 범위에서만 수행한다.
- 파괴적 Git 명령과 광범위 삭제는 실행하지 않는다.
