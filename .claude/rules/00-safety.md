# Safety Rules

- `.env`, token, credential, cookie, keychain 내용을 읽거나 출력하지 않는다.
- `raw/` 원본 자료를 수정·이동·삭제하지 않는다.
- 기존 checkpoint, full-run artifact, 논문 PDF를 사용자 승인 없이 덮어쓰지 않는다.
- `rm -rf`, `git reset --hard`, 강제 checkout 등 복구하기 어려운 명령을 실행하지 않는다.
- Colab/GPU 실행, 외부 업로드, Notion 동기화는 사용자 요청 범위 안에서만 수행한다.
- 장시간 작업 전에 smoke test와 출력 경로를 확인한다.
