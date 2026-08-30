---
paths:
  - "implementations/lite_globe/colab/**"
  - "scripts/package_*_colab.py"
  - "README*COLAB.md"
---

# Colab and Packaging

- bundle 생성과 원격 실행을 구분한다. 로컬 bundle 생성은 GPU session을 시작하지 않는다.
- package script의 required path와 checkpoint 존재를 실행 전에 확인한다.
- bundle에 `.env`, local caches, unrelated artifacts를 포함하지 않는다.
- notebook 명령은 현재 root layout과 README 명령에 맞춘다.
- full GPU 실행은 사용자의 명시적 실행 요청 없이 시작하지 않는다.
- 생성 ZIP은 `unzip -t`와 파일 목록으로 검증한다.
