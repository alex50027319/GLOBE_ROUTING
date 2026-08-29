# Project Map

## 저장소 루트

- `ResearchAIWorkspace/`: 연구 지식베이스와 Lite-GLOBE 구현·실험
- `paper/`: 영문 논문 LaTeX 및 결과 그림
- `kci_paper/`, `paper_kci/`: 국내 논문 변형
- `professor_presentation/`: 교수님 보고용 Markdown, PPTX, 그림
- `docs/`: 현재 브랜치에서 공유하는 방법론 문서

## Lite-GLOBE 구현

- `implementations/lite_globe/env/`: FANET 환경과 관측·행동 정의
- `implementations/lite_globe/models/`: teacher/student 신경망
- `implementations/lite_globe/algorithms/`: 학습과 정책 로직
- `implementations/lite_globe/experiments/`: phase별 campaign
- `implementations/lite_globe/evaluation/`: 지표 집계와 보고서 생성
- `implementations/lite_globe/config/`: phase별 YAML 설정
- `tests/lite_globe/`: 체크포인트가 없어도 실행 가능한 코드 검증
- `scripts/`: Colab 패키징·원격 실행·seed 병합 도구
- `artifacts/lite_globe/`: 실행 산출물. 소스 코드로 취급하지 않는다.

경로가 현재 브랜치에 없다고 해서 다른 브랜치의 파일을 임의로 복원하지 않는다.
