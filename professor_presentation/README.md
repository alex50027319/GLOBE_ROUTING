# Professor Presentation Package

이 폴더는 교수님께 현재 GLOBE++ / Lite-GLOBE routing 연구를 보고하기 위한 Markdown 기반 발표 패키지다.

## 파일 구성

| File | Purpose |
| --- | --- |
| `project_audit.md` | 현재 프로젝트 감사 결과 |
| `presentation.md` | 교수님께 설명할 전체 발표 원문 |
| `slides_marp.md` | Marp 기반 슬라이드 초안 |
| `speaker_notes.md` | 슬라이드별 발표 대본 |
| `one_page_summary.md` | 1페이지 연구 요약 |
| `qna_preparation.md` | 예상 질문과 답변 |
| `research_status.md` | 구현/실험/문서화 상태 |
| `simulation_settings.md` | 시뮬레이션 환경 및 metric 정의 |
| `novelty_analysis.md` | novelty, 한계, 비교 연구 포지셔닝 |
| `future_plan.md` | 단기/중기/장기 계획 |
| `evidence_map.md` | 발표 주장과 근거 파일 연결 |
| `missing_items.md` | 부족한 실험/문헌/그림 목록 |
| `figures/` | 발표에 사용할 그림 복사본 |

## 선택한 발표 전략

전략은 **C. 논문 초안 검토형**을 기본으로 하되, **B. 구현 진행상황 보고형** 톤을 섞는다.

이유:

- simulator, teacher, student, distillation, risk-switch, baseline, figure, KCI draft가 이미 존재한다.
- Phase 12 full multi-seed 결과는 발표 근거로 사용할 수 있다.
- Phase 13 P+는 구현과 smoke 검증은 있으나, full multi-seed 결과가 최종 확정된 상태로 확인되지는 않았다.
- 따라서 교수님께는 “논문으로 갈 수 있는 구조와 결과가 있지만, 최종 주장 전 추가 검증이 필요하다”는 균형 잡힌 보고가 적절하다.

## 사용 방법

Marp가 설치되어 있다면 다음처럼 PDF 또는 PPTX로 변환할 수 있다.

```bash
marp professor_presentation/slides_marp.md --pdf
marp professor_presentation/slides_marp.md --pptx
```

발표 대본은 `speaker_notes.md`, 한 장 요약은 `one_page_summary.md`를 사용한다.

