# GLOBE++ — ETRI Journal 투고 초안

## 이 zip의 내용
- `main.tex` : IEEEtran(journal, onecolumn) + 더블스페이싱으로 구성한 메인 파일
- `sections/` : 영문 본문 10개 절 (`paper/archive/v1_english_phase12/` 초안 기반, 이미
  Phase 13 P+ 서술까지 반영된 최신 버전임을 확인 후 그대로 사용)
- `figures/`, `tables/overall_results.tex` : Phase 12 결과 그림 3종(.pdf/.svg/.png) +
  방법론 개요도 + 종합 성능 표
- `main.pdf` : 이 세션에서 컴파일 확인한 미리보기 PDF (15쪽, 에러/경고 없음)

## 원본 대비 제가 추가/수정한 것
- `sections/04_method.tex`에 원래 `Algorithm~\ref{alg:risk_switch}`로 참조만 되고 실제
  정의는 빠져 있던 알고리즘 블록을 새로 작성해 채워 넣었습니다. **본문에 이미 정의된 수식
  ($D_i$, $Q_i$, $G(a_P,a_N)$, $S$, 에너지 tie-break, drop suppression)을 그대로 의사코드로
  옮긴 것**이며, 새로운 수치나 로직을 만들어 넣지 않았습니다.
- 저자/소속 블록: 알고 있는 사실(김경은, 아주대학교 ACE Lab)만 채우고 나머지는 TODO로
  남겼습니다.

## 템플릿 관련 중요 안내 (needs-verification)
ETRI Journal(Wiley)의 전용 LaTeX 클래스(`wiley-article.cls` 계열)는 Overleaf/Wiley
플랫폼에서만 배포되고 공개 다운로드가 되지 않아, 이 세션에서는 받아올 수 없었습니다.
대신 어디서나 컴파일되는 IEEEtran + 더블스페이싱으로 구성했습니다. Wiley 저자 가이드
(authors.wiley.com)를 확인한 결과, 심사 단계에서는 "표준 LaTeX + 더블스페이싱"이 대체로
허용됩니다. **최종 게재가 확정되면** Overleaf에서 공식 "Wiley Journal Template" 갤러리를 열고
이 안의 섹션 내용을 그대로 옮겨 붙이시면 됩니다. 투고 전 ETRI Journal 공식 Author Guidelines
(onlinelibrary.wiley.com/page/journal/22337326/homepage/forauthors.html)에서 최신 페이지
제한·참고문헌 스타일을 반드시 재확인하세요 (이 세션에서는 해당 페이지가 403으로 막혀
직접 확인하지 못했습니다).

## Overleaf에서 열기
1. Overleaf → New Project → Upload Project → 이 zip 업로드
2. 컴파일러 **pdfLaTeX**
