# GLOBE++ — IEEE Access 투고 초안

## 이 zip의 내용
- `main.tex` : IEEEtran(journal, twocolumn) 기반 + IEEE Access 전용 매크로
  (`\history`, `\doi`, `\tfootnote`, `\authorrefmark` 등) 호환 shim 포함
- `sections/` : 영문 본문 10개 절 (ETRI Journal 초안과 동일 소스, 알고리즘 블록도 동일하게 보강)
- `figures/`, `tables/overall_results.tex` : Phase 12 결과 그림 3종(.pdf/.svg/.png) + 방법론 개요도 + 성능 표
- `main.pdf` : 이 세션에서 컴파일 확인한 미리보기 PDF (10쪽, 에러/경고 없음)

## 템플릿 관련 중요 안내 (needs-verification)
IEEE Access 공식 클래스는 `\documentclass{ieeeaccess}`이며, 관련 파일은 IEEE 저자센터/
Overleaf 공식 갤러리("IEEE Access LaTeX template")에서만 배포되어 이 세션에서 직접
받아올 수 없었습니다. 대신 어디서나 컴파일되는 IEEEtran을 기반으로 하되, `main.tex` 상단에
공식 템플릿과 **동일한 이름의 매크로**(`\history`, `\doi`, `\tfootnote`,
`\IEEEmembership`(IEEEtran 기본 제공), `\address`, `\authorrefmark`)를 안전하게
재정의해 두었습니다 (`=== IEEE ACCESS 호환 shim 시작/끝 ===` 블록).

**최종 투고 시**: Overleaf에서 공식 "IEEE Access" 템플릿을 새로 열고, 이 shim 블록만
삭제한 뒤 본문(타이틀/저자/섹션/그림/표/참고문헌/biography)을 그대로 옮겨 붙이면 매크로
이름이 동일하므로 추가 수정 없이 이식됩니다.

## 투고 전 반드시 채워야 할 것 (TODO — 임의로 만들지 않음)
1. **저자 biography (필수)** — IEEE Access는 참고문헌 뒤에 전 저자의 약력(사진 포함 권장)을
   요구합니다. 본문 하단에 자리표시자만 넣어두었으니 실제 내용으로 교체하세요.
2. **저자/소속/이메일** — 김경은/아주대 ACE Lab까지만 채웠고 나머지는 TODO입니다.
3. **DOI, history(접수/게재일자)** — 편집부가 부여하는 값이라 비워두었습니다.
4. 페이지 수는 10쪽으로, IEEE Access 권장 상한(20쪽)에 여유 있게 들어옵니다.

## Overleaf에서 열기
1. Overleaf → New Project → Upload Project → 이 zip 업로드
2. 컴파일러 **pdfLaTeX**
