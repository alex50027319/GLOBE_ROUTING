from pathlib import Path
import re
root=Path(__file__).resolve().parent
old=root.parent
for venue in ('ieee_access','mdpi_drones'):
    src=root/venue/'source'
    main=(src/'main.tex').read_text()
    main=main.replace('Corresponding author: Howon Lee (\\AuthorCheck{institutional e-mail to be provided}).','Corresponding author: Howon Lee; \\AuthorCheck{e-mail pending}.')
    # Explicitly separate publication declarations from optional deletions.
    if venue=='ieee_access':
        main=main.replace('\\section*{Data Availability}', '\\section*{Funding}\n\\AuthorCheck{Confirm funding agency and grant number, or the absence of external funding.}\n\n\\section*{Data Availability}')
        main=main.replace('is with the Department of Military Digital Convergence, Ajou University, Suwon, Republic of Korea. Research interests include UAV and FANET routing, reinforcement learning, graph neural networks, and deployment-oriented optimization of distributed networking systems.', '\\AuthorCheck{Author-verified affiliation, biography, and research interests must be supplied.}')
        main=main.replace('is with the Department of Military Digital Convergence, Ajou University, Suwon, Republic of Korea. Research interests include wireless and mobile networking, UAV communications, distributed systems, and artificial-intelligence-enabled network control.', '\\AuthorCheck{Author-verified affiliation, biography, and research interests must be supplied.}')
    (src/'main.tex').write_text(main)
    supp=(old/venue/'source/supplementary.tex').read_text()
    for a,b in [('Fused Exact SwitchGLOBE','DiSwitch'),('FastSwitchGLOBE','Fast-DiSwitch'),('SwitchGLOBE Exact','DiSwitch'),('Fused Exact','DiSwitch'),('Exact--Fast','DiSwitch--Fast-DiSwitch'),('predeclared','study-defined'),('Anonymous Author(s)','Kyung Eun Kim and Howon Lee'),('Affiliation withheld for double-blind draft preparation','Department of Military Digital Convergence, Ajou University, Suwon, Republic of Korea')]:
        supp=supp.replace(a,b)
    (src/'supplementary.tex').write_text(supp)
    cover=r'''\documentclass{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{xcolor}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.8em}
\begin{document}
Editors, VENUE

Dear Editors,

Please consider our manuscript, ``DiSwitch: Global-to-Local Policy Distillation with Risk-Switched Predictive Recovery for Decentralized UAV Routing.'' It studies next-hop selection using a local distilled policy and predictive risk switching.

The common-simulator comparison includes eight implementations and five training seeds across 14 scenarios. DiSwitch achieved scenario-macro connected-pair PDR of 0.9053 and deadline delivery of 0.8376. A separately scoped software benchmark reports batch-one p95 latency of 5.836 ms on A100 CUDA and 2.208 ms on its host CPU. The manuscript identifies limitations of the simulator, adapted baselines, and hardware scope, and discloses AI assistance.

\textcolor{blue}{Before use, the authors must confirm originality, exclusive submission, author approval, funding, conflicts, and the archival release details. This cover letter remains a draft.}

Sincerely,\\Howon Lee\\Department of Military Digital Convergence, Ajou University
\end{document}
'''.replace('VENUE','IEEE Access' if venue=='ieee_access' else 'Drones')
    (src/'cover_letter.tex').write_text(cover)
    (src/'SUPPLEMENT_STATUS.md').write_text('보충자료는 이전 실험 스냅샷을 기반으로 저널별 클래스와 DiSwitch 명칭을 맞춘 동반 소스입니다. 실험 재실행은 하지 않았습니다. 이번 최종 PDF 검증 대상은 main.tex입니다. 보충자료 본문은 main.tex의 정밀한 통계·구현 설명을 기준으로 투고 전 동기화해야 합니다. Cover letter는 저널별 초안이며 저자의 제출 선언을 임의 확정하지 않았습니다.\n')
    # Make all main figures inspectable from the report with stable labels and line anchors.
    rows=[]
    for m in re.finditer(r'\\includegraphics[^\n]+',main):
        rows.append(f'- main.tex:{main[:m.start()].count(chr(10))+1} — `{m.group(0)}`')
    (src/'REVISION_FIGURE_INDEX.md').write_text('# 그림 편집 위치\n\n'+'\n'.join(rows)+'\n')
