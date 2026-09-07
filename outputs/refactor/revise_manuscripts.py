"""Build isolated editorial revisions from the existing refactor snapshots."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'review_revision'
SOURCE = ROOT / 'ieee_access/source'

def unwrap(text, name):
    needle = '\\' + name + '{'
    while needle in text:
        start = text.index(needle)
        pos = start + len(needle)
        end, depth = pos, 1
        while depth:
            if text[end] == '{' and text[end-1] != '\\': depth += 1
            if text[end] == '}' and text[end-1] != '\\': depth -= 1
            end += 1
        text = text[:start] + text[pos:end-1] + text[end:]
    return text

base = (SOURCE / 'main.tex').read_text()
base = re.sub(r'^\\newcommand\{\\(?:MustEdit|VerifyEdit)\}.*\n', '', base, flags=re.M)
for macro in ('MustEdit', 'VerifyEdit'): base = unwrap(base, macro)
base = re.sub(r'^.*Editing-guide copy.*\n', '', base, flags=re.M)
base = base.replace('\\color{ReviewBlue}\n', '')
base = re.sub(r'^% (?:EDITING-GUIDE|RED =).*\n', '', base, flags=re.M)

review_macros = r'''
% Red marks editorial deletion/move candidates; blue marks author-input gaps.
\newif\ifreviewmarks
\reviewmarkstrue
\newcommand{\DeleteCandidate}[2]{\ifreviewmarks\par\noindent{\color{ReviewRed}\textbf{Deletion/move candidate #1.} #2}\par\fi}
\newcommand{\AuthorCheck}[1]{\textcolor{ReviewBlue}{#1}}
\newcommand{\ReviewLegend}{\par\noindent{\small\textbf{Editorial review copy.} \textcolor{ReviewRed}{Red text and red-framed figures are candidates for deletion or relocation to the supplement.} \textcolor{ReviewBlue}{Blue text identifies information requiring author verification.} Black text is revised manuscript content.}\par}
'''
base = base.replace('\\begin{document}', review_macros + '\n\\begin{document}', 1)
base = base.replace('\\maketitle', '\\maketitle\n\\ReviewLegend', 1)
abstract = r'''Flying ad hoc networks require reliable next-hop selection under changing connectivity and limited local information. We present DiSwitch (Distilled Policy Switch), which transfers a graph-based reinforcement-learning teacher's preferences to a local student and combines a geographic-residual policy with a predictive prior through a risk switch. The teacher is used offline; online decisions consume relay and one-hop features under the stated observation contract. In a common simulator, eight implementations were evaluated using five training seeds, 14 scenarios, and 200 episodes per seed--scenario cell. DiSwitch achieved a scenario-macro connected-pair packet delivery ratio of 0.9053 and a deadline-delivery ratio of 0.8376. Relative to adapted Evo-QGeo, the paired differences were 1.84 and 2.23 percentage points, respectively. Successful-packet delay is reported in simulator steps and separately from software decision latency. In one Google Colab A100 session, mean seed-specific batch-one p95 decision latency was 5.836 ms on CUDA and 2.208 ms on the host CPU. A compact distilled variant reduced CUDA latency to 2.005 ms but did not satisfy the study's 0.5-percentage-point reliability tolerance. The findings support the evaluated local policy composition and expose a reliability--latency trade-off. They do not establish performance on airborne hardware or superiority over independently reproduced wireless protocol implementations.'''
base = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: '\\begin{abstract}\n' + abstract + '\n\\end{abstract}', base, count=1, flags=re.S)
base = base.replace('predeclared', 'study-defined').replace('predefined', 'study-defined')
base = base.replace('confirmatory runtime environment', 'runtime measurement environment')
base = base.replace('rather than a ritual $p<0.05$ threshold', 'rather than a binary significance threshold')
base = base.replace('62.16\\% faster on the A100-session host CPU than on CUDA at p95', '62.16\\% lower in p95 latency on the A100-session host CPU than on CUDA')
base = base.replace('The fixed launch and synchronization costs dominate the compute saved by parallel arithmetic.', 'This pattern is consistent with launch and synchronization overhead offsetting the benefit of parallel arithmetic; the measurements do not isolate a causal contribution for each operator.')
base = base.replace('suggesting a policy-family limitation rather than a threshold-only defect.', 'showing a persistent failure in the tested interventions. These interventions do not establish whether the limiting cause is the policy family, training coverage, or simulator dynamics.')
base = base.replace('the result supports the architectural interpretation:', 'the result is consistent with the architectural interpretation:')
base = base.replace('The result supports the architectural interpretation:', 'The result is consistent with the architectural interpretation:')
base = base.replace('\\item We introduce a semantics-preserving computation-reuse path that eliminates repeated diagnostic forwards and preserves the branch logits, mask, switch rule, and selected action.', '\\item We specify a single-evaluation implementation of the two-branch decision rule and measure its batch-one runtime. Removing repeated diagnostic calls is an implementation optimization, not a separate routing algorithm contribution.')
start = base.index('The systems contribution is the semantics-preserving execution path')
end = base.index('\n\n', start)
base = base[:start] + r'''The deployed implementation evaluates each branch once and reuses its outputs for action selection and diagnostics. This implementation detail supports reproducible timing. The routing contribution is the combination of local distilled preferences, predictive risk features, and calibrated switching; a comparison with an earlier instrumented wrapper is reported only as an internal implementation check.''' + base[end:]
start = base.index('\\begin{align}', base.index('\\subsection{Objective and Constraints}'))
end = base.index('Approximate speed variants', start)
base = base[:start] + r'''The design prioritizes connected-pair and deadline delivery and reports execution time and conditional successful delay as separate outcomes. We do not claim to solve a calibrated scalar optimization problem that combines these quantities in different units. ''' + base[end:]
base = base.replace(r'\mathrm{LCB}_{0.95}', r'\mathrm{LB}_{95\%,\mathrm{two\text{-}sided}}')
base = base.replace('and delay and energy show no material directional degradation.', 'where the bound is the lower endpoint of the two-sided 95\\% $t$ interval used in the archived analysis. The tolerance is 0.005 in ratio units (0.5 percentage points). Delay and energy provide supporting diagnostics; no numeric acceptance margins were specified for them. We do not claim prospective preregistration of this rule. Failure to establish non-inferiority is not, by itself, a proof that degradation exceeds the tolerance.')
base = base.replace("below the study-defined $-0.005$ non-inferiority floor.", "so its lower endpoint fails the study-defined $-0.005$ non-inferiority criterion.")
base = base.replace('Auxiliary masked cross-entropy terms may supervise teacher action, shortest-path action, and risk-aware local action. Dataset partitions are made by scenario and episode seed, rather than by individual hops, so that transitions from one trajectory do not leak across partitions.', r'''The auxiliary supervision sources considered in the project include teacher actions, shortest-path actions, and local risk-aware actions. \AuthorCheck{The release must identify the active loss weights, distillation temperature, reward terms, training budgets, and checkpoint-selection rule for every reported checkpoint.} A split by scenario and episode is required to prevent trajectory leakage; independence of tuning and evaluation must be documented with split manifests before a prospective validation claim is made.''')
base = base.replace('The principal distillation term is', r'''Here $M(a)\in\{0,1\}$, $\log 0=-\infty$, and normalization is over valid actions, with $\tau>0$. The KL divergence is evaluated on that common support; DROP must remain a defined fallback when no neighbor is valid. The principal distillation term is''')
base = base.replace('The deployment contract excludes the full adjacency matrix, remote hidden states, and teacher queries.', 'The deployment contract excludes the full adjacency matrix, remote hidden states, and teacher queries. However, simulator-local features do not by themselves establish that those features are locally measurable on a UAV. Destination state, neighbor queue information, and onward-connectivity summaries require an acquisition and freshness model; this study does not measure the radio cost of obtaining them.')
base = base.replace('Let $a_N=\\arg\\max_a z^N_a$ and $a_P=\\arg\\max_a z^P_a$ after masking. Define the safety gain', r'''Let $K$ be the padded candidate capacity, with candidate slots $0,\ldots,K-1$ and DROP index $K$. Let $a_N=\arg\max_a z^N_a$ and $a_P=\arg\max_a z^P_a$ after masking. For consistency with the evaluated code, define $\iota(a)=\min(a,K-1)$ and interpret the risk features indexed by $a$ below as those at slot $\iota(a)$. This is an indexing convention for DROP, not a physical link-risk estimate. Define the safety gain''')
base = base.replace('The final logits are $z=(1-S)z^N+Sz^P$', 'When no valid neighbor exists, $S=0$; if risk features are absent, the evaluated switch also defaults to the normal branch. The indexing convention can affect a predictive DROP choice and must be covered by regression tests before any change. The switch is a heuristic selection rule, not a guarantee of link safety. The final logits are $z=(1-S)z^N+Sz^P$')
base = base.replace('from local risk features\n\\State $S', 'using the slot map $\\iota$\n\\State $S')
base = base.replace('Network end-to-end delay in steps and local decision latency in milliseconds are deliberately reported as different metrics.', 'The step count is a simulator route-duration statistic. No conversion to physical end-to-end delay in milliseconds is supported without a calibrated link, MAC, queueing, and time-step model. The reported aggregate p95 is a mean of cell-level quantiles, not a quantile of the pooled packet population.')
base = base.replace('partly because they solve no neural inference problem and because unsuccessful packets are excluded from the conditional delay statistic.', 'with the latter conditional on each method\'s own delivered-packet subset. These simulator-step differences cannot be attributed to neural inference time, which was benchmarked separately.')
base = base.replace('Energy is a simulator proxy; p95 delay is conditional on successful delivery.', 'Energy is a simulator proxy; p95 delay is conditional on successful delivery. Bold identifies the proposed method, not the winner of each column.')
base = base.replace('The repository also contains a pooled full-analysis table', 'The intervals are unadjusted for multiple comparisons. The strongest comparator was identified within the evaluated set, so these contrasts are descriptive rather than a prospectively specified family-wise superiority test. Zero between-training-seed variance for deterministic policies does not imply zero deployment uncertainty. A negative lower $t$ bound for a nonnegative energy proxy reflects an unconstrained small-sample interval, not negative energy; a future replication should report a support-respecting interval.\n\nThe repository also contains a pooled full-analysis table')
base = base.replace('The project history includes additional negative interventions:', '\\DeleteCandidate{D04}{The project history includes additional negative interventions:')
base = base.replace('motivate worst-scenario reweighting and action-space shielding.', 'motivate worst-scenario reweighting and action-space shielding.}')
base = base.replace('An attempted expanded rerun for additional p90, maximum, coefficient-of-variation, and peak-device-memory fields was interrupted when the backend reclaimed the session; no values are imputed from that failed job.', r'\DeleteCandidate{D05}{An attempted expanded rerun for additional p90, maximum, coefficient-of-variation, and peak-device-memory fields was interrupted when the backend reclaimed the session; no values are imputed from that failed job.}')
base = base.replace('The result supports', 'The result is consistent with')
base = base.replace('Figure~\\ref{fig:ablation} reports the seed-level contribution analysis.', 'Figure~\\ref{fig:ablation} reports paired component contrasts. Comparisons between historical checkpoints can confound architecture, training data, and tuning budget. They do not isolate the causal contribution of distillation without a matched no-distillation control trained under the same budget.')
base = base.replace('We recommend separately recording', 'This requires separately recording')
base = base.replace('Both authors approved the manuscript. Confirm these roles against the final CRediT statement.', r'\AuthorCheck{Both authors must verify the contribution assignments and approve the final version before submission.}')
base = base.replace('Confirm the target journal\'s current AI-disclosure policy and revise or relocate this disclosure before submission.', '')
base = base.replace('insert the final institutional e-mail address before submission', r'\AuthorCheck{institutional e-mail to be provided}')
base = base.replace('Replace this sentence with the final public repository URL and archival DOI, or state the approved access conditions.', r'\AuthorCheck{A public repository URL and archival DOI, or approved access conditions, remain to be supplied.}')
base = base.replace('Confirm this statement for both authors immediately before submission.', r'\AuthorCheck{Both authors must verify this declaration before submission.}')

# Remove the internal speedup narrative from the abstract and concluding headline.
a = base.index('This paper presented', base.index('\\section{Conclusions}'))
b = base.index('\n\n', a)
base = base[:a] + r'''DiSwitch combines global-to-local policy distillation with risk-switched predictive recovery for UAV next-hop selection. In the evaluated common simulator, it achieved 0.9053 connected-pair PDR and 0.8376 deadline delivery and had positive paired differences relative to adapted Evo-QGeo. The batch-one benchmark gave mean seed-specific p95 decision latencies of 5.836 ms on A100 CUDA and 2.208 ms on the session's CPU. Fast-DiSwitch reduced CUDA latency to 2.005 ms but failed the study-defined reliability criterion. These findings are restricted to the archived checkpoints, simulator scenarios, and measured session. Establishing operational utility requires a reproducible release, controlled ablations, and target-device and wireless-network evaluation.''' + base[b:]

delete_figures = {'fusion': ('D01', 'Internal wrapper comparison; move to implementation appendix.'), 'litcoverage': ('D02', 'Convenience-corpus counts; omit or move to a literature-audit appendix.'), 'decision': ('D03', 'Redundant with the reliability--latency plot and acceptance discussion.')}
for label, (key, why) in delete_figures.items():
    pattern = r'\\begin\{figure\*\}\[t\](?:(?!\\end\{figure\*\}).)*?\\label\{fig:' + label + r'\}.*?\\end\{figure\*\}'
    def mark(m):
        block = m.group(0)
        block = re.sub(r'\\includegraphics\[width=[^]]+\]\{([^}]+)\}', lambda g: r'\fcolorbox{ReviewRed}{white}{\includegraphics[width=0.94\textwidth]{' + g.group(1) + '}}', block)
        block = block.replace('\\caption{', '\\caption{\\textcolor{ReviewRed}{Deletion/move candidate '+key+'. '+why+'} ')
        return '\\ifreviewmarks\n' + block + '\n\\fi'
    base, n = re.subn(pattern, mark, base, flags=re.S)
    assert n == 1, (label, n)

base = base.replace('Figure~\\ref{fig:fusion} distinguishes the policy from its runtime wrapper.', 'The implementation uses the same logits for action selection and diagnostic extraction.')
base = base.replace('Figure~\\ref{fig:litcoverage} visualizes the metric imbalance.', 'The literature counts summarize a project convenience sample.')
base = base.replace('This is an implementation equivalence argument, not a claim of equivalence under stochastic training mode or stateful layers.', 'This is an implementation equivalence argument under deterministic evaluation. Its associated wrapper-comparison diagram is marked for relocation to supplementary material.')

for venue in ('ieee_access', 'mdpi_drones'):
    dst = OUT / venue / 'source'
    if dst.exists(): raise RuntimeError(f'Refusing to overwrite {dst}')
    shutil.copytree(SOURCE, dst, ignore=shutil.ignore_patterns('rendered', 'build', '__pycache__', '*.pdf.bak', '.DS_Store', 'main_clean_snapshot.tex'))
    text = base
    if venue == 'mdpi_drones':
        shutil.copytree(ROOT / 'mdpi_drones/source/Definitions', dst / 'Definitions')
        body = base[base.index('\\section{Introduction}'):base.index('\\section*{Author Contributions}')]
        for env in ('figure', 'table', 'algorithm'):
            body = body.replace('{'+env+'*}', '{'+env+'}')
        # Keep readable full-width layout in the MDPI single-column text area.
        body = body.replace('[t]', '[H]')
        pre = r'''\documentclass[drones,article,submit,moreauthors]{Definitions/mdpi}
\firstpage{1}
\pubvolume{1}\issuenum{1}\articlenumber{0}\pubyear{2026}\copyrightyear{2026}
\datereceived{}\daterevised{}\dateaccepted{}\datepublished{}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,multirow}
\usepackage{graphicx,xcolor,enumitem,algorithm,algpseudocode}
\setlength{\headheight}{24pt}
\newcommand{\DS}{DiSwitch}\newcommand{\FastDS}{Fast-DiSwitch}
\newcommand{\drop}{\mathrm{DROP}}\newcommand{\pdr}{\mathrm{PDR}}
\newcommand{\E}{\mathbb{E}}\newcommand{\R}{\mathbb{R}}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\definecolor{ReviewRed}{RGB}{190,25,35}\definecolor{ReviewBlue}{RGB}{0,82,155}
'''+review_macros+r'''
\Title{DiSwitch: Global-to-Local Policy Distillation with Risk-Switched Predictive Recovery for Decentralized UAV Routing}
\Author{Kyung Eun Kim $^{1}$ and Howon Lee $^{1,}$*}
\AuthorNames{Kyung Eun Kim and Howon Lee}
\address{ $^{1}$ Department of Military Digital Convergence, Ajou University, Suwon 16499, Republic of Korea}
\corres{Correspondence: \AuthorCheck{institutional e-mail to be provided}}
'''
        tail = r'''
\authorcontributions{\AuthorCheck{Kyung Eun Kim and Howon Lee must confirm the final CRediT assignments and approval of the manuscript.}}
\funding{\AuthorCheck{Provide the funding agency and grant number, or confirm that no external funding was received.}}
\institutionalreview{Not applicable to the reported simulation study.}
\informedconsent{Not applicable.}
\dataavailability{The publication package includes figure-source data and analysis summaries. \AuthorCheck{Provide the archival DOI and repository URL for code, configurations, checkpoints, and raw results, or specify approved access conditions.}}
\acknowledgments{OpenAI Codex assisted with English drafting, LaTeX restructuring, and figure-layout preparation in the abstract, methods, related work, results, and supplementary material \cite{openai2026codex}. The authors are responsible for verifying the final content.}
\conflictsofinterest{\AuthorCheck{Both authors must verify the final conflict-of-interest declaration.}}
\reftitle{References}
\bibliography{references}
\end{document}
'''
        body = body.replace(r'\PARstart{U}{nmanned}', 'Unmanned')
        text = pre + '\n\\abstract{'+abstract+'}\n\\keyword{UAV routing; policy distillation; reinforcement learning; FANET; decision latency}\n\\begin{document}\n\\ReviewLegend\n' + body + tail
    (dst / 'main.tex').write_text(text)
    # Preserve supplements as clearly identified supporting snapshots, not silently revised evidence.
    (dst / 'SUPPLEMENT_STATUS.md').write_text('보충자료와 cover letter는 이전 IEEE DiSwitch 소스 스냅샷입니다. 이번 심사 대응 편집은 main.tex에 적용되었습니다. MDPI 최종 제출 시 supplementary의 저널 형식 및 cover letter 수신 저널을 별도로 맞춰야 합니다. 원시 실험이나 체크포인트는 이번 작업에서 재실행하지 않았습니다.\n')
    (dst / 'README.md').write_text('Main document: main.tex. Compiler: pdfLaTeX. Red text / red frames mark deletion or supplement-relocation candidates. Blue marks author input. The manuscript is an editorial review copy. Removing red candidates requires also editing adjacent explanations; review the supplied reviewer report. Scientific data were not changed or rerun.\n')
print(OUT)
