"""LaTeX version of the submission report: reports/latex/UIDAI_review.tex (+ figures/). Same numbers as the PDF builder."""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

P = ROOT
# Reuse the numbers computed by build_submission.py (everything before its marker).
shared_code = open(f"{P}/src/build_submission.py").read().split("\n# ==== END OF SHARED DATA")[0]
g = {"__file__": __file__}
exec(shared_code, g)
FL, J, real, syn, AF, ROT, bench = g["FL"], g["J"], g["real"], g["syn"], g["AF"], g["ROT"], g["bench"]
N, t768, jip_sec, jip_years, AFsec, pr = g["N"], g["t768"], g["jip_sec"], g["jip_years"], g["AFsec"], g["pr"]
afr_e, jip_same, r1 = g["afr_e"], g["jip_same"], g["r1"]
S = pd.read_csv(f"{P}/experiments/summary.csv")


def e(x):  # escape for LaTeX
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("≈", r"$\approx$"),
        ("×", r"$\times$"),
        ("≥", r"$\geq$"),
        ("→", r"$\rightarrow$"),
        ("°", r"$^\circ$"),
        ("–", "--"),
        ("·", r"$\cdot$"),
        ("’", "'"),
        ("“", "``"),
        ("”", "''"),
        ("⌈", r"$\lceil$"),
        ("⌉", r"$\rceil$"),
    ]:
        x = x.replace(a, b)
    return x


def sci(x, k=1):
    ex = int(f"{x:e}".split("e")[1])
    return f"${x/10**ex:.{k}f} \\times 10^{{{ex}}}$"


P_ = lambda v, f: e(f.format(v))
NAME = {
    "2002_DB1": "FVC2002 DB1 (optical)",
    "2002_DB2": "FVC2002 DB2 (optical)",
    "2002_DB3": "FVC2002 DB3 (capacitive)",
    "2002_DB4": "FVC2002 DB4 (synthetic)",
    "2004_DB1": "FVC2004 DB1 (optical)",
    "2004_DB2": "FVC2004 DB2 (optical)",
    "2004_DB3": "FVC2004 DB3 (thermal)",
    "2004_DB4": "FVC2004 DB4 (synthetic)",
}
NOTE = {
    "2004_DB1": "JIPNet trained on it",
    "2004_DB2": "JIPNet trained on it",
    "2002_DB1": "paper's test set",
    "2002_DB3": "paper's test set",
}
ORDER = list(NAME)


def kv(rows):
    s = "\\begin{tabularx}{\\textwidth}{@{}>{\\bfseries\\raggedright}p{2.9cm}X@{}}\\toprule\n"
    s += "\\\\\\midrule\n".join(f"{k} & {v}" for k, v in rows) + "\\\\\\bottomrule\\end{tabularx}\n"
    return s


def fc(rows):
    s = "\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright}p{3.4cm}>{\\raggedright}X>{\\raggedright\\arraybackslash}X@{}}\\toprule\n\\textbf{Issue} & \\textbf{Impact at Aadhaar scale} & \\textbf{Mitigation in the pipeline}\\\\\\midrule\n"
    s += (
        "\\\\\\midrule\n".join(f"\\textbf{{{e(a)}}} ({e(src)}) & {e(b)} & {e(c)}" for a, src, b, c in rows)
        + "\\\\\\bottomrule\\end{tabularx}\n"
    )
    return s


def fig(f, cap, w="\\textwidth"):
    return f"\\begin{{figure}}[H]\\centering\\includegraphics[width={w},height=0.36\\textheight,keepaspectratio]{{figures/{f}}}\\caption*{{\\small {cap}}}\\end{{figure}}\n"


jrows = ""
for d in ORDER:
    r = S[(S.model == "JIPNet") & (S.db == d)].iloc[0]
    jrows += f"{e(NAME[d])}{(' \\newline {\\footnotesize ' + e(NOTE[d]) + '}') if d in NOTE else ''} & \\textbf{{{P_(r.EER,'{:.1%}')}}} ({P_(r.EER_lo,'{:.1%}')}--{P_(r.EER_hi,'{:.1%}')}) & \\textbf{{{P_(r.TAR_0p1,'{:.0%}')}}} ({P_(r.TAR_lo,'{:.0%}')}--{P_(r.TAR_hi,'{:.0%}')})\\\\\n"
prows = ""
for d in ORDER:
    r = S[(S.model == "JIPNet") & (S.db == d)].iloc[0]
    prows += f"{e(NAME[d])} & {r.turn_med:.1f}$^\\circ$ & {r.turn_p90:.0f}$^\\circ$ & {r.shift_med:.1f} px ({r.shift_med*0.0508:.2f} mm) & {r.shift_p90:.0f} px\\\\\n"
paperJ = {"2002_DB1": "1\\% / 93\\%", "2002_DB3": "3\\% / 67\\%"}
paperA = {"2002_DB1": "29\\% / 3\\%", "2002_DB3": "35\\% / 1\\%"}
crows = "".join(
    f"{e(NAME[d])} & \\textbf{{{P_(AF[d]['jip_same400_EER'],'{:.1%}')}}} / {P_(AF[d]['jip_same400_TAR'],'{:.0%}')} & {paperJ[d]} & \\textbf{{{P_(AF[d]['afr_EER'],'{:.1%}')}}} / {P_(AF[d]['afr_TAR'],'{:.0%}')} & {paperA[d]}\\\\\n"
    for d in ["2002_DB1", "2002_DB3"]
)
nc = 32 * -(-N // 4096)
tex = (
    r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=16mm,footskip=8mm]{geometry}
\usepackage[T1]{fontenc}\usepackage[scaled=0.92]{helvet}\usepackage{microtype}
\usepackage{graphicx,booktabs,tabularx,array,enumitem,float,caption,xcolor,fancyhdr,titlesec}
\usepackage[hidelinks]{hyperref}
\renewcommand{\familydefault}{\sfdefault}
\definecolor{accent}{HTML}{2F6DB5}
\titleformat{\section}{\large\bfseries\color{accent}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{10pt}{4pt}\titlespacing*{\subsection}{0pt}{8pt}{3pt}
\setlist{nosep,leftmargin=1.2em}\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\renewcommand{\arraystretch}{1.25}
\pagestyle{fancy}\fancyhf{}\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\scriptsize Sreejith Markkassery \textperiodcentered{} Review of AI/ML research for fingerprint deduplication \textperiodcentered{} page \thepage}
\begin{document}
\noindent\begin{tabularx}{\textwidth}{@{}X r@{}}\textbf{Review of AI/ML research for fingerprint deduplication} & Sreejith Markkassery\\ AFR-Net \textperiodcentered{} JIPNet \textperiodcentered{} HERS & 26 October 2026\end{tabularx}\par\vspace{-2pt}\rule{\textwidth}{1pt}
{\LARGE\bfseries AFR-Net, JIPNet and HERS for billion-scale fingerprint deduplication\par}\vspace{4pt}
\textbf{(a)} Applicability to billion-scale gallery development: design choices, latency and failure cases, with a proposed pipeline combining the three. \textbf{(b)} JIPNet and an AFR-Net-style baseline on FVC2002/2004 partial prints: EER, TAR at FAR 0.1\%, JIPNet pose error and failure analysis.

{\footnotesize Sources of figures: \emph{reported} in the cited paper; \emph{measured} in this study (test machine described in Part (b)); \emph{derived} by calculation from measured or reported figures.}

\section*{Executive summary}
\textbf{(a) Applicability to a billion-scale gallery} ($\approx$ 14.5 billion finger records). The three papers are complementary:
\begin{itemize}
\item \textbf{AFR-Net}: a fixed-length template computed once per print and searchable across the full gallery. Role: \textbf{candidate retrieval}.
\item \textbf{JIPNet}: accurate pairwise verification with pose estimation, strongest on partial prints. Role: \textbf{candidate verification}.
\item \textbf{HERS}: search over encrypted templates. Role: \textbf{template protection}.
\item Combined in a staged pipeline (a.4), they cover retrieval, verification and privacy.
\end{itemize}
\textbf{(b) Partial-print experiment} (FVC2002/2004, 160$\times$160 patches):
\begin{itemize}
\item \textbf{JIPNet accuracy} on the six real-sensor databases: EER """
    + e(pr(real.EER.min(), real.EER.max(), "{:.1%}"))
    + r""", TAR at FAR 0.1\% of """
    + e(pr(real.TAR_0p1.min(), real.TAR_0p1.max(), "{:.0%}"))
    + r""", in line with the paper.
\item \textbf{JIPNet pose error} (typical): """
    + e(pr(J.turn_med.min(), J.turn_med.max(), "{:.0f}"))
    + r"""$^\circ$ in rotation and """
    + e(pr(J.shift_med.min(), J.shift_med.max(), "{:.0f}"))
    + f" px ({J.shift_med.min()*0.0508:.1f}--{J.shift_med.max()*0.0508:.1f} mm)"
    + r""" in translation.
\item \textbf{Comparison}: JIPNet clearly outperformed the AFR-Net-style fixed-length baseline on the same pairs, as also reported in the JIPNet paper. This confirms the value of pose estimation for partial prints and supports orientation normalisation before template extraction.
\item \textbf{Main factors affecting accuracy}: fingerprint coverage within the patch, overlap between patches, rotation, and synthetic versus real data.
\end{itemize}

\section*{Part (a) \textperiodcentered{} Applicability to billion-scale gallery development}
\textbf{Scale:} 145.32 crore Aadhaar holders $\times$ 10 fingers $\approx$ 14.5 billion finger records (UIDAI dashboard). Every new enrolment is compared against all of them.

\textbf{Implication:} search time and the expected number of false matches both grow in proportion to gallery size. A billion-scale system therefore needs (1) a reusable template for fast full-gallery retrieval, (2) a more accurate verifier applied to a short candidate list, and (3) fusion across fingers.

\textbf{Notation} used in the latency and complexity entries: $N$ = number of finger records in the gallery ($\approx$ 14.5 billion); $d$ = numbers per template; $K$ = number of candidates passed to verification; $P$ = local descriptors per print in AFR-Net realignment; $n$ = templates packed into one encrypted block (ciphertext) in HERS; $f$ = fraction of the gallery scanned when an index is used. $O(\cdot)$ describes how cost grows with these quantities, independent of hardware. GFLOP / TFLOP = $10^9$ / $10^{12}$ floating-point operations (refer Appendix A0).

\subsection*{a.1 \textperiodcentered{} AFR-Net: fixed-length fingerprint representation}
"""
    + fig(
        "diagram_afrnet.png",
        "AFR-Net: shared trunk with CNN and attention heads; realignment only for uncertain scores.",
        "0.85\\textwidth",
    )
    + kv(
        [
            (
                "Approach",
                r"\textbf{Network}: shared ResNet50 trunk with a CNN head and an attention head (85\,M parameters, trained on 1.3\,M images).\newline \textbf{Template}: 384 + 384 numbers per print, compared by cosine similarity.\newline \textbf{Realignment}: a pairwise refinement applied only to uncertain scores (0.3--0.6; 17.9\% of comparisons in the authors' tests).",
            ),
            (
                "Reported results",
                r"\textbf{FVC2002 DB1 / DB2 / DB3} (full prints, standard FVC protocol): TAR at FAR 0.1\% of 100\% / 99.96\% / 99.36\%.\newline \textbf{NIST SD14} (rolled prints): rank-1 identification 99.78\%.\newline \textbf{NIST SD27} (latent prints vs a 100{,}000-print gallery): rank-1 identification 53.10\%.",
            ),
            (
                "Design choice at scale",
                r"The template is computed once per print, stored, indexed and compatible with encryption, which makes it well suited to \textbf{candidate retrieval} over the full gallery.",
            ),
            (
                "Latency",
                f"\\textbf{{Complexity}} (machine-independent): template creation $O(1)$ per print, $\\approx$ {FL['AFRNet_template_GFLOP']:.1f} GFLOP (measured operation count); full-gallery search $O(N\\cdot d)$ $\\approx$ {2*N*768/1e12:.0f} TFLOP per enrolee at $d$ = 768 (derived); realignment $O(K\\cdot P^2)$ with $P$ = 196.\\newline \\textbf{{Creating a template}} (once per print, at enrolment): 8.42 ms on a GPU, as reported; {AFsec:.1f} s on the CPU test machine (measured).\\newline \\textbf{{Comparing two templates}}: {bench['768']*1e9:.0f} ns on one CPU core (measured). Searching all 14.5 billion templates therefore takes $\\approx$ {t768:.0f} minutes on one core (derived), and proportionally less with more cores or an index.\\newline \\textbf{{Realignment}} (only for uncertain scores): 29.36 ms per pair, as reported, so it is applied to candidate lists only.",
            ),
        ]
    )
    + "\n\\textbf{Failure cases and impact}\\par\n"
    + fc(
        [
            (
                "Very low image quality",
                "authors",
                "Worn, dry or wet fingers produce unreliable templates; a duplicate missed at retrieval is not recovered later.",
                "Quality check and re-capture at enrolment (stage 0); fusion across 10 fingers.",
            ),
            (
                "Very small overlap",
                "authors",
                "Partial or off-centre placement reduces the shared area between enrolment and query prints.",
                "Supervised full-print capture at enrolment; JIPNet verification for partial captures (stage 2).",
            ),
            (
                "Rotation on small patches",
                "refer Part (b)",
                "Template similarity drops as rotation grows (EER 36.5% on rotated 160×160 patches; the JIPNet paper also reports 29–35% for AFR-Net on such patches).",
                "Orientation normalisation before template extraction (stage 0).",
            ),
            (
                "Largest evaluated gallery 100,000",
                "authors",
                "False matches grow with gallery size; behaviour at 14.5 billion is not yet demonstrated.",
                "Validation at million-scale before rollout; multi-finger fusion (stage 3).",
            ),
        ]
    )
    + r"""
\subsection*{a.2 \textperiodcentered{} JIPNet: joint pairwise verification and pose estimation}
"""
    + fig(
        "diagram_jipnet.png",
        "JIPNet: both patches are processed together; outputs match probability and relative pose.",
        "0.85\\textwidth",
    )
    + kv(
        [
            (
                "Approach",
                r"\textbf{Network}: shared CNN encoder for both patches, followed by a transformer with self- and cross-attention between them.\newline \textbf{Output}: match probability and relative pose (rotation and translation), estimated jointly.",
            ),
            (
                "Reported results",
                r"\textbf{Mixed partial-print test set} (160$\times$160 patches): EER 0.04, TAR at FAR 0.1\% of 0.71 (VeriFinger: 0.18 and 0.56).\newline \textbf{FVC2002 DB1} (160$\times$160 patches): EER 0.01, TAR at FAR 0.1\% of 0.93.",
            ),
            (
                "Design choice at scale",
                r"A pairwise model with high accuracy on partial prints and an explicit pose output, well suited to \textbf{candidate verification} and to aligning prints for review.",
            ),
            (
                "Latency",
                f"\\textbf{{Complexity}} (machine-independent): $O(K)$ network runs for $K$ candidates, $\\approx$ {FL['JIPNet_pair_GFLOP']:.1f} GFLOP per pair (measured operation count); no reusable template, so a full-gallery search would be $O(N)$ runs $\\approx$ {sci(N*FL['JIPNet_pair_GFLOP']*1e9)} FLOP (derived).\\newline \\textbf{{Comparing two patches}}: {jip_sec:.2f} s per pair on the CPU test machine (measured); no figure is reported in the paper.\\newline \\textbf{{Verifying a candidate list}} of 1{{,}}000: $\\approx$ {jip_sec*1000/60:.0f} minutes on the test machine (derived), and considerably less on a GPU.",
            ),
        ]
    )
    + "\n\\textbf{Failure cases and impact}\\par\n"
    + fc(
        [
            (
                "Severe defects, weak texture",
                "authors",
                "Low-quality enrolments may be rejected by verification even when they are true duplicates.",
                "Fusion with other fingers (and iris/face); manual review queue for borderline cases.",
            ),
            (
                "Skin distortion",
                "authors",
                "Pressure and skin changes between enrolments years apart; not covered by the single-capture test (refer Part (b)).",
                "Validation on pairs from different captures before deployment.",
            ),
            (
                "Small overlap",
                "authors; refer Part (b)",
                "Pose error and missed matches increase as overlap decreases.",
                "Minimum-coverage check at capture; several patches or fingers per decision.",
            ),
            (
                "Occasional 180° pose",
                "refer Part (b)",
                "At most 1.3% of genuine pairs, with high match scores; could support a wrong alignment during review.",
                "Pose consistency check across fingers (stage 2).",
            ),
            (
                "One-to-one only",
                "authors",
                f"Cannot search the gallery directly: 14.5 billion pairs would take ≈ {jip_years:,.0f} years on the test machine.",
                "Used only after candidate retrieval (stage 1).",
            ),
        ]
    )
    + r"""
\subsection*{a.3 \textperiodcentered{} HERS: search over homomorphically encrypted templates}
"""
    + fig(
        "diagram_hers.png",
        "HERS: templates are reduced, encrypted and scored without decryption; only the key holder reads the scores.",
        "0.85\\textwidth",
    )
    + kv(
        [
            (
                "Approach",
                r"\textbf{Template}: any fixed-length template, reduced to 64, 32 or 16 numbers.\newline \textbf{Encryption}: Fan--Vercauteren homomorphic scheme (Microsoft SEAL); similarity is computed without decrypting the gallery or the query.",
            ),
            (
                "Reported results",
                r"\textbf{Search time} (100 million gallery, 10 CPU cores): 740 s per query at 32 numbers; $\approx$ 500 s with a two-stage search.\newline \textbf{Accuracy} (NIST SD4 vs 100 million): rank-1 92.3\% at 192 and 64 numbers, 92.2\% at 32, 78.6\% at 16.\newline \textbf{Storage}: 140 GB at 32 numbers per 100 million templates. Reported as 275$\times$ faster than earlier encrypted search.",
            ),
            (
                "Design choice at scale",
                r"Templates remain protected at rest and during search, and any fixed-length template can be used, which makes it the \textbf{privacy layer} for candidate retrieval.",
            ),
            (
                "Latency",
                f"\\textbf{{Complexity}} (machine-independent): $O(d\\cdot\\lceil N/n\\rceil)$ encrypted multiplications per query, with $n$ = 4{{,}}096 templates per ciphertext (paper setting): $\\approx$ {sci(nc)} at $d$ = 32 and $N$ = 14.5 billion (derived); storage $O(N\\cdot d)$.\\newline \\textbf{{Reported}}: 740 s per query for a 100 million gallery on 10 CPU cores (32 numbers per template).\\newline \\textbf{{Scaled to 14.5 billion}} ($\\times$ 145): $\\approx$ 30 hours per query for a full scan (derived). With an index that limits the search to 1\\% of the gallery: $\\approx$ 18 minutes (design target).",
            ),
        ]
    )
    + "\n\\textbf{Failure cases and impact}\\par\n"
    + fc(
        [
            (
                "Accuracy drop at 16 numbers",
                "authors",
                "Limits how far templates can be compressed to save time and storage.",
                "Use 32–64 numbers per template (≈ 20 TB encrypted at 32 numbers for 14.5 billion).",
            ),
            (
                "Encrypted search cost",
                "authors",
                "A full encrypted scan of 14.5 billion is ≈ 30 hours per query on 10 cores.",
                "Index that narrows the search (design target 1% of the gallery, ≈ 18 minutes).",
            ),
            (
                "Server cannot rank results",
                "authors",
                "Scores must be decrypted by the key holder to select candidates.",
                "Ranking inside the secure processing zone, with controlled key management.",
            ),
            (
                "Synthetic evaluation gallery",
                "authors; refer Part (b)",
                "Synthetic prints behave differently from real ones (JIPNet EER 5.9–10% vs 1.6–3.2%), so real-print accuracy at scale is unverified.",
                "Validation on a large real-print gallery.",
            ),
        ]
    )
    + r"""
\subsection*{a.4 \textperiodcentered{} Proposed pipeline}
"""
    + fig("diagram_pipeline.png", "Proposed staged pipeline.", "\\textwidth")
    + r"""
\begin{tabularx}{\textwidth}{@{}>{\raggedright}p{2.6cm}>{\raggedright}X>{\raggedright\arraybackslash}p{5.2cm}@{}}\toprule
\textbf{Stage} & \textbf{Supporting evidence} & \textbf{Time per enrolee}\\\midrule
0 $\cdot$ Capture checks & \textbf{Coverage:} JIPNet EER on FVC2002 DB1 improves from 2.5\% to 1.1\% when both patches are at least 70\% fingerprint.\newline \textbf{Orientation:} fixed-length templates perform best on consistently oriented prints.\newline \textbf{Liveness:} synthetic prints score highly against each other (refer Part (b)). & Negligible\\\midrule
1 $\cdot$ Candidate retrieval & \textbf{Scalability:} fixed-length templates can be searched across the full gallery (a.1).\newline \textbf{Privacy:} templates can stay encrypted during search (a.3).\newline \textbf{Compactness:} 32--64 numbers per template retain accuracy (HERS). & Unencrypted: $\approx$ """
    + f"{t768:.0f}"
    + r""" minutes on one core for a full scan, reduced by more cores and indexing. Encrypted: $\approx$ 18 minutes on 10 cores if an index limits the search to 1\% (derived).\\\midrule
2 $\cdot$ Candidate verification & \textbf{Accuracy on partial prints:} on the same pairs, JIPNet EER """
    + e(pr(min(jip_same), max(jip_same), "{:.1%}"))
    + " versus "
    + e(pr(min(afr_e), max(afr_e), "{:.1%}"))
    + r""" for a fixed-length template (refer Part (b)).\newline \textbf{Pose output:} supports alignment checks and examiner review. & For 1{,}000 candidates: $\approx$ 29 s with AFR-Net realignment; $\approx$ """
    + f"{jip_sec*1000/60:.0f}"
    + r""" minutes with JIPNet on the CPU test machine.\\\midrule
3 $\cdot$ Decision & \textbf{Multi-finger fusion:} combining 10 fingers adds independent evidence and reduces false matches at gallery scale. & Negligible\\\bottomrule
\end{tabularx}

\textbf{Machine-independent cost per new enrolee}\par
\begin{tabularx}{\textwidth}{@{}>{\raggedright}p{4.6cm}>{\raggedright}p{4.4cm}>{\raggedright\arraybackslash}X@{}}\toprule
\textbf{Step} & \textbf{Complexity} & \textbf{Operations at $N$ = 14.5 billion}\\\midrule
Template creation (AFR-Net) & $O(1)$ per print & $\approx$ """
    + f"{FL['AFRNet_template_GFLOP']:.1f}"
    + r""" GFLOP per print, $\times$ 10 fingers\\
Full-gallery search, plain & $O(N\cdot d)$ & $\approx$ """
    + f"{2*N*768/1e12:.0f}"
    + r""" TFLOP at $d$ = 768; $\approx$ """
    + f"{2*N*32/1e12:.1f}"
    + r""" TFLOP at $d$ = 32\\
Indexed search (fraction $f$) & $O(f\cdot N\cdot d)$ & $f$ = 1\%: $\approx$ """
    + f"{2*N*32*0.01/1e9:.0f}"
    + r""" GFLOP at $d$ = 32\\
Full-gallery search, encrypted (HERS) & $O(d\cdot\lceil N/n\rceil)$ ciphertext mult. & $\approx$ """
    + sci(nc)
    + r""" at $d$ = 32, $n$ = 4{,}096\\
Candidate verification (JIPNet) & $O(K)$ & $K$ = 1{,}000: $\approx$ """
    + f"{1000*FL['JIPNet_pair_GFLOP']/1e3:.1f}"
    + r""" TFLOP\\
Decision (fusion) & $O(K\cdot\mathrm{fingers})$ & negligible\\\bottomrule
\end{tabularx}
{\footnotesize Network counts measured with PyTorch's operation counter (AFR-Net: the JIPNet authors' re-implementation at 224$\times$224 input). These counts are independent of hardware; the wall-clock times above are one example on the test machine. Verifying 1{,}000 candidates with JIPNet requires more operations than a plain scan of the entire gallery at $d$ = 32, which is why the candidate list must remain short.\par}

\textbf{Operating modes}
\begin{itemize}
\item \textbf{Full-print enrolment} (supervised capture of full impressions): AFR-Net reports TAR at FAR 0.1\% of 99.36--100\% on full FVC2002 prints, where its built-in alignment can orient the print. An AFR-Net-type template can therefore handle both retrieval and verification, entirely in encrypted form.
\item \textbf{Partial or low-quality captures} (small sensors, poor placement): JIPNet verifies the candidate list inside the secure processing zone, where its pose estimation gives the largest benefit (refer Part (b), section b.2).
\end{itemize}
\begin{tabularx}{\textwidth}{@{}>{\raggedright}p{3.6cm}>{\raggedright}X>{\raggedright\arraybackslash}X@{}}\toprule
 & \textbf{AFR-Net-type template} & \textbf{JIPNet}\\\midrule
Stored per finger & 768 numbers $\approx$ 3 KB (float32) & Fingerprint image (e.g.\ 388$\times$374 px $\approx$ 145 KB uncompressed)\\
Full gallery, 14.5 B fingers & $\approx$ 45 TB unencrypted; $\approx$ 20 TB encrypted at 32 numbers (derived) & Image archive for all fingers\\
Matching in encrypted form & Supported (HERS) & Requires decrypted images, within the secure zone\\\bottomrule
\end{tabularx}

\section*{Part (b) \textperiodcentered{} JIPNet and AFR-Net-style baseline on FVC2002/2004}
\begin{itemize}
\item \textbf{Data:} FVC2002 and FVC2004, 8 databases (6 real sensors, 2 synthetic), each with 100 fingers $\times$ 8 impressions.
\item \textbf{Test pairs:} 160$\times$160 patches, following the JIPNet protocol (patch centres up to 100 px apart, rotation between $-180^\circ$ and $180^\circ$). Per database: 800 genuine pairs (same finger) and 2{,}400 impostor pairs (different fingers).
\item \textbf{Ground-truth pose:} both patches of a genuine pair are cut from one impression with a known shift and rotation, so the true pose is exact. Results are therefore an upper bound for pairs taken from different captures.
\item \textbf{JIPNet:} official code and weights; outputs reproduced to 2 decimals on the authors' examples. Run on all pairs of all 8 databases.
\item \textbf{AFR-Net-style baseline:} AFR-Net weights released by the JIPNet authors, used as a fixed-length template (score = 0.2 $\times$ CNN similarity + 0.8 $\times$ attention similarity). Run on a subsample: 400 genuine pairs each from FVC2002 DB1 and DB3, against all different-finger combinations.
\item \textbf{Metrics:} \emph{EER}, the error rate at the threshold where false accepts equal false rejects (lower is better). \emph{TAR at FAR 0.1\%}, the share of genuine pairs accepted when only 1 in 1{,}000 impostor pairs is accepted (higher is better). \emph{Pose error}, the difference between estimated and true rotation (degrees) and translation (pixels).
\item \textbf{Test machine:} 2-core Intel Xeon CPU (2.1 GHz), no GPU. Measured timings refer to this machine; a GPU server is considerably faster.
\end{itemize}

\subsection*{b.1 \textperiodcentered{} EER and TAR at FAR 0.1\% (JIPNet)}
\begin{tabularx}{\textwidth}{@{}>{\raggedright}X>{\raggedright}p{4.2cm}>{\raggedright\arraybackslash}p{4.2cm}@{}}\toprule
\textbf{Database} & \textbf{EER} & \textbf{TAR at FAR 0.1\%}\\\midrule
"""
    + jrows
    + r"""\bottomrule\end{tabularx}
{\footnotesize Values in brackets: range across random re-draws of the test pairs (90\% interval).\par}

\textbf{Comparison with the paper} (FVC2002 DB1: EER 1\%, TAR 93\%; DB3: EER 3\%, TAR 67\%). TAR at FAR 0.1\% is set by the few highest-scoring impostor pairs, so it varies widely between samples (78--92\% here on DB1). The remaining difference is explained by blank area in the patches: when both patches are at least 70\% fingerprint, JIPNet reaches EER 1.1\% and TAR 89\% on FVC2002 DB1, in line with the paper. On DB3 the measured TAR (82\%) is higher than the paper's (67\%).
"""
    + fig("b_bars.png", "EER and TAR at FAR 0.1\\% per database.")
    + r"""
\subsection*{b.2 \textperiodcentered{} Comparison with the AFR-Net-style baseline}
\begin{tabularx}{\textwidth}{@{}>{\raggedright}X>{\raggedright}p{2.8cm}>{\raggedright}p{2.2cm}>{\raggedright}p{3.2cm}>{\raggedright\arraybackslash}p{2.4cm}@{}}\toprule
\textbf{Database} & \textbf{JIPNet, measured} & \textbf{JIPNet, paper} & \textbf{AFR-Net-style, measured} & \textbf{AFR-Net, paper*}\\\midrule
"""
    + crows
    + r"""\bottomrule\end{tabularx}
{\footnotesize Each cell: EER / TAR at FAR 0.1\%. Measured values use the same 400 genuine pairs per database for both models. Paper values: JIPNet paper, Table V, same protocol. *The paper's AFR-Net result adds a separate alignment network in front of AFR-Net.\par}
\begin{itemize}
\item \textbf{Agreement with the paper:} both measured models are close to the values reported in the JIPNet paper. The paper likewise reports a high EER for AFR-Net on these partial patches (29\% on DB1, 35\% on DB3), even with an added alignment network; AFR-Net's near-perfect published results refer to full prints.
\item \textbf{Effect of rotation:} the fixed-length template is not rotation-invariant. On FVC2002 DB1 its typical genuine score is """
    + f"{r1[0]['afr_med']:.2f}"
    + r""" for rotations under 15$^\circ$ and """
    + f"{r1[-1]['afr_med']:.2f}"
    + r""" for rotations of 120--180$^\circ$. JIPNet estimates the rotation itself and stays near 0.95 at all angles.
\end{itemize}
"""
    + fig("b_afr_rotation.png", "Typical genuine score by true rotation between the patches.")
    + r"""
\subsection*{b.3 \textperiodcentered{} JIPNet pose error}
\begin{tabularx}{\textwidth}{@{}>{\raggedright}X ccc c@{}}\toprule
\textbf{Database} & \textbf{Rotation, typical} & \textbf{Rotation, worst 10\%} & \textbf{Translation, typical} & \textbf{Translation, worst 10\%}\\\midrule
"""
    + prows
    + r"""\bottomrule\end{tabularx}
{\footnotesize Genuine pairs only. Typical = median (half the pairs have a smaller error). Worst 10\% = the error exceeded by only 1 pair in 10. 1 px = 0.0508 mm at 500 dpi. Pose error grows as overlap decreases (Figure A4).\par}

\subsection*{b.4 \textperiodcentered{} Failure analysis}
\begin{tabularx}{\textwidth}{@{}>{\bfseries\raggedright}p{3.2cm}X@{}}\toprule
\textbf{Factor} & \textbf{Observation and mitigation}\\\midrule
Fingerprint coverage & JIPNet EER on FVC2002 DB1 improves from 2.5\% to 1.1\%, matching the paper, when both patches are $\geq$ 70\% fingerprint (Figure A5). Lowest-scoring genuine pairs often contain blank regions (Figure A6).\\\midrule
Overlap & Rotation and translation errors increase as the overlap between patches decreases (Figure A4).\\\midrule
Synthetic prints & EER """
    + e(pr(syn.EER.min(), syn.EER.max(), "{:.1%}"))
    + " versus "
    + e(pr(real.EER.min(), real.EER.max(), "{:.1%}"))
    + r""" on real sensors; different synthetic fingers receive high scores (Figure A7). This suggests synthetic galleries are not representative of real performance, and that resistance to synthetic prints (e.g.\ ``MasterPrint'' attacks, Roy et al.\ 2017; Bontrager et al.\ 2018) should be part of evaluation.\\\midrule
180$^\circ$ pose estimates & At most 1.3\% of genuine pairs (0--0.2\% on real sensors), with high match scores; a pose consistency check addresses this.\\\midrule
Rotation (fixed-length) & Accuracy of the AFR-Net-style template decreases with rotation (refer Part (b), section b.2); orientation normalisation before template extraction addresses this.\\\midrule
Sensor type & Consistent performance on real sensors not seen in training (optical FVC2002 DB2, thermal FVC2004 DB3).\\\bottomrule
\end{tabularx}

\clearpage
\section*{Appendix}
\textbf{A0 \textperiodcentered{} Note on operation counts (GFLOP).} A \emph{FLOP} is one floating-point operation: a single addition or multiplication of two decimal numbers. A \emph{GFLOP} is one billion such operations. For example, creating one AFR-Net template takes $\approx$ """
    + f"{FL['AFRNet_template_GFLOP']:.1f}"
    + r""" GFLOP, and comparing one JIPNet pair takes $\approx$ """
    + f"{FL['JIPNet_pair_GFLOP']:.1f}"
    + r""" GFLOP. The operation count is fixed by the method and does not depend on the hardware, which is why it is used here as a machine-independent measure of cost. Hardware speed is expressed separately in GFLOPS (operations per second). The time for a step is approximately the operation count divided by the hardware speed, so the same count gives shorter times on faster hardware such as a GPU.

\subsection*{Supporting figures}
"""
    + "".join(
        fig(f, f"\\textbf{{Figure A{k}.}} {c}")
        for k, (f, c) in enumerate(
            [
                (
                    "b_hills_JIPNet.png",
                    "JIPNet score distributions per database: same-finger (green) and different-finger (red) pairs.",
                ),
                (
                    "b_afr_vs_jip_hills.png",
                    "Score distributions on FVC2002 DB1/DB3: AFR-Net-style baseline (top) and JIPNet (bottom).",
                ),
                (
                    "b_pose.png",
                    "JIPNet: share of genuine pairs within a given rotation or translation error.",
                ),
                ("b_pose_overlap.png", "JIPNet: pose error by overlap between the two patches."),
                (
                    "b_coverage.png",
                    "Effect of fingerprint coverage within the patch on EER and TAR at FAR 0.1\\%.",
                ),
                (
                    "b_fail_JIPNet_2002_DB1.png",
                    "JIPNet lowest-scoring genuine pairs, highest-scoring impostor pairs and largest rotation errors, FVC2002 DB1.",
                ),
                ("b_fail_JIPNet_2002_DB4.png", "As Figure A6, synthetic FVC2002 DB4."),
                (
                    "b_fail_AFR_2002_DB1.png",
                    "AFR-Net-style baseline lowest-scoring genuine and highest-scoring impostor pairs, FVC2002 DB1.",
                ),
            ],
            1,
        )
    )
    + r"""
\vspace{6pt}{\footnotesize \textbf{Sources:} AFR-Net, arXiv 2211.13897 \textperiodcentered{} JIPNet, arXiv 2405.03959; code github.com/XiongjunGuan/JIPNet \textperiodcentered{} HERS, arXiv 2003.12197 \textperiodcentered{} Roy, Memon, Ross, arXiv 1705.07386 \textperiodcentered{} Bontrager et al., IEEE BTAS 2018 \textperiodcentered{} UIDAI enrolment dashboard \textperiodcentered{} FVC2002/2004, bias.csr.unibo.it.\par}
\end{document}
"""
)
open(f"{P}/reports/latex/UIDAI_review.tex", "w").write(tex)
print("ok", len(tex))
