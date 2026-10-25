"""Final deliverable: reports/UIDAI_report.html = (a) billion-scale review + (b) partial-print experiment.
Regenerate after new results: python src/report_figs.py && python src/build_final.py"""

import base64
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

P = ROOT
FIG = f"{P}/reports/figures"
tpl = open(f"{P}/reports/assets/page_template.html").read()
style = tpl[tpl.index("<style>") : tpl.index("</style>") + 8]


def svg(label):
    i = tpl.index(f'aria-label="{label}"')
    a = tpl.rfind("<svg", 0, i)
    b = tpl.index("</svg>", i) + 6
    out = tpl[a:b].replace("postcode", "code")
    for x, y in [
        ("texture expert → 384", "CNN head → 384"),
        ("12 attention blocks → 384", "12 transformer blocks → 384"),
        (
            "Second look — only when the score is in the “unsure” zone 0.3–0.6",
            "Realignment — only for scores in the uncertain range 0.3–0.6",
        ),
        ("= “trying the jigsaw fit”", ""),
        ("reads each patch alone", "encodes each patch"),
        ("locked gallery", "encrypted gallery"),
        ("🔒 lock", "encrypt"),
        ("🔓 unlock", "decrypt"),
        ("locked query", "encrypted query"),
        ("locked scores", "encrypted scores"),
        ("locked gallery × locked query", "encrypted gallery × encrypted query"),
        (
            "The server cannot find the maximum itself — the lock only allows add and multiply, not “compare”.",
            "The server cannot compute the maximum: the scheme supports addition and multiplication, not comparison.",
        ),
        ("pick best K;", "select top K;"),
        ("re-check (stage 2)", "verify (stage 2)"),
        (
            "Pre-training task: the CNN first learns to “clean up” (enhance) prints, which makes its features sharper.",
            "Pre-training task: fingerprint enhancement, used to initialise the CNN encoder.",
        ),
        ("“A looks at B and back”", ""),
        ("cross-attention: A looks at B and back", "cross-attention between A and B"),
        ("self-attention: look within A, within B", "self-attention within A and within B"),
        ("straighten", "spatial"),
        ("(alignment)", "alignment"),
        ("Key holder (client)", "Key holder (client)"),
        ("Server (no key)", "Server (no decryption key)"),
    ]:
        out = out.replace(x, y)
    return out


SVG_AFR, SVG_JIP, SVG_HERS = (
    svg("AFR-Net block diagram"),
    svg("JIPNet block diagram"),
    svg("HERS encrypted search flow"),
)
S = pd.read_csv(f"{P}/experiments/summary.csv")
bench = json.load(
    open(f"{P}/experiments/bench.json")
)  # measured ns per code comparison (src/bench_compare.py)
img = lambda f: (
    ("data:image/png;base64," + base64.b64encode(open(f"{FIG}/{f}", "rb").read()).decode())
    if os.path.exists(f"{FIG}/{f}")
    else ""
)
ORDER = ["2002_DB1", "2002_DB2", "2002_DB3", "2002_DB4", "2004_DB1", "2004_DB2", "2004_DB3", "2004_DB4"]
NAME = {
    "2002_DB1": "FVC2002 DB1 · optical",
    "2002_DB2": "FVC2002 DB2 · optical",
    "2002_DB3": "FVC2002 DB3 · capacitive",
    "2002_DB4": "FVC2002 DB4 · synthetic",
    "2004_DB1": "FVC2004 DB1 · optical",
    "2004_DB2": "FVC2004 DB2 · optical",
    "2004_DB3": "FVC2004 DB3 · thermal",
    "2004_DB4": "FVC2004 DB4 · synthetic",
}
NOTE = {
    "2004_DB1": "JIPNet trained on it",
    "2004_DB2": "JIPNet trained on it",
    "2002_DB1": "paper’s test set",
    "2002_DB3": "paper’s test set",
}


def get(m, db):
    r = S[(S.model == m) & (S.db == db)]
    return None if r.empty else r.iloc[0]


def fmt_rate(r, k, f):
    if r is None:
        return '<span class="mini">running</span>'
    lo, hi = (r.EER_lo, r.EER_hi) if k == "EER" else (r.TAR_lo, r.TAR_hi)
    part = "" if r.n_gen >= 800 else f'<br><span class="mini">partial: {int(r.n_gen)}/800 genuine</span>'
    return f'<b>{f.format(r[k])}</b> <span class="mini">({f.format(lo)}–{f.format(hi)})</span>{part}'


rows = "".join(
    f'<tr><td>{NAME[d]}<br><span class="mini">{NOTE.get(d, "")}</span></td><td>{fmt_rate(get("JIPNet", d), "EER", "{:.1%}")}</td>'
    f'<td>{fmt_rate(get("JIPNet", d), "TAR_0p1", "{:.0%}")}</td></tr>'
    for d in ORDER
)


def pose(d):
    r = get("JIPNet", d)
    if r is None:
        return f'<tr><td>{NAME[d]}</td><td colspan="4" class="mini">running</td></tr>'
    return f'<tr><td>{NAME[d]}</td><td>{r.turn_med:.1f}°</td><td>{r.turn_p90:.0f}°</td><td>{r.shift_med:.1f} px <span class="mini">({r.shift_med*0.0508:.2f} mm)</span></td><td>{r.shift_p90:.0f} px</td></tr>'


prow = "".join(pose(d) for d in ORDER)
AF = (
    json.load(open(f"{P}/experiments/afr_summary.json"))
    if os.path.exists(f"{P}/experiments/afr_summary.json")
    else {}
)
ROT = (
    json.load(open(f"{P}/experiments/afr_rotation.json"))
    if os.path.exists(f"{P}/experiments/afr_rotation.json")
    else {}
)
AFsec = min(v["sec_per_patch"] for v in AF.values()) if AF else float("nan")
r1 = ROT.get("2002_DB1", [])
rot_line = (
    (
        f"on FVC2002 DB1 its typical same-finger score is {r1[0]['afr_med']:.2f} for turns under 15° but {r1[-1]['afr_med']:.2f} for turns of 120–180°, against a typical different-finger score of about 0.20"
    )
    if r1
    else "see figure"
)
rot_html = (
    (
        f'<p><b>Cause of the baseline’s low accuracy:</b> rotation. Its representation is not rotation-invariant, and its built-in alignment module does not compensate for large rotations of small patches. The test pairs are turned anywhere from −180° to 180°, the JIPNet paper’s protocol, which is wider than typical real captures. Even when the turn is under 15°, the baseline’s EER is {r1[0]["afr_eer"]:.1%} on FVC2002 DB1, against 2.2% for JIPNet on all turns. {rot_line[0].upper() + rot_line[1:]}.</p><figure class="fig"><img src="{img("b_afr_rotation.png")}"><figcaption>Typical same-finger score by true turn. The baseline (blue) declines towards the different-finger level (red dashed) as the turn increases. JIPNet (orange) stays high at every angle.</figcaption></figure>'
    )
    if r1 and img("b_afr_rotation.png")
    else ""
)


def afr_rows():
    out = ""
    for d in ["2002_DB1", "2002_DB3"]:
        j = get("JIPNet", d)
        a = AF.get(d)
        jt = f"<b>{j.EER:.1%}</b> / <b>{j.TAR_0p1:.0%}</b>" if j is not None else "—"
        at = (
            f'<b>{a["afr_EER"]:.1%}</b> <span class="mini">({a["afr_EER_lo"]:.1%}–{a["afr_EER_hi"]:.1%})</span> / <b>{a["afr_TAR"]:.0%}</b> <span class="mini">({a["afr_TAR_lo"]:.0%}–{a["afr_TAR_hi"]:.0%})</span>'
            if a
            else '<span class="mini">running</span>'
        )
        same = f'{a["jip_same400_EER"]:.1%} / {a["jip_same400_TAR"]:.0%}' if a else "—"
        paper = {"2002_DB1": ("1% / 93%", "29% / 3%"), "2002_DB3": ("3% / 67%", "35% / 1%")}[d]
        out += f"<tr><td>{NAME[d]}</td><td>{jt}</td><td>{same}</td><td>{paper[0]}</td><td>{at}</td><td>{paper[1]}</td></tr>"
    return out


J = S[(S.model == "JIPNet") & ~S.db.str.startswith("ALL")]
A = S[(S.model == "AFRNet") & ~S.db.str.startswith("ALL")]
done_afr = len(A[A.n_gen >= 800])
done_jip = len(J[J.n_gen >= 800])
part_dbs = [
    NAME[d] + f" ({int(get('JIPNet', d).n_gen)} of 800 genuine pairs)"
    for d in ORDER
    if get("JIPNet", d) is not None and get("JIPNet", d).n_gen < 800
]
status = (
    ""
    if not part_dbs
    else f'<div class="callout"><b>Coverage note:</b> JIPNet ran on all 800 genuine + 2,400 impostor pairs of 7 databases; {", ".join(part_dbs)} was stopped early to meet the deadline and is reported on that subset. The AFR-Net-style baseline ran on FVC2002 DB1 and DB3, the paper’s two test sets.</div>'
)
N = 14.532e9  # 145.32 crore Aadhaar x 10 fingers (UIDAI dashboard)
t768 = bench["768"] * N / 60
t192 = bench["192"] * N / 60
jip_sec = J.sec_med.median() if len(J) else float("nan")
afr_sec = A.sec_med.median() if len(A) else 1.65
# ==== END OF SHARED DATA (reused by later report builders) ====
jip_years = N * jip_sec / 3.156e7
a1 = AF.get("2002_DB1", {})
a3 = AF.get("2002_DB3", {})
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UIDAI Fingerprint Review</title>{style}
<style>.fig img{{width:100%;height:auto;display:block;background:#fff;border-radius:8px}} td b{{font-size:15px}} .q{{font:600 12px system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--blue);margin:22px 0 4px}} h3{{margin-top:26px}}</style></head><body>
<svg width="0" height="0" style="position:absolute"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs></svg>
<div class="wrap">
<header class="hero"><div class="kicker">UIDAI · AI/ML fingerprint deduplication · Research review</div>
<h1>AFR-Net, JIPNet and HERS: fit for a billion-scale gallery, and a partial-print test</h1>
<p class="lede"><b>(a)</b> Applicability of each paper to billion-scale gallery development: design choices, latency, failure cases, followed by a proposed pipeline that combines them. <b>(b)</b> JIPNet and an AFR-Net-style baseline run on FVC2002/2004 partial prints: EER, TAR at FAR 0.1%, JIPNet pose error, failure analysis.</p>
<p class="src"><span class="tag paper">paper</span>reported in the cited paper &nbsp; <span class="tag derived">measured</span>measured in this study on the test machine (2-core Intel Xeon CPU at 2.1 GHz, no GPU); timings are for relative comparison, and a GPU server would be considerably faster &nbsp; <span class="tag illus">derived</span>arithmetic on measured or reported figures</p></header>

<h2><span class="n">PART (a)</span>Applicability to billion-scale gallery development</h2>
<p><b>Scale considered.</b> 145.32 crore Aadhaar holders (UIDAI dashboard) × 10 fingers ≈ <b>14.5 billion finger records</b>. Deduplicating one new enrolee means comparing against all of them. Two properties determine whether a method can operate at this scale:</p>
<div class="grid2">
<div class="card"><b class="sans">Search cost</b><p>Time per new enrolee = time per comparison × 14.5 billion. A method that must process both prints jointly for every comparison cannot be applied to the full gallery. A full-gallery search requires a template that is computed once per print and reused.</p></div>
<div class="card"><b class="sans">False-match accumulation</b><p>Expected false matches per search = 14.5 billion × false-match rate per comparison. At a rate of 1 in 1,000 this is 14.5 million false matches. Single-finger matchers therefore require multi-finger fusion and a second verification stage.</p></div>
</div>

<h3>a.1 · AFR-Net: fixed-length fingerprint representation</h3>
<p class="mini">Grosz &amp; Jain, “AFR-Net: Attention-Driven Fingerprint Recognition Network”, IEEE T-BIOM 2024</p>
<figure class="fig card">{SVG_AFR}<figcaption>Each print is aligned, passed through a shared CNN trunk, and split into two heads: a CNN head capturing local texture and an attention head capturing global structure. Each gives 384 numbers. The joined code is compared by similarity. A pairwise realignment step is applied only when the score falls in the uncertain range 0.3–0.6.</figcaption></figure>
<div class="grid2">
<div class="card"><b class="sans">Strengths</b><ul>
<li>Fixed-length code: computed once at enrolment, stored, indexed, and can be encrypted (it is the kind of template HERS works on).</li>
<li>Complementary local (CNN) and global (attention) representations, trained with ArcFace on 1.3 M images.</li>
<li>Realignment is applied only to uncertain scores (17.9% of comparisons in the authors’ tests). <span class="tag paper">AFR-Net</span></li>
<li>Robust across sensors and to latent prints; the authors report it ahead of the baselines on most of 11 test sets.</li></ul></div>
<div class="card"><b class="sans">Reported results</b> <span class="tag paper">AFR-Net</span><ul>
<li>FVC2002 DB1/DB2/DB3, TAR at FAR 0.1%: 100% / 99.96% / 99.36% (full prints, with realignment).</li>
<li>NIST SD14 rolled-to-rolled identification: rank-1 99.78%.</li>
<li>NIST SD27 latents vs 100 K gallery: rank-1 53.10% (Verifinger 55.04%).</li>
<li>8.42 ms per print (RTX 2080 Ti); 15.18 ms on average including realignment.</li></ul></div>
</div>
<div class="card"><b class="sans">Failure cases</b><p>Named by the authors: very poor image quality and very small overlap; no use of minutiae or other fingerprint domain knowledge; realignment is slow (brute-force matching). Found in part (b): on 160×160 partial patches turned relative to each other, the code fails (EER {a1.get("afr_EER", float("nan")):.1%} on FVC2002 DB1), because the representation is not rotation-invariant.</p></div>
<div class="tablewrap card"><table><tr><th colspan="2">Billion-scale applicability</th></tr>
<tr><td><b>Design choice</b></td><td>Suited to the <b>first stage</b> (full-gallery candidate retrieval): templates are computed once and searched repeatedly. Prints must first be brought to a common orientation, or the representation must be made rotation-invariant.</td></tr>
<tr><td><b>Latency</b></td><td>Code: 8.42 ms per print on GPU <span class="tag paper">AFR-Net</span> ({AFsec:.1f} s per print <span class="tag derived">measured</span>), paid once per print. Comparing two 768-number codes: {bench["768"]*1e9:.0f} ns on one CPU core <span class="tag derived">measured</span>, so a full scan of 14.5 B takes ≈ {t768:.0f} core-minutes per enrolee, before any index cuts it down. Realignment: 29.36 ms per pair, so it is applicable only to a candidate list.</td></tr>
<tr><td><b>Failure cases at scale</b></td><td>Evidence stops at a 100 K gallery, which is 145,000× smaller than the target. A duplicate missed in the first stage cannot be recovered later. Turned or partial captures must be handled before coding.</td></tr></table></div>

<h3>a.2 · JIPNet: joint pairwise verification and pose estimation</h3>
<p class="mini">Guan, Pan, Feng, Zhou, “Joint Identity Verification and Pose Alignment for Partial Fingerprints”, IEEE TIFS 2025</p>
<figure class="fig card">{SVG_JIP}<figcaption>Both patches enter together. A shared CNN encodes each patch; a transformer with self- and cross-attention relates the two; two heads output the match probability and the relative pose.</figcaption></figure>
<div class="grid2">
<div class="card"><b class="sans">Strengths</b><ul>
<li>Verification and pose solved together, each helping the other.</li>
<li>Built for small, partial, shifted and turned captures.</li>
<li>Pose output can align the pair for an examiner or for further checks.</li>
<li>Official code and weights are public (MIT licence).</li></ul></div>
<div class="card"><b class="sans">Reported results</b> <span class="tag paper">JIPNet Table V</span><ul>
<li>Mixed partial test set, 160×160: EER 0.04, TAR 0.71 at FAR 0.1% (VeriFinger 0.18 / 0.56).</li>
<li>FVC2002 DB1, 160×160: EER 0.01, TAR 0.93 at FAR 0.1%.</li>
<li>Smaller 96×96 patches: EER 0.08, so accuracy drops with patch size.</li>
<li>Pose error shown only as a figure; no latency figure in the text.</li></ul></div>
</div>
<div class="card"><b class="sans">Failure cases</b><p>Named by the authors: severe image defects, skin distortion, extremely weak texture, small overlap. One-to-many search is listed as future work. Found in part (b): blank area inside a patch, small overlap, synthetic prints, and rare 180° pose errors given with high confidence.</p></div>
<div class="tablewrap card"><table><tr><th colspan="2">Billion-scale applicability</th></tr>
<tr><td><b>Design choice</b></td><td>Pairwise, with no stored template, so it is suited to the <b>second stage</b>: verification of a candidate list, particularly for partial or low-quality captures.</td></tr>
<tr><td><b>Latency</b></td><td>{jip_sec:.2f} s per pair <span class="tag derived">measured</span>. A full scan of 14.5 B pairs would take ≈ {jip_years:,.0f} years on the test machine <span class="tag illus">derived</span>. A 1,000-candidate list takes ≈ {jip_sec*1000/60:.0f} min on the test machine, and much less on a GPU.</td></tr>
<tr><td><b>Failure cases at scale</b></td><td>Per-comparison error of 1.6–3.2% EER on real sensors (part b) is adequate for ranking a candidate list but insufficient for a decision at 14.5 B scale on its own; scores should be fused across fingers.</td></tr></table></div>

<h3>a.3 · HERS: search over homomorphically encrypted templates</h3>
<p class="mini">Engelsma, Jain, Boddeti, “HERS: Homomorphically Encrypted Representation Search”, IEEE T-BIOM 2022</p>
<figure class="fig card">{SVG_HERS}<figcaption>The code is shortened (192 → 64/32/16 numbers), encrypted, and scored against the encrypted gallery without decryption. Only the key holder can read the scores and pick the best match.</figcaption></figure>
<div class="grid2">
<div class="card"><b class="sans">Strengths</b><ul>
<li>Templates are never readable on the server, at rest or during search. A leak exposes nothing usable.</li>
<li>Works with any fixed-length code (AFR-Net’s included).</li>
<li>Shortening to 32 numbers keeps accuracy while cutting time and storage.</li>
<li>Reported 275× faster than earlier encrypted search. <span class="tag paper">HERS</span></li></ul></div>
<div class="card"><b class="sans">Reported results</b> <span class="tag paper">HERS</span><ul>
<li>100 M gallery, 10 CPU cores: 740 s per query at 32 numbers, 1,480 s at 64; two-stage search ≈ 500 s.</li>
<li>Rank-1 (NIST SD4 vs 100 M): 92.3% at 192 and 64 numbers, 92.2% at 32, 78.6% at 16.</li>
<li>Encrypted gallery: 140 GB at 32 numbers, 2.2 TB at 512 (per 100 M).</li></ul></div>
</div>
<div class="card"><b class="sans">Failure cases</b><p>Accuracy collapses at 16 numbers. Encrypted search is orders of magnitude slower than plain search. The server cannot find the best match itself (the key holder must). The 100 M gallery was synthetic, and part (b) shows synthetic prints behave differently from real ones (EER 5.9–10% vs 1.6–3.2% for JIPNet).</p></div>
<div class="tablewrap card"><table><tr><th colspan="2">Billion-scale applicability</th></tr>
<tr><td><b>Design choice</b></td><td>The <b>privacy layer</b> for the first stage: the gallery of codes stays encrypted.</td></tr>
<tr><td><b>Latency</b></td><td>Scaled to 14.5 B (×145): ≈ 30 hours per query on 10 cores at 32 numbers for a full scan <span class="tag illus">derived</span>. Workable only if an index narrows the scan: scanning 1% of the gallery is ≈ 18 min.</td></tr>
<tr><td><b>Failure cases at scale</b></td><td>≈ 20 TB of encrypted gallery at 32 numbers (140 GB × 145) <span class="tag illus">derived</span>. Accuracy on real prints at this size is untested.</td></tr></table></div>

<h3>a.4 · Proposed architecture: a staged pipeline combining the three</h3>
<p>None of the three papers addresses billion-scale deduplication end to end. Each addresses one stage; the findings of part (b) add two low-cost pre-processing checks.</p>
<figure class="fig card"><svg viewBox="0 0 900 300" role="img" aria-label="Proposed funnel">
<rect x="10" y="95" width="120" height="80" rx="8" class="boxb"/><text x="70" y="124" text-anchor="middle" class="d-ink" font-size="12.5">new enrolee</text><text x="70" y="142" text-anchor="middle" class="d-soft" font-size="11">10 fingers</text><text x="70" y="158" text-anchor="middle" class="d-soft" font-size="11">(+ iris, face)</text>
<path d="M135 135 L160 135" class="arrow"/>
<rect x="165" y="85" width="150" height="100" rx="10" class="box"/><text x="240" y="112" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">0 · Capture checks</text><text x="240" y="132" text-anchor="middle" class="d-soft" font-size="11">print coverage high enough?</text><text x="240" y="149" text-anchor="middle" class="d-soft" font-size="11">else re-capture</text><text x="240" y="166" text-anchor="middle" class="d-soft" font-size="11">orient to a common frame</text>
<path d="M320 135 L345 135" class="arrow"/>
<rect x="350" y="70" width="190" height="130" rx="10" class="boxb"/><text x="445" y="97" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">1 · Candidate retrieval</text><text x="445" y="117" text-anchor="middle" class="d-soft" font-size="11">AFR-Net-style code, 32–64 nos.</text><text x="445" y="134" text-anchor="middle" class="d-soft" font-size="11">encrypted (HERS)</text><text x="445" y="151" text-anchor="middle" class="d-soft" font-size="11">index → scan part of gallery</text><text x="445" y="168" text-anchor="middle" class="d-soft" font-size="11">14.5 B → candidate list</text>
<path d="M545 135 L570 135" class="arrow"/>
<rect x="575" y="70" width="180" height="130" rx="10" class="boxo"/><text x="665" y="97" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">2 · Candidate verification</text><text x="665" y="117" text-anchor="middle" class="d-soft" font-size="11">JIPNet / AFR-Net realignment</text><text x="665" y="134" text-anchor="middle" class="d-soft" font-size="11">in a protected zone</text><text x="665" y="151" text-anchor="middle" class="d-soft" font-size="11">pose consistency check</text><text x="665" y="168" text-anchor="middle" class="d-soft" font-size="11">fuse all fingers</text>
<path d="M760 135 L785 135" class="arrow"/>
<rect x="790" y="95" width="100" height="80" rx="10" class="box"/><text x="840" y="124" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">3 · Decide</text><text x="840" y="142" text-anchor="middle" class="d-soft" font-size="11">unique /</text><text x="840" y="158" text-anchor="middle" class="d-soft" font-size="11">duplicate / review</text>
<text x="450" y="245" text-anchor="middle" class="d-soft" font-size="12">Stages are ordered from lowest to highest cost per comparison. Templates remain encrypted wherever they are stored or searched.</text>
</svg><figcaption>Proposed pipeline. The evidence for each stage is given in the table below.</figcaption></figure>
<div class="tablewrap card"><table>
<tr><th>Stage</th><th>Why it is there (evidence)</th><th>Time per new enrolee</th></tr>
<tr><td>0 · Capture checks</td><td>Blank area causes misses: JIPNet EER on FVC2002 DB1 halves (2.5% → 1.1%) when both patches are ≥ 70% fingerprint. The AFR-Net-style code fails when patches are turned. Synthetic prints score highly against each other, so capture should include presentation-attack (liveness) detection. (part b)</td><td>Negligible: a texture test per image.</td></tr>
<tr><td>1 · Candidate retrieval</td><td>Only a stored, fixed-length template scales to 14.5 B records (a.1). Encryption protects the gallery (a.3). Shortening to 32–64 numbers keeps accuracy (HERS: rank-1 92.2–92.3%).</td><td>Plain codes: ≈ {t768:.0f} core-minutes for a full scan, divided across cores and further by an index. Encrypted: ≈ 18 min on 10 cores if the index limits the scan to 1% <span class="tag illus">derived</span>.</td></tr>
<tr><td>2 · Candidate verification</td><td>Pairwise models are the most accurate on partial prints (JIPNet EER 2.2–3.0% vs 36.5–37.5% for the code alone on the same pairs, part b), but affordable only on a candidate list. A pose consistency check addresses the rare high-confidence 180° errors.</td><td>1,000 candidates: ≈ 29 s with AFR-Net realignment, ≈ {jip_sec*1000/60:.0f} min with JIPNet on the test machine (faster on a GPU).</td></tr>
<tr><td>3 · Decide</td><td>Per-finger errors are too high for 14.5 B on their own (see “False-match accumulation” above). Fusion across 10 fingers combines independent evidence.</td><td>Negligible.</td></tr>
</table></div>

<h2><span class="n">PART (b)</span>JIPNet and an AFR-Net-style baseline on FVC2002/2004 partial prints</h2>
<h3>Setup</h3>
<ul>
<li><b>Data:</b> FVC2002 and FVC2004, all 8 databases (100 fingers × 8 impressions each; 6 real sensors, 2 synthetic).</li>
<li><b>Partial pairs</b>, built with the JIPNet authors’ own patch recipe: 160×160 patches, centres up to 100 px apart, random turn −180° to 180°. Per database: 800 genuine pairs (both patches from the same impression with a known shift and turn, so the true pose is exact) and 2,400 impostor pairs (different fingers).</li>
<li><b>JIPNet:</b> official code and weights; reproduces the authors’ saved outputs to 2 decimals. Run on all pairs of all 8 databases.</li>
<li><b>AFR-Net-style baseline:</b> AFR-Net weights released by the JIPNet authors, used as a fixed-length code (details below). Run on a subsample: 400 genuine pairs each from FVC2002 DB1 and DB3 (the JIPNet paper’s test sets), against all different-finger combinations.</li>
<li><b>Test machine:</b> 2-core Intel Xeon CPU at 2.1 GHz, no GPU. All measured timings refer to this machine.</li>
<li><b>Reading the numbers:</b> both patches of a genuine pair come from one capture, so the results are a best case for real cross-capture matching. JIPNet’s training data included FVC2004 DB1/DB2.</li>
</ul>

<h3>b.1 · EER and TAR at FAR 0.1% (JIPNet, all databases)</h3>
<div class="tablewrap card"><table>
<tr><th>Database</th><th>EER</th><th>TAR at FAR 0.1%</th></tr>{rows}</table>
<p class="src" style="margin-top:8px"><span class="tag derived">measured</span>Brackets: range covering 90% of random re-draws of the pairs. With 2,400 impostors, FAR 0.1% allows 2 false accepts, so TAR moves more than EER. The paper reports EER 1% and TAR 93% on FVC2002 DB1 <span class="tag paper">JIPNet Table V</span>; the gap is explained under b.4.</p></div>
<figure class="fig"><img src="{img('b_bars.png')}"><figcaption>EER and TAR at FAR 0.1% per database.</figcaption></figure>
<figure class="fig"><img src="{img('b_hills_JIPNet.png')}"><figcaption>JIPNet score distributions: same-finger (green) vs different-finger (red) pairs. The less the green and red overlap, the better the separation.</figcaption></figure>

<h3>b.2 · JIPNet vs the AFR-Net-style baseline</h3>
<p>The baseline uses AFR-Net as designed: each patch becomes one fixed-length code (CNN 384 + attention 384 numbers, with AFR-Net’s built-in alignment), and two patches are compared by code similarity, score = 0.2 × CNN + 0.8 × attention (as in the JIPNet authors’ AFR-Net script).</p>
<div class="tablewrap card"><table><tr><th>Database</th><th>JIPNet EER / TAR, measured (800 genuine)</th><th>JIPNet, measured (same 400 genuine)</th><th>JIPNet, paper</th><th>AFR-Net-style, measured</th><th>AFR-Net*, paper</th></tr>{afr_rows()}</table>
<p class="src" style="margin-top:8px"><span class="tag derived">measured</span>Same genuine pairs for both models. The baseline compares every code with every other, giving about 160,000 impostor pairs per database, so its TAR at FAR 0.1% is measured more precisely. Paper columns: JIPNet Table V, 160×160 patches, same ±180° turn protocol. The paper’s AFR-Net* adds PFVNet’s AlignNet in front of AFR-Net, because, in the authors’ words, AFR-Net’s own pose rectification “cannot be performed as expected” on partial prints and its discriminative ability “is not satisfactory in this small image patch scenario”. <span class="tag paper">JIPNet §V, Table V</span></p></div>
<p><b>Agreement with the paper.</b> Both measured models land close to the JIPNet paper’s own figures on the same two test sets: JIPNet strong, AFR-Net weak on small rotated patches. The large gap between them is therefore a property of the methods, not of this experiment.</p>
{f'<figure class="fig"><img src="{img("b_afr_vs_jip_hills.png")}"><figcaption>Score distributions: baseline (top) vs JIPNet (bottom). The baseline’s same-finger and different-finger score distributions overlap substantially.</figcaption></figure>' if img("b_afr_vs_jip_hills.png") and AF else ""}
{rot_html}

<h3>b.3 · JIPNet pose error</h3>
<div class="tablewrap card"><table><tr><th>Database</th><th>Turn error, typical</th><th>Turn error, worst 10%</th><th>Shift error, typical</th><th>Shift error, worst 10%</th></tr>{prow}</table>
<p class="src" style="margin-top:8px"><span class="tag derived">measured</span>Genuine pairs only. Typical = median; worst 10% = 9 in 10 pairs do better. 1 px = 0.0508 mm at 500 dpi. The paper shows pose error only as a figure.</p></div>
<figure class="fig"><img src="{img('b_pose.png')}"><figcaption>Share of genuine pairs whose turn or shift error stays below a given value.</figcaption></figure>
<figure class="fig"><img src="{img('b_pose_overlap.png')}"><figcaption>Pose error grows as the two patches overlap less.</figcaption></figure>

<h3>b.4 · Failure analysis</h3>
<div class="tablewrap card"><table>
<tr><th>Failure</th><th>Evidence</th></tr>
<tr><td><b>Blank area in the patch</b> (JIPNet)</td><td>On FVC2002 DB1, EER falls from 2.5% to 1.1%, the paper’s level, when both patches are at least 70% fingerprint. This explains the gap to the paper’s figure. Missed same-finger pairs are often half-empty patches.</td></tr>
<tr><td><b>Small overlap</b> (JIPNet)</td><td>Pose error and missed matches rise steadily as overlap falls.</td></tr>
<tr><td><b>Synthetic prints</b> (JIPNet)</td><td>On both SFinGe databases the different-finger scores spread across the whole range, and false alarms occur on full, clean patches. EER is 5.9–10% vs 1.6–3.2% on real sensors.</td></tr>
<tr><td><b>Upside-down pose</b> (JIPNet)</td><td>≈180° turn errors in at most 1.3% of genuine pairs (0–0.2% on real sensors), while the match score stays high (median ≈ 0.9).</td></tr>
<tr><td><b>Rotation</b> (AFR-Net-style)</td><td>The code loses the match as the patches are turned apart: {rot_line}. JIPNet estimates the turn itself, so its scores stay high at every angle.</td></tr>
<tr><td><b>Synthetic prints: security implication</b> (interpretation)</td><td>JIPNet gives high scores to pairs of <em>different</em> synthetic fingers, so generated ridge texture alone can raise similarity. This was not tested against real galleries, but it points to a vulnerability rather than a safeguard. Partial-print matchers are the known target of synthetic “MasterPrint” attacks, where one generated print matches many enrolled users (Roy, Memon, Ross 2017; Bontrager et al., IEEE BTAS 2018). Resistance to synthetic prints should be tested explicitly before deployment.</td></tr>
<tr><td><b>Sensor change</b></td><td>No clear failure: on real sensors JIPNet never saw (optical FVC2002 DB2, thermal FVC2004 DB3) it performs like on the paper’s test sets.</td></tr>
</table></div>
<figure class="fig"><img src="{img('b_coverage.png')}"><figcaption>Errors fall when both patches are mostly fingerprint.</figcaption></figure>
<figure class="fig"><img src="{img('b_fail_JIPNet_2002_DB1.png')}"><figcaption>JIPNet worst cases, FVC2002 DB1: missed matches, false alarms, biggest turn errors.</figcaption></figure>
<figure class="fig"><img src="{img('b_fail_JIPNet_2002_DB4.png')}"><figcaption>JIPNet worst cases, synthetic FVC2002 DB4.</figcaption></figure>
{f'<figure class="fig"><img src="{img("b_fail_AFR_2002_DB1.png")}"><figcaption>AFR-Net-style baseline worst cases, FVC2002 DB1.</figcaption></figure>' if img("b_fail_AFR_2002_DB1.png") and AF else ""}

<footer><b>Sources</b>: AFR-Net, arXiv 2211.13897 · JIPNet, arXiv 2405.03959; code github.com/XiongjunGuan/JIPNet · HERS, arXiv 2003.12197 · MasterPrint: Roy, Memon, Ross, arXiv 1705.07386 · DeepMasterPrints: Bontrager et al., IEEE BTAS 2018 · UIDAI enrolment dashboard (145.32 crore Aadhaar) · FVC2002/2004, bias.csr.unibo.it.</footer>
</div></body></html>"""

open(f"{P}/reports/UIDAI_report.html", "w").write(html)
print("ok", len(html))
