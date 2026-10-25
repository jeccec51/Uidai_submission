"""Concise submission report (A4). Reuses the computed tables/figures from build_final.py."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from paths import ROOT
import json, pandas as pd
P = ROOT + ""
code = open(f"{P}/src/build_final.py").read(); code = code[:code.index("jip_years = ")] if "jip_years = " in code else code
ns = {"__file__": __file__}; exec(code[:code.rindex('html = f')] if 'html = f' in code else code, ns)
g = ns; img, style = g["img"], g["style"]; SVG_AFR, SVG_JIP, SVG_HERS = g["SVG_AFR"], g["SVG_JIP"], g["SVG_HERS"]
for x, y in [("find shift/turn", "estimate shift and rotation"), ("hide non-overlap", "mask non-overlap"), ("CNN head → 384", "384 numbers"), ("12 transformer blocks → 384", "12 transformer blocks · 384 numbers"),
             ("code (768)", "template (768)"), ("compare", "similarity"), ("shrink to", "reduce to"), ("many codes packed into one box,", "templates packed into ciphertexts"),
             ("multiplied in one go (“batching”)", "and multiplied together (batching)"), ("P(same finger)", "match probability"), ("cos θ, sin θ, shift x, shift y", "rotation and translation")]:
    SVG_AFR, SVG_JIP, SVG_HERS = SVG_AFR.replace(x, y), SVG_JIP.replace(x, y), SVG_HERS.replace(x, y)
rows, prow, afr_rows = g["rows"], g["prow"], g["afr_rows"]; AF, ROT, bench = g["AF"], g["ROT"], g["bench"]
N = 14.532e9; t768 = bench["768"] * N / 60
S = pd.read_csv(f"{P}/experiments/summary.csv"); J = S[(S.model == "JIPNet") & ~S.db.str.startswith("ALL")].set_index("db")
real = J.drop(["2002_DB4", "2004_DB4"]); syn = J.loc[["2002_DB4", "2004_DB4"]]
FL = json.load(open(f"{P}/experiments/flops.json")); jip_sec = J.sec_med.median(); jip_years = N * jip_sec / 3.156e7; AFsec = min(v["sec_per_patch"] for v in AF.values())
def sci(x, k=1):
    e = int(f"{x:e}".split("e")[1]); m = x / 10**e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{k}f} × 10{sup}"
pr = lambda a, b, f: f"{f.format(a)}–{f.format(b)}"
afr_e = [AF[d]["afr_EER"] for d in AF]; jip_same = [AF[d]["jip_same400_EER"] for d in AF]; r1 = ROT["2002_DB1"]
M = '<span class="tag derived">measured</span>'; D = '<span class="tag illus">derived</span>'; Pp = lambda t: f'<span class="tag paper">{t}</span>'
def fig(f, cap):
    return f'<figure class="fig"><img src="{img(f)}"><figcaption>{cap}</figcaption></figure>' if img(f) else ""
app_list = [("b_hills_JIPNet.png", "JIPNet score distributions per database: same-finger (green) and different-finger (red) pairs."),
            ("b_afr_vs_jip_hills.png", "Score distributions on FVC2002 DB1/DB3: AFR-Net-style baseline (top) and JIPNet (bottom)."),
            ("b_pose.png", "JIPNet: share of genuine pairs within a given rotation or translation error."),
            ("b_pose_overlap.png", "JIPNet: pose error by overlap between the two patches."),
            ("b_coverage.png", "Effect of fingerprint coverage within the patch on EER and TAR at FAR 0.1%."),
            ("b_fail_JIPNet_2002_DB1.png", "JIPNet lowest-scoring genuine pairs, highest-scoring impostor pairs and largest rotation errors, FVC2002 DB1."),
            ("b_fail_JIPNet_2002_DB4.png", "As Figure A6, synthetic FVC2002 DB4."),
            ("b_fail_AFR_2002_DB1.png", "AFR-Net-style baseline lowest-scoring genuine and highest-scoring impostor pairs, FVC2002 DB1.")]
appendix = "".join(fig(f, f"<b>Figure A{k}.</b> {c}") for k, (f, c) in enumerate(app_list, 1))
pipeline = '''<svg viewBox="0 0 900 210" role="img" aria-label="Proposed pipeline">
<rect x="10" y="55" width="120" height="80" rx="8" class="boxb"/><text x="70" y="84" text-anchor="middle" class="d-ink" font-size="12.5">new enrolee</text><text x="70" y="102" text-anchor="middle" class="d-soft" font-size="11">10 fingers</text><text x="70" y="118" text-anchor="middle" class="d-soft" font-size="11">(+ iris, face)</text>
<path d="M135 95 L160 95" class="arrow"/>
<rect x="165" y="45" width="150" height="100" rx="10" class="box"/><text x="240" y="72" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">0 · Capture checks</text><text x="240" y="92" text-anchor="middle" class="d-soft" font-size="11">coverage and quality</text><text x="240" y="109" text-anchor="middle" class="d-soft" font-size="11">orientation normalisation</text><text x="240" y="126" text-anchor="middle" class="d-soft" font-size="11">liveness detection</text>
<path d="M320 95 L345 95" class="arrow"/>
<rect x="350" y="30" width="190" height="130" rx="10" class="boxb"/><text x="445" y="57" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">1 · Candidate retrieval</text><text x="445" y="77" text-anchor="middle" class="d-soft" font-size="11">AFR-Net-type template</text><text x="445" y="94" text-anchor="middle" class="d-soft" font-size="11">encrypted search (HERS)</text><text x="445" y="111" text-anchor="middle" class="d-soft" font-size="11">indexed gallery</text><text x="445" y="128" text-anchor="middle" class="d-soft" font-size="11">14.5 B → candidate list</text>
<path d="M545 95 L570 95" class="arrow"/>
<rect x="575" y="30" width="180" height="130" rx="10" class="boxo"/><text x="665" y="57" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">2 · Candidate verification</text><text x="665" y="77" text-anchor="middle" class="d-soft" font-size="11">JIPNet / AFR-Net realignment</text><text x="665" y="94" text-anchor="middle" class="d-soft" font-size="11">secure processing zone</text><text x="665" y="111" text-anchor="middle" class="d-soft" font-size="11">pose consistency check</text><text x="665" y="128" text-anchor="middle" class="d-soft" font-size="11">multi-finger fusion</text>
<path d="M760 95 L785 95" class="arrow"/>
<rect x="790" y="55" width="100" height="80" rx="10" class="box"/><text x="840" y="84" text-anchor="middle" class="d-ink" font-size="12.5" font-weight="700">3 · Decision</text><text x="840" y="102" text-anchor="middle" class="d-soft" font-size="11">unique /</text><text x="840" y="118" text-anchor="middle" class="d-soft" font-size="11">duplicate / review</text>
</svg>'''
html = f'''<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fingerprint Deduplication Review</title>{style}
<style>
.titleblock{{display:flex;justify-content:space-between;font:13px/1.5 system-ui,sans-serif;border-bottom:2px solid #1f2328;padding-bottom:8px;margin-bottom:14px}}
.fig img{{width:100%;height:auto;display:block;background:#fff}} table.kv > tbody > tr > td:first-child, table.kv > tr > td:first-child{{width:120px;font-weight:600;font-family:system-ui,sans-serif}} table.fc{{width:100%;margin:0;font-size:inherit}} table.fc th{{font-size:0.92em;padding:2px 4px}} table.fc td{{padding:3px 4px;font-weight:400;width:auto}} table.fc td:first-child{{width:28%}}
.fig.card svg{{max-height:44mm;width:auto;max-width:100%;margin:0 auto;display:block}}
@page{{size:A4;margin:15mm 14mm}}
@media print{{body{{font-size:11px;line-height:1.42;background:#fff}}.wrap{{max-width:none;padding:0}}header.hero{{padding:0 0 8px;border:0}}
h1{{font-size:21px;margin:4px 0 6px}}h2{{font-size:17px;margin:18px 0 4px}}h3{{font-size:13.5px;margin:12px 0 4px}}.lede{{font-size:12px}}
.card,figure,table,tr{{break-inside:avoid}}.pb{{break-before:page}}table{{font-size:10.5px}}th,td{{padding:3px 6px}}.card{{padding:8px 11px;margin:8px 0}}
.fig{{margin:6px 0}}.fig figcaption{{font-size:10px}}.fig img{{max-height:72mm;object-fit:contain}}h2,h3{{break-after:avoid}}td b{{font-size:11px !important}}p{{margin:5px 0}}.split,.split table.kv{{break-inside:auto !important}}.split table.kv > tbody > tr{{break-inside:avoid}}}}
</style></head><body>
<svg width="0" height="0" style="position:absolute"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink)"/></marker></defs></svg>
<div class="wrap">
<header class="hero"><div class="titleblock"><div><b>Review of AI/ML research for fingerprint deduplication</b><br>AFR-Net · JIPNet · HERS</div><div style="text-align:right">Sreejith Markkassery<br>26 October 2026</div></div>
<h1>AFR-Net, JIPNet and HERS for billion-scale fingerprint deduplication</h1>
<p class="lede"><b>(a)</b> Applicability to billion-scale gallery development: design choices, latency and failure cases, with a proposed pipeline combining the three. <b>(b)</b> JIPNet and an AFR-Net-style baseline on FVC2002/2004 partial prints: EER, TAR at FAR 0.1%, JIPNet pose error and failure analysis.</p>
<p class="src">Sources of figures: {Pp("paper")} reported in the cited paper &nbsp;·&nbsp; {M} measured in this study (test machine described in Part b) &nbsp;·&nbsp; {D} calculated from measured or reported figures</p></header>

<h2>Executive summary</h2>
<div class="card"><p><b>(a) Applicability to a billion-scale gallery</b> (≈ 14.5 billion finger records). The three papers are complementary:</p>
<ul style="margin:2px 0 6px">
<li><b>AFR-Net</b>: a fixed-length template computed once per print and searchable across the full gallery. Role: <b>candidate retrieval</b>.</li>
<li><b>JIPNet</b>: accurate pairwise verification with pose estimation, strongest on partial prints. Role: <b>candidate verification</b>.</li>
<li><b>HERS</b>: search over encrypted templates. Role: <b>template protection</b>.</li>
<li>Combined in a staged pipeline (a.4), they cover retrieval, verification and privacy.</li></ul>
<p><b>(b) Partial-print experiment</b> (FVC2002/2004, 160×160 patches):</p>
<ul style="margin:2px 0">
<li><b>JIPNet accuracy</b> on the six real-sensor databases: EER {pr(real.EER.min(), real.EER.max(), "{:.1%}")}, TAR at FAR 0.1% of {pr(real.TAR_0p1.min(), real.TAR_0p1.max(), "{:.0%}")}, in line with the paper.</li>
<li><b>JIPNet pose error</b> (typical): {pr(J.turn_med.min(), J.turn_med.max(), "{:.0f}")}° in rotation and {pr(J.shift_med.min(), J.shift_med.max(), "{:.0f}")} px ({J.shift_med.min()*0.0508:.1f}–{J.shift_med.max()*0.0508:.1f} mm) in translation.</li>
<li><b>Comparison</b>: JIPNet clearly outperformed the AFR-Net-style fixed-length baseline on the same pairs, as also reported in the JIPNet paper. This confirms the value of pose estimation for partial prints and supports orientation normalisation before template extraction.</li>
<li><b>Main factors affecting accuracy</b>: fingerprint coverage within the patch, overlap between patches, rotation, and synthetic versus real data.</li></ul></div>

<h2>Part (a) · Applicability to billion-scale gallery development</h2>
<p><b>Scale:</b> 145.32 crore Aadhaar holders × 10 fingers ≈ 14.5 billion finger records (UIDAI dashboard). Every new enrolment is compared against all of them.</p><p><b>Implication:</b> search time and the expected number of false matches both grow in proportion to gallery size. A billion-scale system therefore needs (1) a reusable template for fast full-gallery retrieval, (2) a more accurate verifier applied to a short candidate list, and (3) fusion across fingers.</p><p><b>Notation</b> used in the latency and complexity entries: <b>N</b> = number of finger records in the gallery (≈ 14.5 billion); <b>d</b> = numbers per template (template length); <b>K</b> = number of candidates passed to verification; <b>P</b> = local descriptors per print in AFR-Net realignment; <b>n</b> = templates packed into one encrypted block (ciphertext) in HERS; <b>f</b> = fraction of the gallery scanned when an index is used. <b>O(·)</b> describes how cost grows with these quantities, independent of hardware. <b>GFLOP / TFLOP</b> = 10<sup>9</sup> / 10<sup>12</sup> floating-point operations (refer Appendix A0).</p>

<div class="sec"><h3>a.1 · AFR-Net: fixed-length fingerprint representation</h3>
<figure class="fig card">{SVG_AFR}</figure>
<div class="tablewrap card split"><table class="kv">
<tr><td>Approach</td><td><b>Network</b>: shared ResNet50 trunk with a CNN head and an attention head (85 M parameters, trained on 1.3 M images).<br><b>Template</b>: 384 + 384 numbers per print, compared by cosine similarity.<br><b>Realignment</b>: a pairwise refinement applied only to uncertain scores (0.3–0.6; 17.9% of comparisons in the authors’ tests).</td></tr>
<tr><td>Reported results</td><td><b>FVC2002 DB1 / DB2 / DB3</b> (full prints, standard FVC protocol): TAR at FAR 0.1% of 100% / 99.96% / 99.36%.<br><b>NIST SD14</b> (rolled prints): rank-1 identification 99.78%.<br><b>NIST SD27</b> (latent prints vs a 100,000-print gallery): rank-1 identification 53.10%.</td></tr>
<tr><td>Design choice at scale</td><td>The template is computed once per print, stored, indexed and compatible with encryption, which makes it well suited to <b>candidate retrieval</b> over the full gallery.</td></tr>
<tr><td>Latency</td><td><b>Complexity</b> (machine-independent): template creation O(1) per print, ≈ {FL["AFRNet_template_GFLOP"]:.1f} GFLOP (measured operation count); full-gallery search O(N·d) ≈ 2·N·d operations ≈ {2*N*768/1e12:.0f} TFLOP per enrolee at d = 768 (derived); realignment O(K·P²) with P = 196 local descriptors.<br><b>Creating a template</b> (once per print, at enrolment): 8.42 ms on a GPU, as reported in the paper; {AFsec:.1f} s on the CPU test machine (measured).<br><b>Comparing two templates</b>: {bench["768"]*1e9:.0f} ns on one CPU core (measured). Searching all 14.5 billion templates therefore takes ≈ {t768:.0f} minutes on one core (derived), and proportionally less with more cores or an index.<br><b>Realignment</b> (only for uncertain scores): 29.36 ms per pair, as reported, so it is applied to candidate lists only.</td></tr>
<tr><td>Failure cases and impact</td><td><table class="fc"><tr><th>Issue</th><th>Impact at Aadhaar scale</th><th>Mitigation in the pipeline</th></tr><tr><td><b>Very low image quality</b> (authors)</td><td>Worn, dry or wet fingers produce unreliable templates; a duplicate missed at retrieval is not recovered later.</td><td>Quality check and re-capture at enrolment (stage 0); fusion across 10 fingers.</td></tr><tr><td><b>Very small overlap</b> (authors)</td><td>Partial or off-centre placement reduces the shared area between enrolment and query prints.</td><td>Supervised full-print capture at enrolment; JIPNet verification for partial captures (stage 2).</td></tr><tr><td><b>Rotation on small patches</b> (refer Part (b))</td><td>Template similarity drops as rotation grows (EER 36.5% on rotated 160×160 patches; the JIPNet paper also reports 29–35% for AFR-Net on such patches).</td><td>Orientation normalisation before template extraction (stage 0).</td></tr><tr><td><b>Largest evaluated gallery 100,000</b> (authors)</td><td>False matches grow with gallery size; behaviour at 14.5 billion is not yet demonstrated.</td><td>Validation at million-scale before rollout; multi-finger fusion (stage 3).</td></tr></table></td></tr></table></div></div>

<div class="sec"><h3>a.2 · JIPNet: joint pairwise verification and pose estimation</h3>
<figure class="fig card">{SVG_JIP}</figure>
<div class="tablewrap card split"><table class="kv">
<tr><td>Approach</td><td><b>Network</b>: shared CNN encoder for both patches, followed by a transformer with self- and cross-attention between them.<br><b>Output</b>: match probability and relative pose (rotation and translation), estimated jointly.</td></tr>
<tr><td>Reported results</td><td><b>Mixed partial-print test set</b> (160×160 patches): EER 0.04, TAR at FAR 0.1% of 0.71 (VeriFinger: 0.18 and 0.56).<br><b>FVC2002 DB1</b> (160×160 patches): EER 0.01, TAR at FAR 0.1% of 0.93.</td></tr>
<tr><td>Design choice at scale</td><td>A pairwise model with high accuracy on partial prints and an explicit pose output, well suited to <b>candidate verification</b> and to aligning prints for review.</td></tr>
<tr><td>Latency</td><td><b>Complexity</b> (machine-independent): O(K) network runs for K candidates, ≈ {FL["JIPNet_pair_GFLOP"]:.1f} GFLOP per pair (measured operation count); no reusable template, so a full-gallery search would be O(N) runs ≈ {sci(N*FL["JIPNet_pair_GFLOP"]*1e9)} FLOP (derived).<br><b>Comparing two patches</b>: {jip_sec:.2f} s per pair on the CPU test machine (measured); no figure is reported in the paper.<br><b>Verifying a candidate list</b> of 1,000: ≈ {jip_sec*1000/60:.0f} minutes on the test machine (derived), and considerably less on a GPU.</td></tr>
<tr><td>Failure cases and impact</td><td><table class="fc"><tr><th>Issue</th><th>Impact at Aadhaar scale</th><th>Mitigation in the pipeline</th></tr><tr><td><b>Severe defects, weak texture</b> (authors)</td><td>Low-quality enrolments may be rejected by verification even when they are true duplicates.</td><td>Fusion with other fingers (and iris/face); manual review queue for borderline cases.</td></tr><tr><td><b>Skin distortion</b> (authors)</td><td>Pressure and skin changes between enrolments years apart; not covered by the single-capture test (refer Part (b)).</td><td>Validation on pairs from different captures before deployment.</td></tr><tr><td><b>Small overlap</b> (authors; refer Part (b))</td><td>Pose error and missed matches increase as overlap decreases.</td><td>Minimum-coverage check at capture; several patches or fingers per decision.</td></tr><tr><td><b>Occasional 180° pose</b> (refer Part (b))</td><td>At most 1.3% of genuine pairs, with high match scores; could support a wrong alignment during review.</td><td>Pose consistency check across fingers (stage 2).</td></tr><tr><td><b>One-to-one only</b> (authors)</td><td>Cannot search the gallery directly: 14.5 billion pairs would take ≈ {jip_years:,.0f} years on the test machine.</td><td>Used only after candidate retrieval (stage 1).</td></tr></table></td></tr></table></div></div>

<div class="sec"><h3>a.3 · HERS: search over homomorphically encrypted templates</h3>
<figure class="fig card">{SVG_HERS}</figure>
<div class="tablewrap card split"><table class="kv">
<tr><td>Approach</td><td><b>Template</b>: any fixed-length template, reduced to 64, 32 or 16 numbers.<br><b>Encryption</b>: Fan–Vercauteren homomorphic scheme (Microsoft SEAL); similarity is computed without decrypting the gallery or the query.</td></tr>
<tr><td>Reported results</td><td><b>Search time</b> (100 million gallery, 10 CPU cores): 740 s per query at 32 numbers; ≈ 500 s with a two-stage search.<br><b>Accuracy</b> (NIST SD4 vs 100 million): rank-1 92.3% at 192 and 64 numbers, 92.2% at 32, 78.6% at 16.<br><b>Storage</b>: 140 GB at 32 numbers per 100 million templates. Reported as 275× faster than earlier encrypted search.</td></tr>
<tr><td>Design choice at scale</td><td>Templates remain protected at rest and during search, and any fixed-length template can be used, which makes it the <b>privacy layer</b> for candidate retrieval.</td></tr>
<tr><td>Latency</td><td><b>Complexity</b> (machine-independent): O(d·⌈N/n⌉) encrypted multiplications per query, with n = 4,096 templates packed per ciphertext (paper setting): ≈ {sci(32*-(-N//4096))} at d = 32 and N = 14.5 billion (derived); storage O(N·d).<br><b>Reported</b>: 740 s per query for a 100 million gallery on 10 CPU cores (32 numbers per template).<br><b>Scaled to 14.5 billion</b> (× 145): ≈ 30 hours per query for a full scan (derived). With an index that limits the search to 1% of the gallery: ≈ 18 minutes (design target).</td></tr>
<tr><td>Failure cases and impact</td><td><table class="fc"><tr><th>Issue</th><th>Impact at Aadhaar scale</th><th>Mitigation in the pipeline</th></tr><tr><td><b>Accuracy drop at 16 numbers</b> (authors)</td><td>Limits how far templates can be compressed to save time and storage.</td><td>Use 32–64 numbers per template (≈ 20 TB encrypted at 32 numbers for 14.5 billion).</td></tr><tr><td><b>Encrypted search cost</b> (authors)</td><td>A full encrypted scan of 14.5 billion is ≈ 30 hours per query on 10 cores.</td><td>Index that narrows the search (design target 1% of the gallery, ≈ 18 minutes).</td></tr><tr><td><b>Server cannot rank results</b> (authors)</td><td>Scores must be decrypted by the key holder to select candidates.</td><td>Ranking inside the secure processing zone, with controlled key management.</td></tr><tr><td><b>Synthetic evaluation gallery</b> (authors; refer Part (b))</td><td>Synthetic prints behave differently from real ones (JIPNet EER 5.9–10% vs 1.6–3.2%), so real-print accuracy at scale is unverified.</td><td>Validation on a large real-print gallery.</td></tr></table></td></tr></table></div></div>

<div><h3>a.4 · Proposed pipeline</h3>
<figure class="fig card">{pipeline}</figure>
<div class="tablewrap card"><table><tr><th>Stage</th><th>Supporting evidence</th><th>Time per enrolee</th></tr>
<tr><td>0 · Capture checks</td><td><b>Coverage:</b> JIPNet EER on FVC2002 DB1 improves from 2.5% to 1.1% when both patches are at least 70% fingerprint.<br><b>Orientation:</b> fixed-length templates perform best on consistently oriented prints.<br><b>Liveness:</b> synthetic prints score highly against each other (refer Part (b)).</td><td>Negligible</td></tr>
<tr><td>1 · Candidate retrieval</td><td><b>Scalability:</b> fixed-length templates can be searched across the full gallery (a.1).<br><b>Privacy:</b> templates can stay encrypted during search (a.3).<br><b>Compactness:</b> 32–64 numbers per template retain accuracy (HERS).</td><td>Unencrypted: ≈ {t768:.0f} minutes on one core for a full scan, reduced by more cores and indexing. Encrypted: ≈ 18 minutes on 10 cores if an index limits the search to 1% (derived).</td></tr>
<tr><td>2 · Candidate verification</td><td><b>Accuracy on partial prints:</b> on the same pairs, JIPNet EER {pr(min(jip_same), max(jip_same), "{:.1%}")} versus {pr(min(afr_e), max(afr_e), "{:.1%}")} for a fixed-length template (refer Part (b)).<br><b>Pose output:</b> supports alignment checks and examiner review.</td><td>For 1,000 candidates: ≈ 29 s with AFR-Net realignment; ≈ {jip_sec*1000/60:.0f} minutes with JIPNet on the CPU test machine.</td></tr>
<tr><td>3 · Decision</td><td><b>Multi-finger fusion:</b> combining 10 fingers adds independent evidence and reduces false matches at gallery scale.</td><td>Negligible</td></tr></table></div>
<div class="tablewrap card"><b class="sans">Machine-independent cost per new enrolee</b>
<table><tr><th>Step</th><th>Complexity</th><th>Operations at N = 14.5 billion</th></tr>
<tr><td>Template creation (AFR-Net)</td><td>O(1) per print</td><td>≈ {FL["AFRNet_template_GFLOP"]:.1f} GFLOP per print, × 10 fingers</td></tr>
<tr><td>Full-gallery search, plain</td><td>O(N·d)</td><td>≈ {2*N*768/1e12:.0f} TFLOP at d = 768; ≈ {2*N*32/1e12:.1f} TFLOP at d = 32</td></tr>
<tr><td>Indexed search (scans a fraction f)</td><td>O(f·N·d)</td><td>f = 1%: ≈ {2*N*32*0.01/1e9:.0f} GFLOP at d = 32</td></tr>
<tr><td>Full-gallery search, encrypted (HERS)</td><td>O(d·⌈N/n⌉) ciphertext multiplications</td><td>≈ {sci(32*-(-N//4096))} at d = 32, n = 4,096</td></tr>
<tr><td>Candidate verification (JIPNet)</td><td>O(K)</td><td>K = 1,000: ≈ {1000*FL["JIPNet_pair_GFLOP"]/1e3:.1f} TFLOP</td></tr>
<tr><td>Decision (fusion)</td><td>O(K·fingers)</td><td>negligible</td></tr></table>
<p class="src">N = gallery size (finger records), d = numbers per template, K = candidate-list length, n = templates per ciphertext. GFLOP/TFLOP = 10<sup>9</sup>/10<sup>12</sup> floating-point operations; network counts measured with PyTorch’s operation counter (AFR-Net: the JIPNet authors’ re-implementation at 224×224 input). These counts are independent of hardware; the wall-clock times above are one example on the test machine. Note that verifying 1,000 candidates with JIPNet requires more operations than a plain scan of the entire gallery at d = 32, which is why the candidate list must remain short.</p></div>
<div class="card"><b class="sans">Operating modes</b>
<ul style="margin:4px 0 6px"><li><b>Full-print enrolment</b> (supervised capture of full impressions): AFR-Net reports TAR at FAR 0.1% of 99.36–100% on full FVC2002 prints, where its built-in alignment can orient the print. An AFR-Net-type template can therefore handle both retrieval and verification, entirely in encrypted form.</li>
<li><b>Partial or low-quality captures</b> (small sensors, poor placement): JIPNet verifies the candidate list inside the secure processing zone, where its pose estimation gives the largest benefit (refer Part (b), section b.2).</li></ul>
<table><tr><th></th><th>AFR-Net-type template</th><th>JIPNet</th></tr>
<tr><td>Stored per finger</td><td>768 numbers ≈ 3 KB (float32)</td><td>Fingerprint image (e.g. 388×374 px ≈ 145 KB uncompressed)</td></tr>
<tr><td>Full gallery, 14.5 B fingers</td><td>≈ 45 TB unencrypted; ≈ 20 TB encrypted at 32 numbers (derived)</td><td>Image archive for all fingers</td></tr>
<tr><td>Matching in encrypted form</td><td>Supported (HERS)</td><td>Requires decrypted images, within the secure zone</td></tr></table></div></div>

<h2>Part (b) · JIPNet and AFR-Net-style baseline on FVC2002/2004</h2>
<div class="card"><ul style="margin:2px 0">
<li><b>Data:</b> FVC2002 and FVC2004, 8 databases (6 real sensors, 2 synthetic), each with 100 fingers × 8 impressions.</li>
<li><b>Test pairs:</b> 160×160 patches, following the JIPNet protocol (patch centres up to 100 px apart, rotation between −180° and 180°). Per database: 800 genuine pairs (same finger) and 2,400 impostor pairs (different fingers).</li>
<li><b>Ground-truth pose:</b> both patches of a genuine pair are cut from one impression with a known shift and rotation, so the true pose is exact. Results are therefore an upper bound for pairs taken from different captures.</li>
<li><b>JIPNet:</b> official code and weights; outputs reproduced to 2 decimals on the authors’ examples. Run on all pairs of all 8 databases.</li>
<li><b>AFR-Net-style baseline:</b> AFR-Net weights released by the JIPNet authors, used as a fixed-length template (score = 0.2 × CNN similarity + 0.8 × attention similarity). Run on a subsample: 400 genuine pairs each from FVC2002 DB1 and DB3, against all different-finger combinations.</li>
<li><b>Metrics:</b> <i>EER</i>, the error rate at the threshold where false accepts equal false rejects (lower is better). <i>TAR at FAR 0.1%</i>, the share of genuine pairs accepted when only 1 in 1,000 impostor pairs is accepted (higher is better). <i>Pose error</i>, the difference between estimated and true rotation (degrees) and translation (pixels).</li>
<li><b>Test machine:</b> 2-core Intel Xeon CPU (2.1 GHz), no GPU. Measured timings refer to this machine; a GPU server is considerably faster.</li></ul></div>

<h3>b.1 · EER and TAR at FAR 0.1% (JIPNet)</h3>
<div class="tablewrap card"><table><tr><th>Database</th><th>EER</th><th>TAR at FAR 0.1%</th></tr>{rows}</table>
<p class="src">Values in brackets: range across random re-draws of the test pairs (90% interval).</p><p><b>Comparison with the paper (FVC2002 DB1: EER 1%, TAR 93%; DB3: EER 3%, TAR 67%).</b> TAR at FAR 0.1% is set by the few highest-scoring impostor pairs, so it varies widely between samples (78–92% here on DB1). The remaining difference is explained by blank area in the patches: when both patches are at least 70% fingerprint, JIPNet reaches EER 1.1% and TAR 89% on FVC2002 DB1, in line with the paper. On DB3 the measured TAR (82%) is higher than the paper’s (67%).</p></div>
{fig("b_bars.png", "EER and TAR at FAR 0.1% per database.")}

<h3>b.2 · Comparison with the AFR-Net-style baseline</h3>
<div class="tablewrap card"><table><tr><th>Database</th><th>JIPNet, measured</th><th>JIPNet, paper</th><th>AFR-Net-style, measured</th><th>AFR-Net, paper*</th></tr>{"".join(f"<tr><td>{g['NAME'][d]}</td><td><b>{AF[d]['jip_same400_EER']:.1%}</b> / {AF[d]['jip_same400_TAR']:.0%}</td><td>{ {'2002_DB1':'1% / 93%','2002_DB3':'3% / 67%'}[d] }</td><td><b>{AF[d]['afr_EER']:.1%}</b> / {AF[d]['afr_TAR']:.0%}</td><td>{ {'2002_DB1':'29% / 3%','2002_DB3':'35% / 1%'}[d] }</td></tr>" for d in ["2002_DB1","2002_DB3"])}</table>
<p class="src">Each cell: EER / TAR at FAR 0.1%. Measured values use the same 400 genuine pairs per database for both models. Paper values: JIPNet paper, Table V, same protocol. *The paper’s AFR-Net result adds a separate alignment network in front of AFR-Net.</p></div>
<ul>
<li><b>Agreement with the paper:</b> both measured models are close to the values reported in the JIPNet paper. The paper likewise reports a high EER for AFR-Net on these partial patches (29% on DB1, 35% on DB3), even with an added alignment network; AFR-Net’s near-perfect published results refer to full prints.</li>
<li><b>Effect of rotation:</b> the fixed-length template is not rotation-invariant. On FVC2002 DB1 its typical genuine score is {r1[0]["afr_med"]:.2f} for rotations under 15° and {r1[-1]["afr_med"]:.2f} for rotations of 120–180°. JIPNet estimates the rotation itself and stays near 0.95 at all angles.</li></ul>
{fig("b_afr_rotation.png", "Typical genuine score by true rotation between the patches.")}

<h3>b.3 · JIPNet pose error</h3>
<div class="tablewrap card"><table><tr><th>Database</th><th>Rotation, typical</th><th>Rotation, worst 10%</th><th>Translation, typical</th><th>Translation, worst 10%</th></tr>{prow}</table>
<p class="src">Genuine pairs only. Typical = median (half the pairs have a smaller error). Worst 10% = the error exceeded by only 1 pair in 10. 1 px = 0.0508 mm at 500 dpi. Pose error grows as overlap decreases (Figure A4).</p></div>

<h3>b.4 · Failure analysis</h3>
<div class="tablewrap card"><table><tr><th>Factor</th><th>Observation and mitigation</th></tr>
<tr><td>Fingerprint coverage</td><td>JIPNet EER on FVC2002 DB1 improves from 2.5% to 1.1%, matching the paper, when both patches are ≥ 70% fingerprint (Figure A5). Lowest-scoring genuine pairs often contain blank regions (Figure A6).</td></tr>
<tr><td>Overlap</td><td>Rotation and translation errors increase as the overlap between patches decreases (Figure A4).</td></tr>
<tr><td>Synthetic prints</td><td>EER {pr(syn.EER.min(), syn.EER.max(), "{:.1%}")} versus {pr(real.EER.min(), real.EER.max(), "{:.1%}")} on real sensors; different synthetic fingers receive high scores (Figure A7). This suggests synthetic galleries are not representative of real performance, and that resistance to synthetic prints (e.g. “MasterPrint” attacks, Roy et al. 2017; Bontrager et al. 2018) should be part of evaluation.</td></tr>
<tr><td>180° pose estimates</td><td>At most 1.3% of genuine pairs (0–0.2% on real sensors), with high match scores; a pose consistency check addresses this.</td></tr>
<tr><td>Rotation (fixed-length)</td><td>Accuracy of the AFR-Net-style template decreases with rotation (refer Part (b), section b.2); orientation normalisation before template extraction addresses this.</td></tr>
<tr><td>Sensor type</td><td>Consistent performance on real sensors not seen in training (optical FVC2002 DB2, thermal FVC2004 DB3).</td></tr></table></div>

<h2 class="pb">Appendix</h2>
<div class="card"><b class="sans">A0 · Note on operation counts (GFLOP)</b>
<p>A <b>FLOP</b> is one floating-point operation: a single addition or multiplication of two decimal numbers. A <b>GFLOP</b> is one billion such operations. For example, creating one AFR-Net template takes ≈ {FL["AFRNet_template_GFLOP"]:.1f} GFLOP, and comparing one JIPNet pair takes ≈ {FL["JIPNet_pair_GFLOP"]:.1f} GFLOP.</p>
<p>The operation count is fixed by the method and does not depend on the hardware, which is why it is used here as a machine-independent measure of cost. Hardware speed is expressed separately in <b>GFLOPS</b> (operations per second). The time for a step is approximately the operation count divided by the hardware speed, so the same count gives shorter times on faster hardware such as a GPU.</p></div>
<h3>Supporting figures</h3>{appendix}
<footer style="margin-top:14px"><b>Sources</b>: AFR-Net, arXiv 2211.13897 · JIPNet, arXiv 2405.03959; code github.com/XiongjunGuan/JIPNet · HERS, arXiv 2003.12197 · Roy, Memon, Ross, arXiv 1705.07386 · Bontrager et al., IEEE BTAS 2018 · UIDAI enrolment dashboard · FVC2002/2004, bias.csr.unibo.it.</footer>
</div></body></html>'''
open(f"{P}/reports/UIDAI_review_submission.html", "w").write(html); print("ok", len(html))
