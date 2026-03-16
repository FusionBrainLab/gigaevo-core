"""Baseline dashboard — polished starting point for evolution.

Strong visual foundation:
- 2-column layout, each run card is wide and spacious
- Prominent fitness chart (160px) with both frontier + mean lines, legend, axes
- 1D archive strip with gradient coloring and occupancy badge
- Clean stat grid with large readable numbers
- Top-5 programs table with syntax-highlighted code preview
- Genealogy tree
- Stagnation pulse animation
- AIRI brand colors throughout

No external CDN. All canvas/CSS inline.
"""

import json


def entrypoint(context: dict) -> str:
    runs = context["runs"]
    framework = context.get("framework", "GigaEvo")
    timestamp = context.get("timestamp", "")

    # ------------------------------------------------------------------
    # Global summary stats
    # ------------------------------------------------------------------
    all_best = [r.get("best_fitness", 0.0) for r in runs]
    global_best = max(all_best) if all_best else 0.0
    total_progs = sum(r.get("total_programs", 0) for r in runs)
    stalled_count = sum(1 for r in runs if r.get("status") == "stalled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _archive_html(cells, total=150):
        occupied = {c["bin"]: c["fitness"] for c in cells}
        max_f = max((c["fitness"] for c in cells), default=1.0)
        min_f = min((c["fitness"] for c in cells), default=0.0)
        rng = max(max_f - min_f, 1e-6)
        bins = []
        for i in range(total):
            if i in occupied:
                t = (occupied[i] - min_f) / rng
                r2 = int(t * 47)
                g2 = int(100 + t * 90)
                b2 = int(100 + t * 73)
                col = f"rgb({r2},{g2},{b2})"
                tip = f"bin {i} | f={occupied[i]:.3f}"
                bins.append(
                    f'<div class="abin occ" style="background:{col}" title="{tip}"></div>'
                )
            else:
                bins.append('<div class="abin"></div>')
        return "".join(bins)

    def _genealogy_html(node, depth=0):
        if not node:
            return '<span class="dim">No data</span>'
        pad = depth * 16
        mut = node.get("mutation") or ""
        mut_span = f'<span class="mut-label">{mut}</span>' if mut else ""
        f_val = node.get("fitness", 0.0)
        teal_val = max(80, int(f_val * 200))
        color = f"rgb(0,{teal_val},{teal_val - 20})"
        line = (
            f'<div class="gnode" style="padding-left:{pad}px">'
            f'<span class="gid" style="color:{color}">{node["id"]}</span>'
            f'<span class="gdim"> gen {node.get("generation","?")} · f={f_val:.3f}</span>'
            f"{mut_span}</div>"
        )
        for p in node.get("parents", []):
            line += _genealogy_html(p, depth + 1)
        return line

    # ------------------------------------------------------------------
    # Per-run cards
    # ------------------------------------------------------------------
    chart_data = []
    cards = []

    for i, run in enumerate(runs):
        status = run.get("status", "running")
        pct = run["current_gen"] / max(run["total_gens"], 1) * 100
        best_f = run.get("best_fitness", 0.0)
        gsi = run.get("gens_since_improvement", 0)
        filled = run.get("archive_filled_cells", 0)
        total_cells = run.get("archive_total_cells", 150)
        occ_pct = run.get("archive_occupancy_pct", 0.0)
        valid_p = run.get("valid_programs", 0)
        invalid_p = run.get("total_programs", 0) - valid_p
        valid_rate = run.get("valid_rate", 0.0)
        val_mean = run.get("validator_mean_s", 0.0)
        val_p95 = run.get("validator_p95_s", 0.0)
        accept_rate = run.get("acceptance_rate", 0.0)

        stag_html = ""
        if gsi >= 5:
            stag_html = (
                f'<div class="stag-warn">⚠ Stalled &mdash; {gsi} generations since improvement</div>'
            )

        # Status color
        if status == "stalled":
            status_color = "#E5434D"
            card_border = "border-color:#E5434D"
            card_anim = "animation:pulse 2s infinite"
        elif status == "complete":
            status_color = "#2FBEAD88"
            card_border = "border-color:#2FBEAD44"
            card_anim = ""
        else:
            status_color = "#2FBEAD"
            card_border = "border-color:#2a3347"
            card_anim = ""

        cid_f = f"cf{i}"
        cid_p = f"cp{i}"

        chart_data.append({
            "cf": cid_f, "cp": cid_p,
            "mean": run.get("gen_fitness_mean", []),
            "frontier": run.get("gen_fitness_frontier", []),
            "valid": run.get("gen_valid_count", []),
            "invalid": run.get("gen_invalid_count", []),
        })

        # Top programs
        top_progs = run.get("top_programs", [])
        rows = []
        for p in top_progs:
            mut = (p.get("mutation") or "—")[:22]
            preview = (p.get("code_preview") or "").replace("<", "&lt;").replace(">", "&gt;")[:80]
            hl = ' style="color:#2FBEAD"' if p["rank"] == 1 else ""
            rows.append(
                f'<tr{hl}>'
                f'<td class="tc">{p["rank"]}</td>'
                f'<td class="tc mono">{p["id"]}</td>'
                f'<td class="tc teal">{p["fitness"]:.4f}</td>'
                f'<td class="tc dim">{p["generation"]}</td>'
                f'<td class="tc dim">{mut}</td>'
                f'<td class="tc mono dim code-cell" title="{preview}">{preview[:50]}</td>'
                f'</tr>'
            )
        top_html = (
            '<table class="prog-table">'
            '<thead><tr>'
            '<th>#</th><th>ID</th><th>Fitness</th><th>Gen</th><th>Mutation</th><th>Code preview</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
        ) if rows else '<div class="dim">No programs yet</div>'

        genealogy = run.get("genealogy", {})
        gen_html = f'<div class="genealogy-tree">{_genealogy_html(genealogy)}</div>'

        archive_bins = _archive_html(run.get("archive_cells", []), total_cells)

        inv_color = "#E5434D" if valid_rate < 0.5 else "#2FBEAD"

        cards.append(f"""
<div class="run-card" style="{card_border};{card_anim}">
  <!-- Card header -->
  <div class="card-header">
    <div class="card-title">
      <span class="run-label" style="color:{status_color}">{run['label']}</span>
      <span class="run-name">{run['name']}</span>
    </div>
    <div class="header-right">
      <span class="best-badge">{best_f:.4f}</span>
      <span class="status-badge" style="border-color:{status_color};color:{status_color}">{status.upper()}</span>
    </div>
  </div>

  {stag_html}

  <!-- Progress -->
  <div class="progress-row">
    <span class="dim">Gen {run['current_gen']} / {run['total_gens']}</span>
    <span class="dim">stagnant: {gsi}g</span>
  </div>
  <div class="progress-bg"><div class="progress-fill" style="width:{pct:.1f}%;background:{status_color}"></div></div>

  <!-- Fitness chart -->
  <div class="section-hdr">
    <span>Fitness Trajectory</span>
    <span class="legend-row">
      <span class="leg-dot" style="background:#2FBEAD"></span><span class="dim">frontier</span>
      <span class="leg-dot" style="background:#2FBEAD55;margin-left:8px"></span><span class="dim">mean</span>
    </span>
  </div>
  <canvas id="{cid_f}" height="160" style="display:block;width:100%"></canvas>

  <!-- Valid/Invalid bar chart -->
  <div class="section-hdr">
    <span>Valid / Invalid per Generation</span>
    <span class="legend-row">
      <span class="leg-dot" style="background:#2FBEAD55"></span><span class="dim">valid</span>
      <span class="leg-dot" style="background:#E5434D55;margin-left:8px"></span><span class="dim">invalid</span>
    </span>
  </div>
  <canvas id="{cid_p}" height="50" style="display:block;width:100%"></canvas>

  <!-- Stats row -->
  <div class="stats-grid">
    <div class="stat-cell"><div class="sv teal">{valid_p}</div><div class="sl">valid</div></div>
    <div class="stat-cell"><div class="sv coral">{invalid_p}</div><div class="sl">invalid</div></div>
    <div class="stat-cell"><div class="sv" style="color:{inv_color}">{valid_rate*100:.0f}%</div><div class="sl">valid rate</div></div>
    <div class="stat-cell"><div class="sv">{accept_rate*100:.0f}%</div><div class="sl">accept rate</div></div>
    <div class="stat-cell"><div class="sv">{val_mean:.1f}s</div><div class="sl">val mean</div></div>
    <div class="stat-cell"><div class="sv">{val_p95:.1f}s</div><div class="sl">val p95</div></div>
  </div>

  <!-- Archive strip -->
  <div class="section-hdr">
    <span>1D Archive &mdash; {filled}/{total_cells} bins ({occ_pct:.0f}%)</span>
  </div>
  <div class="archive-strip">{archive_bins}</div>
  <div class="archive-axis"><span>0.0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span></div>

  <!-- Top 5 programs -->
  <div class="section-hdr"><span>Top 5 Programs</span></div>
  {top_html}

  <!-- Genealogy -->
  <div class="section-hdr"><span>Genealogy (rank-1 ancestor chain)</span></div>
  {gen_html}
</div>
""")

    cards_html = "\n".join(cards)
    chart_json = json.dumps(chart_data)

    css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0f1420; color: #F2EEE9;
    font-family: 'Segoe UI', system-ui, Arial, sans-serif;
    font-size: 13px; padding: 24px;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(229,67,77,0); }
    50% { box-shadow: 0 0 12px 3px rgba(229,67,77,0.35); }
}
/* Global header */
.global-header {
    display: flex; align-items: baseline; gap: 32px;
    padding: 16px 24px; background: #1a2030;
    border: 1px solid #2a3347; border-radius: 10px;
    margin-bottom: 24px;
}
.gh-title { font-size: 22px; font-weight: 700; color: #2FBEAD; letter-spacing: -0.3px; }
.gh-sub { font-size: 11px; color: #555; margin-top: 2px; }
.gh-stats { display: flex; gap: 24px; margin-left: auto; }
.gh-stat { text-align: center; }
.gh-val { font-size: 20px; font-weight: 700; color: #F2EEE9; }
.gh-val.teal { color: #2FBEAD; }
.gh-val.coral { color: #E5434D; }
.gh-lbl { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }
/* Grid */
.runs-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}
/* Card */
.run-card {
    background: #1a2030; border: 1px solid #2a3347;
    border-radius: 10px; padding: 20px; overflow: hidden;
}
.card-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 10px;
}
.card-title { display: flex; flex-direction: column; gap: 2px; }
.run-label { font-size: 18px; font-weight: 800; }
.run-name { font-size: 11px; color: #888; }
.header-right { display: flex; align-items: center; gap: 8px; }
.best-badge {
    font-size: 20px; font-weight: 700; color: #2FBEAD;
    background: #2FBEAD11; border: 1px solid #2FBEAD33;
    padding: 2px 10px; border-radius: 6px;
}
.status-badge {
    font-size: 10px; font-weight: 700; letter-spacing: 0.05em;
    padding: 3px 8px; border-radius: 5px; border: 1px solid;
}
.stag-warn {
    background: #E5434D18; color: #E5434D; border: 1px solid #E5434D44;
    padding: 6px 10px; border-radius: 6px; font-size: 11px; margin-bottom: 10px;
}
/* Progress */
.progress-row { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 11px; }
.progress-bg { background: #252d3d; border-radius: 4px; height: 6px; margin-bottom: 14px; }
.progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
/* Section header */
.section-hdr {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
    color: #555; margin: 14px 0 5px;
}
.legend-row { display: flex; align-items: center; gap: 4px; font-size: 10px; text-transform: none; letter-spacing: normal; }
.leg-dot { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
/* Stats */
.stats-grid {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 6px; margin: 12px 0 4px;
}
.stat-cell { background: #141b28; border-radius: 6px; padding: 6px 4px; text-align: center; }
.sv { font-size: 15px; font-weight: 700; }
.sv.teal { color: #2FBEAD; }
.sv.coral { color: #E5434D; }
.sl { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 1px; }
/* Archive */
.archive-strip {
    display: flex; height: 18px; border-radius: 4px; overflow: hidden;
    margin-bottom: 3px;
}
.abin { flex: 0 0 calc(100% / 150); background: #141b28; }
.abin.occ { cursor: default; }
.archive-axis {
    display: flex; justify-content: space-between;
    font-size: 9px; color: #444; margin-bottom: 10px;
}
/* Programs table */
.prog-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 4px; }
.prog-table th {
    color: #555; font-weight: 500; text-align: left;
    padding: 3px 6px; border-bottom: 1px solid #252d3d;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;
}
.prog-table td { padding: 4px 6px; border-bottom: 1px solid #1a2030; }
.tc { }
.teal { color: #2FBEAD; }
.coral { color: #E5434D; }
.dim { color: #666; }
.mono { font-family: 'Consolas', 'Courier New', monospace; font-size: 10px; }
.code-cell { max-width: 140px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
/* Genealogy */
.genealogy-tree { font-size: 11px; line-height: 1.8; padding: 4px 0; }
.gnode { display: flex; align-items: baseline; gap: 6px; }
.gid { font-family: monospace; font-size: 11px; font-weight: 600; }
.gdim { color: #555; font-size: 10px; }
.mut-label {
    font-size: 9px; color: #888; background: #252d3d;
    padding: 1px 5px; border-radius: 3px; margin-left: 4px;
}
"""

    js = """
var CD = """ + chart_json + """;

function drawFitness(cid, mean, frontier) {
    var cv = document.getElementById(cid);
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var W = cv.parentElement.clientWidth || 600;
    cv.width = W; var H = cv.height;
    ctx.fillStyle = '#141b28'; ctx.fillRect(0,0,W,H);

    var all = mean.concat(frontier).filter(function(v){return v!=null;});
    if (all.length < 2) { ctx.fillStyle='#333'; ctx.fillText('No data',10,H/2); return; }
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var pad = {t:16,r:48,b:24,l:40};
    var cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
    var ylo = Math.max(0, lo - 0.05), yhi = Math.min(1, hi + 0.05);
    var yr = yhi - ylo || 0.01;

    // Grid lines
    ctx.strokeStyle = '#252d3d'; ctx.lineWidth = 1;
    for (var g=0; g<=4; g++) {
        var yv = ylo + (yr * g/4);
        var yp = pad.t + ch - (yv - ylo)/yr * ch;
        ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(pad.l+cw, yp); ctx.stroke();
        ctx.fillStyle='#444'; ctx.font='9px Arial'; ctx.textAlign='right';
        ctx.fillText(yv.toFixed(2), pad.l-4, yp+3);
    }
    // X axis labels
    var nx = Math.min(frontier.length || mean.length, 5);
    var n = Math.max(frontier.length, mean.length, 1);
    ctx.fillStyle='#444'; ctx.font='9px Arial'; ctx.textAlign='center';
    for (var xi=0; xi<=4; xi++) {
        var xv = Math.round(xi*(n-1)/4);
        var xp = pad.l + xv/(n-1)*cw;
        ctx.fillText(xv, xp, H-6);
    }

    function line(arr, col, lw, dash) {
        if (!arr || arr.length < 2) return;
        ctx.strokeStyle=col; ctx.lineWidth=lw;
        if (dash) ctx.setLineDash(dash); else ctx.setLineDash([]);
        ctx.beginPath();
        for (var i=0; i<arr.length; i++) {
            var x = pad.l + i/(arr.length-1)*cw;
            var y = pad.t + ch - (arr[i]-ylo)/yr*ch;
            i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Mean shaded area
    if (mean && mean.length > 1) {
        ctx.fillStyle = 'rgba(47,190,173,0.08)';
        ctx.beginPath();
        for (var i=0; i<mean.length; i++) {
            var x = pad.l + i/(mean.length-1)*cw;
            var y = pad.t + ch - (mean[i]-ylo)/yr*ch;
            i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }
        ctx.lineTo(pad.l+cw, pad.t+ch); ctx.lineTo(pad.l, pad.t+ch);
        ctx.closePath(); ctx.fill();
    }

    line(mean, 'rgba(47,190,173,0.5)', 1.5, [4,3]);
    line(frontier, '#2FBEAD', 2.5);

    // Latest values
    if (frontier && frontier.length > 0) {
        var fy = pad.t + ch - (frontier[frontier.length-1]-ylo)/yr*ch;
        ctx.fillStyle='#2FBEAD'; ctx.font='bold 10px Arial'; ctx.textAlign='left';
        ctx.fillText(frontier[frontier.length-1].toFixed(3), pad.l+cw+3, fy+4);
    }
    if (mean && mean.length > 0) {
        var my = pad.t + ch - (mean[mean.length-1]-ylo)/yr*ch;
        ctx.fillStyle='rgba(47,190,173,0.7)'; ctx.font='10px Arial'; ctx.textAlign='left';
        ctx.fillText(mean[mean.length-1].toFixed(3), pad.l+cw+3, my+4);
    }
}

function drawPop(cid, valid, invalid) {
    var cv = document.getElementById(cid);
    if (!cv) return;
    var ctx = cv.getContext('2d');
    var W = cv.parentElement.clientWidth || 600;
    cv.width = W; var H = cv.height;
    ctx.fillStyle = '#141b28'; ctx.fillRect(0,0,W,H);
    var n = Math.max(valid.length, invalid.length);
    if (n < 1) return;
    var maxV = 0;
    for (var i=0; i<n; i++) maxV = Math.max(maxV, (valid[i]||0)+(invalid[i]||0));
    if (!maxV) return;
    var bw = (W-2)/n;
    for (var i=0; i<n; i++) {
        var v=valid[i]||0, inv=invalid[i]||0;
        var x = 1+i*bw;
        var vH = (v/maxV)*(H-2), iH = (inv/maxV)*(H-2);
        ctx.fillStyle='#2FBEAD66'; ctx.fillRect(x, H-1-vH, bw-1, vH);
        ctx.fillStyle='#E5434D66'; ctx.fillRect(x, H-1-vH-iH, bw-1, iH);
    }
}

window.addEventListener('load', function() {
    for (var i=0; i<CD.length; i++) {
        drawFitness(CD[i].cf, CD[i].mean, CD[i].frontier);
        drawPop(CD[i].cp, CD[i].valid, CD[i].invalid);
    }
});
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GigaEvo Dashboard</title>
<style>{css}</style>
</head>
<body>

<div class="global-header">
  <div>
    <div class="gh-title">GigaEvo Dashboard</div>
    <div class="gh-sub">{framework} &nbsp;·&nbsp; {timestamp}</div>
  </div>
  <div class="gh-stats">
    <div class="gh-stat">
      <div class="gh-val teal">{global_best:.4f}</div>
      <div class="gh-lbl">Global Best</div>
    </div>
    <div class="gh-stat">
      <div class="gh-val">{len(runs)}</div>
      <div class="gh-lbl">Runs</div>
    </div>
    <div class="gh-stat">
      <div class="gh-val">{total_progs}</div>
      <div class="gh-lbl">Programs</div>
    </div>
    <div class="gh-stat">
      <div class="gh-val {'coral' if stalled_count else ''}">{stalled_count}</div>
      <div class="gh-lbl">Stalled</div>
    </div>
  </div>
</div>

<div class="runs-grid">
{cards_html}
</div>

<script>{js}</script>
</body>
</html>"""
