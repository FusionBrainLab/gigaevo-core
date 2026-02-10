import base64
import csv
import io
import json
import re

import dash
from dash import html, dcc, Input, Output, State, ALL
import dash_cytoscape as cyto
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # headless for image generation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pathlib import Path
# =========================
# CONFIG: ONLY SET THIS
# =========================
INPUT_CSV = Path(__file__).parent / "outputs" / "results_circlpack_n26.csv"
OUTPUT_IDEAS_JSON = Path(__file__).parent / "outputs" / "output.json"

# Optional: if you still want the JSON written to disk for debugging / reuse
WRITE_JSON_TO_DISK = False
OUTPUT_JSON = Path(__file__).parent / "visualisation" / "data_mem.json"

# --- VIS CONSTANTS ---
NODE_SIZE = 800
FONT_SIZE = 180
CANVAS_HEIGHT = "85vh"

BASE_POS_MULTIPLIER = 100
PADDING_FACTOR = 200
SPLIT_GAP_FACTOR = 0.35
HIDE_FITNESS_VALUE = -1000.0
MAX_SELECTED_IDEAS = 10
# Border width for nodes highlighted by selected ideas (base node border is 10)
IDEA_HIGHLIGHT_BORDER_WIDTH = 100
# Multiplier for ring thickness in generated image (>= 1; larger = thicker rings, fewer rings shown)
# 1.0 allows up to 4 rings so 3+ ideas show; increase for thicker rings (max 2 rings).
IDEA_RING_THICKNESS_MULTIPLIER = 1.0
# 10 maximally distinct colors (one red, green, blue, orange, purple, cyan, pink, yellow, indigo, teal)
IDEA_HIGHLIGHT_COLORS = [
    "#C62828", "#2E7D32", "#1565C0", "#E65100", "#6A1B9A",
    "#00838F", "#AD1457", "#F9A825", "#303F9F", "#00695C",
]

# Image size in pixels for generated concentric-ring node backgrounds (2+ ideas)
IDEA_RING_IMAGE_SIZE = 256


def _multi_ring_data_url(hex_colors, size_px=IDEA_RING_IMAGE_SIZE):
    """
    Generate a PNG of concentric rings (innermost = first color, outermost = last).
    Supports up to 10 rings; each ring has thickness 1/n so ring size scales with count.
    """
    if not hex_colors:
        return None
    n = min(len(hex_colors), 10)
    hex_colors = hex_colors[:n]
    dpi = max(72, size_px // 2)
    fig_size = size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    # Figure and axes background = outer ring color (avoids black/transparent when saved)
    outer_color = hex_colors[n - 1]
    fig.patch.set_facecolor(outer_color)
    ax.set_facecolor(outer_color)
    # Draw all rings as circles from outer to inner so inner rings sit on top
    for i in range(n - 1, -1, -1):
        r = (i + 1) / n
        circle = Circle((0, 0), r, facecolor=hex_colors[i], edgecolor="none")
        ax.add_patch(circle)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=outer_color)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# =========================================================
# CSV -> NODE DATA (formerly your "first file")
# =========================================================
def parse_id_list(raw: str):
    """
    Parse parent_ids / children_ids field into a list of IDs.
    Handles formats like:
      - "id1,id2"
      - "id1 id2"
      - "['id1', 'id2']"
      - "" or "[]"  -> []
    """
    if raw is None:
        return []

    s = str(raw).strip()
    if not s or s in ("[]", "null", "None"):
        return []

    s = s.strip("[]")
    s = s.replace("'", "").replace('"', "")
    parts = re.split(r"[,\s]+", s)
    return [p for p in parts if p]


def parse_bool(raw):
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "t"):
        return True
    if s in ("0", "false", "no", "n", "f", ""):
        return False
    return False


def row_to_node(row: dict):
    # ID from program_id
    nid = row.get("program_id") or row.get("id")
    if not nid:
        nid = row.get("name", "").strip() or None
    if nid is None:
        return None

    # generation: prefer 'generation', fallback to 'lineage_generation'
    gen_str = (row.get("generation") or "").strip()
    if not gen_str:
        gen_str = (row.get("lineage_generation") or "").strip()

    try:
        generation = int(gen_str) if gen_str else 0
    except ValueError:
        generation = 0

    # fitness from metric_fitness (only if metric_is_valid is true-ish)
    fitness_str = (row.get("metric_fitness") or "").strip()
    is_valid_str = (row.get("metric_is_valid") or "").strip().lower()

    if is_valid_str in ("0", "false", "no", ""):
        fitness = 0.0
    else:
        try:
            fitness = float(fitness_str) if fitness_str else 0.0
        except ValueError:
            fitness = 0.0

    # ancestors / successors from parent_ids / children_ids
    ancestors = parse_id_list(row.get("parent_ids", ""))
    successors = parse_id_list(row.get("children_ids", ""))

    # human-readable text for info panel
    name = row.get("name", "").strip()
    code = row.get("code", "").strip()
    mutation = row.get("lineage_mutation", "").strip()
    iteration = row.get("metadata_iteration", "").strip()
    strategy = row.get("metadata_strategy_name", "").strip()
    home_island = row.get("metadata_home_island", "").strip()
    current_island = row.get("metadata_current_island", "").strip()
    memory_used = parse_bool(row.get("metadata_memory_used") or row.get("memory_used"))
    model_used = (row.get("metadata_model_used") or "").strip()

    original_insights_lines = []
    if name:
        original_insights_lines.append(f"Name: {name}")
    if code:
        original_insights_lines.append("\n--- Code ---\n")
        original_insights_lines.append(code)

    meta_lines = []
    if iteration:
        meta_lines.append(f"Iteration: {iteration}")
    if mutation:
        meta_lines.append(f"Mutation: {mutation}")
    if strategy:
        meta_lines.append(f"Strategy: {strategy}")
    if home_island or current_island:
        meta_lines.append(
            f"Islands: home={home_island or '-'}, current={current_island or '-'}"
        )
    if model_used:
        meta_lines.append(f"Model: {model_used}")

    if meta_lines:
        original_insights_lines.append("\n--- Metadata ---\n")
        original_insights_lines.append("\n".join(meta_lines))

    original_insights = "\n".join(original_insights_lines) if original_insights_lines else ""

    if name:
        summary_insight = f"{name} (fitness={fitness:.4f})"
    else:
        summary_insight = f"Program {nid} (fitness={fitness:.4f})"

    return {
        "id": str(nid),
        "generation": generation,
        "fitness": fitness,
        "ancestors": ancestors,
        "successors": successors,
        "original_insights": original_insights,
        "summary_insight": summary_insight,
        "memory_used": memory_used,
    }


def load_data_from_csv():
    nodes = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = row_to_node(row)
            if node is not None:
                nodes.append(node)

    if WRITE_JSON_TO_DISK:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
            json.dump(nodes, out, ensure_ascii=False, indent=2)
        print(f"Wrote {len(nodes)} nodes to {OUTPUT_JSON}")

    print(f"Loaded {len(nodes)} nodes from CSV: {INPUT_CSV}")
    return nodes


def load_ideas_from_json():
    """
    Load ideas from output.json. Extract description, avg_fitness, programs from each
    record in top_delta_ideas and top_fitness_ideas. Return a single list sorted by
    avg_fitness descending (best first).
    """
    if not OUTPUT_IDEAS_JSON.exists():
        print(f"Ideas JSON not found: {OUTPUT_IDEAS_JSON}")
        return []

    with open(OUTPUT_IDEAS_JSON, encoding="utf-8") as f:
        data = json.load(f)

    ideas = []
    for source_key in ("top_delta_ideas", "top_fitness_ideas"):
        group = data.get(source_key) or {}
        for idea_id, record in group.items():
            ideas.append({
                "id": idea_id,
                "description": record.get("description", ""),
                "avg_fitness": float(record.get("avg_fitness", 0)),
                "programs": list(record.get("programs") or []),
                "source": source_key,
            })
    ideas.sort(key=lambda x: x["avg_fitness"], reverse=True)
    print(f"Loaded {len(ideas)} ideas from {OUTPUT_IDEAS_JSON}")
    return ideas


# =========================================================
# VIS / DASH (formerly your "second file")
# =========================================================
def normalize_fitness(value, min_val, max_val):
    if max_val <= min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def is_hidden_fitness(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return abs(v - HIDE_FITNESS_VALUE) < 1e-9


def compute_memory_lineage(nodes):
    nodes_by_id = {n["id"]: n for n in nodes}
    memo = {}
    visiting = set()

    def has_memory(nid):
        if nid in memo:
            return memo[nid]

        node = nodes_by_id.get(nid)
        if not node:
            memo[nid] = False
            return False

        if nid in visiting:
            # Cycle guard: fall back to the node's own memory flag.
            return bool(node.get("memory_used"))

        visiting.add(nid)
        if node.get("memory_used"):
            memo[nid] = True
        else:
            memo[nid] = any(has_memory(parent_id) for parent_id in node.get("ancestors", []))
        visiting.remove(nid)
        return memo[nid]

    for node_id in nodes_by_id:
        has_memory(node_id)

    return memo


def get_color(norm_value):
    cmap = matplotlib.colormaps["coolwarm"]
    rgba = cmap(norm_value)
    return mcolors.to_hex(rgba)


base_stylesheet = [
    {
        "selector": "node",
        "style": {
            "content": "data(label)",
            "width": f"{NODE_SIZE}px",
            "height": f"{NODE_SIZE}px",
            "font-size": f"{FONT_SIZE}px",
            "text-valign": "center",
            "text-halign": "center",
            "color": "white",
            "text-outline-width": 8,
            "text-outline-color": "#222",
            "border-width": 10,
            "border-color": "#444",
            "font-weight": "bold",
            "background-color": "data(node_color)",
        },
    },
    {
        "selector": "edge",
        "style": {
            "width": 6,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#ccc",
            "target-arrow-color": "#ccc",
        },
    },
]


app = dash.Dash(__name__)

app.layout = html.Div(
    [
        # Store the parsed CSV data here
        dcc.Store(id="data-store", data=load_data_from_csv()),
        dcc.Store(id="ideas-store", data=load_ideas_from_json()),
        dcc.Store(id="selected-ideas-store", data=[]),  # list of idea indices (max 10)
        html.H2(
            "Интерактивный граф",
            style={"textAlign": "center", "fontFamily": "Arial"},
        ),
        html.Div(
            [
                # LEFT (graph)
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button(
                                    "🔄 Обновить данные",
                                    id="refresh-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "10px 20px",
                                        "fontSize": "16px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#2196F3",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginRight": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="highlight-toggle",
                                    options=[
                                        {"label": " Подсветить путь к предкам", "value": "on"}
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "16px",
                                        "display": "inline-block",
                                    },
                                ),
                                dcc.Checklist(
                                    id="auto-idea-toggle",
                                    options=[
                                        {"label": " Показать идеи", "value": "on"}
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "16px",
                                        "display": "inline-block",
                                        "marginLeft": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="memory-toggle",
                                    options=[
                                        {"label": " Показать использование памяти", "value": "on"}
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "16px",
                                        "display": "inline-block",
                                        "marginLeft": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="memory-split-toggle",
                                    options=[
                                        {
                                            "label": " Отсортировать программы в поколении",
                                            "value": "on",
                                        }
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "16px",
                                        "display": "inline-block",
                                        "marginLeft": "20px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="hide-bad-fitness-toggle",
                                    options=[
                                        {
                                            "label": " Скрыть fitness = -1000",
                                            "value": "on",
                                        }
                                    ],
                                    value=[],
                                    style={
                                        "fontFamily": "Arial",
                                        "fontSize": "16px",
                                        "display": "inline-block",
                                        "marginLeft": "20px",
                                    },
                                ),
                            ],
                            style={
                                "padding": "15px",
                                "backgroundColor": "#f1f1f1",
                                "borderRadius": "5px",
                                "marginBottom": "10px",
                            },
                        ),
                        cyto.Cytoscape(
                            id="cytoscape-graph",
                            elements=[],
                            layout={"name": "preset", "fit": True, "padding": 50},
                            style={"width": "100%", "height": "100%"},
                            stylesheet=base_stylesheet,
                            minZoom=0.001,
                            maxZoom=3.0,
                            wheelSensitivity=0.2,
                            responsive=True,
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0", "height": CANVAS_HEIGHT, "position": "relative"},
                ),
                # RIGHT (info + placeholder list)
                html.Div(
                    [
                        # Stats block
                        html.Div(
                            [
                                html.Button(
                                    "📊 Показать статистику",
                                    id="stats-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "8px 14px",
                                        "fontSize": "14px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#4CAF50",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(id="stats-info"),
                                html.Hr(),
                            ]
                        ),
                        # Scrollable vertical clickable placeholder list + toggle
                        html.Div(
                            [
                                html.Button(
                                    "📋 Скрыть список",
                                    id="toggle-list-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 10px",
                                        "fontSize": "13px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#607D8B",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginRight": "8px",
                                    },
                                ),
                                html.Button(
                                    "Снять выделение",
                                    id="deselect-all-ideas-btn",
                                    n_clicks=0,
                                    style={
                                        "padding": "6px 10px",
                                        "fontSize": "13px",
                                        "cursor": "pointer",
                                        "backgroundColor": "#795548",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                    },
                                ),
                                html.Div(
                                    id="clickable-list-container",
                                    children=[],  # filled by callback from ideas-store
                                    style={
                                        "maxHeight": "250px",
                                        "overflowY": "auto",
                                        "marginTop": "10px",
                                        "border": "1px solid #ddd",
                                        "borderRadius": "5px",
                                        "padding": "5px",
                                        "backgroundColor": "#fefefe",
                                    },
                                ),
                                html.Div(
                                    id="list-click-output",
                                    style={
                                        "marginTop": "8px",
                                        "fontStyle": "italic",
                                        "color": "#555",
                                    },
                                ),
                                html.Hr(),
                            ],
                            style={"marginBottom": "10px"},
                        ),
                        # Node info block
                        html.Div(id="node-info", style={"whiteSpace": "pre-wrap"}),
                    ],
                    style={
                        "width": "400px",
                        "flexShrink": "0",
                        "marginLeft": "20px",
                        "border": "1px solid #ddd",
                        "padding": "20px",
                        "borderRadius": "8px",
                        "backgroundColor": "#f9f9f9",
                        "height": CANVAS_HEIGHT,
                        "overflowY": "auto",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "row", "width": "98%", "margin": "0 auto"},
        ),
    ]
)


@app.callback(
    Output("data-store", "data"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(_n):
    # Reload from CSV each time
    return load_data_from_csv()


@app.callback(
    Output("cytoscape-graph", "elements"),
    [
        Input("data-store", "data"),
        Input("memory-split-toggle", "value"),
        Input("hide-bad-fitness-toggle", "value"),
    ],
)
def update_graph_elements(data, memory_split_toggle, hide_bad_fitness_toggle):
    if not data:
        return []

    hide_on = hide_bad_fitness_toggle and "on" in hide_bad_fitness_toggle
    filtered = [d for d in data if not (hide_on and is_hidden_fitness(d.get("fitness", 0)))]
    if not filtered:
        return []

    fits = [d.get("fitness", 0) for d in filtered]
    positive_fits = [max(0.0, f) for f in fits]
    min_fit_scale = 0.0
    max_fit_scale = max(positive_fits) if positive_fits else 1.0
    if max_fit_scale == 0.0:
        max_fit_scale = 1.0

    gen_counts = {}
    for node in filtered:
        g = node.get("generation", 0)
        gen_counts[g] = gen_counts.get(g, 0) + 1
    max_nodes = max(gen_counts.values()) if gen_counts else 1
    dynamic_scale = BASE_POS_MULTIPLIER + (max_nodes * PADDING_FACTOR)

    G = nx.DiGraph()
    for n in filtered:
        G.add_node(n["id"], generation=n["generation"])
    node_ids = {n["id"] for n in filtered}
    for n in filtered:
        for t in n.get("successors", []):
            if t in node_ids:
                G.add_edge(n["id"], t)

    try:
        pos = nx.multipartite_layout(G, subset_key="generation", align="horizontal", scale=3)
    except Exception:
        pos = {}

    split_on = memory_split_toggle and "on" in memory_split_toggle
    split_gap = dynamic_scale * SPLIT_GAP_FACTOR if split_on else 0

    base_positions = {}
    if split_on:
        # When split is on: sort nodes within each generation left-to-right in ascending order (by fitness, then ID)
        nodes_by_gen = {}
        for node in filtered:
            gen = node.get("generation", 0)
            if gen not in nodes_by_gen:
                nodes_by_gen[gen] = []
            nodes_by_gen[gen].append(node)
        # Sort nodes within each generation by fitness (ascending), then by ID for tie-breaking
        for gen in nodes_by_gen:
            nodes_by_gen[gen].sort(key=lambda n: (n.get("fitness", 0), n.get("id", "")))
        # Get y positions from layout (or compute from generation)
        gen_to_y = {}
        for node in filtered:
            nid = node["id"]
            gen = node.get("generation", 0)
            if nid in pos:
                _, y = pos[nid]
                gen_to_y[gen] = y * dynamic_scale
            else:
                gen_to_y[gen] = gen * dynamic_scale
        # Assign x positions sequentially within each generation (left to right)
        # Use a smaller spacing multiplier to avoid stretching the graph too much
        for gen in sorted(nodes_by_gen.keys()):
            nodes_in_gen = nodes_by_gen[gen]
            y_pos = gen_to_y[gen]
            x_spacing = BASE_POS_MULTIPLIER * 12
            x_start = -(len(nodes_in_gen) - 1) * x_spacing / 2
            for idx, node in enumerate(nodes_in_gen):
                nid = node["id"]
                x_pos = x_start + idx * x_spacing
                base_positions[nid] = (x_pos, y_pos)
    else:
        # Normal layout: use networkx positions
        for node in filtered:
            nid = node["id"]
            if nid in pos:
                x, y = pos[nid]
                x_pos, y_pos = x * dynamic_scale, y * dynamic_scale
            else:
                x_pos, y_pos = 0, 0
            base_positions[nid] = (x_pos, y_pos)

    mem_shift = 0
    nonmem_shift = 0
    if split_on:
        mem_xs = [base_positions[n["id"]][0] for n in filtered if n.get("memory_used")]
        nonmem_xs = [base_positions[n["id"]][0] for n in filtered if not n.get("memory_used")]
        if mem_xs and nonmem_xs:
            mem_min = min(mem_xs)
            nonmem_max = max(nonmem_xs)
            if mem_min <= nonmem_max:
                mem_shift = (nonmem_max - mem_min) + split_gap
            else:
                mem_shift = split_gap

    elements = []
    for node in filtered:
        nid = node["id"]
        raw_fit = node.get("fitness", 0)

        val_for_color = max(0.0, raw_fit)
        norm_val = normalize_fitness(val_for_color, min_fit_scale, max_fit_scale)
        hex_color = get_color(norm_val)

        x_pos, y_pos = base_positions[nid]
        if split_on and mem_shift:
            if node.get("memory_used"):
                x_pos += mem_shift
            else:
                x_pos += nonmem_shift

        elements.append(
            {
                "data": {
                    "id": nid,
                    "label": f"{raw_fit:.3f}",
                    "full_data": node,
                    "node_color": hex_color,
                    "memory_used": node.get("memory_used", False),
                },
                "position": {"x": x_pos, "y": y_pos},
                "classes": "node-base memory-used" if node.get("memory_used") else "node-base",
            }
        )

    for node in filtered:
        src = node["id"]
        for tgt in node.get("successors", []):
            if tgt in node_ids:
                elements.append(
                    {"data": {"id": f"{src}-{tgt}", "source": src, "target": tgt}, "classes": "edge-base"}
                )

    return elements


@app.callback(
    Output("node-info", "children"),
    Input("cytoscape-graph", "tapNodeData"),
    State("data-store", "data"),
)
def show_info(node_data, all_data):
    if not node_data:
        return html.Div([html.H3("Инфо"), html.P("Нажмите на узел.", style={"color": "#777"})])

    d = node_data.get("full_data", {})
    val = d.get("fitness", 0)

    if all_data:
        max_f = max([x.get("fitness", 0) for x in all_data])
        threshold = max(0, max_f) / 2
    else:
        threshold = 0.5

    txt_color = "red" if val > threshold else "blue"

    return html.Div(
        [
            html.H3(f"Узел: {d.get('id')}", style={"borderBottom": "2px solid #333"}),
            html.P([html.Strong("Generation: "), str(d.get("generation"))]),
            html.P(
                [
                    html.Strong("Fitness: "),
                    html.Span(
                        f"{val:.3f}",
                        style={"color": txt_color, "fontSize": "1.2em", "fontWeight": "bold"},
                    ),
                ]
            ),
            html.P(
                [
                    html.Strong("Memory used: "),
                    "yes" if d.get("memory_used") else "no",
                ]
            ),
            html.Hr(),
            html.P([html.Strong("Ancestors: "), ", ".join(d.get("ancestors", [])) or "-"]),
            html.P([html.Strong("Successors: "), ", ".join(d.get("successors", [])) or "-"]),
            html.Hr(),
            html.H4("Original Insights"),
            html.Div(
                d.get("original_insights", ""),
                style={"fontStyle": "italic", "color": "#555", "marginBottom": "10px"},
            ),
            html.H4("Summary Insight"),
            html.Div(
                d.get("summary_insight", ""),
                style={"backgroundColor": "#e3f2fd", "padding": "10px", "borderRadius": "5px"},
            ),
        ]
    )


@app.callback(
    Output("clickable-list-container", "children"),
    Input("ideas-store", "data"),
    Input("selected-ideas-store", "data"),
)
def build_ideas_list(ideas, selected_ideas):
    """Build the scrollable list of idea buttons from ideas-store (sorted by avg_fitness)."""
    if not ideas:
        return [html.Div("Нет идей в output.json", style={"color": "#777", "padding": "8px"})]
    sel_list = selected_ideas if selected_ideas is not None else []
    selected_set = set(sel_list)
    max_desc_len = 55
    buttons = []
    for i, idea in enumerate(ideas):
        desc = (idea.get("description") or "").strip()
        short = (desc[:max_desc_len] + "…") if len(desc) > max_desc_len else desc
        avg = idea.get("avg_fitness", 0)
        label = f"{short} (avg: {avg:.3f})"
        is_selected = i in selected_set
        border_color = IDEA_HIGHLIGHT_COLORS[i % len(IDEA_HIGHLIGHT_COLORS)] if is_selected else "#ddd"
        list_style = {
            "width": "100%",
            "textAlign": "left",
            "marginBottom": "4px",
            "padding": "6px 8px",
            "border": f"2px solid {border_color}",
            "borderRadius": "4px",
            "backgroundColor": "#E8F5E9" if is_selected else "#ffffff",
            "cursor": "pointer",
        }
        buttons.append(
            html.Button(
                label,
                id={"type": "idea-btn", "index": i},
                n_clicks=0,
                style=list_style,
            )
        )
    return buttons


@app.callback(
    Output("clickable-list-container", "style"),
    Output("toggle-list-btn", "children"),
    Input("toggle-list-btn", "n_clicks"),
    State("clickable-list-container", "style"),
)
def toggle_placeholder_list(n_clicks, current_style):
    """
    Show / hide the ideas list with a toggle button.
    """
    base_style = dict(current_style or {})
    visible = (n_clicks or 0) % 2 == 0
    base_style["display"] = "block" if visible else "none"
    button_label = "📋 Скрыть список" if visible else "📋 Показать список"
    return base_style, button_label


@app.callback(
    Output("selected-ideas-store", "data"),
    Input("deselect-all-ideas-btn", "n_clicks"),
    Input({"type": "idea-btn", "index": ALL}, "n_clicks"),
    Input("cytoscape-graph", "tapNodeData"),
    Input("auto-idea-toggle", "value"),
    State("selected-ideas-store", "data"),
    State("ideas-store", "data"),
    prevent_initial_call=True,
)
def update_selected_ideas(deselect_n, idea_n_clicks_list, node_data, auto_idea_toggle, selected, ideas):
    """
    Deselect all on button click, toggle single idea when list item is clicked, or
    auto-select ideas containing a tapped node when the auto toggle is on.
    Max MAX_SELECTED_IDEAS.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "deselect-all-ideas-btn":
        return []
    if trigger_id == "cytoscape-graph":
        if not (auto_idea_toggle and "on" in auto_idea_toggle):
            return dash.no_update
        if not ideas:
            return dash.no_update
        node_id = ""
        if node_data:
            node_id = str(node_data.get("id", "")).strip()
        if not node_id:
            return dash.no_update
        matches = []
        for i, idea in enumerate(ideas):
            programs = idea.get("programs") or []
            for pid in programs:
                if node_id == str(pid).strip():
                    matches.append(i)
                    break
        if len(matches) > MAX_SELECTED_IDEAS:
            matches = matches[:MAX_SELECTED_IDEAS]
        return matches
    if not ideas:
        return dash.no_update
    try:
        id_dict = json.loads(trigger_id)
        idx = id_dict.get("index", 0)
    except (ValueError, KeyError, json.JSONDecodeError):
        return dash.no_update
    # Ignore trigger from idea list when it's due to initial render (n_clicks still 0), not a real click
    idea_n_clicks_list = idea_n_clicks_list or []
    if idx < len(idea_n_clicks_list) and (idea_n_clicks_list[idx] or 0) == 0:
        return dash.no_update
    selected = list(selected or [])
    if idx in selected:
        selected = [i for i in selected if i != idx]
    else:
        if len(selected) >= MAX_SELECTED_IDEAS:
            # At max: do not add; return current selection so list does not clear
            return selected
        selected = selected + [idx]
    return selected


@app.callback(
    Output("list-click-output", "children"),
    Input({"type": "idea-btn", "index": ALL}, "n_clicks"),
    State("ideas-store", "data"),
    prevent_initial_call=True,
)
def on_idea_click(n_clicks_list, ideas):
    """Show full idea description, number of programs, and avg_fitness when a list item is clicked."""
    ctx = dash.callback_context
    if not ctx.triggered or not ideas:
        return dash.no_update
    prop = ctx.triggered[0]["prop_id"]
    if not prop or ".n_clicks" not in prop:
        return dash.no_update
    # prop_id is like '{"type":"idea-btn","index":2}.n_clicks'
    try:
        id_part = prop.split(".")[0]
        id_dict = json.loads(id_part)
        idx = id_dict.get("index", 0)
    except (ValueError, KeyError, json.JSONDecodeError):
        return dash.no_update
    # Ignore triggers from button rebuilds (n_clicks still 0), only show description for real clicks
    n_clicks_list = n_clicks_list or []
    if idx < 0 or idx >= len(ideas) or idx >= len(n_clicks_list):
        return dash.no_update
    if (n_clicks_list[idx] or 0) == 0:
        return dash.no_update
    idea = ideas[idx]
    desc = idea.get("description", "")
    programs = idea.get("programs", [])
    avg = idea.get("avg_fitness", 0)
    n_programs = len(programs)
    return html.Div(
        [
            html.H4("Идея", style={"marginTop": "0", "borderBottom": "1px solid #ccc"}),
            html.P([html.Strong("avg_fitness: "), f"{avg:.4f}"]),
            html.P([html.Strong("Программ: "), str(n_programs)]),
            html.Hr(style={"margin": "8px 0"}),
            html.P([html.Strong("Описание: ")], style={"marginBottom": "4px"}),
            html.Div(
                desc,
                style={
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": "#f5f5f5",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                },
            ),
        ]
    )


@app.callback(Output("stats-info", "children"), Input("stats-btn", "n_clicks"), State("data-store", "data"))
def show_stats(n_clicks, all_data):
    if not n_clicks:
        return html.Div(
            [
                html.H4("Статистика"),
                html.P("Нажмите кнопку, чтобы показать.", style={"color": "#777"}),
            ]
        )

    if not all_data:
        return html.Div([html.H4("Статистика"), html.P("Нет данных.")])

    filtered = [x for x in all_data if not is_hidden_fitness(x.get("fitness", 0))]
    if not filtered:
        return html.Div([html.H4("Статистика"), html.P("Нет данных (все = -1000).")])

    memory_lineage = compute_memory_lineage(all_data)
    best_overall = max([x.get("fitness", 0) for x in filtered])
    mem_vals_lineage = [x.get("fitness", 0) for x in filtered if memory_lineage.get(x["id"], False)]
    nonmem_vals_lineage = [x.get("fitness", 0) for x in filtered if not memory_lineage.get(x["id"], False)]

    mem_vals_direct = [x.get("fitness", 0) for x in filtered if x.get("memory_used")]
    nonmem_vals_direct = [x.get("fitness", 0) for x in filtered if not x.get("memory_used")]

    best_mem_lineage = max(mem_vals_lineage) if mem_vals_lineage else None
    best_nonmem_lineage = max(nonmem_vals_lineage) if nonmem_vals_lineage else None
    avg_mem_lineage = (sum(mem_vals_lineage) / len(mem_vals_lineage)) if mem_vals_lineage else None
    avg_nonmem_lineage = (sum(nonmem_vals_lineage) / len(nonmem_vals_lineage)) if nonmem_vals_lineage else None

    best_mem_direct = max(mem_vals_direct) if mem_vals_direct else None
    best_nonmem_direct = max(nonmem_vals_direct) if nonmem_vals_direct else None
    avg_mem_direct = (sum(mem_vals_direct) / len(mem_vals_direct)) if mem_vals_direct else None
    avg_nonmem_direct = (sum(nonmem_vals_direct) / len(nonmem_vals_direct)) if nonmem_vals_direct else None

    return html.Div(
        [
            html.H4("Статистика (fitness != -1000)"),
            html.P([html.Strong("Best overall: "), f"{best_overall:.3f}"]),
            html.P(
                [
                    html.Strong("Best with memory (direct): "),
                    f"{best_mem_direct:.3f}" if best_mem_direct is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Best without memory (direct): "),
                    f"{best_nonmem_direct:.3f}" if best_nonmem_direct is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Avg with memory (direct): "),
                    f"{avg_mem_direct:.3f}" if avg_mem_direct is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Avg without memory (direct): "),
                    f"{avg_nonmem_direct:.3f}" if avg_nonmem_direct is not None else "-",
                ]
            ),
            html.Hr(),
            html.P("Память по предкам: если есть у узла или у любого предка."),
            html.P(
                [
                    html.Strong("Best with memory (lineage): "),
                    f"{best_mem_lineage:.3f}" if best_mem_lineage is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Best without memory (lineage): "),
                    f"{best_nonmem_lineage:.3f}" if best_nonmem_lineage is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Avg with memory (lineage): "),
                    f"{avg_mem_lineage:.3f}" if avg_mem_lineage is not None else "-",
                ]
            ),
            html.P(
                [
                    html.Strong("Avg without memory (lineage): "),
                    f"{avg_nonmem_lineage:.3f}" if avg_nonmem_lineage is not None else "-",
                ]
            ),
        ]
    )


@app.callback(
    Output("cytoscape-graph", "stylesheet"),
    [
        Input("cytoscape-graph", "tapNodeData"),
        Input("highlight-toggle", "value"),
        Input("memory-toggle", "value"),
        Input("data-store", "data"),
        Input("selected-ideas-store", "data"),
    ],
    State("ideas-store", "data"),
)
def update_styles(node_data, highlight_toggle, memory_toggle, all_data, selected_ideas, ideas):
    new_style = list(base_stylesheet)

    if memory_toggle and "on" in memory_toggle:
        new_style.append(
            {
                "selector": "node.memory-used",
                "style": {
                    "border-color": "#2ECC71",
                    "border-width": 120,
                },
            }
        )

    # Idea-based highlighting: selected ideas -> highlight their program nodes; multiple ideas = multiple rings
    # Color is fixed per idea index (idea_idx % 10) so selection order does not reassign colors or cause lag
    selected_ideas = selected_ideas or []
    ideas_list = ideas or []
    if selected_ideas and ideas_list:
        node_to_color_indices = {}
        for idea_idx in selected_ideas:
            if idea_idx < 0 or idea_idx >= len(ideas_list):
                continue
            color_idx = idea_idx % len(IDEA_HIGHLIGHT_COLORS)
            for nid in ideas_list[idea_idx].get("programs") or []:
                nid_str = str(nid).strip()
                if nid_str:
                    if nid_str not in node_to_color_indices:
                        node_to_color_indices[nid_str] = []
                    if color_idx not in node_to_color_indices[nid_str]:
                        node_to_color_indices[nid_str].append(color_idx)
        if node_to_color_indices:
            new_style.append({"selector": "node", "style": {"opacity": 0.2}})
            new_style.append({"selector": "edge", "style": {"opacity": 0.15}})
            _ring_url_cache = {}
            for nid, color_indices in node_to_color_indices.items():
                # Preserve selection order: first selected = innermost ring, last = outermost (no sort)
                hex_colors = [IDEA_HIGHLIGHT_COLORS[i] for i in color_indices]
                style: dict = {"opacity": 1}
                if len(hex_colors) == 1:
                    style["border-color"] = hex_colors[0]
                    style["border-width"] = IDEA_HIGHLIGHT_BORDER_WIDTH
                else:
                    key = tuple(color_indices)
                    if key not in _ring_url_cache:
                        _ring_url_cache[key] = _multi_ring_data_url(hex_colors)
                    data_url = _ring_url_cache[key]
                    if data_url:
                        style["background-image"] = data_url
                        style["background-fit"] = "cover"
                        style["background-clip"] = "node"
                        style["border-width"] = 0
                    else:
                        style["border-color"] = hex_colors[0]
                        style["border-width"] = IDEA_HIGHLIGHT_BORDER_WIDTH
                new_style.append({"selector": f'node[id="{nid}"]', "style": style})

    # Tap-node ancestor path highlighting (skip when idea selection is active so idea colors stay visible)
    if (not node_data or "on" not in highlight_toggle or not all_data) or selected_ideas:
        return new_style

    nodes_dict = {n["id"]: n for n in all_data}
    sel_id = node_data["id"]
    path_nodes = {sel_id}
    path_edges = set()
    queue = [sel_id]

    while queue:
        curr = queue.pop(0)
        c_data = nodes_dict.get(curr)
        if c_data:
            for anc in c_data.get("ancestors", []):
                if anc not in path_nodes:
                    path_nodes.add(anc)
                    queue.append(anc)
                path_edges.add(f"{anc}-{curr}")

    new_style.extend(
        [
            {"selector": "node", "style": {"opacity": 0.1}},
            {"selector": "edge", "style": {"opacity": 0.05}},
        ]
    )

    for nid in path_nodes:
        new_style.append(
            {"selector": f'node[id="{nid}"]', "style": {"opacity": 1, "border-color": "#000", "border-width": 3}}
        )
    for eid in path_edges:
        new_style.append(
            {"selector": f'edge[id="{eid}"]', "style": {"opacity": 1, "line-color": "#555", "width": 5, "z-index": 10}}
        )

    new_style.append(
        {
            "selector": f'node[id="{sel_id}"]',
            "style": {
                "border-color": "#FFD700",
                "border-width": 6,
                "width": f"{NODE_SIZE+10}px",
                "height": f"{NODE_SIZE+10}px",
            },
        }
    )

    return new_style


if __name__ == "__main__":
    app.run(debug=True)
