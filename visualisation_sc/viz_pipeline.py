import csv
import json
import re

import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import networkx as nx
import matplotlib
import matplotlib.colors as mcolors

# =========================
# CONFIG: ONLY SET THIS
# =========================
INPUT_CSV = "outputs/lyapunov3.csv"

# Optional: if you still want the JSON written to disk for debugging / reuse
WRITE_JSON_TO_DISK = False
OUTPUT_JSON = "visualisation/data_mem.json"

# --- VIS CONSTANTS ---
NODE_SIZE = 800
FONT_SIZE = 180
CANVAS_HEIGHT = "85vh"

BASE_POS_MULTIPLIER = 100
PADDING_FACTOR = 200
SPLIT_GAP_FACTOR = 0.35
HIDE_FITNESS_VALUE = -1000.0


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
        html.H2(
            "Интерактивный граф (CSV → JSON → Dash)",
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
                                            "label": " Разделить: память справа / без памяти слева",
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
                # RIGHT (info)
                html.Div(
                    [
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
    ],
)
def update_styles(node_data, highlight_toggle, memory_toggle, all_data):
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

    if not node_data or "on" not in highlight_toggle or not all_data:
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
