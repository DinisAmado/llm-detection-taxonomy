import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pyvis.network import Network

from analytics import run_analytics

# --- Constants ---
DB_PATH    = "results/sna.db"
OUT_DIR    = Path("graphs")

# Severity palette (darkest = most severe)
CATEGORY_COLOR = {
    "EXTREMIST":   "#d62728",
    "RADICALISM":  "#e6550d",
    "VIOLATED":    "#fd8d3c",
    "THREAT":      "#e7ba52",
    "HATE":        "#9467bd",
    "EMOTIONAL":   "#5254a3",
    "SENTIMENTAL": "#6baed6",
    "NEUTRAL":     "#74c476",
}

ENTITY_TYPE_COLOR = {
    "Person":       "#4e79a7",
    "Institution":  "#f28e2b",
    "Location":     "#59a14f",
    "Group":        "#e15759",
    "OTHER":        "#b07aa1",
}

# --- Database Operations ---
def open_db(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run ingest.py first to populate the database."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_relations(
    conn: sqlite3.Connection,
    langs: list[str] | None = None,
    categories: list[str] | None = None,
    limit: int = 1000,
) -> list[sqlite3.Row]:
    """Retrieves valid relationships from SQLite storage."""
    conditions, params = [], []

    if langs:
        placeholders = ",".join("?" * len(langs))
        conditions.append(f"r.lang IN ({placeholders})")
        params.extend(langs)

    if categories:
        placeholders = ",".join("?" * len(categories))
        conditions.append(f"rel.taxonomy_classification IN ({placeholders})")
        params.extend(categories)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    query = f"""
        SELECT
            rel.source,
            rel.target,
            rel.interaction_type,
            rel.taxonomy_classification,
            rel.confidence_score,
            r.lang,
            r.id AS result_id
        FROM relations rel
        JOIN results r ON rel.result_id = r.id
        {where}
        LIMIT ?
    """
    return conn.execute(query, params).fetchall()

def fetch_entity_types(conn: sqlite3.Connection) -> dict[str, str]:
    """Generates mapping dictionary for entity types."""
    rows = conn.execute("SELECT entity_id, entity_type FROM entities").fetchall()
    mapping: dict[str, str] = {}
    for row in rows:
        eid = row["entity_id"]
        if eid and eid not in mapping:
            mapping[eid] = row["entity_type"] or "OTHER"
    return mapping

# --- Graph Engine ---
def build_graph(
    relations: list[sqlite3.Row],
    entity_types: dict[str, str],
    top_entities: int | None = None,
) -> nx.MultiDiGraph:
    """Builds the directed multigraph, strictly filtering for severe network interactions."""
    G = nx.MultiDiGraph()

    for rel in relations:
        src  = str(rel["source"]).strip()
        tgt  = str(rel["target"]).strip()
        if not src or not tgt or src == tgt:
            continue

        cat = rel["taxonomy_classification"] or "NEUTRAL"
        
        # Discard neutral daily interactions to highlight structural risks
        if cat == "NEUTRAL":
            continue

        itype = rel["interaction_type"] or ""
        score = rel["confidence_score"]
        lang  = rel["lang"]

        for node_id in (src, tgt):
            if node_id not in G:
                etype = entity_types.get(node_id, "OTHER")
                G.add_node(
                    node_id,
                    entity_type=etype,
                    color=ENTITY_TYPE_COLOR.get(etype, "#aaa"),
                    frequency=0,
                )
            G.nodes[node_id]["frequency"] = G.nodes[node_id].get("frequency", 0) + 1

        G.add_edge(
            src, tgt,
            interaction_type=itype,
            category=cat,
            confidence=score,
            lang=lang,
            color=CATEGORY_COLOR.get(cat, "#999"),
        )

    if top_entities and G.number_of_nodes() > top_entities:
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        keep = {n for n, _ in degree_sorted[:top_entities]}
        remove = [n for n in list(G.nodes()) if n not in keep]
        G.remove_nodes_from(remove)

    return G

# --- Export Pipelines ---
def export_html(G: nx.MultiDiGraph, out_path: Path):
    """Renders interactive PyVis HTML dashboard configuration."""
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    for node_id, attrs in G.nodes(data=True):
        freq  = attrs.get("frequency", 1)
        size  = max(10, min(50, 8 + freq * 2))
        etype = attrs.get("entity_type", "OTHER")
        color = attrs.get("color", "#aaa")
        net.add_node(
            node_id,
            label=node_id,
            title=f"<b>{node_id}</b><br>Type: {etype}<br>Connections: {freq}",
            color=color,
            size=size,
            font={"size": 12, "color": "#ffffff"},
        )

    for src, tgt, attrs in G.edges(data=True):
        cat   = attrs.get("category", "NEUTRAL")
        itype = attrs.get("interaction_type", "")
        score = attrs.get("confidence", None)
        score_str = f"{score:.2f}%" if score is not None else "N/A"
        color = attrs.get("color", "#999")
        
        # UI labels emphasize taxonomy/severity. Hover displays specific qualitative verb.
        net.add_edge(
            src, tgt,
            title=f"<b>Interaction:</b> {itype}<br><b>Taxonomy:</b> {cat}<br><b>Confidence:</b> {score_str}",
            label=cat,
            color={"color": color, "opacity": 0.75},
            arrows="to",
            smooth={"type": "curvedCW", "roundness": 0.2},
        )

    legend_items = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{c};border-radius:50%;margin-right:4px"></span>{k}&nbsp;&nbsp;'
        for k, c in CATEGORY_COLOR.items() if k != "NEUTRAL"
    )
    net.html = net.html.replace(
        "</body>",
        f"""
<div style="position:fixed;bottom:12px;left:12px;background:#0d0d1a;
            padding:10px 16px;border-radius:8px;font-family:sans-serif;
            font-size:12px;color:#ccc;z-index:9999">
  <b style="color:#fff">Edge Severity (Filtered)</b><br><br>{legend_items}
</div>
</body>""",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_path))
    print(f"[✓] Interactive graph  → {out_path}")

def export_png(G: nx.MultiDiGraph, out_path: Path):
    """Renders static NetworkX fallback image."""
    if G.number_of_nodes() == 0:
        print("[!] No nodes to draw — skipping PNG.")
        return

    fig, ax = plt.subplots(figsize=(22, 16))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.axis("off")

    if G.number_of_nodes() <= 80:
        pos = nx.kamada_kawai_layout(G.to_undirected())
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    degrees = dict(G.degree())
    max_deg = max(degrees.values(), default=1)
    node_sizes  = [300 + 2000 * (degrees[n] / max_deg) for n in G.nodes()]
    node_colors = [G.nodes[n].get("color", "#aaa") for n in G.nodes()]
    edge_colors = [d.get("color", "#666") for _, _, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=0.5, edgecolors="#ffffff33")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, alpha=0.55, arrows=True, arrowsize=12, width=0.9, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#eeeeee", font_family="monospace")

    sev_patches = [mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLOR.items() if k != "NEUTRAL"]
    leg1 = ax.legend(handles=sev_patches, title="Edge Severity", title_fontsize=9, fontsize=8, loc="lower left", framealpha=0.35, facecolor="#1a1a2e", labelcolor="#dddddd")
    leg1.get_title().set_color("#ffffff")
    ax.add_artist(leg1)

    ent_patches = [mpatches.Patch(color=c, label=etype) for etype, c in ENTITY_TYPE_COLOR.items()]
    ax.legend(handles=ent_patches, title="Entity Type", title_fontsize=9, fontsize=8, loc="lower right", framealpha=0.35, facecolor="#1a1a2e", labelcolor="#dddddd").get_title().set_color("#ffffff")

    ax.set_title("SNA Knowledge Graph (Threat & Sentiment Only)", color="#ffffff", fontsize=18, pad=14, fontweight="bold")

    stats_txt = f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  Components: {nx.number_weakly_connected_components(G)}"
    ax.text(0.5, 0.01, stats_txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#aaaaaa")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Static PNG graph   → {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Build knowledge graphs from sna.db")
    parser.add_argument("--db",           default=DB_PATH)
    parser.add_argument("--limit",        type=int, default=1000)
    parser.add_argument("--top-entities", type=int, default=None)
    parser.add_argument("--out-dir",      default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"[•] Opening database  : {args.db}")
    conn = open_db(args.db)

    relations    = fetch_relations(conn, limit=args.limit)
    entity_types = fetch_entity_types(conn)
    conn.close()

    if not relations:
        print("No relations found. Exiting.")
        return

    print("Building graph (Filtering Neutral)...")
    G = build_graph(relations, entity_types, top_entities=args.top_entities)
    
    if G.number_of_nodes() == 0:
        print("[!] Graph generation aborted: No severe nodes detected after filtering.")
        return

    # Trigger centrality and community analytics on filtered threat network
    run_analytics(G, out_dir) 

    export_html(G, out_dir / "knowledge_graph.html")
    export_png(G, out_dir / "knowledge_graph.png")
    print("\nDone. Outputs written to:", out_dir)

if __name__ == "__main__":
    main()