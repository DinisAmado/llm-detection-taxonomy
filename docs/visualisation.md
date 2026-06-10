# Visualisation & Analytics Layer

This document covers the three scripts responsible for transforming the raw data stored in `results/sna.db` into graphs, metrics, and an interactive dashboard: **`kg.py`**, **`analytics.py`**, and **`app.py`**.

These scripts are designed to run **after** the pipeline has been populated — either via `ingest.py` (batch ingestion) or the `/analyze` API endpoint.

---

## Execution Order

```
ingest.py / api.py  →  results/sna.db
                              │
                         kg.py  (calls analytics.py internally)
                              │
               ┌──────────────┴────────────────┐
    graphs/knowledge_graph.html          graphs/metrics_report.json
    graphs/knowledge_graph.png
    graphs/subgraphs/*.png
                              │
                         app.py  (reads sna.db + graphs/)
                              │
                    Streamlit Dashboard
```

---

## `kg.py` — Knowledge Graph Builder

### What it does

`kg.py` reads all relations from `results/sna.db`, constructs a directed multigraph, and exports it in multiple formats. It also triggers `analytics.py` to compute node-level metrics before rendering.

### Pipeline

1. **Fetch from DB** — Queries the `relations` and `entities` tables with optional filters (language, severity category, row limit).
2. **Build graph** — Constructs a `networkx.MultiDiGraph` where nodes are entities (coloured by type) and edges are relations (coloured by severity category). Optionally trims to the top-N most-connected entities.
3. **Run analytics** — Calls `analytics.py` to inject centrality metrics into the graph and produce `metrics_report.json`.
4. **Export HTML** — Renders an interactive vis.js graph via `pyvis`, with a dark background, Barnes-Hut physics, hover tooltips, and a severity legend.
5. **Export PNG** — Renders a static matplotlib/networkx image with dual legends (edge severity + entity type) and graph stats annotation.
6. **Export subgraphs** — Optionally generates one PNG per taxonomy category (e.g. `subgraph_hate.png`, `subgraph_threat.png`).

### Outputs

| File | Description |
|---|---|
| `graphs/knowledge_graph.html` | Interactive graph (open in any browser) |
| `graphs/knowledge_graph.png` | Static image for reports |
| `graphs/subgraphs/subgraph_<category>.png` | Per-severity subgraphs (with `--subgraphs`) |
| `graphs/metrics_report.json` | Node metrics produced by `analytics.py` |

### CLI Reference

```bash
python src/kg.py [OPTIONS]

Options:
  --db            Path to SQLite database (default: results/sna.db)
  --lang          Filter by language: EN ES PT FR (space-separated)
  --category      Filter by severity: EXTREMIST RADICALISM VIOLATED THREAT HATE ...
  --limit         Max relations to load (default: 1000)
  --top-entities  Keep only the N most-connected nodes
  --out-dir       Output directory (default: graphs/)
  --no-html       Skip interactive HTML export
  --no-png        Skip static PNG export
  --subgraphs     Export one PNG per taxonomy category
```

### Examples

```bash
# Full default run
python src/kg.py

# Portuguese data, threats and hate only, top 30 entities
python src/kg.py --lang PT --category THREAT HATE --top-entities 30

# Fast run for the dashboard — HTML only
python src/kg.py --no-png
```

### Colour Coding

**Node colour** reflects entity type:

| Type | Colour |
|---|---|
| Person | Blue `#4e79a7` |
| Institution | Orange `#f28e2b` |
| Location | Green `#59a14f` |
| Group | Red `#e15759` |

**Edge colour** reflects severity (darkest = most severe):

| Category | Colour |
|---|---|
| EXTREMIST | `#d62728` |
| RADICALISM | `#e6550d` |
| VIOLATED | `#fd8d3c` |
| THREAT | `#e7ba52` |
| HATE | `#9467bd` |
| EMOTIONAL | `#5254a3` |
| SENTIMENTAL | `#6baed6` |
| NEUTRAL | `#74c476` |

---

## `analytics.py` — Graph Metrics Engine

### What it does

`analytics.py` receives the `networkx.MultiDiGraph` built by `kg.py` and computes three families of metrics for every node. It then injects those metrics back into the graph object (for use at render time) and writes a `metrics_report.json` file consumed by the Streamlit dashboard.

> `analytics.py` is not meant to be run directly. It is called automatically by `kg.py`.

### Metrics Computed

| Metric | Description |
|---|---|
| **In-degree** | Number of incoming edges — how many entities target this node. Proxy for how much a node is talked about or acted upon. |
| **Betweenness centrality** | Fraction of shortest paths in the undirected graph that pass through this node. High values indicate bridge nodes that connect otherwise separate clusters. |
| **PageRank (Influence)** | Directed PageRank (α = 0.85). Nodes that receive edges from other high-influence nodes score higher. Used as the primary influence ranking in the dashboard. |

Community detection (greedy modularity on the undirected projection) is also performed, though the community IDs are not currently exposed in the dashboard.

### Output

```json
// graphs/metrics_report.json
{
  "TRUMP": {
    "type": "Person",
    "in_degree": 14,
    "betweenness": 0.3812,
    "eigenvector": 0.0941
  },
  "IMMIGRANTS": {
    "type": "Group",
    "in_degree": 22,
    "betweenness": 0.1204,
    "eigenvector": 0.1573
  }
}
```

---

## `app.py` — Streamlit Decision Support Dashboard

### What it does

`app.py` is the front-end of the system. It reads data from `results/sna.db` and `graphs/metrics_report.json` and presents an operational picture of the social network for analysts.

### Requirements

Before launching, ensure the following files exist:

- `results/sna.db` — populated by `ingest.py` or the `/analyze` API
- `graphs/metrics_report.json` — generated by `kg.py`
- `graphs/knowledge_graph.html` — generated by `kg.py`

### How to run

```bash
streamlit run src/app.py
```

### Dashboard Sections

**Top metrics row (4 KPI cards):**

| Card | What it shows |
|---|---|
| Total Interactions | Total number of classified relation edges in the DB |
| High-Risk Alerts | Relations classified as EXTREMIST, RADICALISM, VIOLATED, THREAT, or HATE |
| Unique Entities | Distinct node count (union of all sources and targets) |
| Most Targeted | The entity that appears most frequently as a relation target |

**Critical Alerts table:**  
Lists the top 5 sources ranked by number of severe incidents (categories above). Useful for identifying the most active hostile actors.

**Network Influencers table:**  
Ranks the top 7 nodes by PageRank influence score, with a blue gradient heatmap. Also shows entity type and influence value (4 decimal places).

**Interactive Knowledge Graph:**  
Embeds `graphs/knowledge_graph.html` directly in the dashboard (800 px tall). Nodes are draggable; hovering shows entity type and connection count; edges show interaction type, severity category, and confidence score.

### Data Sources

| Source | Used for |
|---|---|
| `results/sna.db` → `relations` table | KPI cards, alerts table |
| `graphs/metrics_report.json` | Influencer ranking table |
| `graphs/knowledge_graph.html` | Embedded interactive graph |
