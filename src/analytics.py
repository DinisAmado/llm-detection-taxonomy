import networkx as nx
import json
from pathlib import Path

def run_analytics(G: nx.MultiDiGraph, out_dir: Path):
    """
    Computes graph analytics (Centrality, Influence, Communities)
    and saves a metrics report for the Streamlit dashboard.
    """
    if G.number_of_nodes() == 0:
        return
        
    D = nx.DiGraph(G)
    U = nx.Graph(D)
    
    # Compute Centrality Metrics
    in_degree = dict(D.in_degree())
    
    # Betweenness on undirected graph to find structural bridges
    betweenness = nx.betweenness_centrality(U)
    
    # Eigenvector/PageRank to calculate real network influence
    try:
        eigenvector = nx.pagerank(D, alpha=0.85)
    except Exception:
        eigenvector = {n: 0 for n in D.nodes()}

    # Detect Communities
    try:
        communities = list(nx.community.greedy_modularity_communities(U))
        comm_map = {node: cid for cid, comm in enumerate(communities) for node in comm}
    except Exception:
        comm_map = {n: 0 for n in U.nodes()}

    metrics_report = {}

    # Inject metrics into nodes and build report
    for node in G.nodes():
        G.nodes[node]["in_degree"] = in_degree.get(node, 0)
        G.nodes[node]["betweenness"] = round(betweenness.get(node, 0), 4)
        G.nodes[node]["eigenvector"] = round(eigenvector.get(node, 0), 4)
        
        metrics_report[node] = {
            "type": G.nodes[node].get("entity_type", "OTHER"),
            "in_degree": in_degree.get(node, 0),
            "betweenness": round(betweenness.get(node, 0), 4),
            "eigenvector": round(eigenvector.get(node, 0), 4)
        }

    # Save metrics report for dashboard consumption
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
        
    print(f"[✓] Analytics exported to graphs/metrics_report.json")