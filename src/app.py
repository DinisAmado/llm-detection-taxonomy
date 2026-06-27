import streamlit as st
import sqlite3
import pandas as pd
import json
import streamlit.components.v1 as components
import os
from pyvis.network import Network 

# --- Page Config ---
st.set_page_config(page_title="SNA Decision Support", layout="wide", page_icon="🛡️")

# --- Custom CSS ---
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    iframe {
        border: none !important;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h3 {
        color: #11caa0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    conn = sqlite3.connect("results/sna.db")
    df_rel = pd.read_sql_query(
        "SELECT source, target, interaction_type, taxonomy_classification, confidence_score FROM relations", 
        conn
    )
    conn.close()
    return df_rel

@st.cache_data
def load_metrics():
    metrics_path = "graphs/metrics_report.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

df = load_data()
metrics = load_metrics()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Tactical Filters")
st.sidebar.write("Isolate threats or filter validation context.")

all_taxonomies = df["taxonomy_classification"].unique().tolist()
default_severe = [t for t in ["EXTREMIST", "RADICALISM", "VIOLATED", "THREAT", "HATE"] if t in all_taxonomies]

selected_taxonomies = st.sidebar.multiselect(
    "Risk Level (Taxonomy)",
    options=all_taxonomies,
    default=default_severe
)

search_query = st.sidebar.text_input("Search Entity (e.g., WOMEN, IMMIGRANTS)", "").strip().upper()

# --- Apply Filters ---
filtered_df = df[df["taxonomy_classification"].isin(selected_taxonomies)]
if search_query:
    filtered_df = filtered_df[
        filtered_df["source"].str.contains(search_query, case=False, na=False) |
        filtered_df["target"].str.contains(search_query, case=False, na=False)
    ]

filtered_metrics = {}
if metrics:
    for entity, data in metrics.items():
        if search_query and search_query not in entity.upper():
            continue
        filtered_metrics[entity] = data


# --- Main Layout with Tabs ---
st.title("🛡️ TACTICAL EARLY WARNING SYSTEM")
st.markdown("API-Driven Semantic Network Analysis Dashboard")

tab1, tab2 = st.tabs(["📊 Real-time SNA Operations", "🔬 Model Validation & Performance"])

# OPERATIONAL DASHBOARD (DYNAMIC KG)

with tab1:
    st.divider()
    left_col, right_col = st.columns([1, 2.2])

    with left_col:
        st.subheader("Critical Alerts")
        if not filtered_df.empty:
            top_sources = filtered_df["source"].value_counts().head(5).reset_index()
            top_sources.columns = ["Entity (Source)", "Filtered Incidents"]
            st.dataframe(top_sources, hide_index=True, width="stretch")
        else:
            st.info("No critical alerts found for the selected filters.")
        
        st.subheader("Network Influencers")
        if filtered_metrics:
            df_metrics = pd.DataFrame.from_dict(filtered_metrics, orient="index").reset_index()
            df_metrics.columns = ["Entity", "Type", "In-Degree", "Betweenness", "Influence"]
            df_inf = df_metrics.sort_values(by="Influence", ascending=False).head(7)
            
            styled_inf = df_inf[["Entity", "Type", "Influence"]].style.background_gradient(
                subset=["Influence"], cmap="Blues"
            ).format({"Influence": "{:.2%}"})
            
            st.dataframe(styled_inf, hide_index=True, width="stretch")
        else:
            st.info("No entities match the current search criteria.")

    with right_col:
        st.subheader("Interactive Knowledge Graph")
        
        if not filtered_df.empty:
            # Dynamically instantiate a PyVis network in memory
            net = Network(height="600px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
            
            # Preserve the official severity color palette from kg.py
            edge_colors = {
                "EXTREMIST":   "#d62728",
                "RADICALISM":  "#e6550d",
                "VIOLATED":    "#fd8d3c",
                "THREAT":      "#e7ba52",
                "HATE":        "#9467bd",
                "EMOTIONAL":   "#5254a3",
                "SENTIMENTAL": "#6baed6",
                "NEUTRAL":     "#74c476",
            }
            
            # Populate the network using only the filtered dataframe rows
            for _, row in filtered_df.iterrows():
                src = row["source"]
                tgt = row["target"]
                cls = row["taxonomy_classification"]
                int_type = row["interaction_type"]
                
                # PyVis automatically ignores duplicate nodes
                net.add_node(src, label=src, color="#4e79a7", title=f"Entity: {src}")
                net.add_node(tgt, label=tgt, color="#4e79a7", title=f"Entity: {tgt}")
                
                # Add edge with the corresponding severity color and custom hover tooltip
                net.add_edge(
                    src, tgt, 
                    title=f"Relation: {int_type} ({cls})", 
                    color=edge_colors.get(cls, "#999999"),
                    weight=2
                )
            
            # Generate the HTML structure on-the-fly without touching the disk
            html_string = net.generate_html()
            components.html(html_string, height=650, scrolling=True)
        else:
            st.info("No data available to render the graph based on the current filters.")

# SCIENTIFIC VALIDATION

with tab2:
    st.divider()
    st.subheader("Empirical Model Evaluation (UC Berkeley Dataset)")
    st.markdown("This section presents the real-world validation metrics computed for the multi-tier cascade pipeline.")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Accuracy (Exatidão)", value="71.80%", delta="Baseline Overperformed")
    with m2:
        st.metric(label="Recall (Sensibilidade)", value="91.76%", delta="High Risk Capture", delta_color="normal")
    with m3:
        st.metric(label="Precision (Precisão)", value="57.00%", delta="Over-zealous/Preventive", delta_color="inverse")
    with m4:
        st.metric(label="F1-Score", value="70.32%")
        
    st.divider()
    
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("ROC Curve (Área Sob a Curva)")
        roc_path = "graphs/validation/roc_curve.png" 
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (AUC: 0.63)", use_container_width=True)
        else:
            st.info("ROC Curve image not found at 'graphs/validation/roc_curve.png'.")
            
    with g2:
        st.subheader("Confusion Matrix (Matriz de Confusão)")
        cm_path = "graphs/validation/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix: 15 False Negatives vs 126 False Positives", use_container_width=True)
        else:
            st.info("Confusion Matrix image not found at 'graphs/validation/confusion_matrix.png'.")