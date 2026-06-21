import streamlit as st
import sqlite3
import pandas as pd
import json
import streamlit.components.v1 as components
import os

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
    path = "graphs/metrics_report.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

df = load_data()
metrics = load_metrics()

# --- Main Dashboard ---
st.title("SNA Decision Support Dashboard")
st.markdown("Monitorização de dinâmicas emocionais, alertas de ódio e figuras de influência na rede.")

# Top Level Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Interactions", len(df))
col2.metric("High-Risk Alerts", len(df[df["taxonomy_classification"].isin(["EXTREMIST", "RADICALISM", "VIOLATED", "THREAT", "HATE"])]))
col3.metric("Unique Entities", len(pd.concat([df["source"], df["target"]]).unique()))
col4.metric("Most Targeted", df["target"].mode()[0] if not df.empty else "N/A")

st.divider()

left_col, right_col = st.columns([1, 2.2])

with left_col:
    st.subheader("Critical Alerts")
    df_severe = df[df["taxonomy_classification"].isin(["EXTREMIST", "RADICALISM", "VIOLATED", "THREAT", "HATE"])]
    if not df_severe.empty:
        top_sources = df_severe["source"].value_counts().head(5).reset_index()
        top_sources.columns = ["Entity (Source)", "Severe Incidents"]
        st.dataframe(top_sources, hide_index=True, width="stretch")
    
    st.subheader("Network Influencers")
    if metrics:
        df_metrics = pd.DataFrame.from_dict(metrics, orient="index").reset_index()
        df_metrics.columns = ["Entity", "Type", "In-Degree", "Betweenness", "Influence"]
        df_inf = df_metrics.sort_values(by="Influence", ascending=False).head(7)
        
        # Apply gradient and percentage format to Influence metric
        styled_inf = df_inf[["Entity", "Type", "Influence"]].style.background_gradient(
            subset=["Influence"], cmap="Blues"
        ).format({"Influence": "{:.2%}"})
        
        st.dataframe(styled_inf, hide_index=True, width="stretch")

with right_col:
    st.subheader("Interactive Knowledge Graph")
    html_path = "graphs/knowledge_graph.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            # Render HTML graph in Streamlit iframe
            st.components.v1.html(f.read(), height=800, scrolling=False)

#streamlit run src/app.py