import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------- CONFIG ---------------- #
OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
STRING_API_BASE = "https://version-12-0.string-db.org/api/json"
SPECIES_MAP = {"Homo sapiens": 9606}
TOP_N = 10  # top nodes per centrality

# ---------------- FUNCTIONS ---------------- #
def query_open_targets(disease_name):
    query = """
    query diseaseTargets($diseaseName: String!) {
      search(queryString: $diseaseName, entityNames: ["disease"]) {
        hits {
          object {
            ... on Disease {
              associatedTargets(page: { size: 100, index: 0 }) {
                rows {
                  target { approvedSymbol }
                  score
                }
              }
            }
          }
        }
      }
    }
    """
    variables = {"diseaseName": disease_name}
    resp = requests.post(OT_GRAPHQL_URL, json={"query": query, "variables": variables})
    data = resp.json()
    hits = data["data"]["search"]["hits"]
    if not hits:
        st.error(f"No disease found for '{disease_name}'")
        return []
    assoc = hits[0]["object"]["associatedTargets"]["rows"]
    return [(row["target"]["approvedSymbol"], row["score"]) for row in assoc]


def map_genes_to_string_ids(genes, taxon_id):
    params = {
        "identifiers": "\r".join(genes),
        "species": taxon_id,
        "limit": 1,
        "caller_identity": "my_app"
    }
    resp = requests.post(f"{STRING_API_BASE}/get_string_ids", data=params)
    data = resp.json()
    return {item["queryItem"]: item["stringId"] for item in data if item.get("queryItem") and item.get("stringId")}


def fetch_ppi(string_ids, taxon_id):
    params = {
        "identifiers": "\r".join(string_ids),
        "species": taxon_id,
        "required_score": 400,
        "caller_identity": "my_app"
    }
    resp = requests.post(f"{STRING_API_BASE}/network", data=params)
    return resp.json()


def plot_network_multi(G, top_degree, top_betweenness, top_closeness, top_eigenvector):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for n in G.nodes():
        if n in top_degree: node_colors.append("red")
        elif n in top_betweenness: node_colors.append("orange")
        elif n in top_closeness: node_colors.append("green")
        elif n in top_eigenvector: node_colors.append("purple")
        else: node_colors.append("skyblue")
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, edge_color="gray", alpha=0.7)
    plt.title("PPI Network (Colored by Top 10 Centrality)", fontsize=16, color="darkblue")
    st.pyplot(plt)


# ---------------- STRING DESCRIPTIONS ---------------- #
def fetch_string_descriptions(proteins, taxon_id):
    descriptions = {}
    for p in proteins:
        params = {
            "identifiers": p,
            "species": taxon_id,
            "caller_identity": "my_app"
        }
        try:
            resp = requests.post(f"{STRING_API_BASE}/get_string_ids", data=params)
            data = resp.json()
            if data and "annotation" in data[0]:
                descriptions[p] = data[0]["annotation"]
            elif data and "preferredName" in data[0]:
                descriptions[p] = data[0]["preferredName"]
            else:
                descriptions[p] = "Description not found"
        except:
            descriptions[p] = "Description not found"
    return descriptions


# ---------------- RWR FUNCTIONS ---------------- #
def build_adj_matrix(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    return A, nodes, idx

def random_walk_with_restart(A, seed_indices, restart_prob=0.7, max_iter=100, tol=1e-6):
    n = A.shape[0]
    col_sum = A.sum(axis=0)
    col_sum[col_sum == 0] = 1
    M = A / col_sum

    p0 = np.zeros(n)
    for s in seed_indices:
        p0[s] = 1/len(seed_indices)

    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1 - restart_prob) * M @ p + restart_prob * p0
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new
    return p


# ---------------- STREAMLIT UI ---------------- #


st.set_page_config(page_title="PPI Network Explorer", page_icon="🧬", layout="wide")

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    

    :root {
        --bg-primary: #000000;
        --bg-secondary: #0a0a0a;
        --bg-card: #111111;
        --bg-hover: #1a1a1a;
        --text-primary: #F9F6EE;
        --text-secondary: #F0EAD6;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
        --border: #2a2a2a;
    }
    

    .stApp {
        background-color: var(--bg-primary);
    }
    
    .main {
        background-color: var(--bg-primary);
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: 100%;
    }
    

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    p, span, div, label {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
    }
    

    .header-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid var(--border);
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #F5F5DC 0%, #F5F5DD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    

    .input-section {
        background-color: var(--bg-card);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid var(--border);
    }
    
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-blue);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary);
    }
    

    .stButton > button {
        background: linear-gradient(135deg, #252525 0%, #0D0D0D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(254, 256, 246, 0.2);
    }
    

    .metric-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }
    
    .section-card {
        background-color: var(--bg-card);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin: 1.5rem 0;
    }
    

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid var(--border);
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    

    .dataframe {
        width: 100% !important;
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr th {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 1rem !important;
        border-bottom: 2px solid var(--border) !important;
    }
    
    .dataframe tbody tr td {
        background-color: var(--bg-card) !important;
        color: var(--text-secondary) !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--border) !important;
    }
    
    .dataframe tbody tr:hover td {
        background-color: var(--bg-hover) !important;
    }
    

    .protein-list {
        background-color: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    

    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 8px;
        font-size: 1.25rem;
    }
    
    .icon-blue { background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; }
    .icon-purple { background-color: rgba(139, 92, 246, 0.1); color: #8b5cf6; }
    .icon-green { background-color: rgba(16, 185, 129, 0.1); color: #10b981; }
    .icon-orange { background-color: rgba(245, 158, 11, 0.1); color: #f59e0b; }
    

    .description-box {
        background-color: var(--bg-secondary);
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-blue);
        margin: 1rem 0;
        color: var(--text-secondary);
        line-height: 1.7;
    }
    

    .plot-container {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin: 1rem 0;
    }
    
    .spinner {
        border: 4px solid rgba(59, 130, 246, 0.1);
        border-radius: 50%;
        border-top: 4px solid #3b82f6;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


st.markdown("""
<div class='header-container'>
    <div class='header-title'>PPI Network Explorer</div>
    <div class='header-subtitle'>Centrality Analysis • Random Walk with Restart Algorithm • Essential Protein Prediction</div>
</div>
""", unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([1, 1])

with col1:
    disease = st.text_input("Disease Name", placeholder="e.g., Type 2 Diabetes", key="disease_input", label_visibility="collapsed")
    st.markdown("<small style='color: var(--text-secondary);'>Enter disease name</small>", unsafe_allow_html=True)

with col2:
    species = st.selectbox("Species", list(SPECIES_MAP.keys()), index=0, key="species_input", label_visibility="collapsed")
    st.markdown("<small style='color: var(--text-secondary);'>Select species</small>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
build_button = st.button("Build Network")


loading_placeholder = st.empty()

if build_button:
    
    loading_placeholder.markdown("""
    <div class='section-card' style='text-align: center; padding: 3rem;'>
        <div style='display: inline-block;'>
            <div class='spinner'></div>
        </div>
        <h3 style='color: var(--text-primary); margin-top: 1.5rem;'>Building Network...</h3>
        <p style='color: var(--text-secondary); margin-top: 0.5rem;'>Fetching data and analyzing proteins</p>
    </div>
    """, unsafe_allow_html=True)
    

    try:
        taxon_id = SPECIES_MAP[species]
        target_info = query_open_targets(disease)
        
        if not target_info:
            st.warning("No targets found.")
            st.stop()
        
        genes = [t[0] for t in target_info]
        gene_to_string = map_genes_to_string_ids(genes, taxon_id)
        string_ids = list(gene_to_string.values())
        ppi_data = fetch_ppi(string_ids, taxon_id)
        
        loading_placeholder.empty()
        
        # Build network
        G = nx.Graph()
        for inter in ppi_data:
            a = inter.get("preferredName_A")
            b = inter.get("preferredName_B")
            score = inter.get("score", 0)
            if a and b:
                G.add_edge(a, b, weight=score)
        
        # Disease Display
        st.markdown(f"""
        <div class='section-card'>
            <h2 style='color: var(--text-primary); margin: 0;'>Disease: {disease}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Network Statistics
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-blue'>📊</div>
            <h2 class='section-title'>Network Statistics</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Nodes</div>
                <div class='stat-value'>{G.number_of_nodes()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Edges</div>
                <div class='stat-value'>{G.number_of_edges()}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Density</div>
                <div class='stat-value'>{nx.density(G):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_degree = sum(dict(G.degree()).values())/G.number_of_nodes()
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-label'>Avg Degree</div>
                <div class='stat-value'>{avg_degree:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Centrality Calculations
        deg_c = nx.degree_centrality(G)
        bet_c = nx.betweenness_centrality(G)
        clo_c = nx.closeness_centrality(G)
        eig_c = nx.eigenvector_centrality(G, max_iter=500)
        page_c = nx.pagerank(G)
        
        # Top proteins by centrality
        top_degree = [p for p,_ in sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_betweenness = [p for p,_ in sorted(bet_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_closeness = [p for p,_ in sorted(clo_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_eigenvector = [p for p,_ in sorted(eig_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        
        # Top Proteins Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-purple'>🎯</div>
            <h2 class='section-title'>Top Proteins by Centrality</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4 style='color: #3b82f6;'>⬤ Top Degree Centrality (Hubs)</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='protein-list'>{', '.join(top_degree)}</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #10b981;'>⬤ Top Closeness Centrality (Connectors)</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='protein-list'>{', '.join(top_closeness)}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4 style='color: #f59e0b;'>⬤ Top Betweenness Centrality (Spreaders)</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='protein-list'>{', '.join(top_betweenness)}</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #8b5cf6;'>⬤ Top Eigenvector Centrality (Influencers)</h4>", unsafe_allow_html=True)
            st.markdown(f"<div class='protein-list'>{', '.join(top_eigenvector)}</div>", unsafe_allow_html=True)
        
        # Network Visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-green'>🔗</div>
            <h2 class='section-title'>Network Visualization</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        plot_network_multi(G, top_degree, top_betweenness, top_closeness, top_eigenvector)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Centrality Table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-blue'>📋</div>
            <h2 class='section-title'>Complete Centrality Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        centrality_table = []
        for node in G.nodes():
            centrality_table.append({
                "Protein": node,
                "Degree": deg_c[node],
                "Betweenness": bet_c[node],
                "Closeness": clo_c[node],
                "Eigenvector": eig_c[node],
                "PageRank": page_c[node]
            })
        
        df = pd.DataFrame(centrality_table)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Normalization and Essentiality
        df_norm = df.copy()
        for col in ["Degree", "Betweenness", "Closeness", "Eigenvector", "PageRank"]:
            df_norm[col+"_norm"] = (df_norm[col] - df_norm[col].min())/(df_norm[col].max()-df_norm[col].min())
        
        df_norm["Essentiality"] = (
            0.30 * df_norm["Degree_norm"] +
            0.25 * df_norm["Betweenness_norm"] +
            0.20 * df_norm["Closeness_norm"] +
            0.15 * df_norm["Eigenvector_norm"] +
            0.10 * df_norm["PageRank_norm"]
        )
        
        top_essential = df_norm.sort_values("Essentiality", ascending=False).head(10)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-orange'>⚡</div>
            <h2 class='section-title'>Top 10 Essential Proteins (Weighted Centrality)</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='description-box'>
            Essential proteins are identified using a weighted combination of centrality measures: 
            Degree (30%) + Betweenness (25%) + Closeness (20%) + Eigenvector (15%) + PageRank (10%)
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(top_essential[["Protein", "Essentiality"]], use_container_width=True)
        
        # RWR Section
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-purple'>🔄</div>
            <h2 class='section-title'>Random Walk with Restart (RWR) Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='description-box'>
            Random Walk with Restart (RWR) is a graph-based ranking method that starts from seed proteins 
            and repeatedly walks to their neighbors in the network. At every step, the walker has a fixed 
            probability of jumping back to the seed proteins. Nodes visited more frequently receive higher 
            scores, indicating stronger biological importance and connectivity to seed proteins.
        </div>
        """, unsafe_allow_html=True)
        
        A, nodes, idx = build_adj_matrix(G)
        seed_proteins = top_essential["Protein"].tolist()
        seed_idx = [idx[p] for p in seed_proteins if p in idx]
        rwr_scores = random_walk_with_restart(A, seed_idx)
        
        rwr_df = pd.DataFrame({
            "Protein": nodes,
            "RWR_Score": rwr_scores
        }).sort_values("RWR_Score", ascending=False)
        
        top_rwr = rwr_df.head(10)
        
        st.markdown("""
        <div class='section-header' style='border: none; padding-bottom: 0;'>
            <h3 style='color: var(--text-primary);'>Top 10 RWR Predicted Essential Proteins</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(top_rwr, use_container_width=True)
        
        # RWR with Descriptions
        rwr_desc = fetch_string_descriptions(top_rwr["Protein"].tolist(), taxon_id)
        top_rwr["Description"] = top_rwr["Protein"].map(rwr_desc)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header' style='border: none; padding-bottom: 0;'>
            <h3 style='color: var(--text-primary);'>RWR Proteins with Descriptions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(top_rwr, use_container_width=True, height=400)
        
        # RWR Bar Plot
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: var(--text-primary); margin-bottom: 1rem;'>RWR Score Distribution</h4>", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#111111')
        ax.set_facecolor('#111111')
        sns.barplot(data=top_rwr, x="Protein", y="RWR_Score", palette="inferno", ax=ax)
        plt.xticks(rotation=45, ha='right', color='#F5F5DC')
        plt.yticks(color='#F5F5DC')
        plt.xlabel("Protein", color='#F9F6EE', fontsize=12)
        plt.ylabel("RWR Score", color='#F9F6EE', fontsize=12)
        plt.title("Top RWR Proteins", color='#F9F6EE', fontsize=14, pad=20)
        ax.spines['bottom'].set_color('#2a2a2a')
        ax.spines['top'].set_color('#2a2a2a')
        ax.spines['left'].set_color('#2a2a2a')
        ax.spines['right'].set_color('#2a2a2a')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Centrality Distributions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='icon-box icon-green'>📈</div>
            <h2 class='section-title'>Centrality Distributions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(12, 6), facecolor='#111111')
        ax2.set_facecolor('#111111')
        
        sns.kdeplot(df["Degree"], label="Degree", fill=True, alpha=0.6, ax=ax2)
        sns.kdeplot(df["Betweenness"], label="Betweenness", fill=True, alpha=0.6, ax=ax2)
        sns.kdeplot(df["Closeness"], label="Closeness", fill=True, alpha=0.6, ax=ax2)
        sns.kdeplot(df["Eigenvector"], label="Eigenvector", fill=True, alpha=0.6, ax=ax2)
        sns.kdeplot(df["PageRank"], label="PageRank", fill=True, alpha=0.6, ax=ax2)
        
        plt.xlabel("Value", color='#F9F6EE', fontsize=12)
        plt.ylabel("Density", color='#F9F6EE', fontsize=12)
        plt.title("Distribution of Centrality Measures", color='#F9F6EE', fontsize=14, pad=20)
        plt.xticks(color='#F5F5DC')
        plt.yticks(color='#F5F5DC')
        
        legend = plt.legend(facecolor='#1a1a1a', edgecolor='#2a2a2a')
        for text in legend.get_texts():
            text.set_color('#F5F5DC')
        
        ax2.spines['bottom'].set_color('#2a2a2a')
        ax2.spines['top'].set_color('#2a2a2a')
        ax2.spines['left'].set_color('#2a2a2a')
        ax2.spines['right'].set_color('#2a2a2a')
        
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Error: {e}")