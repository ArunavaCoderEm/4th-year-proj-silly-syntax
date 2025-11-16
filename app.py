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
st.set_page_config(page_title="PPI Network Explorer")
st.title("PPI Network, Centrality, RWR & Essential Protein Predictor")


disease = st.text_input("Disease name", placeholder="e.g., Type 2 Diabetes", key="disease_input")
species = st.selectbox("Species", list(SPECIES_MAP.keys()), index=0, key="species_input")

if st.button("Build Network"):
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

        # Build network
        G = nx.Graph()
        for inter in ppi_data:
            a = inter.get("preferredName_A")
            b = inter.get("preferredName_B")
            score = inter.get("score", 0)
            if a and b:
                G.add_edge(a, b, weight=score)
        # Disease Name display 
        st.subheader(f" Disease: {disease}")
        # ----------- Network Stats ----------- #
        st.subheader(" Network Statistics")
        st.write(f"**Nodes:** {G.number_of_nodes()}")
        st.write(f"**Edges:** {G.number_of_edges()}")
        st.write(f"**Density:** {nx.density(G):.4f}")
        st.write(f"**Average Degree:** {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")

        # ---------------- CENTRALITY ---------------- #
        deg_c = nx.degree_centrality(G)
        bet_c = nx.betweenness_centrality(G)
        clo_c = nx.closeness_centrality(G)
        eig_c = nx.eigenvector_centrality(G, max_iter=500)
        page_c = nx.pagerank(G)

        # ------------- Top based on each centrality ------------- #
        top_degree = [p for p,_ in sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_betweenness = [p for p,_ in sorted(bet_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_closeness = [p for p,_ in sorted(clo_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
        top_eigenvector = [p for p,_ in sorted(eig_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]

        st.subheader("🔴 Top Degree (Hubs)")
        st.write(", ".join(top_degree))

        st.subheader("🟠 Top Betweenness (Spreaders)")
        st.write(", ".join(top_betweenness))

        st.subheader("🟢 Top Closeness (Connectors)")
        st.write(", ".join(top_closeness))

        st.subheader("🟣 Top Eigenvector (Influencers)")
        st.write(", ".join(top_eigenvector))

        # ------------- Plot Network ------------- #
        plot_network_multi(G, top_degree, top_betweenness, top_closeness, top_eigenvector)

        # ------------- Centrality Table ------------- #
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

        st.subheader(" Centrality Table")
        st.dataframe(df)

        # ------------- Normalization ------------- #
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

        st.subheader(" Top 10 Seed Proteins (Weighted Centrality)")
        st.dataframe(top_essential[["Protein", "Essentiality"]])

        # # String Descriptions
        # desc_map = fetch_string_descriptions(top_essential["Protein"].tolist(), taxon_id)
        # top_essential["Description"] = top_essential["Protein"].map(desc_map)

        # st.subheader("📄 Essential Proteins with Descriptions")
        # st.dataframe(top_essential)

        # -------------------- RWR Section -------------------- #
        st.subheader("Random Walk With Restart (RWR)")
        st.write("""
        Random Walk with Restart (RWR) is a graph-based ranking method. 
        It starts from a group of seed proteins and repeatedly walks to their neighbors in the network. 
        At every step, the walker has a fixed probability of jumping back to the seed proteins. 
        Nodes that are visited more often during this process receive higher scores. 
        Proteins with high RWR scores are considered more biologically important and strongly connected to the seed proteins.
        """)

        A, nodes, idx = build_adj_matrix(G)
        seed_proteins = top_essential["Protein"].tolist()
        seed_idx = [idx[p] for p in seed_proteins if p in idx]

        rwr_scores = random_walk_with_restart(A, seed_idx)
        rwr_df = pd.DataFrame({
            "Protein": nodes,
            "RWR_Score": rwr_scores
        }).sort_values("RWR_Score", ascending=False)

        top_rwr = rwr_df.head(10)

        st.subheader(" Top 10 RWR Predicted Essential Proteins")
        st.dataframe(top_rwr)

        # Descriptions
        rwr_desc = fetch_string_descriptions(top_rwr["Protein"].tolist(), taxon_id)
        top_rwr["Description"] = top_rwr["Protein"].map(rwr_desc)

        st.subheader(" RWR Proteins with Descriptions")
        st.dataframe(top_rwr)

        # RWR Bar Plot
        st.subheader(" RWR Score Bar Plot")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=top_rwr, x="Protein", y="RWR_Score", palette="inferno")
        plt.xticks(rotation=45)
        plt.title("Top RWR Proteins")
        st.pyplot(fig)
        #show all distributions of centralitites
        st.subheader(" Centrality Distributions" )
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.kdeplot(df["Degree"], label="Degree", fill=True)
        sns.kdeplot(df["Betweenness"], label="Betweenness", fill=True)
        sns.kdeplot(df["Closeness"], label="Closeness", fill=True)
        sns.kdeplot(df["Eigenvector"], label="Eigenvector", fill=True)
        sns.kdeplot(df["PageRank"], label="PageRank", fill=True)
        plt.legend()
        st.pyplot(fig2)


        # ---------------- Adjacency Matrix ---------------- #
        # st.subheader("🧮 Adjacency Matrix")
        # st.dataframe(pd.DataFrame(A, index=nodes, columns=nodes))

    except Exception as e:
        st.error(f"Error: {e}")
