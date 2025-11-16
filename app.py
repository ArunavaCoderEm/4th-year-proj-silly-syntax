import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
              associatedTargets(page: { size: 50, index: 0 }) {
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

# ---------------- FETCH STRING DESCRIPTIONS ---------------- #
def fetch_string_descriptions(proteins, taxon_id):
    """
    Fetch protein description from STRING database using gene names.
    """
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

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="PPI Network Centrality Explorer")
st.title("🌟 PPI Network, Centrality & Essential Protein Explorer")
st.markdown(
    "<p style='font-size:16px;color:green;'>Visualize the PPI network, centralities, and download the full table.</p>",
    unsafe_allow_html=True
)

disease = st.text_input("Disease name", placeholder="e.g., Breast carcinoma", key="disease_input")
species = st.selectbox("Species", list(SPECIES_MAP.keys()), index=0, key="species_input")

if st.button("Build Network"):
    if not disease.strip():
        st.error("Please enter a disease name")
    else:
        try:
            taxon_id = SPECIES_MAP[species]

            st.markdown(f"<h2 style='color:#1D3557; font-family:Courier New;'>Disease: {disease}</h2>", unsafe_allow_html=True)

            # 1. Open Targets
            target_info = query_open_targets(disease)
            if not target_info:
                st.warning("No targets found.")
            else:
                genes = [t[0] for t in target_info]

                # 2. STRING mapping
                gene_to_string = map_genes_to_string_ids(genes, taxon_id)
                string_ids = list(gene_to_string.values())
                ppi_data = fetch_ppi(string_ids, taxon_id)

                # 3. Build network
                G = nx.Graph()
                for inter in ppi_data:
                    a = inter.get("preferredName_A")
                    b = inter.get("preferredName_B")
                    score = inter.get("score", 0)
                    if a and b:
                        G.add_edge(a, b, weight=score)

                # 4. Centralities
                deg_c = nx.degree_centrality(G)
                bet_c = nx.betweenness_centrality(G)
                clo_c = nx.closeness_centrality(G)
                eig_c = nx.eigenvector_centrality(G, max_iter=500)

                # 5. Top 10 per centrality
                top_degree = [p for p,_ in sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
                top_betweenness = [p for p,_ in sorted(bet_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
                top_closeness = [p for p,_ in sorted(clo_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
                top_eigenvector = [p for p,_ in sorted(eig_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]

                st.subheader("🔴 Top Degree Hubs")
                st.write(", ".join(top_degree))
                st.subheader("🟠 Top Betweenness Spreaders")
                st.write(", ".join(top_betweenness))
                st.subheader("🟢 Top Closeness Connectors")
                st.write(", ".join(top_closeness))
                st.subheader("🟣 Top Eigenvector Influencers")
                st.write(", ".join(top_eigenvector))

                # 6. Plot network
                plot_network_multi(G, top_degree, top_betweenness, top_closeness, top_eigenvector)

                # 7. Centrality Table (all nodes)
                table = []
                for node in G.nodes():
                    table.append({
                        "Protein": node,
                        "Degree": deg_c[node],
                        "Betweenness": bet_c[node],
                        "Closeness": clo_c[node],
                        "Eigenvector": eig_c[node]
                    })
                df = pd.DataFrame(table)
                st.subheader("📊 Protein Centrality Table (All Nodes)")
                st.dataframe(df, height=400)

                # ---------------- ADDITION: ESSENTIALITY SCORE ---------------- #
                df_score = df.copy()
                for col in ["Degree", "Betweenness", "Closeness", "Eigenvector"]:
                    min_val = df_score[col].min()
                    max_val = df_score[col].max()
                    if max_val - min_val > 0:
                        df_score[col + "_norm"] = (df_score[col] - min_val) / (max_val - min_val)
                    else:
                        df_score[col + "_norm"] = 0

                df_score["EssentialityScore"] = df_score[["Degree_norm", "Betweenness_norm", "Closeness_norm", "Eigenvector_norm"]].sum(axis=1)

                top_essential = df_score.sort_values("EssentialityScore", ascending=False).head(TOP_N)
                st.subheader("⭐ Top 10 Essential Proteins")
                st.write(top_essential[["Protein", "EssentialityScore"]])

                # ---------------- FETCH DESCRIPTIONS FROM STRING ---------------- #
                top_essential_desc = fetch_string_descriptions(top_essential["Protein"].tolist(), taxon_id)
                top_essential["Description"] = top_essential["Protein"].map(top_essential_desc)

                st.subheader("📄 Top Essential Proteins with STRING Descriptions")
                st.dataframe(top_essential[["Protein", "EssentialityScore", "Description"]], height=400)

                # ---------------- BAR PLOT OF ESSENTIALITY SCORE ---------------- #
                st.subheader("📊 Essentiality Score Bar Plot")
                fig, ax = plt.subplots(figsize=(10,5))
                sns.barplot(x="Protein", y="EssentialityScore", data=top_essential, palette="viridis", ax=ax)
                plt.xticks(rotation=45)
                plt.ylabel("Essentiality Score (sum of normalized centralities)")
                plt.title("Top 10 Essential Proteins")
                st.pyplot(fig)

                # 8. Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Full Table as CSV", data=csv,
                                   file_name=f"{disease}_ppi_centrality.csv", mime="text/csv")

                # 9. Centrality Distributions
                st.subheader("📈 Centrality Distributions")
                fig, axes = plt.subplots(2,2, figsize=(12,8))
                sns.histplot(list(deg_c.values()), kde=True, ax=axes[0,0], color='skyblue'); axes[0,0].set_title("Degree Centrality")
                sns.histplot(list(bet_c.values()), kde=True, ax=axes[0,1], color='orange'); axes[0,1].set_title("Betweenness Centrality")
                sns.histplot(list(clo_c.values()), kde=True, ax=axes[1,0], color='green'); axes[1,0].set_title("Closeness Centrality")
                sns.histplot(list(eig_c.values()), kde=True, ax=axes[1,1], color='purple'); axes[1,1].set_title("Eigenvector Centrality")
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
