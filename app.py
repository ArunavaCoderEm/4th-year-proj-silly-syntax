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

def plot_network(G, top_nodes_set):
    plt.figure(figsize=(12,10))
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [3000 for _ in G.nodes()]
    node_colors = ["red" if n in top_nodes_set else "skyblue" for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color="gray", alpha=0.7)
    plt.title("PPI Network", fontsize=16, color="darkblue")
    st.pyplot(plt)

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="PPI Network Centrality Explorer", layout="wide")
st.title("🌟 PPI Network & Centrality Explorer")
st.markdown("<p style='font-size:16px;color:green;'>Visualize PPI network, centralities, and download full table.</p>", unsafe_allow_html=True)

# Use keys to keep values when rerunning
disease = st.text_input("Disease name", placeholder="e.g., Breast carcinoma", key="disease_input")
species = st.selectbox("Species", list(SPECIES_MAP.keys()), index=0, key="species_input")

if st.button("Build Network"):
    if not disease.strip():
        st.error("Please enter a disease name")
    else:
        try:
            taxon_id = SPECIES_MAP[species]

            # 1. Query Open Targets
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
                top_degree = sorted(deg_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
                top_betweenness = sorted(bet_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
                top_closeness = sorted(clo_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
                top_eigenvector = sorted(eig_c.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

                st.subheader(f"🔴 Top {TOP_N} Hubs (Degree Centrality)")
                st.write(", ".join([f"{p} ({deg_c[p]:.3f})" for p,_ in top_degree]))
                st.subheader(f"🟠 Top {TOP_N} Spreaders (Betweenness Centrality)")
                st.write(", ".join([f"{p} ({bet_c[p]:.3f})" for p,_ in top_betweenness]))
                st.subheader(f"🟢 Top {TOP_N} Close Connectors (Closeness Centrality)")
                st.write(", ".join([f"{p} ({clo_c[p]:.3f})" for p,_ in top_closeness]))
                st.subheader(f"🟣 Top {TOP_N} Influencers (Eigenvector Centrality)")
                st.write(", ".join([f"{p} ({eig_c[p]:.3f})" for p,_ in top_eigenvector]))

                # 6. Plot network highlighting top degree hubs
                top_nodes_set = set([p for p,_ in top_degree])
                plot_network(G, top_nodes_set)

                # 7. Centrality Table (ALL nodes)
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

                # 8. Download CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="⬇️ Download Full Table as CSV", data=csv, file_name=f"{disease}_ppi_centrality.csv", mime="text/csv")

                # 9. Distribution plots
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
