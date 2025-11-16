# app.py
from flask import Flask, request, jsonify, send_file
import requests
import networkx as nx
import matplotlib.pyplot as plt
import io
import traceback

app = Flask(__name__)

# Globals (session-level)
GLOBAL_DISEASE_ID = None
GLOBAL_GENES = []
GLOBAL_GRAPH = None

# Config
OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
STRING_NETWORK_URL = "https://version-12-0.string-db.org/api/json/network"
TIMEOUT = 15  # seconds for API calls


# -------------------------
# Helper: safe POST JSON
# -------------------------
def safe_post_json(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
    except Exception as e:
        return {"__error__": f"request-exception: {str(e)}"}, None
    # try parse json
    try:
        j = r.json()
    except Exception:
        return {"__error__": "invalid-json-response", "status_code": r.status_code, "text": r.text}, None
    return None, j


# -------------------------
# 1) Search disease -> return disease id + name
# -------------------------
@app.route("/search_disease", methods=["GET"])
def search_disease():
    q = request.args.get("q")
    if not q:
        return jsonify({"error": "provide ?q=<disease name>"}), 400

    query = """
    query SearchDisease($term: String!) {
      search(queryString: $term, entityNames: ["disease"]) {
        hits {
          object {
            ... on Disease {
              id
              name
              description
            }
          }
        }
      }
    }
    """
    payload = {"query": query, "variables": {"term": q}}

    err, data = safe_post_json(OT_GRAPHQL_URL, payload)
    if err:
        return jsonify({"error": "OpenTargets request failed", "details": err}), 502

    # defensive navigation
    hits = data.get("data", {}).get("search", {}).get("hits")
    if not hits:
        # return raw response to help debugging
        return jsonify({"error": "no hits in OpenTargets response", "raw": data}), 404

    # Take first hit's object which should be a Disease object
    first_obj = hits[0].get("object")
    if not first_obj:
        return jsonify({"error": "unexpected OpenTargets hit shape", "raw": data}), 500

    # object might contain id/name directly (depends on API)
    disease_id = first_obj.get("id") or first_obj.get("efoId") or first_obj.get("diseaseId")
    disease_name = first_obj.get("name") or first_obj.get("label")

    if not disease_id:
        # perhaps the hit object is wrapped differently; return raw for inspection
        return jsonify({"error": "couldn't extract disease id", "raw_object": first_obj}), 500

    return jsonify({"disease_id": disease_id, "disease_name": disease_name})


# -------------------------
# 2) Get genes for an EFO disease id (safe)
# -------------------------
@app.route("/get_genes", methods=["POST"])
def get_genes():
    """
    POST JSON: {"disease_id": "EFO_0001360"} OR {"disease": "type 2 diabetes"} (tries search then assoc)
    Response: {"disease_id":..., "genes":[...], "count": N}
    """
    global GLOBAL_DISEASE_ID, GLOBAL_GENES

    body = request.get_json(silent=True) or {}
    disease_id = body.get("disease_id")
    disease_name = body.get("disease")

    # If only disease name provided, search first
    if not disease_id and disease_name:
        # reuse search endpoint logic but inline here for simpler flow
        query = """
        query SearchDisease($term: String!) {
          search(queryString: $term, entityNames: ["disease"]) {
            hits {
              object {
                ... on Disease { id name }
              }
            }
          }
        }
        """
        payload = {"query": query, "variables": {"term": disease_name}}
        err, data = safe_post_json(OT_GRAPHQL_URL, payload)
        if err:
            return jsonify({"error": "OpenTargets search failed", "details": err}), 502
        hits = data.get("data", {}).get("search", {}).get("hits")
        if not hits:
            return jsonify({"error": f"No disease found for '{disease_name}'", "raw": data}), 404
        disease_obj = hits[0].get("object", {})
        disease_id = disease_obj.get("id")

    if not disease_id:
        return jsonify({"error": "Provide disease_id in POST JSON or disease name to search"}), 400

    GLOBAL_DISEASE_ID = disease_id

    # GraphQL query to fetch associatedTargets (safe parsing)
    query2 = """
    query GetAssociatedTargets($id: String!) {
      disease(efoId: $id) {
        associatedTargets(page: { index: 0, size: 200 }) {
          rows {
            score
            target {
              approvedSymbol
              id
            }
          }
        }
      }
    }
    """
    payload2 = {"query": query2, "variables": {"id": disease_id}}
    err, data2 = safe_post_json(OT_GRAPHQL_URL, payload2)
    if err:
        return jsonify({"error": "OpenTargets associatedTargets failed", "details": err}), 502

    # defensive parsing
    disease_block = data2.get("data", {}).get("disease")
    if not disease_block:
        return jsonify({"error": "No 'disease' key in OpenTargets response", "raw": data2}), 500

    assoc = disease_block.get("associatedTargets")
    if not assoc:
        return jsonify({"error": "No 'associatedTargets' in disease block", "raw": data2}), 500

    rows = assoc.get("rows", [])
    genes = []
    for r in rows:
        target = r.get("target", {})
        sym = target.get("approvedSymbol")
        if sym:
            genes.append(sym)

    if not genes:
        return jsonify({"error": "No genes found for disease", "raw": data2}), 404

    # store globally (limit if you want)
    GLOBAL_GENES = genes[:100]  # keep first 100 for safety
    return jsonify({"disease_id": disease_id, "count": len(GLOBAL_GENES), "genes": GLOBAL_GENES})


# -------------------------
# 3) Build PPI network using STRING for the stored genes
# -------------------------
def fetch_string_network_for_identifiers(identifiers_list):
    """
    identifiers_list: list of STRING identifiers or gene names.
    We'll send them joined by \r\n to STRING /network endpoint.
    """
    if not identifiers_list:
        return []
    ids_str = "\r".join(identifiers_list)
    params = {"identifiers": ids_str, "species": 9606, "required_score": 400}
    try:
        r = requests.post(STRING_NETWORK_URL, data=params, timeout=TIMEOUT)
    except Exception as e:
        return {"__error__": f"string-request-exception: {str(e)}"}
    try:
        return r.json()
    except Exception:
        return {"__error__": "invalid-json-from-string", "status_code": r.status_code, "text": r.text}


@app.route("/build_network", methods=["POST"])
def build_network():
    """
    Builds a NetworkX graph from GLOBAL_GENES (must have run /get_genes first).
    Optionally you may pass {"genes": [...]} in POST body to override GLOBAL_GENES.
    """
    global GLOBAL_GRAPH, GLOBAL_GENES

    body = request.get_json(silent=True) or {}
    genes = body.get("genes") or GLOBAL_GENES
    if not genes:
        return jsonify({"error": "No genes available. Run /get_genes or provide genes in POST body."}), 400

    # Call STRING network once with all gene names (STRING accepts gene names too)
    ppi_resp = fetch_string_network_for_identifiers(genes)
    if isinstance(ppi_resp, dict) and ppi_resp.get("__error__"):
        return jsonify({"error": "STRING request failed", "details": ppi_resp}), 502

    # ppi_resp expected to be a list of interaction dicts
    edges_added = 0
    G = nx.Graph()
    for item in ppi_resp:
        a = item.get("preferredName_A")
        b = item.get("preferredName_B")
        score = item.get("score", 0)
        if a and b:
            G.add_edge(a, b, weight=score)
            edges_added += 1

    if G.number_of_nodes() == 0:
        return jsonify({"error": "No interactions returned by STRING", "raw": ppi_resp}), 500

    GLOBAL_GRAPH = G
    return jsonify({"message": "Network built", "nodes": G.number_of_nodes(), "edges": G.number_of_edges()})


# -------------------------
# 4) Visualize (returns PNG)
# -------------------------
@app.route("/visualize", methods=["GET"])
def visualize():
    global GLOBAL_GRAPH
    if GLOBAL_GRAPH is None:
        return jsonify({"error": "Build network first (POST /build_network)"}), 400

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(GLOBAL_GRAPH, seed=42)
    nx.draw(GLOBAL_GRAPH, pos, with_labels=True, node_size=400, font_size=8)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# -------------------------
# 5) Top hubs (degree)
# -------------------------
@app.route("/top_hubs", methods=["GET"])
def top_hubs():
    global GLOBAL_GRAPH
    if GLOBAL_GRAPH is None:
        return jsonify({"error": "Build network first"}), 400
    deg = sorted(GLOBAL_GRAPH.degree(), key=lambda x: x[1], reverse=True)[:10]
    return jsonify({"top_hubs": [{"protein": p, "degree": d} for p, d in deg]})


# -------------------------
# 6) Top connectors (betweenness)
# -------------------------
@app.route("/top_connectors", methods=["GET"])
def top_connectors():
    global GLOBAL_GRAPH
    if GLOBAL_GRAPH is None:
        return jsonify({"error": "Build network first"}), 400
    bc = nx.betweenness_centrality(GLOBAL_GRAPH)
    top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    return jsonify({"top_connectors": [{"protein": p, "betweenness": v} for p, v in top]})


# -------------------------
# Root
# -------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Flask PPI backend running. Use /search_disease, /get_genes, /build_network, /visualize, /top_hubs, /top_connectors"})


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
