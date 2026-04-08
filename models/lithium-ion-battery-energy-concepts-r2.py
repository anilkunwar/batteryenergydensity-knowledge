import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import os
import tempfile
import warnings
from collections import defaultdict
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyvis.network import Network

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & DEVICE SETUP
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model identifiers (<1B constraint)
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pipeline hyperparameters
MIN_CONCEPT_FREQ = 3
MIN_CONCEPT_LENGTH_WORDS = 2
GNN_HIDDEN_DIM = 128
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3

# ==========================================
# MODEL LOADING (CACHED FOR STREAMLIT)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBED_NAME, device=DEVICE)

@st.cache_resource(show_spinner=False)
def load_lightweight_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, 
        torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

# ==========================================
# STEP 1-2: CONCEPT EXTRACTION & NORMALIZATION
# ==========================================
def extract_concepts_from_abstracts(abstracts, tokenizer, model):
    prompt_template = """Extract exactly the core scientific concepts (2+ words) from this abstract. 
Rules: 
- Output ONLY a JSON list of strings.
- Use nominalized form (e.g., 'thermal degradation' not 'degrades thermally').
- Remove filler words ('of', 'in', 'for').
- Standardize chemical formulas (e.g., LiNi0.8Mn0.1Co0.1O2).
- Do not include generic terms like 'research', 'study', 'results'.

Abstract: {text}
Concepts:"""
    
    all_concepts = []
    all_energy_values = []
    
    for i, text in enumerate(abstracts):
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:Wh/kg|W h kg⁻¹|Wh kg-1)', text, re.IGNORECASE)
        energies = [float(m) for m in matches]
        all_energy_values.append(energies)
        
        prompt = prompt_template.format(text=text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=120,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str)]
        except:
            fallback = re.findall(r'\b(?:[A-Za-z]+[\s-]*){2,4}\b', text)
            concepts = [c.lower() for c in fallback if len(c.split()) >= 2 and len(c) > 5]
            
        all_concepts.append(concepts)
        
    return all_concepts, all_energy_values

def normalize_and_filter_concepts(all_concepts):
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc:
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
                
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= MIN_CONCEPT_FREQ and len(c.split()) >= MIN_CONCEPT_LENGTH_WORDS]
    
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: TEMPORAL CONCEPT GRAPH & DISTANCE
# ==========================================
def build_concept_graph(all_concepts, concept_to_id):
    nx_graph = nx.Graph()
    for c in concept_to_id:
        nx_graph.add_node(c)
        
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1)
                    
    valid_nodes = [n for n, d in nx_graph.degree() if d >= MIN_CONCEPT_FREQ]
    graph_filtered = nx_graph.subgraph(valid_nodes).copy()
    d_prev_dict = dict(nx.all_pairs_shortest_path_length(graph_filtered, cutoff=4))
    
    return graph_filtered, d_prev_dict

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id):
    pos_pairs = list(nx_graph.edges())
    neg_pairs = []
    
    valid_ids = [concept_to_id[c] for c in valid_concepts]
    n_nodes = len(valid_ids)
    
    attempts = 0
    target_negs = min(len(pos_pairs) * 2, 2000)
    
    while len(neg_pairs) < target_negs and attempts < 15000:
        u_idx = np.random.randint(n_nodes)
        v_idx = np.random.randint(n_nodes)
        if u_idx == v_idx: continue
            
        u_concept = valid_concepts[u_idx]
        v_concept = valid_concepts[v_idx]
        
        if nx_graph.has_edge(u_concept, v_concept): continue
            
        try:
            dist = d_prev_dict[u_concept][v_concept]
        except:
            dist = float('inf')
            
        if dist == NEG_DPREV_FOCUS:
            neg_pairs.append((u_idx, v_idx))
        elif dist == 2 and np.random.rand() < 0.3:
            neg_pairs.append((u_idx, v_idx))
        attempts += 1
        
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.randint(n_nodes, size=2)
        if u_idx != v_idx and not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            if (u_idx, v_idx) not in neg_pairs and (v_idx, u_idx) not in neg_pairs:
                neg_pairs.append((u_idx, v_idx))
                
    return pos_pairs, neg_pairs

# ==========================================
# STEP 4: SEMANTIC NODE EMBEDDINGS
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

# ==========================================
# STEP 5: PURE PYTORCH SPARSE GRAPHSAGE (NO DGL)
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        # Build sparse adjacency: A
        A = sparse.FloatTensor(adj_indices.t(), adj_values, (num_nodes, num_nodes)).to(h.device)
        
        # Mean aggregation: h_neighbor = A @ h / deg
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs):
    # Prepare edge indices for sparse matrix
    src = torch.tensor([concept_to_id[u] for u, v in pos_pairs + neg_pairs], dtype=torch.long)
    dst = torch.tensor([concept_to_id[v] for u, v in pos_pairs + neg_pairs], dtype=torch.long)
    
    # Remove duplicates and self-loops for adjacency
    unique_edges = set()
    for u, v in pos_pairs:
        if u != v: unique_edges.add((min(u, v), max(u, v)))
            
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    
    model = SparseGraphSAGE(node_features.shape[1], GNN_HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pos_out, neg_out, _ = model(adj_indices, adj_values, len(concept_to_id), node_features, pos_u, pos_v, neg_u, neg_v)
        
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        loss = 0.5 * (pos_loss + neg_loss)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            st.info(f"Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss.item():.4f}")
            
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, len(concept_to_id), node_features, pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1])
        
    return model, final_embeddings.cpu(), adj_indices, adj_values

# ==========================================
# STEP 6: QUANTIFICATION & FEASIBILITY FILTER
# ==========================================
def compute_quantification_layer(valid_concepts, concept_abstract_map, all_energy_values, nx_graph):
    concept_energies = {}
    for c in valid_concepts:
        doc_indices = concept_abstract_map[c]
        energies = []
        for idx in doc_indices:
            energies.extend(all_energy_values[idx])
        concept_energies[c] = np.median(energies) if energies else 0.0
        
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_energies.get(u, 0), concept_energies.get(v, 0)
        w = nx_graph[u][v]['weight']
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.05)
        
    if len(X_feat) > 5:
        ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
    else:
        ridge = None
        
    return concept_energies, ridge

def compute_research_direction_scores(model, final_emb, nx_graph, valid_concepts, concept_energies, ridge, embed_model, d_prev_dict, adj_indices, adj_values):
    n_samples = min(3000, len(valid_concepts) * 5)
    u_ids = np.random.randint(len(valid_concepts), size=n_samples)
    v_ids = np.random.randint(len(valid_concepts), size=n_samples)
    
    filtered_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx: continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c): continue
        filtered_pairs.append((u_idx, v_idx, u_c, v_c))
        
    if not filtered_pairs:
        return pd.DataFrame()
        
    u_idx_tensor = torch.tensor([p[0] for p in filtered_pairs], dtype=torch.long, device=DEVICE)
    v_idx_tensor = torch.tensor([p[1] for p in filtered_pairs], dtype=torch.long, device=DEVICE)
    
    model.eval()
    with torch.no_grad():
        # Re-run forward pass for scoring
        _, _, h2 = model(adj_indices, adj_values, len(valid_concepts), final_emb.to(DEVICE), u_idx_tensor, v_idx_tensor, u_idx_tensor, v_idx_tensor)
        pair_cat = torch.cat([h2, h2], dim=1) # Simplified scoring for unlinked pairs
        gnn_logits = model.decoder(pair_cat).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
        
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
    cos_sims = np.sum(emb_np[u_idx_tensor.cpu().numpy()] * emb_np[v_idx_tensor.cpu().numpy()], axis=1)
    
    scores = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(filtered_pairs):
        try:
            d_prev = d_prev_dict[u_c][v_c]
        except:
            d_prev = 4
            
        if d_prev < 2: continue
        
        p_u = concept_energies.get(u_c, 0)
        p_v = concept_energies.get(v_c, 0)
        w_hist = 1.0
        
        expected_gain = 0
        if ridge is not None:
            expected_gain = float(ridge.predict([[p_u, p_v, w_hist]])[0])
            
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.5)
        
        alpha1, alpha2, alpha3, alpha4 = 0.4, 0.3, 0.2, 0.1
        norm_gain = (expected_gain - 200) / 150
        norm_gain = max(0, min(1, norm_gain))
        
        D_uv = (alpha1 * gnn_scores[i] + 
                alpha2 * semantic_novelty + 
                alpha3 * norm_gain - 
                alpha4 * (1.0 - feasibility))
        
        scores.append({
            'concept_u': u_c, 'concept_v': v_c,
            'd_prev': d_prev,
            'gnn_score': float(gnn_scores[i]),
            'expected_wh_kg': expected_gain,
            'feasibility': float(feasibility),
            'D_uv': float(D_uv)
        })
        
    scores_df = pd.DataFrame(scores).sort_values('D_uv', ascending=False)
    return scores_df.head(50)

# ==========================================
# STEP 7: LLM CURATION & RESEARCH DIRECTIONS
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model):
    results = []
    prompt_template = """You are a materials science research strategist. 
For the novel concept combination: "{u}" + "{v}"
Historical expected energy density: ~{wh:.1f} Wh/kg
Semantic feasibility: {feas:.2f}/1.0
Write exactly 3 sentences:
1. Why this combination is scientifically novel and underexplored.
2. Predicted energy density target and key property trade-off.
3. One concrete experimental validation step (e.g., coin cell testing, XPS, EIS, in situ XRD).
Be concise and technically precise."""

    for _, row in top_pairs_df.iterrows():
        prompt = prompt_template.format(
            u=row['concept_u'], 
            v=row['concept_v'],
            wh=float(row['expected_wh_kg']), 
            feas=float(row['feasibility'])
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inputs.input_ids, max_new_tokens=150, temperature=0.3, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        results.append({
            'Pair': f"{row['concept_u']} + {row['concept_v']}",
            'D_UV Score': f"{row['D_uv']:.3f}",
            'Expected Wh/kg': f"{row['expected_wh_kg']:.1f}",
            'Feasibility': f"{row['feasibility']:.2f}",
            'Research Direction': text
        })
    return pd.DataFrame(results)

# ==========================================
# STREAMLIT UI & PIPELINE ORCHESTRATION
# ==========================================
def main():
    st.set_page_config(page_title="LIB Concept Graph Predictor", layout="wide")
    st.title("🔋 Lightweight LLM + Concept Graph for LIB Research Directions")
    st.caption("Predicts novel, quantified research directions from scientific abstracts using <1B parameter models & Pure PyTorch GraphSAGE.")
    
    with st.sidebar:
        st.header("Configuration")
        st.slider("Min Concept Frequency", 2, 10, 3, key="min_freq")
        
    abstract_input = st.text_area("Paste scientific abstracts (one per line, or separated by blank lines):", height=200, 
                                  placeholder="Enter 10-50 recent battery materials abstracts...")
    
    if st.button("🚀 Run Prediction Pipeline", type="primary"):
        if not abstract_input.strip():
            st.error("Please enter at least one abstract.")
            return
            
        abstracts = [t.strip() for t in re.split(r'\n\s*\n', abstract_input) if t.strip()]
        if len(abstracts) < 5:
            st.warning("Pipeline works best with 10+ abstracts. Proceeding with available text.")
            
        progress = st.progress(0)
        status = st.status("Initializing models...", expanded=True)
        
        try:
            with status:
                st.write("Loading embeddings & LLM (<1B)...")
                embed_model = load_embedding_model()
                tokenizer, model = load_lightweight_llm()
                st.success("Models loaded successfully.")
            progress.progress(10)
            
            with st.status("Extracting concepts & energy metrics..."):
                all_concepts, all_energies = extract_concepts_from_abstracts(abstracts, tokenizer, model)
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts)
                st.write(f"✅ Extracted {len(valid_concepts)} unique concepts.")
            progress.progress(25)
            
            if len(valid_concepts) < 5:
                st.error("Too few valid concepts. Add more abstracts or reduce frequency filter.")
                return
                
            with st.status("Building concept graph & computing distances..."):
                nx_graph, d_prev_dict = build_concept_graph(all_concepts, concept_to_id)
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id)
                st.write(f"✅ Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges.")
            progress.progress(40)
            
            with st.status("Generating semantic embeddings..."):
                node_features = generate_embeddings(valid_concepts, embed_model)
            progress.progress(50)
            
            with st.status("Training Pure PyTorch GraphSAGE..."):
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs)
                st.success("GNN training complete.")
            progress.progress(65)
            
            with st.status("Computing energy density proxies & feasibility..."):
                concept_energies, ridge = compute_quantification_layer(valid_concepts, concept_abstract_map, all_energies, nx_graph)
                top_scores = compute_research_direction_scores(gnn_model, final_emb, nx_graph, valid_concepts, concept_energies, ridge, embed_model, d_prev_dict, adj_indices, adj_values)
            progress.progress(80)
            
            with st.status("Generating LLM-curated research directions..."):
                directions_df = generate_research_directions(top_scores, tokenizer, model)
            progress.progress(100)
            status.update(label="Pipeline complete!", state="complete", expanded=False)
            
            st.subheader("📊 Top Predicted Research Directions")
            st.dataframe(directions_df, use_container_width=True)
            
            st.subheader("🌐 Concept Graph Visualization")
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            net.barnes_hut()
            for node in nx_graph.nodes():
                net.add_node(node, label=node, title=f"Degree: {nx_graph.degree(node)}")
            for u, v in nx_graph.edges():
                net.add_edge(u, v)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                net.save_graph(tmp.name)
                with open(tmp.name, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=650, scrolling=True)
                    
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
