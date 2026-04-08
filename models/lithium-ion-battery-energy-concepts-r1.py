import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import math
import os
import tempfile
import warnings
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
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
GNN_LAYERS = 2
TRAIN_EPOCHS = 50
LR = 1e-3
NEG_DPREV_FOCUS = 3  # Hard negative sampling target
MIXTURE_ALPHA_INIT = 0.5

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
    """Extracts and normalizes concepts using lightweight LLM + regex fallback."""
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
        # Energy density extraction (Wh/kg)
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:Wh/kg|W h kg⁻¹|Wh kg-1)', text, re.IGNORECASE)
        energies = [float(m) for m in matches]
        all_energy_values.append(energies)
        
        # LLM extraction
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
        
        # Parse JSON safely
        concepts = []
        try:
            parsed = json.loads(response.replace("'", '"').strip())
            if isinstance(parsed, list):
                concepts = [c.strip().lower().rstrip('.') for c in parsed if isinstance(c, str)]
        except:
            # Fallback: regex noun phrases
            fallback = re.findall(r'\b(?:[A-Za-z]+[\s-]*){2,4}\b', text)
            concepts = [c.lower() for c in fallback if len(c.split()) >= 2 and len(c) > 5]
            
        all_concepts.append(concepts)
        
    return all_concepts, all_energy_values

def normalize_and_filter_concepts(all_concepts):
    """Filters concepts by frequency and length, maps to unique IDs."""
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc:
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
                
    # Filter by MIN_CONCEPT_FREQ and MIN_CONCEPT_LENGTH_WORDS
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= MIN_CONCEPT_FREQ and len(c.split()) >= MIN_CONCEPT_LENGTH_WORDS]
    
    # Map to integer IDs
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

# ==========================================
# STEP 3: TEMPORAL CONCEPT GRAPH & DISTANCE
# ==========================================
def build_concept_graph(all_concepts, concept_to_id):
    """Builds co-occurrence graph and computes d_prev distances."""
    nx_graph = nx.Graph()
    
    # Add nodes
    for c in concept_to_id:
        nx_graph.add_node(c)
        
    # Add edges from co-occurrence
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1)
                    
    # Filter nodes by degree
    valid_nodes = [n for n, d in nx_graph.degree() if d >= MIN_CONCEPT_FREQ]
    graph_filtered = nx_graph.subgraph(valid_nodes).copy()
    
    # Precompute shortest paths for d_prev (expensive but necessary)
    d_prev_dict = dict(nx.all_pairs_shortest_path_length(graph_filtered, cutoff=4))
    
    return graph_filtered, d_prev_dict

def sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id):
    """Samples positive (existing) and hard negative (d_prev=3) edges."""
    pos_pairs = list(nx_graph.edges())
    neg_pairs = []
    
    valid_ids = [concept_to_id[c] for c in valid_concepts]
    n_nodes = len(valid_ids)
    
    # Hard negative sampling: focus on d_prev == 3
    attempts = 0
    target_negs = min(len(pos_pairs) * 2, 2000)  # Cap negatives for speed
    
    while len(neg_pairs) < target_negs and attempts < 10000:
        u_idx = np.random.randint(n_nodes)
        v_idx = np.random.randint(n_nodes)
        if u_idx == v_idx:
            continue
            
        u_concept = valid_concepts[u_idx]
        v_concept = valid_concepts[v_idx]
        
        if nx_graph.has_edge(u_concept, v_concept):
            continue
            
        # Compute d_prev
        try:
            dist = d_prev_dict[u_concept][v_concept]
        except:
            dist = float('inf')
            
        if dist == NEG_DPREV_FOCUS:
            neg_pairs.append((u_idx, v_idx))
        elif dist == 2 and np.random.rand() < 0.3:  # Some d_prev=2 for balance
            neg_pairs.append((u_idx, v_idx))
        attempts += 1
        
    # If hard negatives insufficient, fill with random non-edges
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.randint(n_nodes, size=2)
        if u_idx != v_idx and not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            if (u_idx, v_idx) not in neg_pairs and (v_idx, u_idx) not in neg_pairs:
                neg_pairs.append((u_idx, v_idx))
                
    pos_labels = np.ones(len(pos_pairs), dtype=np.float32)
    neg_labels = np.zeros(len(neg_pairs), dtype=np.float32)
    
    return pos_pairs, neg_pairs, np.concatenate([pos_labels, neg_labels])

# ==========================================
# STEP 4: SEMANTIC NODE EMBEDDINGS
# ==========================================
def generate_embeddings(valid_concepts, embed_model):
    """Generates mean-pooled embeddings for all valid concepts."""
    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False)
    return torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

# ==========================================
# STEP 5: GNN LINK PREDICTION (GraphSAGE)
# ==========================================
class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean')
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, g, h, pos_u, pos_v, neg_u, neg_v):
        h1 = F.relu(self.conv1(g, h))
        h2 = self.conv2(g, h1)
        
        # Positive scores
        pos_h_u = h2[pos_u]
        pos_h_v = h2[pos_v]
        pos_scores = self.decoder(torch.cat([pos_h_u, pos_h_v], dim=1)).squeeze(1)
        
        # Negative scores
        neg_h_u = h2[neg_u]
        neg_h_v = h2[neg_v]
        neg_scores = self.decoder(torch.cat([neg_h_u, neg_h_v], dim=1)).squeeze(1)
        
        return pos_scores, neg_scores, h2

def train_gnn(g_dgl, node_features, pos_pairs, neg_pairs):
    """Trains GraphSAGE link predictor."""
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=DEVICE)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=DEVICE)
    
    model = GraphSAGELinkPredictor(node_features.shape[1], GNN_HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pos_out, neg_out, _ = model(g_dgl, node_features, pos_u, pos_v, neg_u, neg_v)
        
        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
        loss = 0.5 * (pos_loss + neg_loss)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            st.info(f"Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss.item():.4f}")
            
    # Get final node embeddings
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(g_dgl, node_features, pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1])
        
    return model, final_embeddings.cpu()

# ==========================================
# STEP 6: QUANTIFICATION & FEASIBILITY FILTER
# ==========================================
def compute_quantification_layer(valid_concepts, concept_abstract_map, all_energy_values, nx_graph):
    """Computes historical energy proxy, expected gain, and feasibility."""
    concept_energies = {}
    for c in valid_concepts:
        doc_indices = concept_abstract_map[c]
        energies = []
        for idx in doc_indices:
            energies.extend(all_energy_values[idx])
        concept_energies[c] = np.median(energies) if energies else 0.0
        
    # Train ridge regressor for expected synergy
    X_feat, y_target = [], []
    for u, v in nx_graph.edges():
        pu, pv = concept_energies.get(u, 0), concept_energies.get(v, 0)
        # Use edge weight as proxy for historical synergy
        w = nx_graph[u][v]['weight']
        X_feat.append([pu, pv, w])
        y_target.append(max(pu, pv) * 1.05)  # Simple target assumption
        
    if len(X_feat) > 5:
        X = np.array(X_feat)
        y = np.array(y_target)
        ridge = Ridge(alpha=1.0).fit(X, y)
    else:
        ridge = None
        
    return concept_energies, ridge

def compute_research_direction_scores(model, final_emb, nx_graph, concept_to_id, valid_concepts, concept_energies, ridge, embed_model, d_prev_dict, pos_pairs):
    """Computes D_uv score for all unlinked pairs."""
    scores = []
    
    # Convert concept pairs to IDs for lookup
    concept_pairs = list(nx_graph.edges())
    positive_set = set((min(u,v), max(v,u)) for u,v in concept_pairs)
    
    model.eval()
    with torch.no_grad():
        # Prepare decoder inputs for all pairs
        all_ids = list(range(len(valid_concepts)))
        
        # Sample candidate pairs efficiently
        n_samples = min(5000, len(valid_concepts) * 5)
        u_ids = np.random.randint(len(valid_concepts), size=n_samples)
        v_ids = np.random.randint(len(valid_concepts), size=n_samples)
        
        filtered_pairs = []
        for u_idx, v_idx in zip(u_ids, v_ids):
            if u_idx == v_idx: continue
            u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
            if nx_graph.has_edge(u_c, v_c): continue
            filtered_pairs.append((u_idx, v_idx, u_c, v_c))
            
        if not filtered_pairs:
            return []
            
        u_idx_tensor = torch.tensor([p[0] for p in filtered_pairs], dtype=torch.long, device=DEVICE)
        v_idx_tensor = torch.tensor([p[1] for p in filtered_pairs], dtype=torch.long, device=DEVICE)
        
        # GNN scores
        h_u = final_emb[u_idx_tensor].to(DEVICE)
        h_v = final_emb[v_idx_tensor].to(DEVICE)
        pair_cat = torch.cat([h_u, h_v], dim=1)
        gnn_logits = model.decoder(pair_cat).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).cpu().numpy()
        
        # Semantic novelty
        emb_np = embed_model.encode(valid_concepts, show_progress_bar=False)
        cos_sims = np.sum(emb_np[u_idx_tensor.numpy()] * emb_np[v_idx_tensor.numpy()], axis=1)
        
        # Quantification & Feasibility
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
                expected_gain = ridge.predict([[p_u, p_v, w_hist]])[0]
                
            semantic_novelty = 1.0 - cos_sims[i]
            feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.5)
            
            # D_uv formula
            alpha1, alpha2, alpha3, alpha4 = 0.4, 0.3, 0.2, 0.1
            norm_gain = (expected_gain - 200) / 150  # Rough normalization to ~[0,1]
            norm_gain = max(0, min(1, norm_gain))
            
            D_uv = (alpha1 * gnn_scores[i] + 
                    alpha2 * semantic_novelty + 
                    alpha3 * norm_gain - 
                    alpha4 * (1.0 - feasibility))
            
            scores.append({
                'concept_u': u_c, 'concept_v': v_c,
                'd_prev': d_prev,
                'gnn_score': gnn_scores[i],
                'expected_wh_kg': expected_gain,
                'feasibility': feasibility,
                'D_uv': D_uv
            })
            
    # Sort and return top
    scores_df = pd.DataFrame(scores).sort_values('D_uv', ascending=False)
    return scores_df.head(50)

# ==========================================
# STEP 7: LLM CURATION & RESEARCH DIRECTIONS
# ==========================================
def generate_research_directions(top_pairs_df, tokenizer, model):
    """Generates curated research directions using lightweight LLM."""
    results = []
    prompt_template = """You are a materials science research strategist. 
For the novel concept combination: "{u}" + "{v}"
Historical expected energy density: ~{wh} Wh/kg
Semantic feasibility: {feas:.2f}/1.0
Write exactly 3 sentences:
1. Why this combination is scientifically novel and underexplored.
2. Predicted energy density target and key property trade-off.
3. One concrete experimental validation step (e.g., coin cell testing, XPS, EIS, in situ XRD).
Be concise and technically precise."""

    for _, row in top_pairs_df.iterrows():
        prompt = prompt_template.format(
            u=row['concept_u'], v=row['concept_v'],
            wh=row['expected_wh_kg']:.1f, feas=row['feasibility']
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
    st.caption("Predicts novel, quantified research directions from scientific abstracts using <1B parameter models & GraphSAGE.")
    
    with st.sidebar:
        st.header("Configuration")
        min_freq = st.slider("Min Concept Frequency", 2, 10, 3)
        st.info("Lower the threshold if your dataset is small (<500 abstracts).")
        
    abstract_input = st.text_area("Paste scientific abstracts (one per line, or separated by blank lines):", height=200, 
                                  placeholder="Enter 5-50 recent battery materials abstracts...")
    
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
            # Load Models
            with status:
                st.write("Loading embeddings & LLM (<1B)...")
                embed_model = load_embedding_model()
                tokenizer, model = load_lightweight_llm()
                st.success("Models loaded successfully.")
            progress.progress(10)
            
            # Step 1-2: Extract
            with st.status("Extracting concepts & energy metrics..."):
                all_concepts, all_energies = extract_concepts_from_abstracts(abstracts, tokenizer, model)
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts)
                st.write(f"✅ Extracted {len(valid_concepts)} unique concepts.")
            progress.progress(25)
            
            if len(valid_concepts) < 5:
                st.error("Too few valid concepts. Add more abstracts or reduce frequency filter.")
                return
                
            # Step 3: Graph
            with st.status("Building concept graph & computing distances..."):
                nx_graph, d_prev_dict = build_concept_graph(all_concepts, concept_to_id)
                pos_pairs, neg_pairs, labels = sample_edges_for_training(nx_graph, d_prev_dict, valid_concepts, concept_to_id)
                st.write(f"✅ Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges.")
                st.write(f"✅ Training pairs: {len(pos_pairs)} pos, {len(neg_pairs)} neg.")
            progress.progress(40)
            
            # Step 4: Embeddings
            with st.status("Generating semantic embeddings..."):
                node_features = generate_embeddings(valid_concepts, embed_model)
            progress.progress(50)
            
            # Step 5: GNN
            with st.status("Training GraphSAGE link predictor..."):
                src, dst = zip(*pos_pairs + neg_pairs)
                src = torch.tensor([concept_to_id[valid_concepts[u]] for u in src], dtype=torch.long)
                dst = torch.tensor([concept_to_id[valid_concepts[v]] for v in dst], dtype=torch.long)
                g_dgl = dgl.graph((src, dst), num_nodes=len(valid_concepts))
                g_dgl = dgl.add_reverse_edges(g_dgl)
                
                gnn_model, final_emb = train_gnn(g_dgl, node_features, pos_pairs, neg_pairs)
                st.success("GNN training complete.")
            progress.progress(65)
            
            # Step 6: Quantification
            with st.status("Computing energy density proxies & feasibility..."):
                concept_energies, ridge = compute_quantification_layer(valid_concepts, concept_abstract_map, all_energies, nx_graph)
                top_scores = compute_research_direction_scores(gnn_model, final_emb, nx_graph, concept_to_id, valid_concepts, concept_energies, ridge, embed_model, d_prev_dict, pos_pairs)
            progress.progress(80)
            
            # Step 7: Curation
            with st.status("Generating LLM-curated research directions..."):
                directions_df = generate_research_directions(top_scores, tokenizer, model)
            progress.progress(100)
            status.update(label="Pipeline complete!", state="complete", expanded=False)
            
            # Display Results
            st.subheader("📊 Top Predicted Research Directions")
            st.dataframe(directions_df, use_container_width=True)
            
            # Interactive Graph
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
