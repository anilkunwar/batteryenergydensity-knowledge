#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RechargeableBatteries-ConceptGraph: Advanced Battery & Electrocatalysis Concept Graph Builder
==================================================================================
Concept graph extraction for Zinc-Air Batteries (ZABs), Sodium-Ion Batteries (SIBs),
Seawater Batteries, and Electrocatalysis (HER/OER/ORR) research from JSON metadata.
Optimized for small-to-large corpora (9-3000+ abstracts) with robust statistical methods.

Features:
- Robust JSON/JSONL/CSV loading with BOM handling and error recovery
- Domain-optimized concept extraction (TF-IDF, semantic clustering, PageRank)
- Advanced battery & electrocatalysis filtering (ZAB, SIB, HER, OER, ORR, MOF, MXene, etc.)
- Unicode unit handling for Scopus exports (mAh∙g−1 → mAh/g)
- Interactive PyVis/Plotly 2D/3D visualizations with 50+ colormaps
- Statistical validation: modularity, silhouette, centrality, bootstrap CIs
- GNN-powered (GraphSAGE) research direction scoring with PyTorch
- Export: GraphML, JSON, CSV, HTML, SVG, PNG
- Streamlit UI with session persistence and crash prevention

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn

Run: streamlit run lib_concept_graph_updated.py

Place JSON files in ./json_metadatabase/ folder next to this script.
"""
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
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
from collections import defaultdict, Counter
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patches as mpatches
import seaborn as sns

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RechargeableBatteries-ConceptGraph: Advanced Battery & Electrocatalysis Explorer",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PATHS & DIRECTORIES
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ==========================================
# COLORMAP REGISTRY (50+)
# ==========================================
SUPPORTED_COLORMAPS = {
    "viridis": "Viridis", "plasma": "Plasma", "inferno": "Inferno", "magma": "Magma",
    "cividis": "Cividis", "turbo": "Turbo", "jet": "Jet", "rainbow": "Rainbow",
    "hsv": "Hsv", "nipy_spectral": "NipySpectral", "gist_rainbow": "GistRainbow",
    "coolwarm": "Coolwarm", "RdBu": "RdBu", "seismic": "Seismic", "Spectral": "Spectral",
    "tab10": "Set1", "tab20": "Set2", "tab20b": "Set3", "Accent": "Accent",
    "Dark2": "Dark2", "Paired": "Paired", "Pastel1": "Pastel1", "Pastel2": "Pastel2",
    "cubehelix": "Cubehelix", "bone": "Bone", "gray": "Gray", "pink": "Pink",
    "spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter",
    "cool": "Cool", "hot": "Hot", "twilight": "Twilight", "copper": "Copper",
    "YlOrRd": "YlOrRd", "OrRd": "OrRd", "PuRd": "PuRd", "RdPu": "RdPu",
    "BuPu": "BuPu", "GnBu": "GnBu", "YlGnBu": "YlGnBu", "PuBuGn": "PuBuGn",
    "BuGn": "BuGn", "YlGn": "YlGn", "Greys": "Greys", "afmhot": "Afmhot",
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean"
}

def get_colormap_colors(cmap_name: str, n: int) -> List[str]:
    """Convert matplotlib colormap to list of hex colors for Plotly/PyVis"""
    try:
        cmap = cm.get_cmap(cmap_name, n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        cmap = cm.get_cmap("viridis", n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]

# ==========================================
# ROBUST JSON LOADER (from JSON Metadata Explorer)
# ==========================================
def robust_load_file(filepath: Path):
    """Try multiple strategies to load a file that claims to be JSON."""
    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    sanitized = re.sub(r'\bNaN\b', 'null', text)
    sanitized = re.sub(r'\bInfinity\b', 'null', sanitized)
    sanitized = re.sub(r'\b-Infinity\b', 'null', sanitized)
    sanitized = re.sub(r',(\s*[}\]])', r'\1', sanitized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        pass
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if records:
        return records
    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient="records")
    except Exception:
        pass
    preview = text[:300]
    raise ValueError(f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}...")

@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    """Load every .json file in directory and return a list of (filepath, records)."""
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        return []
    loaded = []
    for fp in files:
        try:
            data = robust_load_file(fp)
            if isinstance(data, list):
                loaded.append((str(fp.name), data))
            elif isinstance(data, dict):
                loaded.append((str(fp.name), [data]))
            else:
                loaded.append((str(fp.name), []))
        except Exception as e:
            st.error(f"Error loading `{fp.name}`: {e}")
            try:
                raw_bytes = fp.read_bytes()[:300]
                hex_str = raw_bytes.hex()
                formatted = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
                st.code(f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}", language="text")
            except Exception:
                pass
    return loaded

@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    """Flatten all records into one DataFrame."""
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            rec["_source_file"] = fname
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

# ==========================================
# ADVANCED BATTERIES & ELECTROCATALYSIS DOMAIN CONFIGURATION
# ==========================================
PERFORMANCE_METRICS_KEYWORDS = [
    "energy density", "power density", "specific energy", "wh/kg", "wh/l", "mah/g", "mah/cm2",
    "overpotential", "half-wave potential", "current density", "cell voltage", "open circuit voltage",
    "cycling stability", "durability", "cycle life", "capacity retention", "coulombic efficiency",
    "conversion efficiency", "solar to hydrogen", "corrosion resistance", "stability"
]

BATTERY_ELECTRODE_MATERIALS = [
    "sn", "tin", "sb", "antimony", "mof", "metal-organic framework", "terephthalic acid",
    "ti3c2tx", "mxene", "g-c3n4", "carbon nitride", "negatrode", "positrode",
    "zn", "zinc", "air cathode", "air-cathode", "nis", "cos2", "nd", "neodymium", 
    "mo2n", "cop", "ceo2", "cnt", "carbon nanotube", "cuo", "co-cn", "fe-se-c", 
    "nio", "iro2", "mo4o11", "co-ni", "single atom", "single-atom", "fe-n-c", 
    "n-doped carbon", "zif67", "nc", "nc-cnt"
]

REACTIONS_KEYWORDS = [
    "her", "oer", "orr", "water splitting", "seawater splitting", "overall water splitting",
    "hydrogen evolution", "oxygen evolution", "oxygen reduction", "electrocatalysis", 
    "electrolysis", "redox"
]

STRUCTURES_KEYWORDS = [
    "heterostructure", "heterointerface", "bilayer", "nanosheet", "nanowire", "nanocube", 
    "nanofiber", "nanofoam", "nano foam", "hollow carbon", "hierarchical", "core-shell",
    "single-atom catalyst", "sa catalyst", "bifunctional", "trifunctional"
]

ELECTROLYTE_KEYWORDS = [
    "alkaline", "seawater", "freshwater", "aqueous", "non-aqueous", "liquid electrolyte",
    "solid electrolyte", "zinc-air", "zab", "sib", "lib", "pouch cell", "pouch battery"
]

# Combine them all for the validation function
ALL_DOMAIN_KEYWORDS = (PERFORMANCE_METRICS_KEYWORDS + BATTERY_ELECTRODE_MATERIALS + 
                       REACTIONS_KEYWORDS + STRUCTURES_KEYWORDS + ELECTROLYTE_KEYWORDS)

# Keep old variable names pointing to new lists so the rest of the script doesn't break
ENERGY_DENSITY_KEYWORDS = PERFORMANCE_METRICS_KEYWORDS
CATHODE_MATERIALS = BATTERY_ELECTRODE_MATERIALS
ANODE_MATERIALS = BATTERY_ELECTRODE_MATERIALS

# Robust patterns that handle Scopus's weird Unicode minus signs (−) and dots (∙)
BATTERY_PATTERNS = [
    # Capacity and Energy Density (robust to unicode minus/superscripts)
    r'\d+(?:\.\d+)?\s*mah.{0,3}?g.{0,3}?1',
    r'\d+(?:\.\d+)?\s*wh.{0,3}?kg.{0,3}?1',
    r'\d+(?:\.\d+)?\s*mah.{0,3}?cm.{0,3}?2',
    # Power Density and Current Density
    r'\d+(?:\.\d+)?\s*mw.{0,3}?cm.{0,3}?2',
    r'\d+(?:\.\d+)?\s*ma.{0,3}?cm.{0,3}?2',
    r'\d+(?:\.\d+)?\s*a.{0,3}?cm.{0,3}?2',
    # Overpotential, Voltage, and Potential
    r'\d+(?:\.\d+)?\s*(?:mv|v)',
    r'(?:overpotential|half-wave potential|cell voltage|open circuit voltage|operating voltage)',
    # Specific Chemical Formulas & Materials from your JSON
    r'(?:Ti3C2Tx|g-C3N4|NiS|CoS2|Mo2N|CoP|CeO2|CuO|Fe-Se-C|NiO|IrO2|Mo4O11|ZIF67|Co-CN|NC-CNT)',
    # Battery Types & Configurations
    r'(?:zinc-air|zab|sodium-ion|sib|lithium-ion|lib|zn-seawater|seawater battery|pouch cell|pouch battery|half-pouch|full-pouch)',
    # Electrocatalysis & Reactions
    r'(?:her|oer|orr|water splitting|seawater splitting|overall water splitting|hydrogen evolution|oxygen evolution|oxygen reduction)',
    # Structural Keywords
    r'(?:single-atom|single atom|sa catalyst|mof|metal-organic framework|heterostructure|heterointerface|nanosheet|nanowire|nanocube|bilayer|hollow carbon|n-doped|nitrogen-doped|nanofoam|nano foam)',
    # Electrode roles
    r'(?:negatrode|positrode|air cathode|air-cathode|anode|cathode|electrocatalyst|bifunctional catalyst|trifunctional electrocatalyst)'
]

BATTERY_CATEGORY_MAPPING = {
    r'zinc.?air|zab|zn.?air': 'zinc_air_battery',
    r'sodium.?ion|sib|na.?ion': 'sodium_ion_battery',
    r'lithium.?ion|lib|li.?ion': 'lithium_ion_battery',
    r'zn.?seawater|seawater.?battery': 'seawater_battery',
    r'water.?splitting|seawater.?splitting|electrolysis': 'water_splitting',
    r'her|hydrogen.?evolution': 'hydrogen_evolution',
    r'oer|oxygen.?evolution': 'oxygen_evolution',
    r'orr|oxygen.?reduction': 'oxygen_reduction',
    r'overpotential|half-wave|current.?density|power.?density': 'performance_metric',
    r'mof|metal.?organic.?framework': 'mof_material',
    r'single.?atom|sa.?catalyst|fe.?n.?c': 'single_atom_catalyst',
    r'heterostructure|heterointerface|bilayer|core.?shell': 'heterostructure',
    r'nanosheet|nanowire|nanocube|nanofiber|nanoframe|hollow': 'nanostructure',
    r'ti3c2tx|mxene|g.?c3n4': 'mxene_carbon_nitride',
    r'negatrode|anode': 'anode_material',
    r'positrode|cathode|air.?cathode': 'cathode_material'
}
# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 1,         # Changed from 2 to 1
            "MIN_CONCEPT_LENGTH_WORDS": 1, # Changed from 2 to 1
            "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.72, "COOCCURRENCE_WEIGHT": 0.5,
            "SEMANTIC_WEIGHT": 0.5, "CLUSTER_SIMILARITY": 0.75,
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.9,
            "SEMANTIC_WEIGHT": 0.1, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10
        }
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# ==========================================
# CONCEPT EXTRACTION & NORMALIZATION
# ==========================================
def is_valid_battery_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in BATTERY_PATTERNS)
    generic = {'study', 'analysis', 'effect', 'role', 'investigation', 'research', 
               'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
               'new', 'recent', 'various', 'different', 'significant', 'important'}
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    # CHANGED: Allow single-word concepts (needed for small corpora like 9 papers)
    if len(words) < 1 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic
def normalize_battery_term(concept: str) -> str:
    concept = concept.lower().strip()

    # NEW: Standardize units with weird unicode characters (bullet, minus, superscripts)
    concept = re.sub(r'mah[\s\W]*g[\s\W]*1', 'mah/g', concept)
    concept = re.sub(r'wh[\s\W]*kg[\s\W]*1', 'wh/kg', concept)
    concept = re.sub(r'mah[\s\W]*cm[\s\W]*2', 'mah/cm2', concept)
    concept = re.sub(r'mw[\s\W]*cm[\s\W]*2', 'mw/cm2', concept)
    concept = re.sub(r'ma[\s\W]*cm[\s\W]*2', 'ma/cm2', concept)
    concept = re.sub(r'a[\s\W]*cm[\s\W]*2', 'a/cm2', concept)

    # ORIGINAL: Standardize standard units
    concept = re.sub(r'wh\s*/\s*kg', 'wh/kg', concept)
    concept = re.sub(r'wh\s*/\s*l', 'wh/l', concept)
    concept = re.sub(r'mah\s*/\s*g', 'mah/g', concept)

    # Keep original normalization for LiB terms (backward compatibility)
    concept = re.sub(r'ncm\s*(\d+(?:\.\d+)?(?:\d+)?)', r'ncm', concept)
    concept = re.sub(r'nmc\s*(\d+(?:\.\d+)?(?:\d+)?)', r'nmc', concept)
    concept = re.sub(r'lfp', 'lfp', concept)
    concept = re.sub(r'lco', 'lco', concept)
    concept = re.sub(r'nca', 'nca', concept)
    concept = re.sub(r'18650', '18650', concept)
    concept = re.sub(r'21700', '21700', concept)
    concept = re.sub(r'4680', '4680', concept)
    concept = re.sub(r'si\s*ox?', 'siox', concept)
    concept = re.sub(r'silicon\s*oxide', 'siox', concept)
    concept = re.sub(r'fec', 'fec', concept)
    concept = re.sub(r'vc', 'vc', concept)
    concept = re.sub(r'sei', 'sei', concept)
    concept = re.sub(r'solid[-\s]?state', 'solid state', concept)
    concept = re.sub(r'fast[-\s]?charge', 'fast charging', concept)
    concept = re.sub(r'li[-\s]?metal', 'lithium metal', concept)
    return concept
def extract_concepts_from_text(text: str) -> List[str]:
    concepts = set()
    text_lower = text.lower()
    for pattern in BATTERY_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    # NEW: Extract single-word domain keywords (ZAB, MOF, HER, OER, ORR, etc.)
    for keyword in ALL_DOMAIN_KEYWORDS:
        kw_lower = keyword.lower()
        # Match as whole word, case-insensitive
        if re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower):
            concepts.add(kw_lower)
    noun_pattern = r'(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:electrode|electrolyte|battery|cell|anode|cathode|material|composite|coating|layer|film|particle|structure|morphology|performance|property|capacity|density|conductivity|resistance|impedance|stability|degradation|mechanism|process|method|technique|analysis|simulation|model|design|optimization)'
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_battery_concept(concept):
            concepts.add(concept)
    for keyword in ENERGY_DENSITY_KEYWORDS:
        for match in re.finditer(r'' + re.escape(keyword) + r'', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(r'([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|for|in|with|using|via|through|by|to|and|or)\s+' + re.escape(keyword) + r'', context)
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_battery_concept(concept):
                    concepts.add(concept)
    material_prop_pattern = r'([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:wh/kg|mah/g|wh/l|\%|percent|fold|times|x))'
    matches = re.findall(material_prop_pattern, text, re.I)
    for material, value in matches:
        concept = f"{material.lower()} {value.lower()}"
        if is_valid_battery_concept(concept):
            concepts.add(concept)
    return list(concepts)
def extract_concepts_from_abstracts(df: pd.DataFrame, text_columns: List[str]) -> Tuple[List[List[str]], List[Dict]]:
    all_concepts = []
    all_metrics = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        metrics = {}
        # Energy density (with unicode handling)
        ed_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg|wh kg-1|wh kg⁻¹|wh[\s\W]*kg[\s\W]*1)', combined_text, re.I)
        if ed_matches: metrics['energy_density_wh_kg'] = [float(m) for m in ed_matches]
        # Capacity (with unicode handling)
        cap_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g|mah g-1|mah g⁻¹|mah[\s\W]*g[\s\W]*1)', combined_text, re.I)
        if cap_matches: metrics['capacity_mah_g'] = [float(m) for m in cap_matches]
        # Areal capacity
        cap_cm2_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/cm²|mah cm-2|mah cm⁻²|mah[\s\W]*cm[\s\W]*2)', combined_text, re.I)
        if cap_cm2_matches: metrics['capacity_mah_cm2'] = [float(m) for m in cap_cm2_matches]
        # Voltage
        volt_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:V|volt)', combined_text, re.I)
        if volt_matches: metrics['voltage_v'] = [float(m) for m in volt_matches]
        # Cycle life
        cycle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:cycles|cycle)', combined_text, re.I)
        if cycle_matches: metrics['cycle_life'] = [float(m) for m in cycle_matches]
        # C-rate
        crate_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:C|c-rate)', combined_text, re.I)
        if crate_matches: metrics['c_rate'] = [float(m) for m in crate_matches]
        # Efficiency
        eff_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:efficiency|retention|coulombic)', combined_text, re.I)
        if eff_matches: metrics['efficiency_pct'] = [float(m) for m in eff_matches]
        # NEW: Power density (mW/cm²) - important for ZABs
        pd_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mw/cm²|mw cm-2|mw cm⁻²|mw[\s\W]*cm[\s\W]*2)', combined_text, re.I)
        if pd_matches: metrics['power_density_mw_cm2'] = [float(m) for m in pd_matches]
        # NEW: Current density (mA/cm², A/cm²)
        cd_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ma/cm²|ma cm-2|ma cm⁻²|ma[\s\W]*cm[\s\W]*2|a/cm²|a cm-2|a cm⁻²|a[\s\W]*cm[\s\W]*2)', combined_text, re.I)
        if cd_matches: metrics['current_density_ma_cm2'] = [float(m) for m in cd_matches]
        # NEW: Overpotential (mV, V) - important for electrocatalysis
        op_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mv|v)\s*(?:overpotential|overpotential|η|over-potential)', combined_text, re.I)
        if op_matches: metrics['overpotential_mv'] = [float(m) for m in op_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_battery_term(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics
def cluster_similar_concepts(valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75):
    if len(valid_concepts) < 5:
        return valid_concepts, {c: c for c in valid_concepts}
    try:
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1 - similarity_threshold, 
            linkage='average', metric='cosine'
        ).fit(embeddings)
        cluster_members = defaultdict(list)
        concept_to_cluster = {}
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        cluster_representatives = {}
        for label, members in cluster_members.items():
            def score(m):
                domain_hits = sum(1 for kw in ALL_DOMAIN_KEYWORDS if kw.lower() in m.lower())
                return (domain_hits, -len(m))
            representative = max(members, key=score)
            cluster_representatives[label] = representative
        final_mapping = {c: cluster_representatives[label] for c, label in concept_to_cluster.items()}
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        return valid_concepts, {c: c for c in valid_concepts}

def normalize_and_filter_concepts(all_concepts: List[List[str]], config: Dict) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    concept_counts = defaultdict(int)
    concept_abstract_map = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_battery_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq = config.get("MIN_CONCEPT_FREQ", 5)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    max_words = config.get("MAX_CONCEPT_LENGTH", 10)
    valid_concepts = [c for c, cnt in concept_counts.items() 
                      if cnt >= min_freq and min_words <= len(c.split()) <= max_words]
    if config.get("USE_SEMANTIC_CLUSTERING", False) and len(valid_concepts) > 50:
        try:
            embed_model = load_embedding_model()
            valid_concepts, concept_to_cluster = cluster_similar_concepts(
                valid_concepts, embed_model, 
                similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.72)
            )
            new_abstract_map = defaultdict(list)
            for orig_concept, docs in concept_abstract_map.items():
                clustered = concept_to_cluster.get(orig_concept, orig_concept)
                if clustered in valid_concepts:
                    new_abstract_map[clustered].extend(docs)
            concept_abstract_map = new_abstract_map
        except Exception as e:
            st.warning(f"Semantic clustering skipped: {e}")
    valid_concepts = sorted(valid_concepts, key=lambda c: concept_counts[c], reverse=True)
    top_n = config.get("TOP_N_CONCEPTS", 1000)
    if len(valid_concepts) > top_n:
        valid_concepts = valid_concepts[:top_n]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map

def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    concept_to_abstract = {}
    for concept in concepts:
        matched = False
        for pattern, category in BATTERY_CATEGORY_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'\bLi[A-Z]', r'\bNCM', r'\bNMC', r'\bLFP', r'\bLCO']):
                concept_to_abstract[concept] = 'material_specific'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract

# ==========================================
# CONCEPT DISTILLATION
# ==========================================
def compute_concept_distillation(valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]], 
                                  all_texts: List[str]) -> pd.DataFrame:
    distill_data = []
    doc_corpus = []
    for c in valid_concepts:
        doc_text = " ".join([all_texts[i] for i in concept_abstract_map.get(c, []) if i < len(all_texts)])
        doc_corpus.append(doc_text)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english', max_features=5000)
    try:
        tfidf_matrix = tfidf.fit_transform(doc_corpus)
        tfidf_scores = tfidf_matrix.max(axis=1).A1
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    embed_model = load_embedding_model()
    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:50]
                concept_embeddings = embed_model.encode(words, show_progress_bar=False, batch_size=32)
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c, "frequency": freq, "tfidf_weight": semantic_density,
            "semantic_density": semantic_density, "coherence_score": float(coherence),
            "distillation_efficiency": float(semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence))
        })
    return pd.DataFrame(distill_data).sort_values("distillation_efficiency", ascending=False)

# ==========================================
# GRAPH CONSTRUCTION
# ==========================================
def build_hybrid_graph(all_concepts: List[List[str]], valid_concepts: List[str], 
                        concept_to_id: Dict[str, int], embed_model=None, config: Dict = None) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        nx_graph.add_node(c, frequency=0)
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(u, v, weight=1, cooccurrence=1, semantic=0, edge_type='cooccurrence')
                nx_graph.nodes[u]['frequency'] = nx_graph.nodes[u].get('frequency', 0) + 1
                nx_graph.nodes[v]['frequency'] = nx_graph.nodes[v].get('frequency', 0) + 1
    if embed_model and len(valid_concepts) >= 10:
        try:
            embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i+1:], start=i+1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3):
                        nx_graph.add_edge(c1, c2, weight=sim * 2, cooccurrence=0, 
                                         semantic=sim, edge_type='semantic')
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph

def sample_edges_for_training(nx_graph: nx.Graph, valid_concepts: List[str], 
                               concept_to_id: Dict[str, int], config: Dict = None) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs
    target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)
    attempts = 0
    max_attempts = 50000
    try:
        path_lengths = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=3))
    except Exception:
        path_lengths = {}
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            attempts += 1
            continue
        dist = path_lengths.get(u_c, {}).get(v_c, 999)
        if dist == 2 or dist == 3:
            neg_pairs.append((u_idx, v_idx))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((u_idx, v_idx))
        attempts += 1
    while len(neg_pairs) < target_negs:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((u_idx, v_idx))
    return pos_pairs, neg_pairs

# ==========================================
# GNN MODEL
# ==========================================
class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, adj_indices, adj_values, num_nodes, h, pos_u, pos_v, neg_u, neg_v):
        A = sparse.FloatTensor(adj_indices, adj_values, torch.Size([num_nodes, num_nodes])).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1)))
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(torch.cat([h2[pos_u], h2[pos_v]], dim=1)).squeeze(1)
        neg_scores = self.decoder(torch.cat([h2[neg_u], h2[neg_v]], dim=1)).squeeze(1)
        return pos_scores, neg_scores, h2

def train_gnn(node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, 
              progress_callback=None, epochs: int = 50, lr: float = 1e-3):
    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384
    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")
    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32)
    target_device = node_features.device if node_features.numel() > 0 else torch.device('cpu')
    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=target_device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=target_device)
    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else torch.tensor([], dtype=torch.long, device=target_device)
    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if len(neg_pairs) == 0:
            pos_out, _, _ = model(adj_indices, adj_values, num_nodes, node_features, 
                                 pos_u, pos_v, pos_u[:1], pos_v[:1])
            loss = criterion(pos_out, torch.ones_like(pos_out)) * 0.5
        else:
            pos_out, neg_out, _ = model(adj_indices, adj_values, num_nodes, node_features,
                                         pos_u, pos_v, neg_u, neg_v)
            pos_loss = criterion(pos_out, torch.ones_like(pos_out))
            neg_loss = criterion(neg_out, torch.zeros_like(neg_out))
            loss = 0.5 * (pos_loss + neg_loss)
        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())
    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(adj_indices, adj_values, num_nodes, node_features,
                                       pos_u[:1], pos_v[:1], neg_u[:1] if len(neg_pairs) > 0 else pos_u[:1],
                                       neg_v[:1] if len(neg_pairs) > 0 else pos_v[:1])
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()

# ==========================================
# RESEARCH DIRECTION SCORING
# ==========================================
def compute_research_direction_scores(model, node_features, final_emb, nx_graph, 
                                       valid_concepts, concept_properties, ridge, 
                                       embed_model, n_samples: int = 5000) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    v_ids = np.random.randint(n_concepts, size=min(n_samples, n_concepts * 5))
    candidate_pairs = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((u_idx, v_idx, u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat([final_emb[u_tensor], final_emb[v_tensor]], dim=1)
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    emb_np = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
    cos_sims = np.sum(emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1)
    results = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(ridge.predict([[p_u, p_v, 1.0]])[0])
            except:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = np.exp(-0.5 * semantic_novelty) * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = np.clip((expected_improvement - 50) / 200, 0, 1) if expected_improvement > 0 else 0
        D_uv = (alpha['gnn'] * gnn_scores[i] + alpha['novelty'] * semantic_novelty + 
                alpha['gain'] * norm_gain + alpha['feas'] * (1.0 - feasibility))
        results.append({
            'concept_u': u_c, 'concept_v': v_c, 'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty), 'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility), 'composite_score': float(D_uv)
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    return df.head(min(100, len(df)))

# ==========================================
# MATHEMATICAL VALIDATION
# ==========================================
def validate_graph_metrics(nx_graph: nx.Graph, valid_concepts: List[str]) -> Dict[str, Any]:
    metrics = {}
    if nx_graph.number_of_nodes() < 3:
        return metrics
    try:
        from networkx.algorithms import community
        partition = list(community.greedy_modularity_communities(nx_graph))
        metrics["modularity"] = community.modularity(nx_graph, partition)
        metrics["n_communities"] = len(partition)
    except Exception:
        metrics["modularity"] = 0.0
        metrics["n_communities"] = 0
    try:
        embed_model = load_embedding_model()
        embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(partition if 'partition' in locals() else [[]]):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(sum(1 for p in p_values if p < 0.05))
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(list(nx.betweenness_centrality(nx_graph).values()))
        metrics["avg_closeness"] = np.mean(list(nx.closeness_centrality(nx_graph).values()))
    except Exception:
        pass
    return metrics

@st.cache_data(ttl=3600)
def compute_bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05):
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(ci_low), float(ci_high)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def get_battery_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    # Advanced Batteries
    if any(c in concept_lower for c in ['zinc-air', 'zab', 'zn-air']):
        return "#2196F3"  # Blue
    elif any(c in concept_lower for c in ['sodium-ion', 'sib', 'na-ion']):
        return "#FF9800"  # Orange
    elif any(c in concept_lower for c in ['seawater', 'zn-seawater']):
        return "#00BCD4"  # Cyan
    # Electrocatalysis & Reactions
    elif any(c in concept_lower for c in ['her', 'hydrogen evolution']):
        return "#4CAF50"  # Green
    elif any(c in concept_lower for c in ['oer', 'oxygen evolution']):
        return "#F44336"  # Red
    elif any(c in concept_lower for c in ['orr', 'oxygen reduction']):
        return "#9C27B0"  # Purple
    elif any(c in concept_lower for c in ['water splitting', 'electrolysis']):
        return "#FF5722"  # Deep Orange
    # Materials
    elif any(c in concept_lower for c in ['mof', 'metal-organic framework']):
        return "#795548"  # Brown
    elif any(c in concept_lower for c in ['single-atom', 'single atom', 'fe-n-c']):
        return "#E91E63"  # Pink
    elif any(c in concept_lower for c in ['ti3c2tx', 'mxene', 'g-c3n4']):
        return "#3F51B5"  # Indigo
    elif any(c in concept_lower for c in ['heterostructure', 'heterointerface', 'bilayer']):
        return "#607D8B"  # Blue Grey
    elif any(c in concept_lower for c in ['nanosheet', 'nanowire', 'nanocube', 'nanofiber']):
        return "#009688"  # Teal
    # Legacy LiB categories (keep for backward compatibility)
    elif any(c in concept_lower for c in ['cathode', 'ncm', 'nmc', 'lfp', 'lco', 'nca', 'layered', 'olivine', 'spinel']):
        return "#E91E63"
    elif any(a in concept_lower for a in ['anode', 'graphite', 'silicon', 'siox', 'li metal', 'lithium metal', 'titanate', 'lto']):
        return "#3F51B5"
    elif any(e in concept_lower for e in ['electrolyte', 'sei', 'cei', 'solid state', 'sulfide', 'oxide', 'polymer', 'carbonate']):
        return "#00BCD4"
    elif any(ed in concept_lower for ed in ['energy density', 'wh/kg', 'wh/l', 'mah/g', 'power density', 'capacity']):
        return "#FF9800"
    elif any(d in concept_lower for d in ['dendrite', 'safety', 'thermal', 'fire', 'short circuit', 'venting']):
        return "#F44336"
    elif any(m in concept_lower for m in ['manufacturing', 'calendering', 'coating', '3d print', 'additive', 'cell design']):
        return "#9C27B0"
    elif any(comp in concept_lower for comp in ['machine learning', 'neural', 'dft', 'molecular dynamics', 'phase field', 'simulation']):
        return "#4CAF50"
    else:
        return "#607D8B"
def render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=True,
                        min_node_size=8, max_node_size=40, cmap_name="viridis",
                        custom_labels=None, node_label_size=12, top_n_nodes=0):
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#000000",
                  select_menu=True, notebook=False, cdn_resources='remote')
    if physics_enabled:
        net.barnes_hut(gravity=-3000, spring_length=120, spring_strength=0.05, 
                       damping=0.09, overlap=0.5)
    else:
        net.set_options("var options = { physics: { enabled: false } }")
    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.5, min_node_size, max_node_size))
        color = get_battery_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        label = custom_labels.get(node, node) if custom_labels else node
        net.add_node(node, label=label, size=size, color=color,
                     font={'color': '#000000', 'size': node_label_size},
                     title=f"{node}\nDegree: {degree}\nFrequency: {freq}")
    color_map = {'cooccurrence': "#4CAF50", 'semantic': "#2196F3", 'bridge': "#FFC107"}
    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, "#607D8B")
        net.add_edge(u, v, value=float(np.clip(w, 0.5, 5)),
                     width=float(np.clip(w * 0.6, 1, 3)),
                     color=color, smooth={'type': 'curvedCW', 'roundness': 0.2})
    html_content = net.generate_html()
    st.components.v1.html(html_content, height=750, scrolling=True)
    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes,
                          file_name="lib_concept_graph.html", mime="text/html")
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

def render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name="viridis", 
                            custom_labels=None, top_n_nodes=0, node_label_size=10):
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_hover = [], [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        edge_hover.extend([f"<b>{u} ↔ {v}</b><br>Weight: {w:.2f}<br>Type: {edge_type}"] * 2 + [None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=1, color='#888'),
                            hoverinfo='text', hovertext=edge_hover, name='Connections')
    node_x, node_y, node_text, node_size, node_color, node_labels = [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
        node_size.append(max(8, min(35, deg * 2.5 + 10)))
        node_color.append(cmap_colors[i])
        node_labels.append(custom_labels.get(node, node) if custom_labels else node)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            marker=dict(size=node_size, color=node_color, 
                                       line=dict(width=2, color='#ffffff')),
                            text=node_labels, textposition="bottom center",
                            textfont=dict(size=node_label_size, color='#000000'),
                            hovertext=node_text, hoverinfo='text', name='Concepts')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name="viridis", top_n_nodes=0):
    if len(nx_graph.nodes()) < 3:
        st.info("3D view requires ≥3 nodes.")
        return
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos_3d = nx.spring_layout(nx_graph, dim=3, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x, edge_y, edge_z = [], [], []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos_3d[u]; x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None]); edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                              line=dict(width=2, color='#888'), hoverinfo='skip')
    node_x, node_y, node_z, node_text, node_size, node_color, node_labels = [], [], [], [], [], [], []
    for i, node in enumerate(nx_graph.nodes()):
        x, y, z = pos_3d[node]
        node_x.append(x); node_y.append(y); node_z.append(z)
        deg = nx_graph.degree(node); freq = len(concept_abstract_map.get(node, []))
        node_text.append(f"{node}<br>Degree: {deg}<br>Frequency: {freq}")
        node_size.append(max(6, min(25, deg * 2 + 8)))
        node_color.append(cmap_colors[i])
        node_labels.append(node)
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text',
                                marker=dict(size=node_size, color=node_color, opacity=0.9),
                                text=node_labels, textposition="top center",
                                textfont=dict(size=8, color='#000000'),
                                hovertext=node_text, hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(scene=dict(xaxis=dict(showbackground=False),
                                                 yaxis=dict(showbackground=False),
                                                 zaxis=dict(showbackground=False)),
                                     margin=dict(l=0, r=0, b=0, t=0), showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_fallback(nx_graph, concept_abstract_map):
    st.markdown("### 📊 Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    if len(nx_graph.edges()) > 0:
        edge_list = [(u, v, nx_graph[u][v].get('weight', 1)) for u, v in nx_graph.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**🔗 Top 20 Strongest Connections:**")
        for i, (u, v, w) in enumerate(edge_list[:20], 1):
            edge_type = nx_graph[u][v].get('edge_type', 'unknown')
            st.markdown(f"{i}. `{u}` ↔ `{v}` (weight: {w:.2f}, type: {edge_type})")
    if len(concept_abstract_map) > 0:
        freq_data = [(c, len(concept_abstract_map.get(c, []))) for c in nx_graph.nodes()]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**📈 Top Concepts by Frequency:**")
        st.dataframe(pd.DataFrame(freq_data[:15], columns=["Concept", "Abstract Count"]), use_container_width=True)

# ==========================================
# SUNBURST & RADAR CHARTS
# ==========================================
def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))
        hierarchy[category]["children"].append((concept, freq))
        hierarchy[category]["count"] += freq
    for parent in list(hierarchy.keys()):
        children = hierarchy[parent]["children"]
        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children.sort(key=lambda x: x[1], reverse=True)
            children = children[:top_n_per_category]
            hierarchy[parent]["count"] = sum(cnt for _, cnt in children)
            hierarchy[parent]["children"] = children
    labels, parents, values = [], [], []
    for parent, data in hierarchy.items():
        labels.append(parent); parents.append(""); values.append(data["count"])
        for child, cnt in data["children"]:
            labels.append(child); parents.append(parent); values.append(cnt)
    return labels, parents, values

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, width=800, height=600):
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    n_items = len(labels)
    use_remainder = n_items > 80
    unique_ids = []; seen = {}
    for i, lab in enumerate(labels):
        base = lab[:25] + ("…" if len(lab) > 25 else "")
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base); seen[base] = 1
    parent_ids = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    break
            else:
                parent_ids.append("")
    colors = get_colormap_colors(cmap_name, len(unique_ids))
    branchvalues = "remainder" if use_remainder else "total"
    fig = go.Figure(go.Sunburst(
        labels=unique_ids, parents=parent_ids, values=values, ids=unique_ids,
        branchvalues=branchvalues,
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))
    fig.update_layout(
        title="<b>Advanced Battery & Electrocatalysis Research Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white", plot_bgcolor="white",
        width=width, height=height,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_radar_chart(concept_scores_df: pd.DataFrame, top_k: int = 15, cmap_name: str = "viridis"):
    if concept_scores_df.empty or len(concept_scores_df) < 2:
        st.info("Not enough concepts for radar chart.")
        return
    metrics = ['frequency', 'semantic_density', 'coherence_score', 'distillation_efficiency']
    available_metrics = [m for m in metrics if m in concept_scores_df.columns]
    if not available_metrics:
        st.warning("No metrics available for radar chart.")
        return
    top_concepts = concept_scores_df.nlargest(top_k, 'distillation_efficiency')
    normalized = top_concepts.copy()
    for m in available_metrics:
        col = normalized[m]
        if col.max() > col.min():
            normalized[m] = (col - col.min()) / (col.max() - col.min())
        else:
            normalized[m] = 0.5
    categories = available_metrics
    fig = go.Figure()
    colors = get_colormap_colors(cmap_name, len(normalized))
    for idx, (_, row) in enumerate(normalized.iterrows()):
        concept = row['concept']
        values = [row[m] for m in categories]
        values += values[:1]
        angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name=concept[:20],
            line=dict(width=2, color=colors[idx]), fillcolor=colors[idx], opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Top Concepts: Multi-Dimensional Comparison",
        showlegend=True, width=750, height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# EXPORT FUNCTIONS
# ==========================================
def export_graph(nx_graph, concept_abstract_map, format_type: str):
    if format_type == "GraphML":
        try:
            nx.write_graphml_lxml(nx_graph, "lib_graph.graphml")
        except:
            nx.write_graphml(nx_graph, "lib_graph.graphml")
        with open("lib_graph.graphml", "rb") as f:
            return f.read(), "application/graphml+xml", "lib_graph.graphml"
    elif format_type == "JSON":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "lib_graph.json"
    elif format_type == "CSV (Edges)":
        edge_data = []
        for u, v, data in nx_graph.edges(data=True):
            row = {"source": u, "target": v}
            row.update({k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))})
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "lib_edges.csv"
    elif format_type == "CSV (Nodes)":
        node_data = []
        for node in nx_graph.nodes():
            row = {"concept": node, "frequency": len(concept_abstract_map.get(node, [])),
                   "degree": nx_graph.degree(node)}
            row.update({k: v for k, v in nx_graph.nodes[node].items()})
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "lib_nodes.csv"
    elif format_type == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [get_battery_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/png", "lib_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    return None, None, None

# ==========================================
# GRAPH METRICS DASHBOARD
# ==========================================
def compute_graph_metrics(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {}
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0,
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
    }
    try:
        bc = nx.betweenness_centrality(G, normalized=True, k=min(100, G.number_of_nodes()))
        top_bridges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics

def display_metric_dashboard(metrics: dict):
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"]); col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}"); col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric("Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}")
    if metrics.get("top_bridges"):
        st.markdown("**🌉 Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(metrics["top_bridges"], columns=["Concept", "Bridge Score"])
        st.dataframe(bridge_df, use_container_width=True)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("🔋 Advanced Battery & Electrocatalysis Focus Areas")
        st.markdown("- Energy/Power density (Wh/kg, mW/cm²)")
        st.markdown("- Battery types (ZAB, SIB, Seawater)")
        st.markdown("- Electrode materials (MOF, MXene, Single-atom)")
        st.markdown("- Electrocatalysis (HER, OER, ORR)")
        st.markdown("- Nanostructures (heterostructure, nanosheet)")
        st.markdown("- Performance metrics (overpotential, stability)")
        st.subheader("🎨 Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )
        st.session_state['physics_enabled'] = st.checkbox("Enable physics", value=True)
        st.session_state['top_n_graph'] = st.slider("Max nodes in graph", 50, 500, 200, step=50)
        st.session_state['top_n_sunburst'] = st.slider("Max children/category", 10, 100, 40, step=10)
        st.session_state['top_n_radar'] = st.slider("Top K for radar", 5, 30, 15)
        st.subheader("🔧 Graph Parameters")
        #st.session_state['min_freq'] = st.slider("Min concept frequency", 2, 20, 5)
        #st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 1)
        st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 2)
        st.session_state['min_words'] = st.slider("Min words per concept", 1, 5, 1)  # Default 1
        st.session_state['sim_threshold'] = st.slider("Semantic threshold", 0.6, 0.95, 0.85, step=0.05)
        st.session_state['cooc_weight'] = st.slider("Co-occurrence weight", 0.5, 1.0, 0.9, step=0.1)
        st.session_state['sem_weight'] = st.slider("Semantic weight", 0.0, 0.5, 0.1, step=0.1)
        st.subheader("📐 Statistics")
        st.session_state['bootstrap_samples'] = st.slider("Bootstrap samples", 100, 2000, 500, step=100)
        st.session_state['alpha_level'] = st.selectbox("Significance α", [0.01, 0.05, 0.10], index=1)
        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        gpu_info = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"🖥️ Device: {gpu_info}")

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    st.title("🔋 RechargeableBatteries-ConceptGraph: Advanced Battery & Electrocatalysis Explorer")
    st.caption("Concept graph builder for advanced batteries & electrocatalysis research • ZABs, SIBs, Seawater Batteries, HER/OER/ORR")
    render_sidebar()
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None

    # ─── LOAD JSON DATA ───
    st.header("📂 Data Loading")
    st.info(f"Place JSON files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)
    if not file_records:
        st.warning("No .json files found in the directory.")
        st.info("Please place your JSON metadata files in the `json_metadatabase/` folder.")
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error("Files found but none could be parsed. Check error messages above.")
        return
    st.success(f"Loaded {len(successful_files)} file(s) • {len(df)} record(s)")
    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect("Filter by source file", file_names, default=file_names)
    if selected_files:
        df_filtered = df[df["_source_file"].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    st.write(f"Working with **{len(df_filtered)}** records")
    with st.expander("📋 Preview Data Structure"):
        st.dataframe(df_filtered.head(5), use_container_width=True)
        st.markdown("**Available columns:**")
        st.write(list(df_filtered.columns))

    # ─── TEXT COLUMN SELECTION ───
    text_cols = [c for c in df_filtered.columns if any(k in c.lower() for k in ['abstract', 'title', 'summary', 'text', 'content', 'description'])]
    if not text_cols:
        text_cols = [c for c in df_filtered.columns if df_filtered[c].dtype == 'object']
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    # ─── RUN ANALYSIS ───
    if st.button("🚀 Build Concept Graph", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status = st.status("🔄 Initializing analysis...", expanded=True)
        try:
            with status:
                st.write("📦 Preparing text corpus...")
                all_texts = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([str(row[col]) for col in selected_text_cols if col in row and pd.notna(row[col])])
                    all_texts.append(text)
                num_abstracts = len(all_texts)
                st.write(f"✅ Prepared {num_abstracts} documents")
                progress_bar.progress(0.05)
                st.write("🧠 Loading embedding model...")
                embed_model = load_embedding_model()
                st.success("✅ Embedding model loaded")
                progress_bar.progress(0.10)
                config = get_adaptive_config(num_abstracts)
                config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
                config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
                config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.9)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.1)
                st.write(f"📊 Adaptive config: {config}")
                progress_bar.progress(0.15)
                st.write("🔍 Extracting concepts from abstracts...")
                all_concepts, all_metrics = extract_concepts_from_abstracts(df_filtered, selected_text_cols)
                st.write(f"✅ Extracted concepts from {len(all_concepts)} documents")
                progress_bar.progress(0.30)
                st.write("🧹 Filtering and normalizing concepts...")
                valid_concepts, concept_to_id, id_to_concept, concept_abstract_map = normalize_and_filter_concepts(all_concepts, config)
                st.write(f"✅ **{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)
                if len(valid_concepts) < 5:
                    st.error("Too few concepts extracted. Try lowering frequency thresholds.")
                    return
                st.write("🕸️ Building concept graph...")
                nx_graph = build_hybrid_graph(all_concepts, valid_concepts, concept_to_id, embed_model, config)
                try:
                    d_prev_dict = dict(nx.all_pairs_shortest_path_length(nx_graph, cutoff=4))
                except Exception:
                    d_prev_dict = {}
                pos_pairs, neg_pairs = sample_edges_for_training(nx_graph, valid_concepts, concept_to_id, config)
                st.write(f"✅ Graph: {len(valid_concepts)} nodes, {nx_graph.number_of_edges()} edges")
                progress_bar.progress(0.55)
                st.write("🧬 Generating node embeddings...")
                try:
                    embeddings = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
                    node_features = torch.tensor(embeddings, dtype=torch.float32)
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"✅ Node features: {node_features.shape}")
                progress_bar.progress(0.65)
                st.write("🤖 Training GraphSAGE...")
                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(f"📊 Epoch {epoch}/50 | Loss: {loss:.4f}")
                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs, training_progress
                )
                st.success("✅ GNN training complete")
                progress_bar.progress(0.80)
                st.write("📈 Scoring research directions...")
                concept_properties = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            for metric_values in all_metrics[idx].values():
                                values.extend(metric_values)
                    concept_properties[concept] = np.median(values) if values else 0.0
                X_feat, y_target = [], []
                for u, v in nx_graph.edges():
                    pu, pv = concept_properties.get(u, 0), concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
                ridge = None
                if len(X_feat) > 5:
                    ridge = Ridge(alpha=1.0).fit(np.array(X_feat), np.array(y_target))
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph, valid_concepts,
                    concept_properties, ridge, embed_model
                )
                st.write(f"✅ Scored {len(top_scores)} novel pairs")
                progress_bar.progress(0.90)
                st.write("🔬 Computing distillation metrics...")
                distill_df = compute_concept_distillation(valid_concepts, concept_abstract_map, all_texts)
                st.success("✅ Analysis complete!")
                progress_bar.progress(1.00)
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                st.session_state.analysis_data = {
                    "valid_concepts": valid_concepts,
                    "concept_to_id": concept_to_id,
                    "id_to_concept": id_to_concept,
                    "concept_abstract_map": concept_abstract_map,
                    "nx_graph": nx_graph,
                    "concept_properties": concept_properties,
                    "ridge": ridge,
                    "top_scores": top_scores,
                    "distill_df": distill_df,
                    "gnn_model": gnn_model,
                    "final_emb": final_emb,
                    "embed_model": embed_model,
                    "all_metrics": all_metrics,
                    "all_texts": all_texts,
                    "config": config
                }
        except Exception as e:
            st.error(f"❌ Pipeline Error: {e}")
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ─── DISPLAY RESULTS ───
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        valid_concepts = data["valid_concepts"]
        concept_abstract_map = data["concept_abstract_map"]
        nx_graph = data["nx_graph"]
        top_scores = data["top_scores"]
        distill_df = data["distill_df"]
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)
        viz_tab, distill_tab, scores_tab, valid_tab, export_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", "📐 Validation", "📥 Export"
        ])
        with viz_tab:
            st.subheader("🌐 Interactive Concept Graph")
            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            elif nx_graph.number_of_edges() == 0:
                st.warning("No edges — building semantic fallback")
                nx_graph = nx.complete_graph(len(valid_concepts))
                nx_graph = nx.relabel_nodes(nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))})
            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=physics,
                                   cmap_name=cmap, top_n_nodes=top_n_graph)
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n_graph)
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n_graph)
            else:
                render_graph_fallback(nx_graph, concept_abstract_map)
            with st.expander("📊 Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics)
            with st.expander("📈 Domain Hierarchy (Sunburst)"):
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map, 
                                                                    top_n_per_category=st.session_state.get('top_n_sunburst', 40))
                render_sunburst_chart(labels, parents, values, cmap_name=cmap)
            with st.expander("📡 Concept Radar"):
                render_radar_chart(distill_df, top_k=st.session_state.get('top_n_radar', 15), cmap_name=cmap)
        with distill_tab:
            st.subheader("🔍 Concept Distillation Efficiency")
            top_n = st.slider("Show Top N", 10, min(200, len(distill_df)), 50, key="distill_top_n")
            display_df = distill_df.head(top_n)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("**📈 Efficiency vs Frequency:**")
            chart_df = display_df.set_index('concept')[['distillation_efficiency']]
            st.bar_chart(chart_df)
            st.markdown("**📊 Multi-Metric Comparison:**")
            metric_cols = [c for c in ['frequency', 'tfidf_weight', 'semantic_density', 'coherence_score'] 
                           if c in display_df.columns]
            if metric_cols:
                compare_df = display_df[['concept'] + metric_cols].set_index('concept')
                st.line_chart(compare_df)
        with scores_tab:
            st.subheader("🎯 Top Research Direction Recommendations")
            if top_scores.empty:
                st.info("No novel pairs scored. The graph may be too dense or too sparse.")
            else:
                st.write(f"Top {len(top_scores)} novel concept pairs:")
                st.dataframe(top_scores[['concept_u', 'concept_v', 'composite_score', 
                                         'gnn_affinity', 'semantic_novelty', 
                                         'expected_property_gain', 'feasibility_score']].head(20),
                            use_container_width=True)
                csv_scores = top_scores.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Scores (CSV)", data=csv_scores, 
                                  file_name="research_directions.csv", mime="text/csv")
        with valid_tab:
            st.subheader("📐 Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Modularity", f"{val_metrics.get('modularity', 0):.3f}")
            col2.metric("Silhouette", f"{val_metrics.get('silhouette_score', 0):.3f}")
            col3.metric("Communities", val_metrics.get('n_communities', 0))
            col4.metric("Significant Edges", val_metrics.get('edge_significant_count', 0))
            if not top_scores.empty:
                n_boot = st.session_state.get('bootstrap_samples', 500)
                alpha = st.session_state.get('alpha_level', 0.05)
                mean_score, ci_low, ci_high = compute_bootstrap_ci(
                    top_scores['composite_score'].values, n_bootstrap=n_boot, alpha=alpha
                )
                st.success(f"🎯 Composite Score: `{mean_score:.3f}` | {int((1-alpha)*100)}% CI: `[{ci_low:.3f}, {ci_high:.3f}]`")
            X_feat, y_target = [], []
            for u, v in nx_graph.edges():
                pu, pv = data["concept_properties"].get(u, 0), data["concept_properties"].get(v, 0)
                w = nx_graph[u][v].get('weight', 1)
                X_feat.append([pu, pv, w])
                y_target.append(max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0)
            if data["ridge"] is not None and len(X_feat) > 5:
                y_pred = data["ridge"].predict(np.array(X_feat))
                st.markdown("### 🔬 Ridge Regression (Property Prediction)")
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r2_score(y_target, y_pred):.3f}")
                c2.metric("MAE", f"{mean_absolute_error(y_target, y_pred):.2f}")
                c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_target, y_pred)):.2f}")
        with export_tab:
            st.subheader("📥 Export & Post-Processing")
            export_format = st.selectbox("Format:", ["GraphML", "JSON", "CSV (Edges)", "CSV (Nodes)", "PNG"])
            if st.button("📤 Generate Export"):
                result = export_graph(nx_graph, concept_abstract_map, export_format)
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button("💾 Save File", data=data_bytes, file_name=filename, mime=mime)
            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [len(concept_abstract_map.get(c, [])) for c in valid_concepts],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [abstract_concepts_to_categories([c]).get(c, 'general') for c in valid_concepts]
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Concept List (CSV)", data=csv_concepts, 
                              file_name="concepts.csv", mime="text/csv")

if __name__ == "__main__":
    main()
