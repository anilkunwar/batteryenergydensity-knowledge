#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiB-ConceptGraph: Energy Density Concept Graph Builder for Lithium-Ion Batteries
==================================================================================
Enhanced large-corpus concept graph extraction (3000+ abstracts) from JSON/BibTeX/CSV metadata.
No seed injection needed — robust statistical methods for high-volume data.

NEW IN THIS VERSION:
- BibTeX ingestion (.bib files from Mendeley/Zotero)
- Advanced Analytics: keyword bursts, semantic drift, concept genealogy, cross-domain bridges,
  network motifs, centrality comparison
- Interactive Graph Editing with Undo/Redo
- Enhanced visualizations: edge weight labels, t-SNE, community detection, concept timeline,
  co-occurrence heatmap, growth rate, bubble chart
- Publication-ready exports: SVG, 600-DPI PNG, automated Markdown report

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run: streamlit run lib_concept_graph_enhanced.py

Place JSON/BibTeX/CSV files in ./json_metadatabase/ folder next to this script.
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
import io
from collections import defaultdict, Counter, deque
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
from sklearn.manifold import TSNE
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

try:
    import bibtexparser
    BIBTEX_AVAILABLE = True
except ImportError:
    BIBTEX_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LiB-ConceptGraph: Energy Density Explorer",
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
        cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
        return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
    except Exception:
        try:
            cmap = cm.get_cmap(cmap_name, n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]
        except Exception:
            try:
                cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n)
            except Exception:
                cmap = cm.get_cmap("viridis", n)
            return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]

# ==========================================
# ROBUST FILE LOADER (JSON / JSONL / CSV / BibTeX)
# ==========================================
def parse_bibtex_file(filepath: Path):
    """Parse a .bib file and return a list of record dicts."""
    if not BIBTEX_AVAILABLE:
        raise ImportError("bibtexparser not installed. Run: pip install bibtexparser")
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        bib_db = bibtexparser.load(f)
    records = []
    for entry in bib_db.entries:
        rec = {}
        if 'title' in entry:
            rec['Title'] = entry['title']
        if 'abstract' in entry:
            rec['Abstract'] = entry['abstract']
        if 'year' in entry:
            rec['Year'] = entry['year']
        if 'author' in entry:
            rec['Authors'] = entry['author']
        if 'journal' in entry:
            rec['Journal'] = entry['journal']
        if 'doi' in entry:
            rec['DOI'] = entry['doi']
        rec['_source_file'] = filepath.name
        records.append(rec)
    return records

def robust_load_file(filepath: Path):
    """Try multiple strategies to load a file that claims to be JSON/CSV/BibTeX."""
    if filepath.suffix.lower() == '.bib':
        return parse_bibtex_file(filepath)
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
    """Load every supported file in directory and return a list of (filepath, records)."""
    p = Path(directory)
    files = []
    files.extend(sorted(p.glob("*.json")))
    files.extend(sorted(p.glob("*.jsonl")))
    files.extend(sorted(p.glob("*.bib")))
    files.extend(sorted(p.glob("*.csv")))
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
# LITHIUM-ION BATTERY DOMAIN CONFIGURATION
# ==========================================
ENERGY_DENSITY_KEYWORDS = [
    "energy density", "power density", "specific energy", "gravimetric energy",
    "volumetric energy", "wh/kg", "wh/l", "mah/g", "mah/cm3", "capacity retention",
    "coulombic efficiency", "areal capacity", "mass loading", "electrode density",
    "tap density", "packing density", "energy efficiency", "round-trip efficiency",
    "thermal runaway", "heat generation", "adiabatic temperature", "c-rate", "discharge rate",
    "charge rate", "fast charging", "high power", "high energy", "cell design",
    "electrode thickness", "electrode porosity", "binder content", "conductive additive",
    "active material ratio", "n/p ratio", "anode/cathode ratio", "cell voltage",
    "open circuit voltage", "average voltage", "voltage plateau", "polarization",
    "internal resistance", "impedance", "ionic conductivity", "electronic conductivity",
    "diffusion coefficient", "charge transfer resistance", "solid electrolyte interphase",
    "sei", "cei", "electrolyte decomposition", "gassing", "swelling", "calendar life",
    "cycle life", "degradation mechanism", "capacity fade", "impedance growth"
]

CATHODE_MATERIALS = [
    "ncm", "nmc", "lco", "lmo", "lfp", "lmno", "lnmo", "nca", "lno",
    "liNiMnCo", "liNiCoAl", "liFePo4", "liMn2O4", "liCoO2", "liNiO2",
    "high nickel", "low cobalt", "cobalt free", "single crystal", "polycrystalline",
    "core shell", "concentration gradient", "full concentration gradient",
    "layered oxide", "spinel", "olivine", "rock salt", "disordered rocksalt"
]

ANODE_MATERIALS = [
    "graphite", "soft carbon", "hard carbon", "silicon", "silicon oxide",
    "siOx", "tin", "germanium", "lithium metal", "li metal", "lithium foil",
    "lithium alloy", "lithium titanate", "lto", "titanium oxide", "niobium oxide",
    "conversion anode", "alloying anode", "intercalation anode", "prelithiation",
    "artificial sei", "solid electrolyte", "inorganic solid electrolyte",
    "sulfide electrolyte", "oxide electrolyte", "halide electrolyte",
    "polymer electrolyte", "gel polymer", "composite electrolyte", "hybrid electrolyte"
]

ELECTROLYTE_KEYWORDS = [
    "liquid electrolyte", "solid electrolyte", "solid state", "polymer electrolyte",
    "gel electrolyte", "ionic liquid", "superconcentrated", "localized high concentration",
    "fluorinated", "sulfone", "carbonate", "ether", "ester", "additive", "film former",
    "vc", "vec", "fec", "dfec", "lipo2f2", "liodfb", "libob", "litfsi", "lifsi",
    "dual salt", "solvent-in-salt", "water-in-salt", "aqueous", "non-aqueous",
    "propylene carbonate", "ethylene carbonate", "dimethyl carbonate", "ethyl methyl carbonate",
    "diethyl carbonate", "linear carbonate", "cyclic carbonate", "fluoroethylene carbonate"
]

CELL_DESIGN = [
    "cylindrical cell", "prismatic cell", "pouch cell", "18650", "21700", "4680",
    "cell format", "cell geometry", "jelly roll", "stacked electrode", "tab design",
    "current collector", "al foil", "cu foil", "porous current collector",
    "3d current collector", "current collector coating", "cell casing", "vent design",
    "thermal management", "cooling plate", "heat pipe", "phase change material",
    "battery pack", "module design", "cell-to-pack", "cell-to-chassis", "ctp", "ctc"
]

MANUFACTURING = [
    "calendering", "slot die coating", "doctor blade", "spray coating", "dry electrode",
    "solvent free", "binder free", "electrodeposition", "3d printing", "additive manufacturing",
    "electrode slurry", "mixing", "dispersion", "rheology", "viscosity", "solids loading",
    "drying", "solvent evaporation", "nmp", "pvdf", "cmc", "sbr", "paa", "alginate",
    "foil thickness", "electrode loading", "areal loading", "coating uniformity",
    "electrode calendering", "roll pressing", "electrode density control"
]

SAFETY_DEGRADATION = [
    "thermal stability", "overcharge", "overdischarge", "short circuit", "internal short",
    "dendrite", "lithium plating", "lithium whisker", "dead lithium", "gas evolution",
    "venting", "fire", "explosion", "safety vent", "cid", "ptc", "fuse", "bms",
    "state of charge", "state of health", "soc", "soh", "state estimation",
    "electrochemical impedance spectroscopy", "eis", "differential capacity", "dQ/dV",
    "differential voltage", "dV/dQ", "operando", "in-situ", "x-ray tomography",
    "neutron imaging", "cryo-em", "tem", "stem", "electron microscopy"
]

ALL_DOMAIN_KEYWORDS = (ENERGY_DENSITY_KEYWORDS + CATHODE_MATERIALS + ANODE_MATERIALS + 
                       ELECTROLYTE_KEYWORDS + CELL_DESIGN + MANUFACTURING + SAFETY_DEGRADATION)

BATTERY_PATTERNS = [
    r'\b(?:\d+(?:\.\d+)?\s*(?:wh/kg|wh kg-1|wh kg⁻¹|wh l-1|wh l⁻¹|mah/g|mah g-1|mah g⁻¹|mah/cm³|mah cm-3))\b',
    r'\b(?:Li(?:[A-Z][a-z]?\d*)+(?:O\d*)?)\b',
    r'\b(?:NCM|NMC|LCO|LMO|LFP|LMNO|LNMO|NCA|LNO|LTO)\d*(?:\d+(?:\.\d+)?)?\b',
    r'\b(?:18650|21700|4680|26650|14500)\b',
    r'\b(?:solid.?state|all.?solid.?state)\b',
    r'\b(?:fast.?charge|quick.?charge|rapid.?charge)\b',
    r'\b(?:high.?energy|high.?power|long.?life)\b',
    r'\b(?:Si(?:Ox?)?|SiO\d*|silicon.?oxide)\b',
    r'\b(?:prelithiat(?:ed|ion))\b',
    r'\b(?:3D.?print(?:ed|ing)|additive.?manufactur(?:ed|ing))\b'
]

BATTERY_CATEGORY_MAPPING = {
    r'ncm\d*|nmc\d*|li(?:ni)?mn?co|high.?nickel|layered.?oxide': 'cathode_material',
    r'lfp|liFePo4|olivine|phosphate': 'cathode_material',
    r'lco|liCoO2|cobalt.?oxide': 'cathode_material',
    r'nca|liNiCoAl|aluminum.?doped': 'cathode_material',
    r'graphite|soft.?carbon|hard.?carbon|carbon.?anode': 'anode_material',
    r'silicon|siOx|siO\d*|tin|germanium|alloy.?anode|conversion.?anode': 'anode_material',
    r'li.?metal|lithium.?foil|lithium.?anode': 'anode_material',
    r'lto|liTi|titanate|niobium.?oxide': 'anode_material',
    r'liquid.?electrolyte|carbonate|ether|ester|ionic.?liquid': 'liquid_electrolyte',
    r'solid.?electrolyte|sulfide|oxide|halide|garnet|nasicon|lispo|llzo|lagp': 'solid_electrolyte',
    r'polymer.?electrolyte|gel|peo|pan|pmma|pvdf.?hf[pt]': 'polymer_electrolyte',
    r'wh/kg|wh/l|mah/g|specific.?energy|gravimetric|volumetric': 'energy_density_metric',
    r'fast.?charge|quick.?charge|c-rate|charge.?rate|discharge.?rate': 'rate_capability',
    r'cycle.?life|calendar.?life|capacity.?retention|capacity.?fade|degradation': 'lifetime',
    r'thermal.?runaway|safety|fire|explosion|venting|dendrite|short.?circuit': 'safety',
    r'18650|21700|4680|pouch|prismatic|cylindrical|cell.?format': 'cell_design',
    r'calendering|coating|slot.?die|dry.?electrode|3d.?print|additive.?manuf': 'manufacturing',
    r'sei|cei|interphase|interface|passivation|film.?former': 'interphase',
    r'conductive.?additive|carbon.?black|cnt|graphene|cnt|super.?p|acetylene': 'conductive_network',
    r'binder|pvdf|cmc|sbr|paa|alginate|pva|nbr': 'binder_system',
    r'current.?collector|al.?foil|cu.?foil|3d.?current.?collector|porous.?cc': 'current_collector',
    r'bms|state.?of.?charge|state.?of.?health|soc|soh|estimation|algorithm': 'battery_management',
    r'operando|in.?situ|ex.?situ|x.?ray|neutron|tem|stem|cryo|tomography': 'characterization',
    r'phase.?field|molecular.?dynamics|dft|ab.?initio|machine.?learning|neural.?network|graph.?neural': 'computational_method'
}

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2,
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

# ==========================================
# DEVICE & MODEL MANAGEMENT
# ==========================================
@st.cache_resource(show_spinner=False)
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
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic

def normalize_battery_term(concept: str) -> str:
    concept = concept.lower().strip()
    concept = re.sub(r'\bwh\s*/\s*kg\b', 'wh/kg', concept)
    concept = re.sub(r'\bwh\s*/\s*l\b', 'wh/l', concept)
    concept = re.sub(r'\bmah\s*/\s*g\b', 'mah/g', concept)
    concept = re.sub(r'\bncm\s*(\d+(?:\.\d+)?(?:\d+)?)\b', r'ncm\1', concept)
    concept = re.sub(r'\bnmc\s*(\d+(?:\.\d+)?(?:\d+)?)\b', r'nmc\1', concept)
    concept = re.sub(r'\blfp\b', 'lfp', concept)
    concept = re.sub(r'\blco\b', 'lco', concept)
    concept = re.sub(r'\bnca\b', 'nca', concept)
    concept = re.sub(r'\b18650\b', '18650', concept)
    concept = re.sub(r'\b21700\b', '21700', concept)
    concept = re.sub(r'\b4680\b', '4680', concept)
    concept = re.sub(r'\bsi\s*ox?\b', 'siox', concept)
    concept = re.sub(r'\bsilicon\s*oxide\b', 'siox', concept)
    concept = re.sub(r'\bfec\b', 'fec', concept)
    concept = re.sub(r'\bvc\b', 'vc', concept)
    concept = re.sub(r'\bsei\b', 'sei', concept)
    concept = re.sub(r'\bsolid[-\s]?state\b', 'solid state', concept)
    concept = re.sub(r'\bfast[-\s]?charge\b', 'fast charging', concept)
    concept = re.sub(r'\bli[-\s]?metal\b', 'lithium metal', concept)
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
    noun_pattern = r'\b(?:[A-Z][a-z]+(?:\d+(?:\.\d+)?)?[\s\-]?){2,4}(?:electrode|electrolyte|battery|cell|anode|cathode|material|composite|coating|layer|film|particle|structure|morphology|performance|property|capacity|density|conductivity|resistance|impedance|stability|degradation|mechanism|process|method|technique|analysis|simulation|model|design|optimization)\b'
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_battery_concept(concept):
            concepts.add(concept)
    for keyword in ENERGY_DENSITY_KEYWORDS:
        for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+(?:of|for|in|with|using|via|through|by|to|and|or)\s+' + re.escape(keyword) + r'\b', context)
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_battery_concept(concept):
                    concepts.add(concept)
    material_prop_pattern = r'\b([A-Z][a-z]+(?:\d+(?:\.\d+)?)?(?:[\s\-][A-Z][a-z]?\d*)+)\b\s+(?:with|having|exhibiting|showing|demonstrating|achieving|reaching|delivering|providing|offering)\s+(?:a\s+)?([\d\.]+\s*(?:wh/kg|mah/g|wh/l|\%|percent|fold|times|x))\b'
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
        ed_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg|wh kg-1|wh kg⁻¹)', combined_text, re.I)
        if ed_matches: metrics['energy_density_wh_kg'] = [float(m) for m in ed_matches]
        cap_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g|mah g-1|mah g⁻¹)', combined_text, re.I)
        if cap_matches: metrics['capacity_mah_g'] = [float(m) for m in cap_matches]
        volt_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:V|volt)', combined_text, re.I)
        if volt_matches: metrics['voltage_v'] = [float(m) for m in volt_matches]
        cycle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:cycles|cycle)', combined_text, re.I)
        if cycle_matches: metrics['cycle_life'] = [float(m) for m in cycle_matches]
        crate_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:C|c-rate)', combined_text, re.I)
        if crate_matches: metrics['c_rate'] = [float(m) for m in crate_matches]
        eff_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:efficiency|retention|coulombic)', combined_text, re.I)
        if eff_matches: metrics['efficiency_pct'] = [float(m) for m in eff_matches]
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
# ADVANCED ANALYTICS ENGINE
# ==========================================
def detect_keyword_bursts(df, valid_concepts, concept_abstract_map, window=2, threshold=2.0):
    """Identify sudden spikes in concept frequency over publication years."""
    if 'Year' not in df.columns or df['Year'].isna().all():
        return pd.DataFrame()
    year_concept_counts = defaultdict(lambda: defaultdict(int))
    for concept in valid_concepts:
        for doc_idx in concept_abstract_map.get(concept, []):
            if doc_idx < len(df):
                year = df.iloc[doc_idx].get('Year')
                if pd.notna(year):
                    year_concept_counts[int(year)][concept] += 1
    if not year_concept_counts:
        return pd.DataFrame()
    years = sorted(year_concept_counts.keys())
    burst_records = []
    for concept in valid_concepts:
        counts = [year_concept_counts[y].get(concept, 0) for y in years]
        if sum(counts) < 5:
            continue
        for i in range(len(years)):
            if i < window:
                continue
            prev_window = counts[max(0, i-window):i]
            prev_mean = np.mean(prev_window) if prev_window else 0.001
            curr = counts[i]
            if prev_mean > 0 and curr > 0:
                burst_ratio = curr / prev_mean
                if burst_ratio >= threshold:
                    burst_records.append({
                        'concept': concept, 'year': years[i], 'frequency': curr,
                        'prev_mean': prev_mean, 'burst_ratio': burst_ratio,
                        'burst_score': np.log1p(burst_ratio)
                    })
    return pd.DataFrame(burst_records).sort_values('burst_score', ascending=False)

def detect_semantic_drift(valid_concepts, concept_abstract_map, all_texts, df, embed_model, year_threshold=5):
    """Compare concept embeddings in early vs recent papers to detect semantic drift."""
    if 'Year' not in df.columns or df['Year'].isna().all():
        return pd.DataFrame()
    median_year = int(df['Year'].dropna().median())
    early_year_cutoff = median_year - year_threshold
    drift_records = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        early_texts = []
        recent_texts = []
        for idx in doc_indices:
            if idx >= len(df):
                continue
            year = df.iloc[idx].get('Year')
            text = all_texts[idx] if idx < len(all_texts) else ""
            if pd.notna(year):
                if int(year) <= early_year_cutoff:
                    early_texts.append(text)
                else:
                    recent_texts.append(text)
        if len(early_texts) < 2 or len(recent_texts) < 2:
            continue
        try:
            early_emb = embed_model.encode(early_texts, show_progress_bar=False, batch_size=32)
            recent_emb = embed_model.encode(recent_texts, show_progress_bar=False, batch_size=32)
            early_mean = np.mean(early_emb, axis=0)
            recent_mean = np.mean(recent_emb, axis=0)
            sim = float(cosine_similarity([early_mean], [recent_mean])[0][0])
            drift = 1.0 - sim
            drift_records.append({
                'concept': concept, 'drift_score': drift, 'similarity': sim,
                'early_papers': len(early_texts), 'recent_papers': len(recent_texts),
                'early_year': early_year_cutoff, 'recent_year': median_year
            })
        except Exception:
            continue
    return pd.DataFrame(drift_records).sort_values('drift_score', ascending=False)

def build_concept_genealogy(nx_graph, valid_concepts):
    """Classify concepts into Foundational, Intermediate, or Emerging based on PageRank and Degree."""
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    try:
        pr = nx.pagerank(nx_graph, weight='weight')
    except Exception:
        pr = {n: 0 for n in nx_graph.nodes()}
    degrees = dict(nx_graph.degree(weight='weight'))
    pr_vals = np.array(list(pr.values()))
    deg_vals = np.array(list(degrees.values()))
    pr_p80 = np.percentile(pr_vals, 80) if len(pr_vals) > 0 else 0
    pr_p50 = np.percentile(pr_vals, 50) if len(pr_vals) > 0 else 0
    deg_p80 = np.percentile(deg_vals, 80) if len(deg_vals) > 0 else 0
    deg_p30 = np.percentile(deg_vals, 30) if len(deg_vals) > 0 else 0
    records = []
    for concept in valid_concepts:
        if concept not in pr:
            continue
        p = pr[concept]
        d = degrees.get(concept, 0)
        if p >= pr_p80 and d >= deg_p80:
            generation = "Foundational (Parent)"
        elif p <= pr_p50 and d <= deg_p30:
            generation = "Emerging (Child)"
        else:
            generation = "Intermediate"
        records.append({
            'concept': concept, 'pagerank': p, 'degree': d,
            'generation': generation
        })
    return pd.DataFrame(records).sort_values('pagerank', ascending=False)

def detect_cross_domain_bridges(nx_graph, valid_concepts):
    """Find edges that connect concepts from different battery categories."""
    category_map = abstract_concepts_to_categories(valid_concepts)
    bridges = []
    for u, v in nx_graph.edges():
        cat_u = category_map.get(u, 'general')
        cat_v = category_map.get(v, 'general')
        if cat_u != cat_v and cat_u != 'general' and cat_v != 'general':
            bridges.append({
                'concept_u': u, 'concept_v': v,
                'category_u': cat_u, 'category_v': cat_v,
                'weight': nx_graph[u][v].get('weight', 1)
            })
    return pd.DataFrame(bridges).sort_values('weight', ascending=False)

def analyze_network_motifs(nx_graph):
    """Count triangles, cliques, and star motifs in the graph."""
    motifs = {"triangles": 0, "clique_3": 0, "clique_4": 0, "star_motifs": 0}
    if nx_graph.number_of_nodes() < 3:
        return motifs
    try:
        motifs["triangles"] = sum(nx.triangles(nx_graph).values()) // 3
    except Exception:
        pass
    try:
        cliques = list(nx.find_cliques(nx_graph))
        for c in cliques:
            if len(c) >= 3:
                motifs["clique_3"] += 1
            if len(c) >= 4:
                motifs["clique_4"] += 1
    except Exception:
        pass
    try:
        for node in nx_graph.nodes():
            neighbors = list(nx_graph.neighbors(node))
            if len(neighbors) >= 3:
                sub = nx_graph.subgraph(neighbors)
                if sub.number_of_edges() == 0:
                    motifs["star_motifs"] += 1
    except Exception:
        pass
    return motifs

def compute_centrality_comparison(nx_graph):
    """Compute and compare multiple centrality metrics."""
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    cent = {}
    try:
        cent['degree'] = dict(nx.degree_centrality(nx_graph))
    except Exception:
        cent['degree'] = {n: 0 for n in nx_graph.nodes()}
    try:
        cent['betweenness'] = dict(nx.betweenness_centrality(nx_graph, normalized=True))
    except Exception:
        cent['betweenness'] = {n: 0 for n in nx_graph.nodes()}
    try:
        cent['closeness'] = dict(nx.closeness_centrality(nx_graph))
    except Exception:
        cent['closeness'] = {n: 0 for n in nx_graph.nodes()}
    try:
        cent['eigenvector'] = dict(nx.eigenvector_centrality(nx_graph, max_iter=500, weight='weight'))
    except Exception:
        cent['eigenvector'] = {n: 0 for n in nx_graph.nodes()}
    records = []
    for node in nx_graph.nodes():
        records.append({
            'concept': node,
            'degree_centrality': cent['degree'].get(node, 0),
            'betweenness_centrality': cent['betweenness'].get(node, 0),
            'closeness_centrality': cent['closeness'].get(node, 0),
            'eigenvector_centrality': cent['eigenvector'].get(node, 0)
        })
    return pd.DataFrame(records)

# ==========================================
# GRAPH EDIT HISTORY & UNDO/REDO
# ==========================================
class GraphEditHistory:
    def __init__(self, max_history=20):
        self.history = deque(maxlen=max_history)
        self.redo_stack = deque(maxlen=max_history)
        self.current = None

    def push_snapshot(self, nx_graph, concept_abstract_map):
        g_copy = nx.Graph()
        g_copy.add_nodes_from((n, dict(d)) for n, d in nx_graph.nodes(data=True))
        g_copy.add_edges_from((u, v, dict(d)) for u, v, d in nx_graph.edges(data=True))
        cam_copy = {k: list(v) for k, v in concept_abstract_map.items()}
        snapshot = {"graph": g_copy, "concept_abstract_map": cam_copy}
        if self.current is not None:
            self.history.append(self.current)
        self.current = snapshot
        self.redo_stack.clear()

    def undo(self):
        if not self.history:
            return None
        self.redo_stack.append(self.current)
        self.current = self.history.pop()
        return self.current

    def redo(self):
        if not self.redo_stack:
            return None
        self.history.append(self.current)
        self.current = self.redo_stack.pop()
        return self.current

    def can_undo(self):
        return len(self.history) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0

def apply_graph_edits(nx_graph, concept_abstract_map, edits):
    """Apply a dict of edits: remove_nodes, merge_nodes, add_edges, min_degree, min_freq."""
    G = nx_graph.copy()
    CAM = {k: list(v) for k, v in concept_abstract_map.items()}
    # Remove nodes
    for node in edits.get('remove_nodes', []):
        if node in G:
            G.remove_node(node)
            CAM.pop(node, None)
    # Merge nodes: {target: [source1, source2, ...]}
    for target, sources in edits.get('merge_nodes', {}).items():
        if target not in G:
            continue
        for src in sources:
            if src in G and src != target:
                for neighbor in list(G.neighbors(src)):
                    if neighbor != target and not G.has_edge(target, neighbor):
                        w = G[src][neighbor].get('weight', 1)
                        et = G[src][neighbor].get('edge_type', 'cooccurrence')
                        G.add_edge(target, neighbor, weight=w, edge_type=et)
                    elif neighbor != target:
                        G[target][neighbor]['weight'] = G[target][neighbor].get('weight', 0) + G[src][neighbor].get('weight', 0)
                CAM[target] = list(set(CAM.get(target, []) + CAM.get(src, [])))
                G.remove_node(src)
                CAM.pop(src, None)
    # Add edges: list of (u, v, weight)
    for u, v, w in edits.get('add_edges', []):
        if u in G and v in G:
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w, edge_type='manual')
    # Filter by degree
    min_deg = edits.get('min_degree', 0)
    if min_deg > 0:
        to_remove = [n for n, d in G.degree() if d < min_deg]
        for n in to_remove:
            G.remove_node(n)
            CAM.pop(n, None)
    # Filter by frequency
    min_freq = edits.get('min_freq', 0)
    if min_freq > 0:
        to_remove = [n for n in G.nodes() if len(CAM.get(n, [])) < min_freq]
        for n in to_remove:
            G.remove_node(n)
            CAM.pop(n, None)
    return G, CAM

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def get_battery_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    if any(c in concept_lower for c in ['cathode', 'ncm', 'nmc', 'lfp', 'lco', 'nca', 'layered', 'olivine', 'spinel']):
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
                        custom_labels=None, node_label_size=12, top_n_nodes=0,
                        theme=None, physics_preset=None,
                        show_edge_weights=False, edge_label_mode="hover"):
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    # Pre-compute deterministic layout
    pos = {}
    if len(nx_graph.nodes()) > 0:
        try:
            if len(nx_graph.nodes()) < 300:
                pos = nx.kamada_kawai_layout(nx_graph, weight='weight')
            else:
                pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')
        except Exception:
            pos = nx.spring_layout(nx_graph, k=2.5, iterations=200, seed=42, weight='weight')

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))

    net = Network(
        height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
        select_menu=True, notebook=False, cdn_resources='remote'
    )

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
          "physics": {{
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {{
              "gravitationalConstant": {physics_preset['gravity']},
              "centralGravity": {physics_preset['central_gravity']},
              "springLength": {physics_preset['spring_length']},
              "springConstant": {physics_preset['spring_strength']},
              "damping": {physics_preset['damping']},
              "overlap": 0.15
            }},
            "stabilization": {{
              "enabled": true,
              "iterations": {physics_preset['stabilization']},
              "updateInterval": 30,
              "onlyDynamicEdges": false,
              "fit": true
            }}
          }},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 180,
            "hideEdgesOnDrag": false,
            "zoomView": true,
            "dragView": true
          }}
        }}
        """)
    else:
        net.set_options("""
        var options = {
          "physics": { "enabled": false },
          "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true }
        }
        """)

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(min_node_size + freq * 1.2, min_node_size, max_node_size))
        color = get_battery_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        label = custom_labels.get(node, node) if custom_labels else node

        x, y = (pos.get(node, (0, 0))[0] * 1200, pos.get(node, (0, 0))[1] * 1200)

        net.add_node(
            node,
            label=label,
            size=size,
            x=x,
            y=y,
            color={
                'background': color,
                'border': theme['node_border'],
                'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}
            },
            font={
                'color': theme['font'],
                'size': node_label_size,
                'face': 'Inter, Segoe UI, Roboto, sans-serif',
                'strokeWidth': 0,
                'vadjust': -6
            },
            title=(
                f"<div style='font-family:Inter,sans-serif;'>"
                f"<b style='font-size:14px;color:{theme['highlight_bg']};'>{node}</b><br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Degree:</span> {degree}<br>"
                f"<span style='color:{theme['tooltip_text']};opacity:0.7;'>Frequency:</span> {freq}"
                f"</div>"
            ),
            borderWidth=2,
            borderWidthSelected=3,
            shadow={
                'enabled': True,
                'color': theme['shadow_color'],
                'size': 12,
                'x': 4,
                'y': 4
            },
            shape='dot',
            mass=max(1, 1 + freq * 0.05)
        )

    color_map = {
        'cooccurrence': theme['edge_cooccurrence'],
        'semantic':     theme['edge_semantic'],
        'bridge':       theme['edge_bridge'],
        'manual':       theme['edge_bridge'],
        'unknown':      theme['edge_unknown']
    }

    # Determine edge weight threshold for labeling
    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = np.percentile(all_weights, 80) if all_weights else 0

    for u, v in nx_graph.edges():
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        color = color_map.get(edge_type, color_map['unknown'])
        width = float(np.clip(w * 0.4, 0.8, 3.5))

        label_text = ""
        if show_edge_weights:
            if edge_label_mode == "all":
                label_text = f"{w:.1f}"
            elif edge_label_mode == "threshold" and w >= weight_threshold:
                label_text = f"{w:.1f}"
            elif edge_label_mode == "hover":
                label_text = ""

        net.add_edge(
            u, v,
            value=float(np.clip(w, 0.5, 5)),
            width=width,
            label=label_text,
            color={
                'color': color,
                'highlight': theme['highlight_bg'],
                'hover': theme['hover_bg'],
                'opacity': 0.85
            },
            smooth={'type': 'continuous', 'roundness': 0.35},
            title=f"<span style='font-family:Inter,sans-serif;'>Weight: <b>{w:.2f}</b><br>Type: {edge_type}</span>"
        )

    html_content = net.generate_html()

    custom_css = f"""
    <style>
        body {{
            background: {theme['bg']};
            margin: 0;
            padding: 0;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        #mynetwork {{
            border-radius: 16px;
            box-shadow: 0 12px 48px {theme['shadow_color']};
            outline: none;
        }}
        div.vis-tooltip {{
            background: {theme['tooltip_bg']} !important;
            color: {theme['tooltip_text']} !important;
            border: 1px solid {theme['tooltip_border']} !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            font-size: 13px !important;
            line-height: 1.5 !important;
            box-shadow: 0 8px 32px {theme['shadow_color']} !important;
            max-width: 320px !important;
            white-space: normal !important;
        }}
        div.vis-network div.vis-manipulation {{
            background: {theme['tooltip_bg']} !important;
            border-top: 1px solid {theme['tooltip_border']} !important;
            color: {theme['font']} !important;
        }}
    </style>
    """
    html_content = html_content.replace('</head>', custom_css + '</head>')

    st.components.v1.html(html_content, height=790, scrolling=True)

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button("📥 Download Interactive Graph (HTML)", data=html_bytes,
                          file_name="lib_concept_graph.html", mime="text/html")
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

def render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name="viridis",
                            custom_labels=None, top_n_nodes=0, node_label_size=10,
                            theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
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
                            line=dict(width=1, color=theme['edge_unknown']),
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
                                       line=dict(width=2, color=theme['node_border'])),
                            text=node_labels, textposition="bottom center",
                            textfont=dict(size=node_label_size, color=theme['font']),
                            hovertext=node_text, hoverinfo='text', name='Concepts')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0),
                                     plot_bgcolor=theme['plotly_bg'], paper_bgcolor=theme['plotly_paper'],
                                     font=dict(color=theme['font']),
                                     xaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color']),
                                     yaxis=dict(showgrid=True, gridcolor=theme['grid_color'],
                                                zeroline=False, showticklabels=False, linecolor=theme['axis_color'])))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name="viridis", top_n_nodes=0,
                            theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
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
                              line=dict(width=2, color=theme['edge_unknown']), hoverinfo='skip')
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
                                textfont=dict(size=8, color=theme['font']),
                                hovertext=node_text, hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(scene=dict(xaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 yaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color']),
                                                 zaxis=dict(showbackground=False, gridcolor=theme['grid_color'], linecolor=theme['axis_color'])),
                                     margin=dict(l=0, r=0, b=0, t=0), showlegend=False,
                                     paper_bgcolor=theme['plotly_paper']))
    st.plotly_chart(fig, use_container_width=True)

def render_graph_fallback(nx_graph, concept_abstract_map, theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.markdown(f"### 📊 Graph Summary (Text View)")
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
def build_category_hierarchy(valid_concepts: List[str], concept_abstract_map: Dict, top_n_per_category: int = 40, category_filter: Optional[List[str]] = None):
    hierarchy = defaultdict(lambda: {"children": [], "count": 0})
    category_map = abstract_concepts_to_categories(valid_concepts)
    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        if category_filter and category not in category_filter:
            continue
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

def render_sunburst_chart(labels, parents, values, cmap_name="viridis", label_size=11, width=800, height=600, theme=None, branchvalues="total"):
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
    bv = branchvalues if branchvalues in ["total", "remainder"] else ("remainder" if use_remainder else "total")
    fig = go.Figure(go.Sunburst(
        labels=unique_ids, parents=parent_ids, values=values, ids=unique_ids,
        branchvalues=bv,
        marker=dict(colors=colors, line=dict(width=0.5, color="white")),
        textinfo="label+percent entry+value",
        insidetextorientation="radial",
        textfont=dict(size=label_size),
        hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Parent: %{parent}<extra></extra>'
    ))
    fig.update_layout(
        title="<b>LiB Research Domain Hierarchy</b><br><i>Size = concept frequency</i>",
        font=dict(size=label_size, family="Arial"),
        paper_bgcolor="white", plot_bgcolor="white",
        width=width, height=height,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_radar_chart(concept_scores_df: pd.DataFrame, top_k: int = 15, cmap_name: str = "viridis", theme=None):
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
        paper_bgcolor=theme["plotly_paper"] if theme else "#ffffff",
        font=dict(color=theme["font"] if theme else "#000000"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# NEW EXTRA VISUALIZATIONS
# ==========================================
def render_concept_timeline(df, valid_concepts, concept_abstract_map, top_n=10):
    if 'Year' not in df.columns or df['Year'].isna().all():
        st.info("Year data not available for timeline.")
        return
    top_concepts = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:top_n]
    year_data = defaultdict(lambda: defaultdict(int))
    for concept in top_concepts:
        for idx in concept_abstract_map.get(concept, []):
            if idx < len(df):
                y = df.iloc[idx].get('Year')
                if pd.notna(y):
                    year_data[int(y)][concept] += 1
    if not year_data:
        return
    years = sorted(year_data.keys())
    plot_data = []
    for y in years:
        for c in top_concepts:
            plot_data.append({'Year': y, 'Concept': c, 'Count': year_data[y].get(c, 0)})
    plot_df = pd.DataFrame(plot_data)
    fig = px.line(plot_df, x='Year', y='Count', color='Concept', markers=True)
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def render_cooccurrence_heatmap(nx_graph, valid_concepts, top_n=30):
    if nx_graph.number_of_nodes() < 3:
        st.info("Not enough nodes for heatmap.")
        return
    top_nodes = sorted(valid_concepts, key=lambda c: nx_graph.degree(c), reverse=True)[:top_n]
    if len(top_nodes) < 3:
        return
    mat = np.zeros((len(top_nodes), len(top_nodes)))
    node_idx = {n: i for i, n in enumerate(top_nodes)}
    for u, v, d in nx_graph.edges(data=True):
        if u in node_idx and v in node_idx:
            w = d.get('weight', 1)
            mat[node_idx[u]][node_idx[v]] = w
            mat[node_idx[v]][node_idx[u]] = w
    fig = px.imshow(mat, x=top_nodes, y=top_nodes, color_continuous_scale='Viridis')
    fig.update_layout(width=700, height=700)
    st.plotly_chart(fig, use_container_width=True)

def render_tsne_projection(valid_concepts, embed_model, nx_graph):
    if len(valid_concepts) < 5:
        st.info("Need ≥5 concepts for t-SNE.")
        return
    embs = embed_model.encode(valid_concepts, show_progress_bar=False, batch_size=64)
    tsne = TSNE(n_components=2, perplexity=min(30, len(valid_concepts)-1), random_state=42, init='pca')
    coords = tsne.fit_transform(embs)
    cat_map = abstract_concepts_to_categories(valid_concepts)
    df = pd.DataFrame({
        'x': coords[:, 0], 'y': coords[:, 1],
        'concept': valid_concepts,
        'category': [cat_map.get(c, 'general') for c in valid_concepts],
        'degree': [nx_graph.degree(c) for c in valid_concepts]
    })
    fig = px.scatter(df, x='x', y='y', color='category', size='degree',
                     hover_data=['concept'], title='t-SNE Concept Embedding Projection')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def render_community_detection(nx_graph, concept_abstract_map, cmap_name='viridis', theme=None):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if nx_graph.number_of_nodes() < 3:
        st.info("Need ≥3 nodes for community detection.")
        return
    try:
        from networkx.algorithms import community
        comms = list(community.greedy_modularity_communities(nx_graph))
    except Exception:
        st.warning("Community detection failed.")
        return
    node_comm = {}
    for i, comm in enumerate(comms):
        for node in comm:
            node_comm[node] = i
    pos = nx.spring_layout(nx_graph, seed=42)
    colors = get_colormap_colors(cmap_name, max(len(comms), 1))
    edge_x, edge_y = [], []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=0.8, color=theme['edge_unknown']))
    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for node in nx_graph.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_color.append(colors[node_comm.get(node, 0) % len(colors)])
        node_text.append(f"{node}<br>Community: {node_comm.get(node, 'N/A')}")
        node_size.append(max(8, min(30, nx_graph.degree(node) * 2 + 8)))
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            marker=dict(size=node_size, color=node_color,
                                       line=dict(width=1.5, color=theme['node_border'])),
                            text=[n for n in nx_graph.nodes()], textposition='bottom center',
                            textfont=dict(size=8, color=theme['font']),
                            hovertext=node_text, hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False, hovermode='closest',
                                     plot_bgcolor=theme['plotly_bg'], paper_bgcolor=theme['plotly_paper'],
                                     font=dict(color=theme['font']),
                                     xaxis=dict(showgrid=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, showticklabels=False)))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Detected {len(comms)} communities**")

def render_concept_growth_rate(df, valid_concepts, concept_abstract_map):
    if 'Year' not in df.columns or df['Year'].isna().all():
        st.info("Year data required for growth rate analysis.")
        return
    median_year = int(df['Year'].dropna().median())
    records = []
    for concept in valid_concepts:
        docs = concept_abstract_map.get(concept, [])
        early = sum(1 for idx in docs if idx < len(df) and pd.notna(df.iloc[idx].get('Year')) and int(df.iloc[idx]['Year']) < median_year)
        recent = sum(1 for idx in docs if idx < len(df) and pd.notna(df.iloc[idx].get('Year')) and int(df.iloc[idx]['Year']) >= median_year)
        records.append({'concept': concept, 'early': early, 'recent': recent,
                        'growth_rate': (recent - early) / max(early, 1)})
    growth_df = pd.DataFrame(records).sort_values('growth_rate', ascending=False).head(20)
    fig = px.bar(growth_df, x='concept', y='growth_rate', color='recent',
                 labels={'growth_rate': 'Growth Rate (Recent/Early)'})
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

def render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, distill_df):
    records = []
    deg_map = {c: nx_graph.degree(c) for c in valid_concepts}
    freq_map = {c: len(concept_abstract_map.get(c, [])) for c in valid_concepts}
    eff_map = dict(zip(distill_df['concept'], distill_df['distillation_efficiency'])) if not distill_df.empty else {}
    for c in valid_concepts:
        records.append({
            'concept': c, 'degree': deg_map.get(c, 0),
            'frequency': freq_map.get(c, 0),
            'efficiency': eff_map.get(c, 0)
        })
    bubble_df = pd.DataFrame(records)
    fig = px.scatter(bubble_df, x='degree', y='frequency', size='efficiency',
                     color='efficiency', hover_data=['concept'],
                     title='Concept Bubble Chart: Degree vs Frequency',
                     color_continuous_scale='Viridis')
    fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
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
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/png", "lib_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif format_type == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), facecolor='white')
            node_colors = [get_battery_category_color(n) for n in nx_graph.nodes()]
            nx.draw(nx_graph, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                   node_size=400, font_size=7, font_weight='bold', edgecolors='white', linewidths=1)
            buf = io.BytesIO()
            plt.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
            buf.seek(0); plt.close()
            return buf.read(), "image/svg+xml", "lib_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif format_type == "Publication PNG (600 DPI)":
        data = export_publication_figure(nx_graph, concept_abstract_map, dpi=600)
        return data, "image/png", "lib_publication_600dpi.png"
    return None, None, None

def export_publication_figure(nx_graph, concept_abstract_map, filename="lib_publication.png", dpi=300):
    pos = nx.spring_layout(nx_graph, seed=42, k=1.5, iterations=100)
    plt.figure(figsize=(16, 14), dpi=dpi, facecolor='white')
    node_colors = [get_battery_category_color(n) for n in nx_graph.nodes()]
    node_sizes = [max(80, min(800, len(concept_abstract_map.get(n, [])) * 15 + 50)) for n in nx_graph.nodes()]
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, width=0.6, edge_color='gray')
    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors='white', linewidths=1.5, alpha=0.95)
    nx.draw_networkx_labels(nx_graph, pos, font_size=7, font_weight='bold', font_color='#1e293b')
    plt.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0); plt.close()
    return buf.read()

def generate_analysis_report(df, valid_concepts, concept_abstract_map, nx_graph, top_scores,
                             burst_df, drift_df, genealogy_df, bridge_df, motif_data,
                             metrics, config):
    lines = []
    lines.append("# LiB-ConceptGraph Analysis Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Records:** {len(df)} | **Concepts:** {len(valid_concepts)} | **Edges:** {nx_graph.number_of_edges()}\n")
    lines.append("---\n")
    lines.append("## Dataset Overview\n")
    if 'Year' in df.columns:
        yr_range = f"{int(df['Year'].min())}-{int(df['Year'].max())}" if not df['Year'].isna().all() else "N/A"
        lines.append(f"- Year range: {yr_range}\n")
    lines.append(f"- Source files: {df['_source_file'].nunique() if '_source_file' in df.columns else 'N/A'}\n")
    lines.append("---\n")
    lines.append("## Top Concepts by Frequency\n")
    top_c = sorted(valid_concepts, key=lambda c: len(concept_abstract_map.get(c, [])), reverse=True)[:15]
    for i, c in enumerate(top_c, 1):
        lines.append(f"{i}. **{c}** — {len(concept_abstract_map.get(c, []))} abstracts\n")
    lines.append("---\n")
    lines.append("## Graph Validation Metrics\n")
    lines.append(f"- Modularity: {metrics.get('modularity', 0):.3f}\n")
    lines.append(f"- Silhouette: {metrics.get('silhouette_score', 0):.3f}\n")
    lines.append(f"- Communities: {metrics.get('n_communities', 0)}\n")
    lines.append("---\n")
    if not burst_df.empty:
        lines.append("## Keyword Bursts\n")
        for _, row in burst_df.head(10).iterrows():
            lines.append(f"- **{row['concept']}** in {row['year']} (burst ratio: {row['burst_ratio']:.2f})\n")
        lines.append("---\n")
    if not drift_df.empty:
        lines.append("## Semantic Drift\n")
        for _, row in drift_df.head(10).iterrows():
            lines.append(f"- **{row['concept']}**: drift={row['drift_score']:.3f}, similarity={row['similarity']:.3f}\n")
        lines.append("---\n")
    if not genealogy_df.empty:
        lines.append("## Concept Genealogy\n")
        for gen in ['Foundational (Parent)', 'Intermediate', 'Emerging (Child)']:
            subset = genealogy_df[genealogy_df['generation'] == gen].head(5)
            if not subset.empty:
                lines.append(f"### {gen}\n")
                for _, row in subset.iterrows():
                    lines.append(f"- {row['concept']} (PR={row['pagerank']:.4f}, deg={row['degree']})\n")
        lines.append("---\n")
    if not bridge_df.empty:
        lines.append("## Cross-Domain Bridges\n")
        for _, row in bridge_df.head(10).iterrows():
            lines.append(f"- `{row['concept_u']}` ({row['category_u']}) ↔ `{row['concept_v']}` ({row['category_v']}) — weight {row['weight']:.2f}\n")
        lines.append("---\n")
    lines.append("## Network Motifs\n")
    for k, v in motif_data.items():
        lines.append(f"- {k.replace('_', ' ').title()}: {v}\n")
    lines.append("---\n")
    if not top_scores.empty:
        lines.append("## Top Research Directions\n")
        for _, row in top_scores.head(10).iterrows():
            lines.append(f"- **{row['concept_u']} + {row['concept_v']}** | Score: {row['composite_score']:.3f} | Novelty: {row['semantic_novelty']:.3f}\n")
    return "\n".join(lines)

# ==========================================
# THEME & PHYSICS CONFIGURATION
# ==========================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b", "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)", "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b"
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0", "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b", "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)", "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8"
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9", "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)", "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b"
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407", "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626", "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)", "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412"
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16", "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d", "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)", "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534"
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344", "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9", "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)", "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490"
    }
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25, "stabilization": 2500
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30, "stabilization": 1500
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20, "stabilization": 3000
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0, "stabilization": 0
    }
}

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

def display_metric_dashboard(metrics: dict, theme=None):
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

        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0
        )
        theme = THEME_PRESETS[st.session_state['theme']]

        st.subheader("🔋 LiB Focus Areas")
        st.markdown("- Energy density (Wh/kg, Wh/L)")
        st.markdown("- Cathode materials (NCM, LFP, NCA)")
        st.markdown("- Anode materials (Si, graphite, Li metal)")
        st.markdown("- Electrolytes (liquid, solid-state)")
        st.markdown("- Cell design & manufacturing")
        st.markdown("- Safety & degradation")

        st.subheader("🖼️ Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:", ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"], index=0
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:", options=list(SUPPORTED_COLORMAPS.keys()), index=0
        )

        st.subheader("🔧 Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0)
        )

        with st.expander("⚙️ Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider("Damping", 0.05, 0.95, preset["damping"], step=0.05)
            st.session_state['adv_gravity'] = st.slider("Repulsion", -8000, -500, preset["gravity"], step=100)
            st.session_state['adv_spring_length'] = st.slider("Spring length", 40, 300, preset["spring_length"], step=10)
            st.session_state['adv_spring_strength'] = st.slider("Spring strength", 0.01, 0.20, preset["spring_strength"], step=0.01)
            st.session_state['adv_central_gravity'] = st.slider("Central gravity", 0.0, 0.5, preset["central_gravity"], step=0.05)
            st.session_state['adv_stabilization'] = st.slider("Stabilization iter", 0, 5000, preset["stabilization"], step=250)

        # Build effective physics preset from base + overrides
        base_preset = PHYSICS_PRESETS[st.session_state['physics_preset']].copy()
        if st.session_state.get('adv_damping') is not None:
            base_preset["damping"] = st.session_state['adv_damping']
            base_preset["gravity"] = st.session_state['adv_gravity']
            base_preset["spring_length"] = st.session_state['adv_spring_length']
            base_preset["spring_strength"] = st.session_state['adv_spring_strength']
            base_preset["central_gravity"] = st.session_state['adv_central_gravity']
            base_preset["stabilization"] = st.session_state['adv_stabilization']
        st.session_state['effective_physics'] = base_preset

        st.subheader("📊 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state['top_n_graph'] = st.slider(
                "Max nodes", 10, 500, 200, step=10, disabled=all_graph,
                key="top_n_graph_slider"
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0

        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10, disabled=all_sun,
                key="top_n_sunburst_slider"
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0

        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15, disabled=all_radar,
                key="top_n_radar_slider"
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0

        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider("Min concept frequency", 1, 20, 5)
        st.session_state['min_words'] = st.slider("Min words per concept", 2, 5, 2)
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
    st.title("🔋 LiB-ConceptGraph: Energy Density Explorer")
    st.caption("Large-corpus concept graph builder for lithium-ion battery research • 3000+ abstracts optimized")
    render_sidebar()

    # Session state initialization
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "burst_df" not in st.session_state:
        st.session_state.burst_df = None
    if "drift_df" not in st.session_state:
        st.session_state.drift_df = None
    if "genealogy_df" not in st.session_state:
        st.session_state.genealogy_df = None
    if "bridge_df" not in st.session_state:
        st.session_state.bridge_df = None
    if "centrality_df" not in st.session_state:
        st.session_state.centrality_df = None
    if "motif_data" not in st.session_state:
        st.session_state.motif_data = None
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = GraphEditHistory()
    if "edited_graph" not in st.session_state:
        st.session_state.edited_graph = None
    if "edited_cam" not in st.session_state:
        st.session_state.edited_cam = None

    # ─── LOAD JSON DATA ───
    st.header("📂 Data Loading")
    if BIBTEX_AVAILABLE:
        st.info(f"Place JSON/BibTeX/CSV files in: `{JSON_METADATA_DIR}`")
    else:
        st.info(f"Place JSON/CSV files in: `{JSON_METADATA_DIR}` (install `bibtexparser` for .bib support)")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)
    if not file_records:
        st.warning("No supported files found in the directory.")
        st.info("Please place your JSON/BibTeX/CSV metadata files in the `json_metadatabase/` folder.")
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

                # Initialize edit history with original graph
                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)

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
                    "config": config,
                    "df": df_filtered
                }
                st.session_state.edited_graph = nx_graph
                st.session_state.edited_cam = concept_abstract_map
                st.rerun()
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

        # Use edited graph if available
        if st.session_state.edited_graph is not None:
            nx_graph = st.session_state.edited_graph
            concept_abstract_map = st.session_state.edited_cam

        viz_tab, distill_tab, scores_tab, valid_tab, extra_viz_tab, advanced_tab, export_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions", 
            "📐 Validation", "📈 Extra Viz", "🧠 Advanced Analytics", "📥 Export"
        ])

        with viz_tab:
            st.subheader("🌐 Interactive Concept Graph")

            # Graph editing controls
            with st.expander("✏️ Graph Editing (Undo/Redo)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("↩️ Undo", disabled=not st.session_state.edit_history.can_undo()):
                        snapshot = st.session_state.edit_history.undo()
                        if snapshot:
                            st.session_state.edited_graph = snapshot["graph"]
                            st.session_state.edited_cam = snapshot["concept_abstract_map"]
                            st.rerun()
                with col2:
                    if st.button("↪️ Redo", disabled=not st.session_state.edit_history.can_redo()):
                        snapshot = st.session_state.edit_history.redo()
                        if snapshot:
                            st.session_state.edited_graph = snapshot["graph"]
                            st.session_state.edited_cam = snapshot["concept_abstract_map"]
                            st.rerun()
                with col3:
                    if st.button("🔄 Reset"):
                        st.session_state.edited_graph = data["nx_graph"]
                        st.session_state.edited_cam = data["concept_abstract_map"]
                        st.session_state.edit_history = GraphEditHistory()
                        st.session_state.edit_history.push_snapshot(data["nx_graph"], data["concept_abstract_map"])
                        st.rerun()

                st.markdown("---")

                # Remove nodes
                with st.expander("🗑️ Remove Nodes"):
                    nodes_to_remove = st.multiselect("Select nodes to remove", list(nx_graph.nodes()))
                    if st.button("Remove Selected Nodes"):
                        edits = {'remove_nodes': nodes_to_remove}
                        new_g, new_cam = apply_graph_edits(nx_graph, concept_abstract_map, edits)
                        st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)
                        st.session_state.edited_graph = new_g
                        st.session_state.edited_cam = new_cam
                        st.rerun()

                # Merge nodes
                with st.expander("🔀 Merge Nodes"):
                    merge_target = st.selectbox("Target node (keep)", list(nx_graph.nodes()), key="merge_target")
                    merge_sources = st.multiselect("Source nodes (merge into target)", 
                                                   [n for n in nx_graph.nodes() if n != merge_target])
                    if st.button("Merge Nodes"):
                        edits = {'merge_nodes': {merge_target: merge_sources}}
                        new_g, new_cam = apply_graph_edits(nx_graph, concept_abstract_map, edits)
                        st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)
                        st.session_state.edited_graph = new_g
                        st.session_state.edited_cam = new_cam
                        st.rerun()

                # Add edge
                with st.expander("➕ Add Edge"):
                    edge_u = st.selectbox("Source", list(nx_graph.nodes()), key="edge_u")
                    edge_v = st.selectbox("Target", [n for n in nx_graph.nodes() if n != edge_u], key="edge_v")
                    edge_w = st.number_input("Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                    if st.button("Add Edge"):
                        edits = {'add_edges': [(edge_u, edge_v, edge_w)]}
                        new_g, new_cam = apply_graph_edits(nx_graph, concept_abstract_map, edits)
                        st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)
                        st.session_state.edited_graph = new_g
                        st.session_state.edited_cam = new_cam
                        st.rerun()

                # Filter by degree/frequency
                with st.expander("🔍 Filter by Degree / Frequency"):
                    min_deg_filter = st.slider("Min degree", 0, 20, 0)
                    min_freq_filter = st.slider("Min frequency", 0, 20, 0)
                    if st.button("Apply Filters"):
                        edits = {'min_degree': min_deg_filter, 'min_freq': min_freq_filter}
                        new_g, new_cam = apply_graph_edits(nx_graph, concept_abstract_map, edits)
                        st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)
                        st.session_state.edited_graph = new_g
                        st.session_state.edited_cam = new_cam
                        st.rerun()

            # Edge display options
            col_edge1, col_edge2 = st.columns(2)
            with col_edge1:
                show_edge_weights = st.checkbox("Show edge weights", value=False)
            with col_edge2:
                edge_label_mode = st.selectbox("Edge label mode", ["hover", "threshold", "all"])

            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            elif nx_graph.number_of_edges() == 0:
                st.warning("No edges — building semantic fallback")
                nx_graph = nx.complete_graph(len(valid_concepts))
                nx_graph = nx.relabel_nodes(nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))})

            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get('effective_physics', PHYSICS_PRESETS["Stable (Default)"])
            theme = THEME_PRESETS.get(st.session_state.get('theme', 'Bright (Default)'), THEME_PRESETS["Bright (Default)"])

            top_n = st.session_state.get('top_n_graph', 0)

            if viz_choice == "PyVis (Interactive)":
                render_graph_pyvis(nx_graph, concept_abstract_map, physics_enabled=physics,
                                   cmap_name=cmap, top_n_nodes=top_n,
                                   theme=theme, physics_preset=physics_preset,
                                   show_edge_weights=show_edge_weights, edge_label_mode=edge_label_mode)
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                       theme=theme)
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(nx_graph, concept_abstract_map, cmap_name=cmap, top_n_nodes=top_n,
                                        theme=theme)
            else:
                render_graph_fallback(nx_graph, concept_abstract_map, theme=theme)

            with st.expander("📊 Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics, theme=theme)

            with st.expander("📈 Domain Hierarchy (Sunburst)"):
                all_cats = list(set(abstract_concepts_to_categories(valid_concepts).values()))
                selected_cats = st.multiselect("Filter categories", all_cats, default=all_cats)
                bv = st.selectbox("Branch values", ["total", "remainder"])
                labels, parents, values = build_category_hierarchy(valid_concepts, concept_abstract_map,
                                                                    top_n_per_category=st.session_state.get('top_n_sunburst', 0),
                                                                    category_filter=selected_cats)
                render_sunburst_chart(labels, parents, values, cmap_name=cmap, theme=theme, branchvalues=bv)

            with st.expander("📡 Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(distill_df, top_k=radar_k, cmap_name=cmap, theme=theme)

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

        with extra_viz_tab:
            st.subheader("📈 Extra Visualizations")

            with st.expander("📅 Concept Timeline"):
                render_concept_timeline(data["df"], valid_concepts, concept_abstract_map)

            with st.expander("🔥 Co-occurrence Heatmap"):
                n_heat = st.slider("Top N concepts", 5, 50, 20, key="heat_n")
                render_cooccurrence_heatmap(nx_graph, valid_concepts, top_n=n_heat)

            with st.expander("🗺️ t-SNE Projection"):
                render_tsne_projection(valid_concepts, data["embed_model"], nx_graph)

            with st.expander("👥 Community Detection"):
                render_community_detection(nx_graph, concept_abstract_map, cmap_name=cmap, theme=theme)

            with st.expander("📊 Concept Growth Rate"):
                render_concept_growth_rate(data["df"], valid_concepts, concept_abstract_map)

            with st.expander("🫧 Bubble Chart"):
                render_bubble_chart(nx_graph, valid_concepts, concept_abstract_map, distill_df)

        with advanced_tab:
            st.subheader("🧠 Advanced Analytics")

            if st.button("🔬 Run Advanced Analytics", type="primary"):
                with st.spinner("Computing advanced analytics..."):
                    # Keyword bursts
                    st.session_state.burst_df = detect_keyword_bursts(
                        data["df"], valid_concepts, concept_abstract_map
                    )
                    # Semantic drift
                    st.session_state.drift_df = detect_semantic_drift(
                        valid_concepts, concept_abstract_map, data["all_texts"], data["df"], data["embed_model"]
                    )
                    # Genealogy
                    st.session_state.genealogy_df = build_concept_genealogy(nx_graph, valid_concepts)
                    # Cross-domain bridges
                    st.session_state.bridge_df = detect_cross_domain_bridges(nx_graph, valid_concepts)
                    # Network motifs
                    st.session_state.motif_data = analyze_network_motifs(nx_graph)
                    # Centrality comparison
                    st.session_state.centrality_df = compute_centrality_comparison(nx_graph)
                    st.success("Advanced analytics complete!")
                    st.rerun()

            if st.session_state.burst_df is not None and not st.session_state.burst_df.empty:
                st.markdown("### 🔥 Keyword Burst Detection")
                st.dataframe(st.session_state.burst_df.head(20), use_container_width=True)
                fig = px.bar(st.session_state.burst_df.head(15), x='concept', y='burst_score',
                             color='year', title='Top Keyword Bursts')
                fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.drift_df is not None and not st.session_state.drift_df.empty:
                st.markdown("### 🌊 Semantic Drift Detection")
                st.dataframe(st.session_state.drift_df.head(20), use_container_width=True)
                fig = px.scatter(st.session_state.drift_df, x='similarity', y='drift_score',
                                 color='recent_papers', hover_data=['concept'],
                                 title='Semantic Drift: Early vs Recent Context')
                fig.update_layout(paper_bgcolor='white', plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.genealogy_df is not None and not st.session_state.genealogy_df.empty:
                st.markdown("### 🧬 Concept Genealogy")
                st.dataframe(st.session_state.genealogy_df, use_container_width=True)
                gen_counts = st.session_state.genealogy_df['generation'].value_counts()
                fig = px.pie(values=gen_counts.values, names=gen_counts.index, title='Concept Generations')
                fig.update_layout(paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

            if st.session_state.bridge_df is not None and not st.session_state.bridge_df.empty:
                st.markdown("### 🌉 Cross-Domain Bridges")
                st.dataframe(st.session_state.bridge_df.head(20), use_container_width=True)

            if st.session_state.motif_data is not None:
                st.markdown("### 🔷 Network Motifs")
                motif_df = pd.DataFrame([st.session_state.motif_data])
                st.dataframe(motif_df, use_container_width=True)

            if st.session_state.centrality_df is not None and not st.session_state.centrality_df.empty:
                st.markdown("### 📊 Centrality Comparison")
                st.dataframe(st.session_state.centrality_df.head(20), use_container_width=True)
                corr = st.session_state.centrality_df[['degree_centrality', 'betweenness_centrality',
                                                       'closeness_centrality', 'eigenvector_centrality']].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                title='Centrality Correlation Matrix')
                fig.update_layout(paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

        with export_tab:
            st.subheader("📥 Export & Post-Processing")
            export_format = st.selectbox("Format:", ["GraphML", "JSON", "CSV (Edges)", "CSV (Nodes)", "PNG", "SVG", "Publication PNG (600 DPI)"])
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

            # Automated report
            st.markdown("---")
            st.markdown("### 📝 Automated Markdown Report")
            if st.button("📄 Generate Report"):
                val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
                report = generate_analysis_report(
                    data["df"], valid_concepts, concept_abstract_map, nx_graph, top_scores,
                    st.session_state.burst_df or pd.DataFrame(),
                    st.session_state.drift_df or pd.DataFrame(),
                    st.session_state.genealogy_df or pd.DataFrame(),
                    st.session_state.bridge_df or pd.DataFrame(),
                    st.session_state.motif_data or {},
                    val_metrics, data["config"]
                )
                st.download_button("📥 Download Report (.md)", data=report.encode('utf-8'),
                                  file_name="lib_analysis_report.md", mime="text/markdown")
                with st.expander("Preview Report"):
                    st.markdown(report)

if __name__ == "__main__":
    main()
