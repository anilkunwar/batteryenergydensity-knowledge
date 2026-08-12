#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LiB-ConceptGraph: Energy Density Concept Graph Builder for Lithium-Ion Batteries
==================================================================================
Enhanced large-corpus concept graph extraction (3000+ abstracts) from JSON/BibTeX/CSV metadata.
No seed injection needed — robust statistical methods for high-volume data.

NEW IN THIS VERSION (v6.2-style):
- LLM-Guided Query & Dynamic Ontology Expansion (as in Cu@Ag v6.2)
- Priority-driven subgraph extraction and GraphRAG answer generation
- Full integration with Ollama / OpenAI / fallback rule-based analysis
- Complete mutation tracking, undo/reset, query history

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run: streamlit run lib_concept_graph_llm.py

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
import base64
import requests  # for Ollama
import copy
from collections import defaultdict, Counter, deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Any, Set, Iterator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
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
from pyvis.network import Network
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    G = nx_graph.copy()
    CAM = {k: list(v) for k, v in concept_abstract_map.items()}
    for node in edits.get('remove_nodes', []):
        if node in G:
            G.remove_node(node)
            CAM.pop(node, None)
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
    for u, v, w in edits.get('add_edges', []):
        if u in G and v in G:
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w, edge_type='manual')
    min_deg = edits.get('min_degree', 0)
    if min_deg > 0:
        to_remove = [n for n, d in G.degree() if d < min_deg]
        for n in to_remove:
            G.remove_node(n)
            CAM.pop(n, None)
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
# EXTRA VISUALIZATIONS
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
# ==========================================
# ==========================================
# BEGIN LLM-GUIDED QUERY SYSTEM (PORTED FROM Cu@Ag v6.2)
# ==========================================
# ==========================================
# ==========================================

# ----------------------------------------------------------------------------
# 1. LiB DOMAIN ONTOLOGY (for reasoning and relationship inference)
# ----------------------------------------------------------------------------

class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MODEL = "model"
    GENERAL = "general"

class RelationshipType(Enum):
    CAUSES = "causes"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CO_OCCURS = "co_occurs"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    BRIDGE = "bridge"
    CONSTRAINS = "constrains"
    MODIFIES = "modifies"
    CORRECTS = "corrects"
    SELECTS = "selects"
    INITIATES = "initiates"
    DRIVES = "drives"
    TRANSITIONS_TO = "transitions_to"
    REPLACES = "replaces"
    TRAINS = "trains"
    OUTPUTS = "outputs"
    LEARNS = "learns"
    CAPTURES = "captures"
    PARALLELIZES = "parallelizes"
    POSITIONS = "positions"
    IDENTIFIES = "identifies"
    FORMS = "forms"
    PROCESSES = "processes"
    STABILIZES = "stabilizes"
    PRESERVES = "preserves"
    GENERATES = "generates"
    COMPOSES = "composes"
    QUALIFIES = "qualifies"
    ENABLES = "enables"
    DISCOVERS = "discovers"
    PRE_TRAINS = "pre_trains"
    GENERALIZES = "generalizes"
    QUERIES = "queries"
    OPTIMIZES = "optimizes"
    VALIDATES = "validates"
    BOUNDS = "bounds"
    QUANTIFIES = "quantifies"
    EVALUATES = "evaluates"
    COMPARES = "compares"
    COMPUTES = "computes"
    MODELS = "models"
    AVERAGES = "averages"
    MAPS = "maps"
    SIMULATES = "simulates"
    DETECTS = "detects"
    MEASURES = "measures"
    OBSERVES = "observes"
    INTEGRATES = "integrates"
    COUPLES = "couples"
    UPSCALES = "upscales"
    RESOLVES = "resolves"
    SYNCHRONIZES = "synchronizes"
    CHARACTERIZES = "characterizes"
    DECOMPOSES = "decomposes"
    DESIGNS = "designs"
    APPROXIMATES = "approximates"
    STRENGTHENS = "strengthens"
    EXPLAINS = "explains"
    INTERPRETS = "interprets"
    GROUPS = "groups"
    VISUALIZES = "visualizes"
    CONSTRUCTS = "constructs"
    FRAMES = "frames"
    ACCELERATES = "accelerates"
    ENFORCES = "enforces"
    CORRELATES = "correlates"
    PREVENTS = "prevents"

@dataclass
class ConceptNode:
    canonical_name: str
    concept_type: ConceptType
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    related_processes: Set[str] = field(default_factory=set)
    related_properties: Set[str] = field(default_factory=set)
    definition: str = ""
    embedding: Optional[np.ndarray] = None

    def add_synonym(self, synonym: str) -> None:
        self.synonyms.add(synonym.lower().strip())

    def is_match(self, text: str) -> bool:
        text_lower = text.lower().strip()
        if text_lower == self.canonical_name.lower():
            return True
        return text_lower in self.synonyms

@dataclass
class Relationship:
    source: str
    target: str
    rel_type: RelationshipType
    confidence: float = 1.0
    evidence: str = ""
    inferred: bool = False

class DomainOntology:
    """LiB-specific ontology with concepts and causal chains."""
    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self.synonym_to_canonical: Dict[str, str] = {}
        self._build_ontology()

    def _build_ontology(self) -> None:
        # ---- Materials ----
        self._add_concept("cathode", ConceptType.MATERIAL,
            synonyms={"positive electrode", "cathode material"},
            definition="Positive electrode in a Li-ion battery, typically a lithium transition metal oxide.")
        self._add_concept("anode", ConceptType.MATERIAL,
            synonyms={"negative electrode", "anode material"},
            definition="Negative electrode in a Li-ion battery, typically graphite, silicon, or lithium metal.")
        self._add_concept("electrolyte", ConceptType.MATERIAL,
            synonyms={"liquid electrolyte", "solid electrolyte", "polymer electrolyte", "electrolyte solution"},
            definition="Ion-conducting medium separating cathode and anode; can be liquid, solid, or polymer.")
        self._add_concept("separator", ConceptType.MATERIAL,
            synonyms={"separator film", "membrane"},
            definition="Porous membrane that physically separates cathode and anode while allowing ion transport.")
        self._add_concept("current_collector", ConceptType.MATERIAL,
            synonyms={"cc", "al foil", "cu foil"},
            definition="Conductive foil (Al for cathode, Cu for anode) that collects and conducts electrons.")
        self._add_concept("binder", ConceptType.MATERIAL,
            synonyms={"pvdf", "cmc", "sbr", "paa", "alginate"},
            definition="Polymer that holds electrode active materials and conductive additives together.")
        self._add_concept("conductive_additive", ConceptType.MATERIAL,
            synonyms={"carbon black", "acetylene black", "cnt", "graphene", "super p"},
            definition="Electrically conductive additive to improve electrode conductivity.")
        self._add_concept("lithium_transition_metal_oxide", ConceptType.MATERIAL,
            synonyms={"layered oxide", "ncm", "nmc", "lco", "nca", "lno", "lmo", "lfp", "olivine", "spinel"},
            definition="Class of cathode materials based on lithium and transition metal oxides.")
        self._add_concept("graphite", ConceptType.MATERIAL,
            synonyms={"natural graphite", "synthetic graphite", "graphitic carbon"},
            definition="Layered carbon material commonly used as anode, intercalating lithium.")
        self._add_concept("silicon", ConceptType.MATERIAL,
            synonyms={"si", "silicon nanowire", "silicon nanoparticle", "siox", "silicon oxide"},
            definition="High-capacity anode material that alloys with lithium; undergoes large volume change.")
        self._add_concept("lithium_metal", ConceptType.MATERIAL,
            synonyms={"li metal", "lithium foil", "lithium anode"},
            definition="Metallic lithium used as anode in lithium-metal batteries; high energy density but safety challenges.")
        self._add_concept("solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"sulfide electrolyte", "oxide electrolyte", "halide electrolyte", "garnet", "nasicon", "lispo", "llzo", "lagp"},
            definition="Non-liquid electrolyte with high ionic conductivity and mechanical strength, used in solid-state batteries.")
        self._add_concept("interphase", ConceptType.MATERIAL,
            synonyms={"sei", "cei", "solid electrolyte interphase", "cathode electrolyte interphase", "passivation layer"},
            definition="Thin layer formed on electrode surfaces due to electrolyte decomposition; affects kinetics and stability.")
        # ---- Properties ----
        self._add_concept("energy_density", ConceptType.PROPERTY,
            synonyms={"specific energy", "gravimetric energy", "volumetric energy", "wh/kg", "wh/l"},
            definition="Amount of energy stored per unit mass or volume.")
        self._add_concept("power_density", ConceptType.PROPERTY,
            synonyms={"specific power", "w/kg", "w/l"},
            definition="Rate of energy delivery per unit mass or volume.")
        self._add_concept("capacity", ConceptType.PROPERTY,
            synonyms={"areal capacity", "specific capacity", "mah/g", "mah/cm2"},
            definition="Amount of charge stored per unit mass or area.")
        self._add_concept("voltage", ConceptType.PROPERTY,
            synonyms={"cell voltage", "open circuit voltage", "average voltage", "voltage plateau"},
            definition="Electrical potential difference between cathode and anode.")
        self._add_concept("coulombic_efficiency", ConceptType.PROPERTY,
            synonyms={"ce", "charge efficiency", "discharge efficiency"},
            definition="Ratio of discharge capacity to charge capacity.")
        self._add_concept("cycle_life", ConceptType.PROPERTY,
            synonyms={"calendar life", "lifetime", "cycle stability"},
            definition="Number of charge/discharge cycles before capacity drops below a threshold.")
        self._add_concept("rate_capability", ConceptType.PROPERTY,
            synonyms={"c-rate", "fast charging", "high rate", "rate performance"},
            definition="Ability to charge/discharge at high current rates.")
        self._add_concept("safety", ConceptType.PROPERTY,
            synonyms={"thermal stability", "fire safety", "overcharge protection"},
            definition="Resistance to thermal runaway, fire, explosion, dendrite formation, short circuits.")
        # ---- Processes ----
        self._add_concept("lithiation", ConceptType.PROCESS,
            synonyms={"intercalation", "alloying", "lithium insertion"},
            definition="Incorporation of lithium ions into the electrode material.")
        self._add_concept("delithiation", ConceptType.PROCESS,
            synonyms={"deintercalation", "lithium extraction"},
            definition="Removal of lithium ions from the electrode material.")
        self._add_concept("solid_electrolyte_interphase_formation", ConceptType.PROCESS,
            synonyms={"sei formation", "passivation", "film formation"},
            definition="Formation of a passivating layer on anode due to electrolyte decomposition.")
        self._add_concept("dendrite_growth", ConceptType.PROCESS,
            synonyms={"lithium dendrite", "whisker growth"},
            definition="Formation of needle-like lithium deposits on anode, leading to short circuits.")
        self._add_concept("degradation", ConceptType.PROCESS,
            synonyms={"capacity fade", "impedance growth", "aging", "performance decay"},
            definition="Irreversible loss of capacity or power over cycling.")
        self._add_concept("calendering", ConceptType.PROCESS,
            synonyms={"roll pressing", "electrode compaction"},
            definition="Mechanical compression of electrode coating to increase density and adhesion.")
        self._add_concept("coating", ConceptType.PROCESS,
            synonyms={"slot die coating", "doctor blade coating", "spray coating", "dry coating"},
            definition="Deposition of electrode slurry onto current collector foil.")
        self._add_concept("prelithiation", ConceptType.PROCESS,
            synonyms={"lithium compensation", "pre-doping"},
            definition="Addition of extra lithium to compensate for irreversible losses during first cycle.")
        # ---- Parameters ----
        self._add_concept("electrode_thickness", ConceptType.PARAMETER,
            synonyms={"coating thickness", "foil thickness"},
            definition="Thickness of the electrode coating; affects energy density and rate capability.")
        self._add_concept("electrode_porosity", ConceptType.PARAMETER,
            synonyms={"pore fraction", "density", "void fraction"},
            definition="Fraction of void space in the electrode; affects electrolyte wettability and ion transport.")
        self._add_concept("loading", ConceptType.PARAMETER,
            synonyms={"mass loading", "areal loading", "mg/cm2"},
            definition="Amount of active material per unit area.")
        self._add_concept("temperature", ConceptType.PARAMETER,
            synonyms={"operating temperature", "synthesis temperature", "annealing temperature"},
            definition="Temperature during operation or synthesis.")
        self._add_concept("c_rate", ConceptType.PARAMETER,
            synonyms={"charge rate", "discharge rate", "current rate"},
            definition="Rate of charge/discharge expressed as multiple of nominal capacity.")
        # ---- Methods ----
        self._add_concept("eis", ConceptType.METHOD,
            synonyms={"electrochemical impedance spectroscopy", "impedance"},
            definition="Technique to measure resistance and capacitance of battery components.")
        self._add_concept("dQ_dV", ConceptType.METHOD,
            synonyms={"differential capacity", "incremental capacity"},
            definition="Derivative of charge with respect to voltage; used to study phase transitions.")
        self._add_concept("operando", ConceptType.METHOD,
            synonyms={"in-situ", "during operation"},
            definition="Characterization performed during battery operation, capturing dynamic behavior.")

        # ---- Build causal chains (direction: process/parameter -> property) ----
        self._add_relationship("cathode", RelationshipType.INFLUENCES, "energy_density", 0.8)
        self._add_relationship("anode", RelationshipType.INFLUENCES, "energy_density", 0.7)
        self._add_relationship("electrode_thickness", RelationshipType.INFLUENCES, "energy_density", 0.7)
        self._add_relationship("electrode_porosity", RelationshipType.INFLUENCES, "energy_density", -0.6)
        self._add_relationship("lithiation", RelationshipType.CAUSES, "capacity", 1.0)
        self._add_relationship("delithiation", RelationshipType.CAUSES, "capacity", 1.0)
        self._add_relationship("solid_electrolyte_interphase_formation", RelationshipType.CAUSES, "impedance", 0.8)
        self._add_relationship("solid_electrolyte_interphase_formation", RelationshipType.INFLUENCES, "cycle_life", 0.6)
        self._add_relationship("dendrite_growth", RelationshipType.CAUSES, "safety", -0.9)
        self._add_relationship("dendrite_growth", RelationshipType.CAUSES, "cycle_life", -0.8)
        self._add_relationship("degradation", RelationshipType.CAUSES, "capacity", -0.9)
        self._add_relationship("degradation", RelationshipType.CAUSES, "power_density", -0.8)
        self._add_relationship("temperature", RelationshipType.INFLUENCES, "safety", -0.7)
        self._add_relationship("temperature", RelationshipType.INFLUENCES, "cycle_life", -0.5)
        self._add_relationship("c_rate", RelationshipType.INFLUENCES, "rate_capability", 0.9)
        self._add_relationship("c_rate", RelationshipType.INFLUENCES, "capacity", -0.4)
        self._add_relationship("electrolyte", RelationshipType.INFLUENCES, "ion_transport", 0.8)
        self._add_relationship("electrolyte", RelationshipType.INFLUENCES, "safety", 0.6)
        self._add_relationship("calendering", RelationshipType.INFLUENCES, "electrode_porosity", -0.7)
        self._add_relationship("calendering", RelationshipType.INFLUENCES, "energy_density", 0.6)
        self._add_relationship("prelithiation", RelationshipType.CAUSES, "coulombic_efficiency", 0.8)
        self._add_relationship("prelithiation", RelationshipType.CAUSES, "cycle_life", 0.5)
        self._add_relationship("silicon", RelationshipType.INFLUENCES, "capacity", 0.9)
        self._add_relationship("silicon", RelationshipType.CAUSES, "degradation", 0.6)
        self._add_relationship("lithium_metal", RelationshipType.INFLUENCES, "energy_density", 0.9)
        self._add_relationship("lithium_metal", RelationshipType.CAUSES, "safety", -0.8)
        self._add_relationship("solid_electrolyte", RelationshipType.INFLUENCES, "safety", 0.8)
        self._add_relationship("solid_electrolyte", RelationshipType.INFLUENCES, "power_density", -0.3)
        self._add_relationship("interphase", RelationshipType.INFLUENCES, "cycle_life", 0.7)
        self._add_relationship("interphase", RelationshipType.INFLUENCES, "impedance", 0.6)

        self._build_synonym_index()

    def _add_concept(self, canonical_name: str, concept_type: ConceptType,
                     synonyms: Set[str] = None, hypernyms: Set[str] = None,
                     hyponyms: Set[str] = None, definition: str = "",
                     related_processes: Set[str] = None,
                     related_properties: Set[str] = None) -> None:
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
            definition=definition
        )
        self.concepts[canonical_name] = node

    def _add_relationship(self, source: str, rel_type: RelationshipType, target: str, confidence: float = 1.0):
        self.relationships.append(Relationship(source, target, rel_type, confidence))

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical.clear()
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        # Try stripping common suffixes
        for suffix in ['s', 'es', 'ed', 'ing']:
            if text_lower.endswith(suffix):
                stem = text_lower[:-len(suffix)]
                if stem in self.synonym_to_canonical:
                    return self.synonym_to_canonical[stem]
        return None

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].concept_type
        return ConceptType.GENERAL

    def get_definition(self, canonical_name: str) -> str:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def get_related_concepts(self, canonical_name: str, rel_type: Optional[RelationshipType] = None) -> List[Tuple[str, RelationshipType, float]]:
        results = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    results.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    results.append((rel.source, rel.rel_type, rel.confidence))
        return results

# ----------------------------------------------------------------------------
# 2. LiB PROBLEM DEFINITIONS (Analogous to Cu@Ag's CS_PROBLEM_DEFINITIONS)
# ----------------------------------------------------------------------------

class LIBProblem(Enum):
    ENERGY_DENSITY_MAXIMIZATION = "energy_density_maximization"
    CYCLE_LIFE_EXTENSION = "cycle_life_extension"
    FAST_CHARGING_CAPABILITY = "fast_charging_capability"
    SAFETY_THERMAL_RUNAWAY = "safety_thermal_runaway"
    COST_REDUCTION = "cost_reduction"
    SOLID_STATE_BATTERIES = "solid_state_batteries"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"

@dataclass
class LIBProblemDefinition:
    problem_id: LIBProblem
    title: str
    scientific_description: str
    root_cause: str
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]  # source, rel_type_str, target
    solution_directions: List[str]
    relevant_materials: List[str]
    relevant_phenomena: List[str]
    relevant_properties: List[str]
    example_queries: List[str]
    visualization_focus: List[str]

    def get_ontology_concepts(self) -> Set[str]:
        concepts = set(self.key_concepts + self.relevant_materials +
                       self.relevant_phenomena + self.relevant_properties)
        for src, _, tgt in self.key_relationships:
            concepts.add(src); concepts.add(tgt)
        return concepts

LIB_PROBLEM_DEFINITIONS: Dict[LIBProblem, LIBProblemDefinition] = {
    LIBProblem.ENERGY_DENSITY_MAXIMIZATION: LIBProblemDefinition(
        problem_id=LIBProblem.ENERGY_DENSITY_MAXIMIZATION,
        title="Maximizing Energy Density",
        scientific_description="Achieving high specific/volumetric energy while maintaining adequate power and safety.",
        root_cause="Trade-off between electrode thickness, active material fraction, and electrolyte volume.",
        key_concepts=["energy_density", "electrode_thickness", "loading", "porosity", "cathode", "anode"],
        key_relationships=[("electrode_thickness", "INFLUENCES", "energy_density"),
                           ("loading", "INFLUENCES", "energy_density"),
                           ("porosity", "INFLUENCES", "energy_density")],
        solution_directions=["Increase electrode thickness", "Reduce inactive materials", "Use high-capacity anode (Si, Li metal)"],
        relevant_materials=["cathode", "anode", "silicon", "lithium_metal", "graphite"],
        relevant_phenomena=["lithiation", "degradation"],
        relevant_properties=["energy_density", "capacity", "voltage"],
        example_queries=["How can we increase the energy density of Li-ion batteries?", 
                         "What is the optimal electrode thickness for high energy density?"],
        visualization_focus=["energy_density_plot", "electrode_thickness_series"]
    ),
    LIBProblem.CYCLE_LIFE_EXTENSION: LIBProblemDefinition(
        problem_id=LIBProblem.CYCLE_LIFE_EXTENSION,
        title="Extending Cycle Life",
        scientific_description="Reducing capacity fade and impedance growth over many cycles.",
        root_cause="Side reactions, electrode cracking, SEI growth, lithium loss.",
        key_concepts=["cycle_life", "degradation", "interphase", "coulombic_efficiency", "prelithiation"],
        key_relationships=[("degradation", "CAUSES", "cycle_life"),
                           ("interphase", "INFLUENCES", "cycle_life"),
                           ("prelithiation", "CAUSES", "cycle_life")],
        solution_directions=["Optimize electrolyte additives", "Apply prelithiation", "Improve SEI stability"],
        relevant_materials=["electrolyte", "interphase", "binder"],
        relevant_phenomena=["solid_electrolyte_interphase_formation", "degradation"],
        relevant_properties=["cycle_life", "coulombic_efficiency"],
        example_queries=["How can we improve the cycle life of lithium-ion batteries?",
                         "What causes capacity fade in NCM/graphite cells?"],
        visualization_focus=["cycle_life_plot", "capacity_retention"]
    ),
    LIBProblem.FAST_CHARGING_CAPABILITY: LIBProblemDefinition(
        problem_id=LIBProblem.FAST_CHARGING_CAPABILITY,
        title="Enabling Fast Charging",
        scientific_description="Achieving high charge rates without lithium plating or overheating.",
        root_cause="Slow Li+ diffusion in electrodes and electrolyte, high internal resistance.",
        key_concepts=["rate_capability", "c_rate", "impedance", "dendrite_growth", "temperature"],
        key_relationships=[("c_rate", "INFLUENCES", "rate_capability"),
                           ("impedance", "CONSTRAINS", "rate_capability"),
                           ("dendrite_growth", "CAUSES", "safety")],
        solution_directions=["Reduce electrode thickness", "Increase electrode porosity", "Use high-ionic-conductivity electrolyte"],
        relevant_materials=["electrolyte", "anode", "cathode"],
        relevant_phenomena=["lithiation", "dendrite_growth"],
        relevant_properties=["rate_capability", "impedance"],
        example_queries=["How can we enable fast charging of lithium-ion batteries?",
                         "What limits the rate capability of graphite anodes?"],
        visualization_focus=["rate_performance_plot", "impedance_spectra"]
    ),
    LIBProblem.SAFETY_THERMAL_RUNAWAY: LIBProblemDefinition(
        problem_id=LIBProblem.SAFETY_THERMAL_RUNAWAY,
        title="Preventing Thermal Runaway",
        scientific_description="Avoiding exothermic chain reactions that lead to fire/explosion.",
        root_cause="Internal short circuits, overcharge, mechanical damage.",
        key_concepts=["safety", "temperature", "dendrite_growth", "short_circuit", "thermal_management"],
        key_relationships=[("dendrite_growth", "CAUSES", "safety"),
                           ("temperature", "INFLUENCES", "safety"),
                           ("short_circuit", "CAUSES", "safety")],
        solution_directions=["Use solid-state electrolyte", "Add safety vents", "Implement BMS"],
        relevant_materials=["solid_electrolyte", "separator"],
        relevant_phenomena=["dendrite_growth", "degradation"],
        relevant_properties=["safety"],
        example_queries=["How can we improve the safety of Li-ion batteries?",
                         "What causes thermal runaway in lithium-ion cells?"],
        visualization_focus=["safety_plot", "thermal_runaway_path"]
    ),
    LIBProblem.COST_REDUCTION: LIBProblemDefinition(
        problem_id=LIBProblem.COST_REDUCTION,
        title="Reducing Cost",
        scientific_description="Lowering material and manufacturing costs without compromising performance.",
        root_cause="Expensive materials (Co, Ni, Li), complex manufacturing.",
        key_concepts=["cost", "cobalt", "nickel", "manufacturing", "calendering", "coating"],
        key_relationships=[("cobalt", "CONSTRAINS", "cost"),
                           ("manufacturing", "INFLUENCES", "cost")],
        solution_directions=["Reduce cobalt content", "Dry electrode processing", "Simplify cell design"],
        relevant_materials=["cathode", "electrolyte", "binder"],
        relevant_phenomena=[],
        relevant_properties=["cost"],
        example_queries=["How can we reduce the cost of Li-ion batteries?",
                         "What are the main cost drivers in battery production?"],
        visualization_focus=["cost_breakdown", "material_cost_plot"]
    ),
    LIBProblem.SOLID_STATE_BATTERIES: LIBProblemDefinition(
        problem_id=LIBProblem.SOLID_STATE_BATTERIES,
        title="Solid-State Battery Development",
        scientific_description="Replacing liquid electrolyte with solid to improve safety and enable Li metal anodes.",
        root_cause="Interfacial resistance, mechanical stability, manufacturing scalability.",
        key_concepts=["solid_electrolyte", "lithium_metal", "interphase", "safety", "energy_density"],
        key_relationships=[("solid_electrolyte", "INFLUENCES", "safety"),
                           ("lithium_metal", "INFLUENCES", "energy_density")],
        solution_directions=["Develop high-ionic-conductivity solid electrolyte",
                             "Improve interface engineering",
                             "Scale up manufacturing"],
        relevant_materials=["solid_electrolyte", "lithium_metal", "cathode"],
        relevant_phenomena=["interphase_formation", "degradation"],
        relevant_properties=["safety", "energy_density", "power_density"],
        example_queries=["What are the challenges in solid-state batteries?",
                         "How does solid electrolyte affect battery performance?"],
        visualization_focus=["solid_electrolyte_plot", "interface_stability"]
    ),
    LIBProblem.GENERAL: LIBProblemDefinition(
        problem_id=LIBProblem.GENERAL,
        title="General Li-Ion Battery Inquiry",
        scientific_description="General question about Li-ion batteries.",
        root_cause="N/A",
        key_concepts=["lithium_ion_battery"],
        key_relationships=[],
        solution_directions=[],
        relevant_materials=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=["What are lithium-ion batteries?"],
        visualization_focus=["general_overview"]
    ),
    LIBProblem.MULTI_PROBLEM: LIBProblemDefinition(
        problem_id=LIBProblem.MULTI_PROBLEM,
        title="Multi-Problem LiB Inquiry",
        scientific_description="Inquiry spanning multiple core problems.",
        root_cause="N/A",
        key_concepts=[],
        key_relationships=[],
        solution_directions=[],
        relevant_materials=[],
        relevant_phenomena=[],
        relevant_properties=[],
        example_queries=[],
        visualization_focus=["multi_problem_comparison"]
    )
}

# ----------------------------------------------------------------------------
# 3. LLM QUERY ANALYZERS (identical to Cu@Ag but with LiB problem mapping)
# ----------------------------------------------------------------------------

@dataclass
class ConceptPriority:
    concept_name: str
    concept_type: str
    composite_score: float
    direct_score: float
    problem_affinity_score: float
    causal_path_score: float
    is_explicitly_mentioned: bool
    is_inferred: bool
    inference_reason: str = ""
    ppr_score: float = 0.0
    qc_pmi: float = 0.0
    semantic_resonance: float = 0.0
    cde: float = 0.0
    causal_proximity: float = 0.0

    def to_dict(self) -> Dict:
        return {**self.__dict__, "score": round(self.composite_score, 3)}

@dataclass
class QueryAnalysisResult:
    original_query: str
    normalized_query: str
    primary_problem: LIBProblem
    secondary_problems: List[LIBProblem]
    problem_confidences: Dict[str, float]
    explicitly_mentioned: List[str]
    inferred_concepts: List[str]
    all_relevant_concepts: List[str]
    concept_priorities: Dict[str, ConceptPriority] = field(default_factory=dict)
    query_type: str = "general"
    emphasis_direction: str = "cause"
    comparison_pairs: List[Tuple[str, str]] = field(default_factory=list)
    subgraph_depth: int = 2
    priority_threshold: float = 0.3
    focus_nodes: List[str] = field(default_factory=list)
    bridge_nodes: List[str] = field(default_factory=list)
    suggested_layout: str = "force"
    highlight_paths: List[List[str]] = field(default_factory=list)
    visualization_focus: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def get_top_concepts(self, n: int = 10) -> List[ConceptPriority]:
        return sorted(self.concept_priorities.values(), key=lambda x: x.composite_score, reverse=True)[:n]

    def get_concepts_above_threshold(self, threshold: float = None) -> List[str]:
        thresh = threshold or self.priority_threshold
        return [name for name, cp in self.concept_priorities.items() if cp.composite_score >= thresh]

# ---- LLM Analyzer base ----
from abc import ABC, abstractmethod
class LLMQueryAnalyzer(ABC):
    @abstractmethod
    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult: pass
    @abstractmethod
    def is_available(self) -> bool: pass

# ---- Fallback analyzer (rule-based) ----
class FallbackAnalyzer(LLMQueryAnalyzer):
    PROBLEM_KEYWORDS = {
        LIBProblem.ENERGY_DENSITY_MAXIMIZATION: {"energy density", "wh/kg", "wh/l", "specific energy", "volumetric energy", "capacity", "loading"},
        LIBProblem.CYCLE_LIFE_EXTENSION: {"cycle life", "calendar life", "capacity retention", "fade", "degradation", "aging"},
        LIBProblem.FAST_CHARGING_CAPABILITY: {"fast charge", "quick charge", "c-rate", "rate capability", "power", "impedance"},
        LIBProblem.SAFETY_THERMAL_RUNAWAY: {"safety", "thermal runaway", "fire", "explosion", "dendrite", "short circuit", "venting"},
        LIBProblem.COST_REDUCTION: {"cost", "cobalt", "nickel", "manufacturing", "expensive", "price"},
        LIBProblem.SOLID_STATE_BATTERIES: {"solid state", "solid electrolyte", "sulfide", "oxide", "garnet", "lithium metal"},
    }

    def is_available(self) -> bool:
        return True

    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        q = query.lower().strip()
        problem_scores = {}
        for p, keywords in self.PROBLEM_KEYWORDS.items():
            problem_scores[p] = sum(1 for kw in keywords if kw in q)
        primary = max(problem_scores, key=problem_scores.get) if sum(problem_scores.values()) > 0 else LIBProblem.GENERAL
        secondary = [p for p, s in sorted(problem_scores.items(), key=lambda x: -x[1]) if s > 0 and p != primary][:2]

        explicitly_mentioned = []
        for canonical, node in ontology.concepts.items():
            if canonical.replace("_", " ") in q or any(syn.replace("_", " ") in q for syn in node.synonyms):
                explicitly_mentioned.append(canonical)

        inferred = []
        if primary != LIBProblem.GENERAL:
            pdef = LIB_PROBLEM_DEFINITIONS[primary]
            for concept in pdef.get_ontology_concepts():
                if concept not in explicitly_mentioned and concept in ontology.concepts:
                    inferred.append(concept)

        all_relevant = list(dict.fromkeys(explicitly_mentioned + inferred))
        priorities = {}
        pdef = LIB_PROBLEM_DEFINITIONS.get(primary, LIB_PROBLEM_DEFINITIONS[LIBProblem.GENERAL])
        problem_concept_set = pdef.get_ontology_concepts()

        for concept in all_relevant:
            is_explicit = concept in explicitly_mentioned
            priorities[concept] = ConceptPriority(
                concept_name=concept,
                concept_type=ontology.get_concept_type(concept).value,
                composite_score=(1.0 if is_explicit else 0.6) * 0.5 + (1.0 if concept in problem_concept_set else 0.4) * 0.5,
                direct_score=1.0 if is_explicit else 0.6,
                problem_affinity_score=1.0 if concept in problem_concept_set else 0.4,
                causal_path_score=0.5,
                is_explicitly_mentioned=is_explicit,
                is_inferred=not is_explicit,
                inference_reason="problem_affinity" if not is_explicit else "explicit_mention"
            )

        query_type = "general"
        if any(w in q for w in ["compare", "vs", "versus", "difference"]):
            query_type = "comparison"
        elif any(w in q for w in ["why", "cause", "reason", "lead to"]):
            query_type = "causal"
        elif any(w in q for w in ["how", "improve", "enhance", "optimize", "strategy"]):
            query_type = "solution"

        highlight_paths = [[src, tgt] for src, rel, tgt in pdef.key_relationships if src in ontology.concepts and tgt in ontology.concepts]
        total = max(sum(problem_scores.values()), 1)

        return QueryAnalysisResult(
            original_query=query,
            normalized_query=q,
            primary_problem=primary,
            secondary_problems=secondary,
            problem_confidences={p.value: s/total for p, s in problem_scores.items()},
            explicitly_mentioned=explicitly_mentioned,
            inferred_concepts=inferred,
            all_relevant_concepts=all_relevant,
            concept_priorities=priorities,
            query_type=query_type,
            emphasis_direction="cause" if query_type == "causal" else "neutral",
            subgraph_depth=2,
            priority_threshold=0.3,
            focus_nodes=explicitly_mentioned[:5],
            bridge_nodes=inferred[:3],
            suggested_layout="force" if query_type != "comparison" else "bisected",
            highlight_paths=highlight_paths,
            visualization_focus=pdef.visualization_focus,
            reasoning_chain=[f"Query normalized: '{q}'", f"Primary problem: {primary.value}"],
            confidence=min(sum(problem_scores.values()) / 3.0, 1.0)
        )

# ---- OpenAI analyzer ----
class OpenAIQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self._client = None
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _get_client(self):
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                st.warning("openai package not installed. Run: pip install openai")
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key) and self._get_client() is not None

    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        client = self._get_client()
        if client is None:
            return FallbackAnalyzer().analyze_query(query, ontology)

        concept_list = list(ontology.concepts.keys())[:50]
        problem_enum_names = [p.value for p in LIBProblem]
        system_prompt = f"""You are an expert in lithium-ion battery materials and electrochemistry. Analyze the user's query and return ONLY valid JSON with:
1. "primary_problem": One of {problem_enum_names}
2. "explicitly_mentioned": List of canonical concept names from the query (use snake_case)
3. "inferred_concepts": List of additional relevant concepts the query implies
4. "query_type": One of: causal, comparison, solution, definition, general
5. "highlight_paths": List of [source, target] concept pairs to highlight
6. "reasoning_chain": List of strings explaining analysis steps
7. "new_concepts": List of objects with "name" (snake_case), "type" (material/property/phenomenon/process/method/parameter), "definition", "synonyms" (list)
8. "new_relationships": List of [source, relationship_type, target, confidence] for NEW relationships between EXISTING concepts."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze: '{query}'. Available concepts: {', '.join(concept_list)}"}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            self._pending_new_concepts = parsed.get("new_concepts", [])
            self._pending_new_relationships = parsed.get("new_relationships", [])

            problem_map = {p.value: p for p in LIBProblem}
            primary = problem_map.get(parsed.get("primary_problem", "general"), LIBProblem.GENERAL)
            explicitly_mentioned = [c for c in parsed.get("explicitly_mentioned", []) if c in ontology.concepts]
            inferred = [c for c in parsed.get("inferred_concepts", []) if c in ontology.concepts and c not in explicitly_mentioned]

            priorities = {c: ConceptPriority(
                c, ontology.get_concept_type(c).value,
                0.9 if c in explicitly_mentioned else 0.6,
                1.0 if c in explicitly_mentioned else 0.5,
                0.8, 0.5,
                c in explicitly_mentioned,
                c not in explicitly_mentioned,
                "llm_inferred"
            ) for c in list(dict.fromkeys(explicitly_mentioned + inferred))}

            return QueryAnalysisResult(
                original_query=query,
                normalized_query=query.lower().strip(),
                primary_problem=primary,
                secondary_problems=[],
                problem_confidences={},
                explicitly_mentioned=explicitly_mentioned,
                inferred_concepts=inferred,
                all_relevant_concepts=list(dict.fromkeys(explicitly_mentioned + inferred)),
                concept_priorities=priorities,
                query_type=parsed.get("query_type", "general"),
                emphasis_direction="cause",
                subgraph_depth=2,
                priority_threshold=0.3,
                focus_nodes=explicitly_mentioned[:5],
                bridge_nodes=inferred[:3],
                suggested_layout="bisected" if parsed.get("query_type") == "comparison" else "force",
                highlight_paths=[[p[0], p[1]] for p in parsed.get("highlight_paths", []) if len(p) >= 2],
                visualization_focus=LIB_PROBLEM_DEFINITIONS[primary].visualization_focus,
                reasoning_chain=parsed.get("reasoning_chain", ["LLM analysis completed"]),
                confidence=0.85
            )
        except Exception as e:
            st.warning(f"OpenAI analysis failed ({e}), falling back to rule-based.")
            return FallbackAnalyzer().analyze_query(query, ontology)

# ---- Local LLM (Ollama / HuggingFace) ----
LOCAL_LLM_REGISTRY: Dict[str, Optional[str]] = {
    "Fallback (Rule-based, no LLM)": None,
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
}

class LocalLLMQueryAnalyzer(LLMQueryAnalyzer):
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self._pipeline = None
        self._loaded = False
        self._is_ollama = model_name.startswith("ollama:")
        self._ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._pending_new_concepts = []
        self._pending_new_relationships = []

    def _load_model(self):
        if self._loaded:
            return
        if self._is_ollama:
            try:
                response = requests.get(f"{self._ollama_url}/api/tags")
                if response.status_code == 200:
                    self._loaded = True
                    st.success(f"✅ Connected to Ollama server at {self._ollama_url}")
                else:
                    st.warning(f"⚠️ Could not connect to Ollama (Status {response.status_code}). Is `ollama serve` running?")
                    self._loaded = False
            except Exception as e:
                st.warning(f"⚠️ Failed to connect to Ollama: {e}. Please start Ollama (`ollama serve`).")
                self._loaded = False
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            st.info(f"⏳ Loading local model: `{self.model_name}`… (first run may take 1–2 min)")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            load_kwargs = {}
            if torch.cuda.is_available():
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = None
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            self._loaded = True
            st.success(f"✅ Model `{self.model_name}` loaded!")
        except Exception as e:
            st.warning(f"⚠️ Failed to load local model `{self.model_name}`: {e}")
            self._loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_available(self) -> bool:
        self._load_model()
        return self._loaded

    def analyze_query(self, query: str, ontology: DomainOntology) -> QueryAnalysisResult:
        if not self.is_available():
            return FallbackAnalyzer().analyze_query(query, ontology)

        problem_enum_names = [p.value for p in LIBProblem]
        prompt = f"""Analyze this Li-ion battery query: '{query}'. Return ONLY valid JSON with:
primary_problem (one of {problem_enum_names}), explicitly_mentioned (list of snake_case concepts),
inferred_concepts (list), query_type, highlight_paths (list of [src,tgt]), reasoning_chain (list)."""
        try:
            if self._is_ollama:
                ollama_model_name = self.model_name.split(":", 1)[1]
                payload = {
                    "model": ollama_model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
                response = requests.post(f"{self._ollama_url}/api/generate", json=payload)
                response.raise_for_status()
                result = response.json().get("response", "")
            else:
                result = self._pipeline(prompt)[0]["generated_text"]
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                fake_openai = OpenAIQueryAnalyzer()
                fake_openai._pending_new_concepts = parsed.get("new_concepts", [])
                fake_openai._pending_new_relationships = parsed.get("new_relationships", [])
                return fake_openai.analyze_query(query, ontology)
        except Exception as e:
            st.warning(f"LLM parsing failed: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return FallbackAnalyzer().analyze_query(query, ontology)

    def unload_model(self) -> None:
        if self._is_ollama:
            try:
                ollama_model_name = self.model_name.split(":", 1)[1]
                requests.post(f"{self._ollama_url}/api/generate", json={"model": ollama_model_name, "keep_alive": 0}, timeout=5)
            except Exception:
                pass
            self._loaded = False
            return
        if self._pipeline is not None:
            if hasattr(self._pipeline, 'tokenizer'): del self._pipeline.tokenizer
            if hasattr(self._pipeline, 'model'): del self._pipeline.model
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class LLMQueryAnalyzerFactory:
    def __init__(self):
        self._openai_cache = None
        self._local_cache = {}
        self._fallback = FallbackAnalyzer()

    def get_analyzer(self, mode: str = "auto", api_key: str = None, local_model: str = None) -> LLMQueryAnalyzer:
        if mode == "openai":
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            return self._openai_cache
        elif mode == "local":
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            return self._local_cache[model]
        elif mode == "fallback":
            return self._fallback
        else:  # auto
            if self._openai_cache is None:
                self._openai_cache = OpenAIQueryAnalyzer(api_key=api_key)
            if self._openai_cache.is_available():
                return self._openai_cache
            model = local_model
            if model is None:
                return self._fallback
            if model not in self._local_cache:
                self._local_cache[model] = LocalLLMQueryAnalyzer(model)
            if self._local_cache[model].is_available():
                return self._local_cache[model]
            return self._fallback

# ----------------------------------------------------------------------------
# 4. DYNAMIC ONTOLOGY EXPANDER (with mutation tracking)
# ----------------------------------------------------------------------------
class DynamicOntologyExpander:
    REL_STR_TO_ENUM = {r.value: r for r in RelationshipType}
    for _k, _v in list(REL_STR_TO_ENUM.items()):
        REL_STR_TO_ENUM[_k.upper()] = _v
    TYPE_STR_TO_ENUM = {t.value: t for t in ConceptType}

    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology
        self.mutation_log: List[Dict[str, Any]] = []
        self.session_concepts_added: Set[str] = set()
        self.session_relationships_added: List[Tuple[str, str, RelationshipType, float]] = []
        self.query_bridge_concepts: Dict[str, str] = {}
        self.priority_overrides: Dict[str, float] = {}
        self._base_concept_count = len(ontology.concepts)
        self._base_rel_count = len(ontology.relationships)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "base_concepts": self._base_concept_count,
            "base_relationships": self._base_rel_count,
            "concepts_added": len(self.session_concepts_added),
            "relationships_added": len(self.session_relationships_added),
            "bridge_concepts": len(self.query_bridge_concepts),
            "total_mutations": len(self.mutation_log)
        }

    def apply_query_analysis(self, analysis: QueryAnalysisResult, analyzer: LLMQueryAnalyzer = None) -> Dict[str, Any]:
        changes = {"concepts_added": [], "relationships_added": [], "bridges_created": []}
        for concept_name, priority in analysis.concept_priorities.items():
            if concept_name in self.ontology.concepts:
                self.priority_overrides[concept_name] = priority.composite_score

        new_concepts_raw = getattr(analyzer, '_pending_new_concepts', []) if hasattr(analyzer, '_pending_new_concepts') else []
        new_rels_raw = getattr(analyzer, '_pending_new_relationships', []) if hasattr(analyzer, '_pending_new_relationships') else []

        for concept_data in new_concepts_raw:
            result = self._add_concept_from_llm(concept_data, analysis.original_query)
            if result:
                changes["concepts_added"].append(result)
        for rel_data in new_rels_raw:
            result = self._add_relationship_from_llm(rel_data, analysis.original_query)
            if result:
                changes["relationships_added"].append(result)

        for concept in analysis.inferred_concepts:
            if concept not in self.ontology.concepts:
                bridge_result = self._create_bridge_concept(concept, analysis.original_query, analysis.primary_problem)
                if bridge_result:
                    changes["bridges_created"].append(bridge_result)

        self.ontology._build_synonym_index()
        return changes

    def _add_concept_from_llm(self, concept_data: Dict, source_query: str) -> Optional[Dict]:
        name = concept_data.get("name", "").strip().lower().replace(" ", "_")
        if not name or name in self.ontology.concepts or name in self.session_concepts_added:
            return None
        concept_type = self.TYPE_STR_TO_ENUM.get(concept_data.get("type", "general"), ConceptType.GENERAL)
        synonyms = set(s.lower().strip() for s in concept_data.get("synonyms", []) if isinstance(s, str))
        definition = concept_data.get("definition", f"LLM-inferred concept from query: {source_query}")
        self.ontology._add_concept(name, concept_type, synonyms=synonyms, definition=definition)
        self.ontology.synonym_to_canonical[name.lower()] = name
        for syn in synonyms:
            self.ontology.synonym_to_canonical[syn] = name
        self.session_concepts_added.add(name)
        for rel_tuple in concept_data.get("relate_to", []):
            if len(rel_tuple) >= 2:
                target, rel_type_str = rel_tuple[0], rel_tuple[1] if len(rel_tuple) > 1 else "influences"
                conf = float(rel_tuple[2]) if len(rel_tuple) > 2 else 0.7
                rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
                if target in self.ontology.concepts:
                    self.ontology.relationships.append(Relationship(name, target, rel_enum, conf))
                    self.session_relationships_added.append((name, target, rel_enum, conf))
        self.mutation_log.append({"type": "add_concept", "concept": name, "concept_type": concept_type.value, "source_query": source_query})
        return {"name": name, "type": concept_type.value, "synonyms": list(synonyms)}

    def _add_relationship_from_llm(self, rel_data: List, source_query: str) -> Optional[Dict]:
        if len(rel_data) < 3:
            return None
        source, rel_type_str, target = str(rel_data[0]).strip().lower().replace(" ", "_"), str(rel_data[1]).upper(), str(rel_data[2]).strip().lower().replace(" ", "_")
        confidence = float(rel_data[3]) if len(rel_data) > 3 else 0.7
        if source not in self.ontology.concepts or target not in self.ontology.concepts:
            return None
        rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
        self.ontology.relationships.append(Relationship(source, target, rel_enum, confidence))
        self.session_relationships_added.append((source, target, rel_enum, confidence))
        self.mutation_log.append({"type": "add_relationship", "source": source, "target": target, "rel_type": rel_enum.value, "source_query": source_query})
        return {"source": source, "target": target, "rel_type": rel_enum.value, "confidence": confidence}

    def _create_bridge_concept(self, missing_concept: str, source_query: str, problem: LIBProblem) -> Optional[Dict]:
        bridge_name = f"query_bridge_{missing_concept.replace(' ', '_').lower()}"
        if bridge_name in self.ontology.concepts:
            return None
        pdef = LIB_PROBLEM_DEFINITIONS.get(problem, LIB_PROBLEM_DEFINITIONS[LIBProblem.GENERAL])
        self.ontology._add_concept(bridge_name, ConceptType.GENERAL, synonyms={missing_concept.lower()}, definition=f"Query-inferred bridge: '{missing_concept}'")
        self.ontology.synonym_to_canonical[bridge_name] = bridge_name
        self.ontology.synonym_to_canonical[missing_concept.lower()] = bridge_name
        connected = []
        for key_concept in pdef.key_concepts[:3]:
            if key_concept in self.ontology.concepts:
                self.ontology.relationships.append(Relationship(bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                self.session_relationships_added.append((bridge_name, key_concept, RelationshipType.BRIDGE, 0.5))
                connected.append(key_concept)
        self.session_concepts_added.add(bridge_name)
        self.query_bridge_concepts[bridge_name] = source_query
        self.mutation_log.append({"type": "create_bridge", "bridge_name": bridge_name, "original_term": missing_concept, "connected_to": connected})
        return {"bridge": bridge_name, "for": missing_concept, "connected_to": connected}

    def undo_last_mutation(self) -> Optional[Dict]:
        if not self.mutation_log:
            return None
        mutation = self.mutation_log.pop()
        if mutation["type"] == "add_concept":
            name = mutation["concept"]
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
                self.session_concepts_added.discard(name)
                self.ontology.relationships = [r for r in self.ontology.relationships if r.source != name and r.target != name]
        elif mutation["type"] == "add_relationship":
            self.ontology.relationships = [r for r in self.ontology.relationships if not (r.source == mutation["source"] and r.target == mutation["target"] and r.rel_type.value == mutation["rel_type"])]
        elif mutation["type"] == "create_bridge":
            bridge_name = mutation["bridge_name"]
            if bridge_name in self.ontology.concepts:
                del self.ontology.concepts[bridge_name]
                self.session_concepts_added.discard(bridge_name)
                self.query_bridge_concepts.pop(bridge_name, None)
        self.ontology._build_synonym_index()
        return mutation

    def reset_to_base(self) -> Dict[str, int]:
        for name in list(self.session_concepts_added):
            if name in self.ontology.concepts:
                del self.ontology.concepts[name]
        self.ontology.relationships = self.ontology.relationships[:self._base_rel_count]
        self.session_concepts_added.clear()
        self.session_relationships_added.clear()
        self.query_bridge_concepts.clear()
        self.priority_overrides.clear()
        self.mutation_log.clear()
        self.ontology._build_synonym_index()
        return {"concepts_removed": len(self.session_concepts_added), "relationships_removed": len(self.ontology.relationships) - self._base_rel_count}

# ----------------------------------------------------------------------------
# 5. PRIORITY-GUIDED SUBGRAPH EXTRACTOR & VISUALIZER
# ----------------------------------------------------------------------------
class PriorityGuidedSubgraphExtractor:
    def __init__(self, full_graph: nx.Graph, ontology: DomainOntology, expander: DynamicOntologyExpander):
        self.full_graph = full_graph
        self.ontology = ontology
        self.expander = expander

    def extract(self, analysis: QueryAnalysisResult, query_embedding: np.ndarray = None) -> nx.Graph:
        raw_seed_nodes = set(analysis.focus_nodes + analysis.get_concepts_above_threshold())
        seed_nodes = {n for n in raw_seed_nodes if n in self.full_graph}
        if not seed_nodes:
            seed_nodes = {n for n, d in self.full_graph.nodes(data=True) if d.get("priority_score", 0) >= 0.3}

        personalization = {n: 1.0 if n in seed_nodes else 0.0 for n in self.full_graph.nodes()}
        try:
            ppr_scores = nx.pagerank(self.full_graph, personalization=personalization, alpha=0.85)
        except Exception:
            ppr_scores = {n: 1.0/len(self.full_graph) for n in self.full_graph.nodes()}

        for node in self.full_graph.nodes():
            ppr = ppr_scores.get(node, 0.0)
            sr = self._compute_semantic_resonance(node, query_embedding) if query_embedding is not None else 0.5
            combined = 0.6 * ppr + 0.4 * sr
            self.full_graph.nodes[node]["priority_score"] = combined
            self.full_graph.nodes[node]["ppr_score"] = ppr
            self.full_graph.nodes[node]["semantic_resonance"] = sr
            if node in analysis.concept_priorities:
                cp = analysis.concept_priorities[node]
                self.full_graph.nodes[node]["is_explicit"] = cp.is_explicitly_mentioned
                self.full_graph.nodes[node]["is_inferred"] = cp.is_inferred
            elif node in self.expander.session_concepts_added:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = True
                self.full_graph.nodes[node]["is_llm_added"] = True
            else:
                self.full_graph.nodes[node]["is_explicit"] = False
                self.full_graph.nodes[node]["is_inferred"] = False

        threshold = 0.1
        selected_nodes = {n for n, d in self.full_graph.nodes(data=True) if d.get("priority_score", 0) >= threshold}
        selected_nodes.update(seed_nodes)
        for node in list(selected_nodes):
            for neighbor in self.full_graph.neighbors(node):
                if self.full_graph.degree(neighbor) > 2:
                    selected_nodes.add(neighbor)

        subgraph = self.full_graph.subgraph(selected_nodes).copy()
        return subgraph

    def _compute_semantic_resonance(self, concept: str, query_emb: np.ndarray) -> float:
        embed_model = st.session_state.get('embed_model')
        if embed_model is None:
            return 0.5
        try:
            concept_emb = embed_model.encode(concept, convert_to_numpy=True)
            sim = np.dot(query_emb, concept_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(concept_emb) + 1e-8)
            return float(np.clip(sim, 0, 1))
        except Exception:
            return 0.5

class QueryDrivenVisualizer:
    def __init__(self, ontology: DomainOntology):
        self.ontology = ontology
        self.type_colors = {
            "material": "#E91E63", "property": "#3F51B5", "phenomenon": "#00BCD4",
            "method": "#4CAF50", "parameter": "#FF9800", "process": "#9C27B0",
            "model": "#607D8B", "general": "#795548"
        }

    def render_pyvis(self, subgraph: nx.Graph, analysis: QueryAnalysisResult,
                     height: str = "700px", physics_enabled: bool = True,
                     gravity: float = -800, central_gravity: float = 0.1,
                     spring_length: float = 120, spring_strength: float = 0.02,
                     damping: float = 0.95) -> str:
        net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
        if physics_enabled:
            net.barnes_hut(
                gravity=gravity,
                central_gravity=central_gravity,
                spring_length=spring_length,
                spring_strength=spring_strength,
                damping=damping,
                overlap=0.1
            )
        else:
            net.set_options('{"physics": {"enabled": false}, "interaction": {"hover": true, "dragNodes": true, "dragView": true, "zoomView": true}}')
        for node, attrs in subgraph.nodes(data=True):
            concept_type = attrs.get("concept_type", "general")
            priority = attrs.get("priority_score", 0.2)
            is_explicit = attrs.get("is_explicit", False)
            is_llm_added = attrs.get("is_llm_added", False)
            size = 15 + priority * 35
            color = self.type_colors.get(concept_type, "#795548")
            if is_explicit:
                border_width, border_color, shape = 4, "#FF0000", "dot"
            elif is_llm_added:
                border_width, border_color, shape = 3, "#00FF00", "diamond"
            else:
                border_width, border_color, shape = 1, "#666666", "dot"
            title = f"<b>{node}</b><br>Type: {concept_type}<br>Priority: {priority:.2f}"
            if is_llm_added:
                title += "<br>⚠️ LLM-inferred concept"
            defn = attrs.get("definition", "")
            if defn:
                title += f"<br><i>{defn[:150]}...</i>"
            net.add_node(
                node,
                label=node.replace("_", " ").title(),
                size=size,
                color=color,
                border_width=border_width,
                border_color=border_color,
                shape=shape,
                title=title,
                font={"size": 10 + priority * 6}
            )
        for u, v, attrs in subgraph.edges(data=True):
            color = attrs.get("color", "#888888")
            width = attrs.get("width", 1.0)
            highlighted = any(len(p) >= 2 and ((p[0] == u and p[1] == v) or (p[1] == u and p[0] == v)) for p in analysis.highlight_paths)
            if highlighted:
                color, width = "#FF0000", max(width, 4.0)
            net.add_edge(u, v, color=color, width=width,
                         dashes=attrs.get("style") == "dashed" or attrs.get("inferred", False),
                         title=f"{u} → {v}<br>Type: {attrs.get('edge_type','unknown')}",
                         arrows="to")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            return Path(f.name).read_text(encoding='utf-8')

# ----------------------------------------------------------------------------
# 6. GRAPHRAG ANSWER GENERATOR
# ----------------------------------------------------------------------------
class GraphRAGAnswerGenerator:
    def __init__(self, analyzer: LLMQueryAnalyzer):
        self.analyzer = analyzer

    def generate_ground_response(self, query: str, analysis: QueryAnalysisResult,
                                 subgraph: nx.Graph,
                                 concept_abstract_map: Dict[str, List[int]],
                                 all_texts: Union[List[str], Dict[int, str]],
                                 max_docs_per_concept: int = 2) -> str:
        top_nodes = sorted(subgraph.nodes(data=True), key=lambda x: x[1].get("priority_score", 0.0), reverse=True)[:5]
        evidence_snippets = []
        for node, attrs in top_nodes:
            doc_indices = concept_abstract_map.get(node, [])[:max_docs_per_concept]
            for idx in doc_indices:
                if isinstance(all_texts, dict):
                    text = all_texts.get(idx, "")
                else:
                    text = all_texts[idx] if 0 <= idx < len(all_texts) else ""
                if text:
                    clean_text = re.sub(r'\s+', ' ', text).strip()[:400]
                    evidence_snippets.append(f"- **{node}**: {clean_text}...")
        nl = "\n"
        prompt = f"You are an expert in Li-ion battery materials and electrochemistry. Answer the user's query based *strictly* on the provided graph context and evidence snippets.\n"
        prompt += f"User Query: {repr(query)}\n"
        prompt += f"Identified Core Problem: {analysis.primary_problem.value.replace('_', ' ').title()}\n"
        prompt += f"Key Graph Concepts: {', '.join([n for n, _ in top_nodes])}\n"
        prompt += "Evidence Snippets from Literature:\n"
        if evidence_snippets:
            prompt += nl.join(evidence_snippets) + nl
        else:
            prompt += "No direct text snippets found. Rely on your general knowledge of Li-ion batteries but note the lack of specific retrieved context.\n"
        prompt += "Instructions:\n"
        prompt += "1. Provide a direct, scientifically accurate answer (2-3 paragraphs).\n"
        prompt += "2. Explicitly mention how the key concepts interact (e.g., causal chains like 'electrode thickness influences energy density').\n"
        prompt += "3. If the retrieved evidence is insufficient, state what specific data is missing."

        if isinstance(self.analyzer, OpenAIQueryAnalyzer) and self.analyzer.is_available():
            return self._call_llm_for_answer(prompt, self.analyzer, query, analysis, top_nodes, evidence_snippets)
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _call_llm_for_answer(self, prompt: str, analyzer: LLMQueryAnalyzer,
                             query: str, analysis: QueryAnalysisResult,
                             top_nodes, evidence_snippets) -> str:
        client = analyzer._get_client()
        if client:
            try:
                response = client.chat.completions.create(
                    model=analyzer.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800
                )
                return response.choices[0].message.content
            except Exception as e:
                fallback = self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)
                return f"⚠️ LLM API Error: {e}\n\n{fallback}"
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _generate_fallback_answer(self, query: str, analysis: Optional[QueryAnalysisResult],
                                  top_nodes, snippets: List[str]) -> str:
        nl = "\n"
        fallback_text = f"### Analysis of: '{query}'\n\n"
        if analysis is not None:
            primary = getattr(analysis, 'primary_problem', None)
            fallback_text += f"**Core Problem Identified:** {primary.value.replace('_', ' ').title() if primary else 'Unknown'}\n\n"
        else:
            fallback_text += "**Core Problem Identified:** (analysis unavailable)\n\n"
        fallback_text += "**Key Concepts in Focus:**\n"
        fallback_text += nl.join([f"- **{node}** ({attrs.get('concept_type', 'general')}): Priority Score {attrs.get('priority_score', 0):.2f}" for node, attrs in top_nodes])
        if snippets:
            fallback_text += nl + "**Retrieved Evidence Context:**\n" + nl.join(snippets[:3]) + nl
        else:
            fallback_text += nl + "*Note: No direct text snippets were linked to these concepts in the current dataset.*\n"
        fallback_text += nl + "**System Reasoning Chain:**\n"
        if analysis is not None:
            fallback_text += nl.join(["- " + step for step in analysis.reasoning_chain])
        else:
            fallback_text += "- No reasoning chain available (analysis was None).\n"
        return fallback_text

# ----------------------------------------------------------------------------
# 7. QUERY SESSION MANAGER
# ----------------------------------------------------------------------------
class QuerySessionManager:
    SESSION_KEY = "lib_query_session"
    @classmethod
    def init_session(cls) -> Dict[str, Any]:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {
                "query_history": [],
                "analysis_history": [],
                "mutation_history": [],
                "analyzer_mode": "auto",
                "total_concepts_added": 0,
                "total_relationships_added": 0
            }
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def record_query(cls, query: str, analysis: QueryAnalysisResult, mutations: Dict[str, Any]) -> None:
        session = cls.init_session()
        session["query_history"].append(query)
        session["analysis_history"].append({
            "query": query,
            "primary_problem": analysis.primary_problem.value,
            "query_type": analysis.query_type,
            "concepts_found": len(analysis.all_relevant_concepts),
            "explicit": len(analysis.explicitly_mentioned),
            "inferred": len(analysis.inferred_concepts),
            "confidence": analysis.confidence,
            "timestamp": datetime.now().isoformat()
        })
        session["mutation_history"].append({
            "query": query,
            "concepts_added": len(mutations.get("concepts_added", [])),
            "relationships_added": len(mutations.get("relationships_added", [])),
            "bridges_created": len(mutations.get("bridges_created", [])),
            "timestamp": datetime.now().isoformat()
        })
        session["total_concepts_added"] += len(mutations.get("concepts_added", []))
        session["total_relationships_added"] += len(mutations.get("relationships_added", []))

    @classmethod
    def get_session(cls) -> Dict[str, Any]:
        return cls.init_session()

    @classmethod
    def clear_session(cls) -> None:
        if cls.SESSION_KEY in st.session_state:
            del st.session_state[cls.SESSION_KEY]

# ----------------------------------------------------------------------------
# 8. UI RENDERERS FOR LLM PANEL (sidebar + main tab)
# ----------------------------------------------------------------------------
def render_llm_query_panel(ontology: DomainOntology, expander: DynamicOntologyExpander,
                           full_graph: nx.Graph) -> Optional[QueryAnalysisResult]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 LLM-Guided Query")
    st.sidebar.caption("Ask a question to dynamically expand the ontology and focus the graph")

    session = QuerySessionManager.get_session()
    mode = st.sidebar.selectbox("Analysis Engine", ["auto", "fallback", "openai", "local"],
                                index=["auto","fallback","openai","local"].index(session.get("analyzer_mode","auto")),
                                key="llm_mode_select_lib")
    session["analyzer_mode"] = mode

    api_key = None
    if mode in ("auto", "openai"):
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password",
                                        value=os.environ.get("OPENAI_API_KEY",""),
                                        key="openai_key_input_lib")

    local_model = None
    if mode in ("auto", "local"):
        st.sidebar.markdown("#### 🖥️ Local LLM Model")
        st.sidebar.caption("🦙 Ollama mode: models run externally via HTTP. Pick any size your Ollama host can handle.")
        model_display_names = list(LOCAL_LLM_REGISTRY.keys())
        selected_display = st.sidebar.selectbox("Select model:", options=model_display_names, index=0, key="local_model_select_lib")
        local_model = LOCAL_LLM_REGISTRY[selected_display]
        st.session_state['selected_local_model'] = local_model
        if local_model and local_model.startswith("ollama:") and any(x in local_model for x in [":14b", ":70b", ":72b"]):
            st.sidebar.warning("⚠️ Large Ollama models (>14B) require significant host RAM/VRAM.")

    example_queries = [q for pdef in LIB_PROBLEM_DEFINITIONS.values() for q in pdef.example_queries[:1]]
    selected_example = st.sidebar.selectbox("Or select an example:", [""] + example_queries, key="example_query_select_lib")
    query = st.sidebar.text_area("Your LiB question:", value=selected_example, height=100,
                                 key="llm_query_input_lib", placeholder="e.g., How can we increase the energy density of lithium-ion batteries?")
    submitted = st.sidebar.button("🚀 Analyze & Expand Ontology", type="primary", key="llm_submit_lib")
    if not submitted or not query.strip():
        return None

    factory = LLMQueryAnalyzerFactory()
    analyzer = factory.get_analyzer(mode=mode, api_key=api_key, local_model=local_model)
    if isinstance(analyzer, OpenAIQueryAnalyzer):
        st.sidebar.info("🤖 Using **OpenAI GPT-4o-mini**")
    elif isinstance(analyzer, LocalLLMQueryAnalyzer):
        st.sidebar.info("🖥️ Using **Local LLM**")
    else:
        st.sidebar.info("📋 Using **Rule-based fallback**")

    with st.spinner("🔍 Analyzing query via LLM..."):
        analysis = analyzer.analyze_query(query, ontology)
    with st.spinner("🧬 Expanding ontology..."):
        mutations = expander.apply_query_analysis(analysis, analyzer)

    if hasattr(analyzer, 'unload_model'):
        analyzer.unload_model()
    del analyzer
    gc.collect()

    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(expander.session_concepts_added)
    whitelist.update(expander.query_bridge_concepts.keys())
    st.session_state['last_query_analysis'] = analysis
    st.session_state['last_query_text'] = query
    st.session_state['last_query_whitelist'] = whitelist
    st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
    st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

    QuerySessionManager.record_query(query, analysis, mutations)

    st.sidebar.success(f"✅ Analysis complete (confidence: {analysis.confidence:.0%})")
    st.sidebar.caption(f"Primary problem: **{analysis.primary_problem.value}**")
    st.sidebar.caption(f"Explicit concepts: {len(analysis.explicitly_mentioned)} | Inferred: {len(analysis.inferred_concepts)}")
    if mutations["concepts_added"]:
        st.sidebar.warning(f"🆕 {len(mutations['concepts_added'])} new concept(s) added")
        for c in mutations["concepts_added"]:
            st.sidebar.markdown(f"  - `{c['name']}` ({c['type']})")
    if mutations["bridges_created"]:
        st.sidebar.info(f"🌉 {len(mutations['bridges_created'])} bridge concept(s) created")
        for b in mutations["bridges_created"]:
            st.sidebar.markdown(f"  - `{b['bridge']}` ← `{b['for']}`")
    return analysis

def render_mutation_controls(expander: DynamicOntologyExpander) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Ontology Mutations")
    stats = expander.stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Concepts +", stats["concepts_added"])
    col2.metric("Relations +", stats["relationships_added"])
    if stats["total_mutations"] > 0:
        with st.sidebar.expander("📋 Mutation Log", expanded=False):
            for i, mut in enumerate(expander.mutation_log[-10:], 1):
                if mut["type"] == "add_concept":
                    st.sidebar.markdown(f"{i}. ➕ `{mut['concept']}`")
                elif mut["type"] == "add_relationship":
                    st.sidebar.markdown(f"{i}. 🔗 `{mut['source']}` → `{mut['target']}`")
                elif mut["type"] == "create_bridge":
                    st.sidebar.markdown(f"{i}. 🌉 `{mut['bridge_name']}`")
        col_undo, col_reset = st.sidebar.columns(2)
        if col_undo.button("↩️ Undo Last", key="undo_mutation_lib"):
            undone = expander.undo_last_mutation()
            if undone:
                st.sidebar.toast(f"Undone: {undone['type']}")
                st.rerun()
        if col_reset.button("🔄 Reset All", key="reset_mutations_lib"):
            result = expander.reset_to_base()
            st.sidebar.toast(f"Reset: {result['concepts_removed']} concepts, {result['relationships_removed']} relations removed")
            st.rerun()

def render_query_history() -> None:
    session = QuerySessionManager.get_session()
    if not session["query_history"]:
        return
    st.sidebar.markdown("---")
    with st.sidebar.expander("📜 Query History", expanded=False):
        for i, entry in enumerate(reversed(session["analysis_history"][-10:]), 1):
            st.sidebar.markdown(f"**{i}.** {entry['query'][:60]}...")
            st.sidebar.caption(f"  Problem: {entry['primary_problem']} | Type: {entry['query_type']} | Concepts: {entry['concepts_found']}")

def render_analysis_details(analysis: QueryAnalysisResult) -> None:
    st.markdown("## 📊 Query Analysis Results")
    with st.expander("🧠 Reasoning Chain", expanded=True):
        for step in analysis.reasoning_chain:
            st.markdown(f"→ {step}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Problem", analysis.primary_problem.value.replace("_", " "))
    col2.metric("Query Type", analysis.query_type)
    col3.metric("Confidence", f"{analysis.confidence:.0%}")
    st.markdown("### Concept Priority Rankings")
    top = analysis.get_top_concepts(15)
    if top:
        df = pd.DataFrame([cp.to_dict() for cp in top])
        def highlight_row(row):
            if row.get("explicit", False):
                return ["background-color: #d4edda"] * len(row)
            elif row.get("inferred", False):
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

def render_llm_qa_tab(analysis_data: Dict, ontology: DomainOntology) -> None:
    st.subheader("🤖 LLM-Guided Graph Q&A")
    st.markdown("Ask a specific scientific question about Li-ion batteries. The system will dynamically expand the ontology, extract a relevant subgraph, and generate a grounded answer using retrieved literature snippets.")

    if "qa_factory" not in st.session_state:
        st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if "qa_expander" not in st.session_state:
        st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if "qa_generator" not in st.session_state:
        st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    factory = st.session_state.qa_factory
    expander = st.session_state.qa_expander
    generator = st.session_state.qa_generator

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your research question:", placeholder="e.g., How does electrode thickness affect energy density?")
    with col2:
        mode = st.selectbox("Engine", ["auto", "openai", "local", "fallback"], index=0)

    if st.button("🔍 Analyze & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        local_model = st.session_state.get('selected_local_model')
        analyzer = factory.get_analyzer(mode=mode, local_model=local_model)
        generator.analyzer = analyzer

        with st.spinner("🧠 Analyzing query and expanding ontology..."):
            analysis = analyzer.analyze_query(query, ontology)
            mutations = expander.apply_query_analysis(analysis, analyzer)

            whitelist = set(analysis.explicitly_mentioned)
            whitelist.update(analysis.inferred_concepts)
            whitelist.update(expander.session_concepts_added)
            whitelist.update(expander.query_bridge_concepts.keys())
            st.session_state['last_query_analysis'] = analysis
            st.session_state['last_query_text'] = query
            st.session_state['last_query_whitelist'] = whitelist
            st.session_state['last_query_dynamic_concepts'] = expander.session_concepts_added
            st.session_state['last_query_bridge_concepts'] = expander.query_bridge_concepts

            if st.session_state.get('query_focused_build', False):
                st.success(f"✅ Query analysis complete. Whitelist contains {len(whitelist)} concepts.")
                if st.button("🔧 Rebuild Graph for This Query", type="primary", key="rebuild_for_query_btn_lib"):
                    st.session_state['force_rebuild'] = True
                    st.rerun()

        with st.spinner("🕸️ Extracting priority-guided subgraph..."):
            full_graph = analysis_data["nx_graph"]
            extractor = PriorityGuidedSubgraphExtractor(full_graph, ontology, expander)
            embed_model = analysis_data.get("embed_model")
            st.session_state['embed_model'] = embed_model
            query_embedding = None
            if embed_model is not None:
                try:
                    with torch.no_grad():
                        query_embedding = embed_model.encode(query, convert_to_numpy=True)
                except Exception:
                    pass
            subgraph = extractor.extract(analysis, query_embedding)

        with st.spinner("📚 Retrieving evidence and generating answer..."):
            answer = generator.generate_ground_response(
                query=query,
                analysis=analysis,
                subgraph=subgraph,
                concept_abstract_map=analysis_data["concept_abstract_map"],
                all_texts=analysis_data.get("all_texts", []),
                max_docs_per_concept=2
            )

        if hasattr(analyzer, 'unload_model'):
            analyzer.unload_model()
        del analyzer
        gc.collect()

        st.markdown("### 💡 Generated Answer")
        st.markdown(answer)
        st.markdown("---")
        st.markdown("### 🕸️ Focused Subgraph Visualization")
        with st.expander("⚙️ Subgraph Physics Settings (Prevent Jiggling)", expanded=False):
            phys_preset = st.selectbox(
                "Physics Preset",
                ["Stable (No Jiggle)", "Fluid", "Tight", "Off"],
                index=0,
                key="subgraph_phys_preset_lib"
            )
            presets = {
                "Stable (No Jiggle)": {"gravity": -800, "central_gravity": 0.1, "spring_length": 120, "spring_strength": 0.02, "damping": 0.95},
                "Fluid": {"gravity": -500, "central_gravity": 0.2, "spring_length": 150, "spring_strength": 0.04, "damping": 0.8},
                "Tight": {"gravity": -2000, "central_gravity": 0.3, "spring_length": 80, "spring_strength": 0.08, "damping": 0.6},
                "Off": {"gravity": 0, "central_gravity": 0, "spring_length": 100, "spring_strength": 0, "damping": 0.99}
            }
            p = presets[phys_preset]
            col1, col2 = st.columns(2)
            with col1:
                grav = st.slider("Gravity (Repulsion)", -5000, 0, p["gravity"], step=100, key="sub_grav_lib")
                spring_len = st.slider("Spring Length", 50, 300, p["spring_length"], step=10, key="sub_slen_lib")
                damp = st.slider("Damping (Anti-jiggle)", 0.1, 0.99, p["damping"], step=0.01, key="sub_damp_lib")
            with col2:
                cent_grav = st.slider("Central Gravity", 0.0, 1.0, p["central_gravity"], step=0.05, key="sub_cgrav_lib")
                spring_str = st.slider("Spring Strength", 0.0, 0.5, p["spring_strength"], step=0.01, key="sub_sstr_lib")
                phys_on = st.checkbox("Enable Physics", value=(phys_preset != "Off"), key="sub_phys_on_lib")
        visualizer = QueryDrivenVisualizer(ontology)
        html = visualizer.render_pyvis(
            subgraph, analysis,
            physics_enabled=phys_on,
            gravity=grav,
            central_gravity=cent_grav,
            spring_length=spring_len,
            spring_strength=spring_str,
            damping=damp
        )
        st.components.v1.html(html, height=600, scrolling=True)
        with st.expander("🔧 Behind the Scenes: Ontology Mutations & Reasoning"):
            st.markdown("**Reasoning Chain:**")
            for step in analysis.reasoning_chain:
                st.markdown("- " + step)
            if mutations.get("concepts_added") or mutations.get("bridges_created"):
                st.markdown("**Dynamic Ontology Updates:**")
                for c in mutations.get("concepts_added", []):
                    st.markdown("➕ Added Concept: `" + c['name'] + "` (" + c['type'] + ")")
                for b in mutations.get("bridges_created", []):
                    st.markdown("🌉 Created Bridge: `" + b['bridge'] + "` for `" + b['for'] + "`")

# ==========================================
# END LLM PORT
# ==========================================

# ==========================================
# SIDEBAR CONFIGURATION (updated to include LLM panels)
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

        # ---- LLM Query Panel (sidebar) ----
        if 'ontology' in st.session_state:
            ontology = st.session_state.ontology
            if 'qa_expander' not in st.session_state:
                st.session_state.qa_expander = DynamicOntologyExpander(ontology)
            expander = st.session_state.qa_expander
            full_graph = st.session_state.analysis_data.get("nx_graph") if st.session_state.get('analysis_data') else nx.Graph()
            render_llm_query_panel(ontology, expander, full_graph)
            render_mutation_controls(expander)
            render_query_history()

# ==========================================
# MAIN APPLICATION (updated to include LLM tab)
# ==========================================
def main():
    st.title("🔋 LiB-ConceptGraph: Energy Density Explorer")
    st.caption("Large-corpus concept graph builder for lithium-ion battery research • LLM-Guided Q&A • 3000+ abstracts optimized")
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
    if "ontology" not in st.session_state:
        st.session_state.ontology = DomainOntology()

    ontology = st.session_state.ontology

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
                    "df": df_filtered,
                    "ontology": ontology  # include ontology for LLM
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

        if st.session_state.edited_graph is not None:
            nx_graph = st.session_state.edited_graph
            concept_abstract_map = st.session_state.edited_cam

        viz_tab, distill_tab, scores_tab, valid_tab, extra_viz_tab, advanced_tab, export_tab, llm_tab = st.tabs([
            "🎨 Visualization", "📊 Distillation", "🎯 Research Directions",
            "📐 Validation", "📈 Extra Viz", "🧠 Advanced Analytics", "📥 Export", "🤖 LLM-Guided Q&A"
        ])

        with viz_tab:
            st.subheader("🌐 Interactive Concept Graph")

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
                with st.expander("🗑️ Remove Nodes"):
                    nodes_to_remove = st.multiselect("Select nodes to remove", list(nx_graph.nodes()))
                    if st.button("Remove Selected Nodes"):
                        edits = {'remove_nodes': nodes_to_remove}
                        new_g, new_cam = apply_graph_edits(nx_graph, concept_abstract_map, edits)
                        st.session_state.edit_history.push_snapshot(nx_graph, concept_abstract_map)
                        st.session_state.edited_graph = new_g
                        st.session_state.edited_cam = new_cam
                        st.rerun()
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
                    st.session_state.burst_df = detect_keyword_bursts(
                        data["df"], valid_concepts, concept_abstract_map
                    )
                    st.session_state.drift_df = detect_semantic_drift(
                        valid_concepts, concept_abstract_map, data["all_texts"], data["df"], data["embed_model"]
                    )
                    st.session_state.genealogy_df = build_concept_genealogy(nx_graph, valid_concepts)
                    st.session_state.bridge_df = detect_cross_domain_bridges(nx_graph, valid_concepts)
                    st.session_state.motif_data = analyze_network_motifs(nx_graph)
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

        # ─── LLM-Guided Q&A Tab ───
        with llm_tab:
            if "ontology" in data:
                render_llm_qa_tab(data, data["ontology"])
            else:
                st.info("Please build the concept graph with ontology enabled first.")

if __name__ == "__main__":
    main()
