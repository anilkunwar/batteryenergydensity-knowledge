#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Hardware toggle: uncomment BOTH lines ONLY if you need to force CPU mode
# -------------------------------------------------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["FORCE_CPU"] = "1"
# These MUST be set before importing torch or streamlit
# ============================================================================
# FORCE CPU ONLY MODE (Prevents CUDA No Kernel Image Errors)
# ============================================================================
import os
import sys

# ============================================================================
# FORCE CPU ONLY MODE
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["FORCE_CPU"] = "1"

import torch

"""
Lithium‑Ion Battery Concept Graph v7.0 (Local Ollama Edition)
=============================================================
Multi-level reasoning concept graph for Li‑ion battery materials.
Focus: Energy Density, Degradation, Manufacturing, and Performance Descriptors.

This is a TRUE architectural port of the Cu@Ag core‑shell codebase,
preserving every memory‑safe pattern, visualization pattern, and session‑state
management pattern. The domain ontology and extraction patterns have been replaced
with those for lithium‑ion battery quantitative descriptors.

NEW in v7.0 — Domain shift from Cu@Ag to Li‑ion batteries:
- All concepts, relationships, patterns, and problem definitions updated.
- Causal chains now reflect energy‑density‑degradation logic.
- Metrics extraction now captures capacity, energy density, cycle life, impedance, etc.

DOMAIN: Lithium‑Ion Batteries
- Materials: NMC811, LFP, Graphite, Silicon, Solid Electrolytes, etc.
- Manufacturing: Calendering, Slot‑Die Coating, Doping, Surface Coating, Prelithiation, Formation
- Properties: Energy Density, Specific Capacity, Cycle Life, Coulombic Efficiency, Rate Capability, Thermal Stability
- Phenomena: Capacity Fade, SEI Formation, Dendrite Growth, Thermal Runaway, Impedance Growth, TM Dissolution, Cation Mixing, Particle Cracking
- Parameters: Electrode Thickness, N/P Ratio, C‑rate, Temperature, Pressure
- Methods: EIS, dQ/dV, Galvanostatic Cycling, Coin Cell, Pouch Cell

DEPLOYMENT:
pip install streamlit torch transformers sentence-transformers networkx scikit-learn
pip install pyvis plotly pandas numpy kaleido matplotlib scipy seaborn bibtexparser

Run:
    streamlit run lib_concept_graph_v7_qdwa.py

Place JSON/BibTeX/CSV files in ./json_metadatabase/ folder next to this script.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
import torch.optim as optim
import networkx as nx
import numpy as np
import pandas as pd
import re
import json
import math
import os
import sys
import tempfile
import warnings
import traceback
import gc
import hashlib
import functools
import time
import io
import base64
import requests  # Ollama HTTP client
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
from sklearn.metrics import (
    silhouette_score, r2_score, mean_absolute_error,
    mean_squared_error, davies_bouldin_score, pairwise_distances
)
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
import plotly.figure_factory as ff  # for annotated heatmap
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# ============================================================================
# CPU/CUDA DEVICE CONFIGURATION
# ============================================================================
def is_force_cpu() -> bool:
    """Check whether CPU mode is forced via env var or sidebar toggle."""
    if os.environ.get("FORCE_CPU", "0") == "1":
        return True
    try:
        return bool(st.session_state.get("force_cpu", False))
    except Exception:
        return False

def get_device() -> str:
    """Return 'cpu' if forced, else cuda if available."""
    return "cpu" if is_force_cpu() else ("cuda" if torch.cuda.is_available() else "cpu")

def maybe_empty_cache():
    """Only empty CUDA cache if CUDA is actually being used."""
    if not is_force_cpu() and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# PERFORMANCE MONITORING DECORATOR
# ============================================================================
class PerformanceMonitor:
    _timings: Dict[str, float] = {}
    _call_counts: Dict[str, int] = {}

    @classmethod
    def reset(cls) -> None:
        cls._timings.clear()
        cls._call_counts.clear()

    @classmethod
    def get_report(cls) -> str:
        report = []
        for func_name, total_time in sorted(
            cls._timings.items(), key=lambda x: x[1], reverse=True
        ):
            count = cls._call_counts.get(func_name, 1)
            avg_time = total_time / count
            report.append(
                f"  {func_name}: {total_time:.3f}s total "
                f"({count} calls, {avg_time:.4f}s avg)"
            )
        return "\n".join(report)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        func_name = func.__qualname__
        PerformanceMonitor._timings[func_name] = (
            PerformanceMonitor._timings.get(func_name, 0) + elapsed
        )
        PerformanceMonitor._call_counts[func_name] = (
            PerformanceMonitor._call_counts.get(func_name, 0) + 1
        )
        return result
    return wrapper


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Lithium‑Ion Battery Concept Graph v7.0 (QDWA)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)


# ============================================================================
# COLORMAP REGISTRY (50+)
# ============================================================================
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
    "gist_earth": "GistEarth", "terrain": "Terrain", "ocean": "Ocean",
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


# ============================================================================
# ROBUST FILE LOADER (JSON / JSONL / CSV / BibTeX)
# ============================================================================
def robust_load_file(filepath: Path):
    suffix = filepath.suffix.lower()
    if suffix == '.bib':
        return parse_bibtex_file(filepath)

    text = filepath.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"File is empty (0 bytes or only whitespace).")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    sanitized = re.sub(r'NaN', 'null', text)
    sanitized = re.sub(r'Infinity', 'null', sanitized)
    sanitized = re.sub(r'-Infinity', 'null', sanitized)
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
    raise ValueError(
        f"Could not parse {filepath.name}. First 200 chars: {preview[:200]}..."
    )


def parse_bibtex_file(filepath: Path) -> List[Dict]:
    try:
        import bibtexparser
        from bibtexparser.bparser import BibTexParser
        from bibtexparser.customization import convert_to_unicode
        with open(filepath, 'r', encoding='utf-8') as bibfile:
            parser = BibTexParser()
            parser.customization = convert_to_unicode
            bib_database = bibtexparser.load(bibfile, parser=parser)
            records = []
            for entry in bib_database.entries:
                record = {
                    'title': entry.get('title', ''),
                    'abstract': entry.get('abstract', ''),
                    'author': entry.get('author', ''),
                    'year': entry.get('year', ''),
                    'journal': entry.get('journal', entry.get('booktitle', '')),
                    'doi': entry.get('doi', ''),
                    'keywords': entry.get('keywords', ''),
                    'entry_type': entry.get('ENTRYTYPE', ''),
                    'id': entry.get('ID', ''),
                    '_source_file': filepath.name,
                }
                records.append(record)
            return records
    except ImportError:
        st.warning(
            "bibtexparser not installed. Install with: pip install bibtexparser"
        )
        return []
    except Exception as e:
        st.error(f"BibTeX parse error for {filepath.name}: {e}")
        return []


@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    files = (
        sorted(Path(directory).glob("*.json"))
        + sorted(Path(directory).glob("*.bib"))
        + sorted(Path(directory).glob("*.csv"))
    )
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
                formatted = ' '.join(
                    hex_str[i:i + 2] for i in range(0, len(hex_str), 2)
                )
                st.code(
                    f"Hex preview (first {len(raw_bytes)} bytes):\n{formatted}",
                    language="text",
                )
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
    df = df.replace({
        float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA
    })
    year_cols = [c for c in df.columns if 'year' in c.lower()]
    if year_cols:
        df["Year"] = pd.to_numeric(df[year_cols[0]], errors="coerce")
    elif "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


# ============================================================================
# ENHANCED ONTOLOGY & NLP REASONING SYSTEM (Lithium‑Ion Battery)
# ============================================================================
class ConceptType(Enum):
    MATERIAL = "material"
    PROCESS = "process"
    PROPERTY = "property"
    PHENOMENON = "phenomenon"
    METHOD = "method"
    PARAMETER = "parameter"
    MICROSTRUCTURE = "microstructure"
    MODEL = "model"
    GENERAL = "general"


# ============================================================================
# NODE LABEL DISPLAY MODES
# ============================================================================

class NodeLabelMode(Enum):
    FULL_NAME    = "full_name"      # Full concept name inside node
    ANNOTATION   = "annotation"     # N1, N2, … inside node + legend below
    CUSTOM_BLANK = "custom_blank"   # User-typed text inside node (or truly blank)

# Hand-curated abbreviation map for Li‑ion ontology
_SHORT_NAME_MAP: Dict[str, str] = {
    "nmc811": "NMC811",
    "nmc622": "NMC622",
    "nmc532": "NMC532",
    "nmc333": "NMC333",
    "lfp": "LFP",
    "lco": "LCO",
    "nca": "NCA",
    "graphite": "Gr",
    "silicon": "Si",
    "siox": "SiOx",
    "lithium_metal": "Li",
    "lto": "LTO",
    "liquid_electrolyte": "Liq",
    "solid_electrolyte": "Sol",
    "fec": "FEC",
    "vc": "VC",
    "calendering": "Cal",
    "slot_die_coating": "Slot",
    "dry_electrode_processing": "Dry",
    "doping": "Dop",
    "surface_coating": "Coat",
    "prelithiation": "PreLi",
    "formation": "Form",
    "energy_density": "E_d",
    "power_density": "P_d",
    "specific_capacity": "Q_s",
    "cycle_life": "Cyc",
    "coulombic_efficiency": "CE",
    "rate_capability": "Rate",
    "thermal_stability": "T_stab",
    "ionic_conductivity": "σ_ion",
    "electronic_conductivity": "σ_el",
    "electrode_density": "ρ_e",
    "porosity": "ε",
    "volumetric_energy_density": "E_v",
    "capacity_fade": "Fade",
    "sei_formation": "SEI",
    "dendrite_growth": "Dend",
    "thermal_runaway": "TR",
    "impedance_growth": "R_g",
    "transition_metal_dissolution": "TM_diss",
    "cation_mixing": "Mix",
    "particle_cracking": "Crack",
    "electrode_thickness": "t_e",
    "np_ratio": "N/P",
    "c_rate": "C",
    "temperature": "T",
    "pressure": "P",
    "eis": "EIS",
    "dqdv": "dQ/dV",
    "galvanostatic_cycling": "Galv",
    "coin_cell": "Coin",
    "pouch_cell": "Pouch",
}

def get_short_name(canonical_name: str) -> str:
    """Return a concise abbreviation for a concept key."""
    if canonical_name in _SHORT_NAME_MAP:
        return _SHORT_NAME_MAP[canonical_name]
    words = canonical_name.split("_")
    if len(words) <= 2:
        return canonical_name.replace("_", " ").title()[:8]
    return "".join(w[0].upper() for w in words if w)


# Single source of truth for label mode dropdown
LABEL_MODE_OPTIONS = {
    "1. Full Name (concept name inside)":       NodeLabelMode.FULL_NAME,
    "2. Annotations N1, N2… (inside + legend)": NodeLabelMode.ANNOTATION,
    "3. Custom Blank (type your own text)":     NodeLabelMode.CUSTOM_BLANK,
}


class RelationshipType(Enum):
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    HYPONYM = "hyponym"
    CAUSES = "causes"
    RESULTS_IN = "results_in"
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
    DISCOVERS = "disovers"
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

# ============================================================================
# EDGE COLOR REGISTRY — one distinct color per RelationshipType category
# ============================================================================
EDGE_COLOR_REGISTRY: Dict[RelationshipType, str] = {
    # --- Semantic / structural ---
    RelationshipType.SYNONYM:           "#AAAAAA",
    RelationshipType.HYPERNYM:          "#5B9BD5",
    RelationshipType.HYPONYM:           "#5B9BD5",
    RelationshipType.PART_OF:           "#70AD47",
    RelationshipType.HAS_PART:          "#70AD47",
    RelationshipType.CO_OCCURS:         "#BFBFBF",

    # --- Causal / directional ---
    RelationshipType.CAUSES:            "#FF4444",
    RelationshipType.RESULTS_IN:        "#E06040",
    RelationshipType.INFLUENCES:        "#FF8C00",
    RelationshipType.DEPENDS_ON:        "#DAA520",
    RelationshipType.CONSTRAINS:        "#CC5500",
    RelationshipType.MODIFIES:          "#FF6347",
    RelationshipType.CORRECTS:          "#CD5C5C",
    RelationshipType.DRIVES:            "#DC143C",
    RelationshipType.ENABLES:           "#FF7F50",
    RelationshipType.PREVENTS:          "#2E8B57",

    # --- Phase / thermodynamic transitions ---
    RelationshipType.TRANSITIONS_TO:    "#8A2BE2",
    RelationshipType.REPLACES:          "#9932CC",
    RelationshipType.FORMS:             "#9370DB",
    RelationshipType.STABILIZES:        "#7B68EE",
    RelationshipType.PRESERVES:         "#6A5ACD",

    # --- Computation / modeling ---
    RelationshipType.TRAINS:            "#00CED1",
    RelationshipType.OUTPUTS:           "#20B2AA",
    RelationshipType.LEARNS:            "#48D1CC",
    RelationshipType.CAPTURES:          "#40E0D0",
    RelationshipType.COMPUTES:          "#008B8B",
    RelationshipType.SIMULATES:         "#5F9EA0",
    RelationshipType.MODELS:            "#4682B4",
    RelationshipType.APPROXIMATES:      "#87CEEB",
    RelationshipType.MAPS:              "#00BFFF",

    # --- Analysis / evaluation ---
    RelationshipType.QUANTIFIES:        "#32CD32",
    RelationshipType.EVALUATES:         "#228B22",
    RelationshipType.COMPARES:          "#3CB371",
    RelationshipType.VALIDATES:         "#2E8B57",
    RelationshipType.AVERAGES:          "#66CDAA",
    RelationshipType.CORRELATES:        "#00FA9A",

    # --- Structural / architectural ---
    RelationshipType.PARALLELIZES:      "#FFD700",
    RelationshipType.POSITIONS:         "#FFC125",
    RelationshipType.IDENTIFIES:        "#F0E68C",
    RelationshipType.PROCESSES:         "#EEE8AA",
    RelationshipType.GROUPS:            "#DAA520",
    RelationshipType.INTEGRATES:        "#B8860B",
    RelationshipType.COUPLES:           "#CD950C",

    # --- Discovery / optimization ---
    RelationshipType.DISCOVERS:         "#FF69B4",
    RelationshipType.PRE_TRAINS:        "#FF1493",
    RelationshipType.GENERALIZES:       "#DB7093",
    RelationshipType.QUERIES:           "#C71585",
    RelationshipType.OPTIMIZES:         "#FF00FF",
    RelationshipType.DESIGNS:           "#BA55D3",
    RelationshipType.CONSTRUCTS:        "#DA70D6",

    # --- Advanced modeling ---
    RelationshipType.UPSCALES:          "#8B4513",
    RelationshipType.RESOLVES:          "#A0522D",
    RelationshipType.SYNCHRONIZES:      "#D2691E",
    RelationshipType.CHARACTERIZES:     "#CD853F",
    RelationshipType.DECOMPOSES:        "#DEB887",
    RelationshipType.FRAMES:            "#D2B48C",
    RelationshipType.COMPOSES:          "#BC8F8F",
    RelationshipType.QUALIFIES:         "#F4A460",

    # --- Explanation / visualization ---
    RelationshipType.STRENGTHENS:       "#7FFF00",
    RelationshipType.EXPLAINS:          "#ADFF2F",
    RelationshipType.INTERPRETS:        "#7CFC00",
    RelationshipType.VISUALIZES:        "#00FF7F",
    RelationshipType.ACCELERATES:       "#98FB98",
    RelationshipType.ENFORCES:          "#90EE90",

    # --- Generic fallback ---
    RelationshipType.SEMANTIC:          "#808080",
    RelationshipType.INFERRED:          "#A9A9A9",
    RelationshipType.BRIDGE:            "#C0C0C0",
    RelationshipType.SELECTS:           "#D3D3D3",
    RelationshipType.INITIATES:         "#696969",
    RelationshipType.DETECTS:           "#556B2F",
    RelationshipType.MEASURES:          "#6B8E23",
    RelationshipType.OBSERVES:          "#808000",
    RelationshipType.GENERATES:         "#6B8E23",
}

EDGE_COLOR_FALLBACK = "#888888"


def get_edge_color(rel_type: RelationshipType) -> str:
    return EDGE_COLOR_REGISTRY.get(rel_type, EDGE_COLOR_FALLBACK)


def get_edge_width(rel_type: RelationshipType) -> float:
    STRONG = {RelationshipType.CAUSES, RelationshipType.DRIVES,
              RelationshipType.FORMS, RelationshipType.STABILIZES,
              RelationshipType.DEPENDS_ON, RelationshipType.CONSTRAINS,
              RelationshipType.PREVENTS}
    MEDIUM = {RelationshipType.INFLUENCES, RelationshipType.RESULTS_IN,
              RelationshipType.MODIFIES, RelationshipType.ENABLES,
              RelationshipType.TRANSITIONS_TO, RelationshipType.COMPUTES}
    if rel_type in STRONG:
        return 3.0
    elif rel_type in MEDIUM:
        return 2.0
    return 1.0


def get_edge_style(rel_type: RelationshipType) -> str:
    DASHED = {RelationshipType.INFERRED, RelationshipType.CO_OCCURS,
              RelationshipType.SEMANTIC, RelationshipType.BRIDGE}
    return "dashed" if rel_type in DASHED else "solid"


def lighten_hex_color(hex_color: str, factor: float) -> str:
    if not hex_color.startswith('#'):
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


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
    """Comprehensive ontology for Lithium‑Ion Battery Materials."""

    def __init__(self) -> None:
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        self._build_ontology()

    def _build_ontology(self) -> None:
        # ============================================================================
        # INSIDE DomainOntology._build_ontology()
        # ============================================================================
        # --- Cathode Materials ---
        self._add_concept("nmc811", ConceptType.MATERIAL,
            synonyms={"lini0.8mn0.1co0.1o2", "high-nickel nmc", "nmc811 cathode"},
            definition="LiNi0.8Mn0.1Co0.1O2, high‑nickel layered cathode for high energy density")
        self._add_concept("nmc622", ConceptType.MATERIAL,
            synonyms={"lini0.6mn0.2co0.2o2", "nmc622 cathode"},
            definition="LiNi0.6Mn0.2Co0.2O2, balanced cathode")
        self._add_concept("nmc532", ConceptType.MATERIAL,
            synonyms={"lini0.5mn0.3co0.2o2", "nmc532 cathode"},
            definition="LiNi0.5Mn0.3Co0.2O2, stable cathode")
        self._add_concept("nmc333", ConceptType.MATERIAL,
            synonyms={"lin1/3mn1/3co1/3o2", "nmc333 cathode"},
            definition="LiNi1/3Mn1/3Co1/3O2, standard cathode")
        self._add_concept("lfp", ConceptType.MATERIAL,
            synonyms={"lifepo4", "lithium iron phosphate", "olivine"},
            definition="LiFePO4, olivine cathode with good thermal stability")
        self._add_concept("lco", ConceptType.MATERIAL,
            synonyms={"licoo2", "lithium cobalt oxide"},
            definition="LiCoO2, traditional cathode for consumer electronics")
        self._add_concept("nca", ConceptType.MATERIAL,
            synonyms={"linicoalo2", "lithium nickel cobalt aluminum oxide"},
            definition="LiNiCoAlO2, high‑energy cathode")

        # --- Anode Materials ---
        self._add_concept("graphite", ConceptType.MATERIAL,
            synonyms={"graphite anode", "natural graphite", "synthetic graphite"},
            definition="Graphite, standard anode material for Li‑ion batteries")
        self._add_concept("silicon", ConceptType.MATERIAL,
            synonyms={"si anode", "silicon anode", "high-capacity anode"},
            definition="Silicon anode with ultra‑high specific capacity but large volume expansion")
        self._add_concept("siox", ConceptType.MATERIAL,
            synonyms={"silicon oxide", "sio", "siox anode"},
            definition="Silicon oxide anode, better cycling stability than pure Si")
        self._add_concept("lithium_metal", ConceptType.MATERIAL,
            synonyms={"li metal", "lithium anode"},
            definition="Lithium metal anode for high energy density, dendrite issues")
        self._add_concept("lto", ConceptType.MATERIAL,
            synonyms={"li4ti5o12", "lithium titanium oxide"},
            definition="Li4Ti5O12, zero‑strain anode with long cycle life")

        # --- Electrolytes ---
        self._add_concept("liquid_electrolyte", ConceptType.MATERIAL,
            synonyms={"carbonate electrolyte", "ether electrolyte", "li+ electrolyte"},
            definition="Liquid electrolyte containing Li salt and organic solvents")
        self._add_concept("solid_electrolyte", ConceptType.MATERIAL,
            synonyms={"sulfide electrolyte", "oxide electrolyte", "polymer electrolyte"},
            definition="Solid electrolyte enabling all‑solid‑state batteries")
        self._add_concept("fec", ConceptType.MATERIAL,
            synonyms={"fluoroethylene carbonate", "fec additive"},
            definition="Fluoroethylene carbonate, electrolyte additive for SEI enhancement")
        self._add_concept("vc", ConceptType.MATERIAL,
            synonyms={"vinylene carbonate", "vc additive"},
            definition="Vinylene carbonate, electrolyte additive for SEI formation")

        # --- Manufacturing Processes ---
        self._add_concept("calendering", ConceptType.PROCESS,
            synonyms={"electrode calendering", "mechanical pressing", "compaction"},
            definition="Pressing of electrode coating to increase density and energy density")
        self._add_concept("slot_die_coating", ConceptType.PROCESS,
            synonyms={"slurry coating", "electrode coating", "slot‑die coating"},
            definition="Coating of electrode slurry onto current collector")
        self._add_concept("dry_electrode_processing", ConceptType.PROCESS,
            synonyms={"dry coating", "solvent-free fabrication"},
            definition="Solvent‑free electrode fabrication process")
        self._add_concept("doping", ConceptType.PROCESS,
            synonyms={"elemental doping", "cation doping", "anion doping"},
            definition="Introduction of dopants to modify electrode properties")
        self._add_concept("surface_coating", ConceptType.PROCESS,
            synonyms={"protective coating", "coating layer", "surface modification"},
            definition="Deposition of a coating layer on electrode particles")
        self._add_concept("prelithiation", ConceptType.PROCESS,
            synonyms={"pre‑lithiation", "lithium pre-doping"},
            definition="Adding extra Li to compensate for first‑cycle loss")
        self._add_concept("formation", ConceptType.PROCESS,
            synonyms={"formation cycling", "initial cycling", "formation step"},
            definition="Initial charge/discharge cycles to build stable SEI")

        # --- Properties ---
        self._add_concept("energy_density", ConceptType.PROPERTY,
            synonyms={"specific energy", "wh/kg", "gravimetric energy density"},
            definition="Gravimetric energy density in Wh/kg")
        self._add_concept("power_density", ConceptType.PROPERTY,
            synonyms={"specific power", "w/kg", "power density"},
            definition="Gravimetric power density in W/kg")
        self._add_concept("specific_capacity", ConceptType.PROPERTY,
            synonyms={"capacity", "mah/g", "capacity"},
            definition="Specific capacity in mAh/g")
        self._add_concept("cycle_life", ConceptType.PROPERTY,
            synonyms={"longevity", "cycling stability", "cycles to 80%"},
            definition="Number of cycles until capacity retention drops to 80%")
        self._add_concept("coulombic_efficiency", ConceptType.PROPERTY,
            synonyms={"ce", "charge/discharge efficiency", "coulombic efficiency"},
            definition="Ratio of discharge to charge capacity, ideally near 100%")
        self._add_concept("rate_capability", ConceptType.PROPERTY,
            synonyms={"rate performance", "c‑rate capability", "fast charging"},
            definition="Ability to maintain capacity at high charge/discharge rates")
        self._add_concept("thermal_stability", ConceptType.PROPERTY,
            synonyms={"thermal safety", "decomposition temperature", "thermal stability"},
            definition="Resistance to thermal decomposition and runaway")
        self._add_concept("ionic_conductivity", ConceptType.PROPERTY,
            synonyms={"li+ conductivity", "electrolyte conductivity", "ionic conductivity"},
            definition="Ionic conductivity of electrolyte or electrode")
        self._add_concept("electronic_conductivity", ConceptType.PROPERTY,
            synonyms={"electron conductivity", "electronic conductivity"},
            definition="Electronic conductivity of electrode materials")
        self._add_concept("electrode_density", ConceptType.PROPERTY,
            synonyms={"coating density", "compaction density", "electrode density"},
            definition="Density of the electrode coating")
        self._add_concept("porosity", ConceptType.PROPERTY,
            synonyms={"electrode porosity", "void fraction", "porosity"},
            definition="Fraction of void space in the electrode")
        self._add_concept("volumetric_energy_density", ConceptType.PROPERTY,
            synonyms={"volumetric energy", "wh/l", "volumetric energy density"},
            definition="Volumetric energy density in Wh/L")

        # --- Phenomena ---
        self._add_concept("capacity_fade", ConceptType.PHENOMENON,
            synonyms={"capacity loss", "fading", "degradation"},
            definition="Gradual loss of reversible capacity over cycling or storage")
        self._add_concept("sei_formation", ConceptType.PHENOMENON,
            synonyms={"sei growth", "solid electrolyte interphase", "passivation layer"},
            definition="Formation of SEI layer on anode, essential for stability but consumes Li")
        self._add_concept("dendrite_growth", ConceptType.PHENOMENON,
            synonyms={"dendrite formation", "lithium whiskers", "dendrite"},
            definition="Growth of lithium dendrites, can lead to short circuits")
        self._add_concept("thermal_runaway", ConceptType.PHENOMENON,
            synonyms={"uncontrolled self‑heating", "fire", "explosion"},
            definition="Uncontrolled thermal reaction leading to safety failure")
        self._add_concept("impedance_growth", ConceptType.PHENOMENON,
            synonyms={"resistance increase", "polarization growth", "impedance rise"},
            definition="Increase in cell impedance over time")
        self._add_concept("transition_metal_dissolution", ConceptType.PHENOMENON,
            synonyms={"tm dissolution", "ion leaching", "metal dissolution"},
            definition="Dissolution of transition metal ions from cathode")
        self._add_concept("cation_mixing", ConceptType.PHENOMENON,
            synonyms={"cation disorder", "ni/li mixing", "site exchange"},
            definition="Exchange of Ni²⁺ and Li⁺ sites in layered cathodes, reducing capacity")
        self._add_concept("particle_cracking", ConceptType.PHENOMENON,
            synonyms={"particle fracture", "mechanical degradation", "cracking"},
            definition="Cracking of electrode particles due to volume changes")

        # --- Parameters ---
        self._add_concept("electrode_thickness", ConceptType.PARAMETER,
            synonyms={"coating thickness", "t_e", "electrode thickness"},
            definition="Thickness of the electrode coating (μm)")
        self._add_concept("np_ratio", ConceptType.PARAMETER,
            synonyms={"negative/positive capacity ratio", "capacity balance", "n/p ratio"},
            definition="Ratio of negative to positive electrode capacity")
        self._add_concept("c_rate", ConceptType.PARAMETER,
            synonyms={"c‑rate", "charge/discharge rate", "current density"},
            definition="Charge/discharge rate, where 1C = 1 hour")
        self._add_concept("temperature", ConceptType.PARAMETER,
            synonyms={"operating temperature", "cell temperature", "temp"},
            definition="Operating temperature of the cell (°C)")
        self._add_concept("pressure", ConceptType.PARAMETER,
            synonyms={"stack pressure", "external compression", "pressure"},
            definition="External pressure applied to the cell stack")

        # --- Methods ---
        self._add_concept("eis", ConceptType.METHOD,
            synonyms={"electrochemical impedance spectroscopy", "impedance spectroscopy"},
            definition="EIS to measure resistance and impedance growth")
        self._add_concept("dqdv", ConceptType.METHOD,
            synonyms={"differential voltage analysis", "dqv", "differential capacity"},
            definition="dQ/dV analysis to study degradation and phase transitions")
        self._add_concept("galvanostatic_cycling", ConceptType.METHOD,
            synonyms={"constant current cycling", "galvanostatic cycling", "cycling"},
            definition="Galvanostatic cycling to evaluate capacity and cycle life")
        self._add_concept("coin_cell", ConceptType.METHOD,
            synonyms={"half‑cell", "cr2032", "coin cell testing"},
            definition="Coin cell testing for rapid material screening")
        self._add_concept("pouch_cell", ConceptType.METHOD,
            synonyms={"soft pack", "pouch cell", "cell‑level validation"},
            definition="Pouch cell testing for cell‑level validation")

        # Build indices and causal chains
        self._build_synonym_index()
        self._build_causal_chains()

    def _add_concept(
        self,
        canonical_name: str,
        concept_type: ConceptType,
        synonyms: Set[str] = None,
        hypernyms: Set[str] = None,
        hyponyms: Set[str] = None,
        definition: str = "",
        related_processes: Set[str] = None,
        related_properties: Set[str] = None,
    ) -> None:
        node = ConceptNode(
            canonical_name=canonical_name,
            concept_type=concept_type,
            synonyms=synonyms or set(),
            hypernyms=hypernyms or set(),
            hyponyms=hyponyms or set(),
            related_processes=related_processes or set(),
            related_properties=related_properties or set(),
            definition=definition,
        )
        self.concepts[canonical_name] = node

    def _build_synonym_index(self) -> None:
        self.synonym_to_canonical: Dict[str, str] = {}
        for canonical, node in self.concepts.items():
            self.synonym_to_canonical[canonical.lower()] = canonical
            for syn in node.synonyms:
                self.synonym_to_canonical[syn.lower()] = canonical

    def _build_causal_chains(self) -> None:
        # ============================================================================
        # INSIDE DomainOntology._build_causal_chains()
        # ============================================================================
        causal_chains = [
            # Manufacturing → Structure/Properties
            ("calendering", RelationshipType.INFLUENCES, "electrode_density", 0.80),
            ("calendering", RelationshipType.INFLUENCES, "energy_density", 0.75),
            ("calendering", RelationshipType.INFLUENCES, "porosity", -0.70),
            ("slot_die_coating", RelationshipType.INFLUENCES, "electrode_thickness", 0.85),
            ("dry_electrode_processing", RelationshipType.INFLUENCES, "electrode_density", 0.70),
            ("doping", RelationshipType.INFLUENCES, "thermal_stability", 0.65),
            ("doping", RelationshipType.INFLUENCES, "cycle_life", 0.60),
            ("surface_coating", RelationshipType.STABILIZES, "cycle_life", 0.75),
            ("surface_coating", RelationshipType.PREVENTS, "transition_metal_dissolution", 0.70),
            ("prelithiation", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.80),
            ("formation", RelationshipType.ENABLES, "sei_formation", 0.90),

            # Material → Properties
            ("silicon", RelationshipType.INFLUENCES, "specific_capacity", 0.95),
            ("silicon", RelationshipType.CAUSES, "particle_cracking", 0.85),
            ("silicon", RelationshipType.CAUSES, "capacity_fade", 0.80),
            ("nmc811", RelationshipType.INFLUENCES, "specific_capacity", 0.85),
            ("nmc811", RelationshipType.INFLUENCES, "thermal_stability", -0.75),
            ("lfp", RelationshipType.INFLUENCES, "thermal_stability", 0.90),
            ("lfp", RelationshipType.INFLUENCES, "cycle_life", 0.80),
            ("solid_electrolyte", RelationshipType.INFLUENCES, "thermal_stability", 0.85),
            ("solid_electrolyte", RelationshipType.INFLUENCES, "ionic_conductivity", 0.70),
            ("fec", RelationshipType.ENABLES, "sei_formation", 0.80),
            ("fec", RelationshipType.INFLUENCES, "cycle_life", 0.75),
            ("vc", RelationshipType.ENABLES, "sei_formation", 0.70),
            ("vc", RelationshipType.INFLUENCES, "cycle_life", 0.70),

            # Parameters → Properties
            ("electrode_thickness", RelationshipType.INFLUENCES, "energy_density", 0.85),
            ("electrode_thickness", RelationshipType.INFLUENCES, "power_density", -0.80),
            ("np_ratio", RelationshipType.INFLUENCES, "cycle_life", 0.70),
            ("np_ratio", RelationshipType.INFLUENCES, "coulombic_efficiency", 0.65),
            ("c_rate", RelationshipType.INFLUENCES, "rate_capability", 0.90),
            ("temperature", RelationshipType.INFLUENCES, "thermal_stability", -0.70),
            ("pressure", RelationshipType.INFLUENCES, "electrode_density", 0.60),

            # Phenomena → Properties
            ("sei_formation", RelationshipType.CAUSES, "impedance_growth", 0.80),
            ("dendrite_growth", RelationshipType.CAUSES, "thermal_runaway", 0.75),
            ("dendrite_growth", RelationshipType.CAUSES, "capacity_fade", 0.70),
            ("thermal_runaway", RelationshipType.CAUSES, "safety_failure", 0.90),
            ("transition_metal_dissolution", RelationshipType.CAUSES, "capacity_fade", 0.80),
            ("cation_mixing", RelationshipType.CAUSES, "specific_capacity", -0.85),
            ("particle_cracking", RelationshipType.CAUSES, "capacity_fade", 0.75),

            # Methods ↔ Properties
            ("eis", RelationshipType.MEASURES, "impedance_growth", 0.90),
            ("dqdv", RelationshipType.MEASURES, "capacity_fade", 0.85),
            ("galvanostatic_cycling", RelationshipType.MEASURES, "cycle_life", 0.95),
            ("coin_cell", RelationshipType.MEASURES, "specific_capacity", 0.90),
            ("pouch_cell", RelationshipType.MEASURES, "energy_density", 0.85),
        ]
        for source, rel_type, target, confidence in causal_chains:
            self.relationships.append(
                Relationship(source, target, rel_type, abs(confidence))
            )

    def resolve_concept(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if text_lower in self.synonym_to_canonical:
            return self.synonym_to_canonical[text_lower]
        normalized = self._normalize_text(text_lower)
        if normalized in self.synonym_to_canonical:
            return self.synonym_to_canonical[normalized]
        variants = [
            text_lower.replace("-", " "),
            text_lower.replace(" ", "-"),
            text_lower.replace(" of ", " "),
            text_lower.replace(" for ", " "),
            text_lower.replace(" in ", " "),
            re.sub(r'\bs\b', '', text_lower),
            re.sub(r'\bes\b', '', text_lower),
        ]
        for variant in variants:
            if variant in self.synonym_to_canonical:
                return self.synonym_to_canonical[variant]
        return None

    def _normalize_text(self, text: str) -> str:
        text = re.sub(
            r'\b(the|a|an|of|for|in|with|by|to|and|or|on|at)\b', ' ', text
        )
        text = ' '.join(text.split())
        return text.strip()

    def get_concept_type(self, canonical_name: str) -> ConceptType:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].concept_type
        return ConceptType.GENERAL

    def get_hypernyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hypernyms
        return set()

    def get_hyponyms(self, canonical_name: str) -> Set[str]:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].hyponyms
        return set()

    def get_definition(self, canonical_name: str) -> str:
        if canonical_name in self.concepts:
            return self.concepts[canonical_name].definition
        return ""

    def infer_path(
        self, source: str, target: str, max_depth: int = 3
    ) -> List[List[str]]:
        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(current: str, target: str, path: List[str], depth: int) -> None:
            if depth > max_depth:
                return
            if current == target:
                paths.append(path.copy())
                return
            if current in visited:
                return
            visited.add(current)
            for rel in self.relationships:
                if rel.source == current and rel.confidence > 0.5:
                    path.append(rel.target)
                    dfs(rel.target, target, path, depth + 1)
                    path.pop()
            if current in self.concepts:
                for hyp in self.concepts[current].hypernyms:
                    path.append(hyp)
                    dfs(hyp, target, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(source, target, [source], 0)
        return paths

    def get_related_concepts(
        self, canonical_name: str, rel_type: RelationshipType = None
    ) -> List[Tuple[str, RelationshipType, float]]:
        related: List[Tuple[str, RelationshipType, float]] = []
        for rel in self.relationships:
            if rel.source == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.target, rel.rel_type, rel.confidence))
            elif rel.target == canonical_name:
                if rel_type is None or rel.rel_type == rel_type:
                    related.append((rel.source, rel.rel_type, rel.confidence))
        return related


# ============================================================================
# ADVANCED CONCEPT RESOLVER
# ============================================================================


# ============================================================================
# HIERARCHY LABEL BUILDER — enriches flat concept names with ancestor path
# ============================================================================

_HIERARCHY_PARENTS = {
    # --- Root domain ---
    "lithium_ion_battery": (None, 0),
    # --- Tier 1: Cathode Materials ---
    "nmc811":   ("Cathode Materials", 1),
    "nmc622":   ("Cathode Materials", 1),
    "nmc532":   ("Cathode Materials", 1),
    "nmc333":   ("Cathode Materials", 1),
    "lfp":      ("Cathode Materials", 1),
    "lco":      ("Cathode Materials", 1),
    "nca":      ("Cathode Materials", 1),
    # --- Tier 1: Anode Materials ---
    "graphite":        ("Anode Materials", 1),
    "silicon":         ("Anode Materials", 1),
    "siox":            ("Anode Materials", 1),
    "lithium_metal":   ("Anode Materials", 1),
    "lto":             ("Anode Materials", 1),
    # --- Tier 1: Electrolytes ---
    "liquid_electrolyte": ("Electrolytes", 1),
    "solid_electrolyte":  ("Electrolytes", 1),
    "fec":                ("Electrolytes", 1),
    "vc":                 ("Electrolytes", 1),
    # --- Tier 1: Manufacturing ---
    "calendering":           ("Manufacturing Processes", 1),
    "slot_die_coating":      ("Manufacturing Processes", 1),
    "dry_electrode_processing": ("Manufacturing Processes", 1),
    "doping":                ("Manufacturing Processes", 1),
    "surface_coating":       ("Manufacturing Processes", 1),
    "prelithiation":         ("Manufacturing Processes", 1),
    "formation":             ("Manufacturing Processes", 1),
    # --- Tier 1: Properties ---
    "energy_density":          ("Energy Metrics", 1),
    "power_density":           ("Energy Metrics", 1),
    "specific_capacity":       ("Capacity Metrics", 1),
    "cycle_life":              ("Capacity Metrics", 1),
    "coulombic_efficiency":    ("Capacity Metrics", 1),
    "rate_capability":         ("Kinetic Metrics", 1),
    "thermal_stability":       ("Safety Metrics", 1),
    "ionic_conductivity":      ("Kinetic Metrics", 1),
    "electronic_conductivity": ("Kinetic Metrics", 1),
    "electrode_density":       ("Structural Metrics", 1),
    "porosity":                ("Structural Metrics", 1),
    "volumetric_energy_density": ("Energy Metrics", 1),
    # --- Tier 1: Phenomena ---
    "capacity_fade":               ("Degradation Phenomena", 1),
    "sei_formation":               ("Degradation Phenomena", 1),
    "dendrite_growth":             ("Degradation Phenomena", 1),
    "thermal_runaway":             ("Degradation Phenomena", 1),
    "impedance_growth":            ("Degradation Phenomena", 1),
    "transition_metal_dissolution":("Degradation Phenomena", 1),
    "cation_mixing":               ("Degradation Phenomena", 1),
    "particle_cracking":           ("Degradation Phenomena", 1),
    # --- Tier 1: Parameters ---
    "electrode_thickness": ("Design Parameters", 1),
    "np_ratio":            ("Design Parameters", 1),
    "c_rate":              ("Design Parameters", 1),
    "temperature":         ("Design Parameters", 1),
    "pressure":            ("Design Parameters", 1),
    # --- Tier 1: Methods ---
    "eis":                 ("Characterization Methods", 1),
    "dqdv":                ("Characterization Methods", 1),
    "galvanostatic_cycling": ("Testing Methods", 1),
    "coin_cell":           ("Testing Methods", 1),
    "pouch_cell":          ("Testing Methods", 1),
}


def get_hierarchy_label(concept_key: str,
                        style: str = "arrow") -> str:
    """
    Build a human-readable hierarchy label for a concept.
    style: "arrow" → "Energy Metrics → Energy Density"
    """
    SEPARATOR = {
        "arrow": " → ",
        "bracket": " [",
        "dot": " · ",
        "leaf": "",
    }
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)
    if entry is None or entry[0] is None or style == "leaf":
        return leaf
    parent_label = entry[0]
    sep = SEPARATOR.get(style, " → ")
    if style == "bracket":
        return f"{parent_label}{sep}{leaf}]"
    return f"{parent_label}{sep}{leaf}"


def get_hierarchy_path(concept_key: str) -> List[str]:
    leaf = concept_key.replace("_", " ").title()
    entry = _HIERARCHY_PARENTS.get(concept_key)
    if entry is None or entry[0] is None:
        return ["Lithium‑Ion Batteries", leaf]
    parent_label = entry[0]
    return ["Lithium‑Ion Batteries", parent_label, leaf]


def build_sunburst_data(
    graph: nx.Graph,
    node_weights: Optional[Dict[str, float]] = None,
    min_weight: float = 0.0,
) -> Tuple[List[str], List[str], List[float], List[str]]:
    ids: List[str] = []
    labels: List[str] = []
    values: List[float] = []
    parents: List[str] = []

    root_id = "Lithium‑Ion Batteries"
    ids.append(root_id)
    labels.append("Lithium‑Ion Batteries")
    values.append(0)
    parents.append("")

    category_children: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for node in graph.nodes:
        if node not in _HIERARCHY_PARENTS:
            continue
        parent_label = _HIERARCHY_PARENTS[node][0]
        if parent_label is None:
            continue
        w = (node_weights or {}).get(node, 1.0)
        if w < min_weight:
            continue
        category_children[parent_label].append((node, w))

    for cat_label, children in sorted(category_children.items()):
        cat_id = cat_label
        cat_value = sum(w for _, w in children)
        ids.append(cat_id)
        labels.append(cat_label)
        values.append(cat_value)
        parents.append(root_id)

        for child_key, child_w in sorted(children, key=lambda x: -x[1]):
            child_label = child_key.replace("_", " ").title()
            child_id = child_key
            ids.append(child_id)
            labels.append(child_label)
            values.append(child_w)
            parents.append(cat_id)

    return ids, labels, values, parents


class AdvancedConceptResolver:
    """
    Multi-level concept resolution using ontology, embeddings, and context.
    """

    def __init__(
        self,
        ontology: DomainOntology,
        embed_model,
        cache_max: int = 2000,
    ) -> None:
        self.ontology = ontology
        self.embed_model = embed_model
        self.resolution_cache: Dict[str, str] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        # v6.1: bounded caches
        self._cache_max = max(100, int(cache_max))
        self.similarity_threshold = 0.85
        self.ontology_concepts_list: Optional[List[str]] = None
        self.ontology_embedding_matrix: Optional[np.ndarray] = None
        self._precompute_ontology_embeddings()

    def _trim_embedding_cache(self) -> None:
        if len(self.embedding_cache) > self._cache_max:
            keys = list(self.embedding_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.embedding_cache[k]
            gc.collect()

    def _trim_resolution_cache(self) -> None:
        if len(self.resolution_cache) > self._cache_max * 4:
            keys = list(self.resolution_cache.keys())
            for k in keys[:int(len(keys) * 0.3)]:
                del self.resolution_cache[k]

    def _precompute_ontology_embeddings(self) -> None:
        concepts: List[str] = []
        all_texts: List[str] = []
        text_counts: List[int] = []

        for canonical, node in self.ontology.concepts.items():
            concepts.append(canonical)
            texts = [canonical] + list(node.synonyms)
            all_texts.extend(texts)
            text_counts.append(len(texts))

        if not all_texts:
            self.ontology_concepts_list = []
            self.ontology_embedding_matrix = np.empty((0, 0))
            return

        with torch.no_grad():
            all_embeddings = self.embed_model.encode(
                all_texts,
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )

        embeddings: List[np.ndarray] = []
        idx = 0
        for count in text_counts:
            concept_embs = all_embeddings[idx:idx + count]
            embeddings.append(np.mean(concept_embs, axis=0))
            idx += count

        del all_embeddings
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()

        self.ontology_concepts_list = concepts
        self.ontology_embedding_matrix = (
            np.array(embeddings) if embeddings else np.empty((0, 0))
        )

    @timed
    def resolve(
        self, text: str, context: str = "", use_embedding: bool = True
    ) -> Optional[str]:
        self._trim_resolution_cache()
        text_lower = text.lower().strip()
        if text_lower in self.resolution_cache:
            return self.resolution_cache[text_lower]

        canonical = self.ontology.resolve_concept(text)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        canonical = self._substring_match(text_lower)
        if canonical:
            self.resolution_cache[text_lower] = canonical
            return canonical

        if use_embedding and self.ontology_embedding_matrix.size > 0:
            canonical = self._embedding_match(text, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        if context:
            canonical = self._context_disambiguation(text_lower, context)
            if canonical:
                self.resolution_cache[text_lower] = canonical
                return canonical

        return None

    @timed
    def resolve_batch(
        self, phrases: List[str], context: str = ""
    ) -> Dict[str, Optional[str]]:
        results: Dict[str, Optional[str]] = {}
        need_embedding: List[str] = []

        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            if phrase_lower in self.resolution_cache:
                results[phrase] = self.resolution_cache[phrase_lower]
                continue
            canonical = self.ontology.resolve_concept(phrase)
            if canonical:
                self.resolution_cache[phrase_lower] = canonical
                results[phrase] = canonical
                continue
            sub_match = self._substring_match(phrase_lower)
            if sub_match:
                self.resolution_cache[phrase_lower] = sub_match
                results[phrase] = sub_match
                continue
            need_embedding.append(phrase)

        if need_embedding and self.ontology_embedding_matrix.size > 0:
            query_texts = [
                p if not context else f"{p} in context of {context}"
                for p in need_embedding
            ]
            with torch.no_grad():
                query_embs = self.embed_model.encode(
                    query_texts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sims = cosine_similarity(query_embs, self.ontology_embedding_matrix)
            best_indices = np.argmax(sims, axis=1)
            best_scores = np.max(sims, axis=1)
            for idx, phrase in enumerate(need_embedding):
                if best_scores[idx] > self.similarity_threshold:
                    canonical = self.ontology_concepts_list[best_indices[idx]]
                    self.resolution_cache[phrase.lower().strip()] = canonical
                    results[phrase] = canonical
                else:
                    results[phrase] = None
            del query_embs, sims, best_indices, best_scores
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_cache()
        else:
            for phrase in need_embedding:
                results[phrase] = None

        self._trim_resolution_cache()
        return results

    def _substring_match(self, text: str) -> Optional[str]:
        for canonical, node in self.ontology.concepts.items():
            all_forms = {canonical.lower()} | node.synonyms
            for form in all_forms:
                if form in text or text in form:
                    if len(form) > 4 and len(text) > 4:
                        return canonical
        return None

    def _embedding_match(self, text: str, context: str = "") -> Optional[str]:
        try:
            query_text = (
                text if not context else f"{text} in context of {context}"
            )
            if query_text not in self.embedding_cache:
                with torch.no_grad():
                    self.embedding_cache[query_text] = self.embed_model.encode(
                        query_text,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
            query_emb = self.embedding_cache[query_text]
            sims = cosine_similarity(
                [query_emb], self.ontology_embedding_matrix
            )[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] > self.similarity_threshold:
                return self.ontology_concepts_list[best_idx]
            return None
        except Exception:
            return None
        finally:
            self._trim_embedding_cache()

    def _context_disambiguation(self, text: str, context: str) -> Optional[str]:
        context_lower = context.lower()
        lib_indicators = [
            'battery', 'li', 'lithium', 'electrode', 'cathode', 'anode',
            'capacity', 'energy density', 'cycle life', 'charge', 'discharge',
            'sei', 'impedance', 'voltage', 'current'
        ]
        if any(ind in context_lower for ind in lib_indicators):
            if 'energy' in text and 'density' in text:
                return "energy_density"
            if 'capacity' in text:
                return "specific_capacity"
            if 'cycle' in text and 'life' in text:
                return "cycle_life"
            if 'impedance' in text or 'resistance' in text:
                return "impedance_growth"
            if 'sei' in text:
                return "sei_formation"
        return None

    def find_equivalent_concepts(
        self, concepts: List[str]
    ) -> Dict[str, str]:
        equivalence_map: Dict[str, str] = {}
        for concept in concepts:
            canonical = self.resolve(concept)
            if canonical:
                equivalence_map[concept] = canonical
            else:
                equivalence_map[concept] = concept
        return equivalence_map

    def compute_semantic_similarity(
        self, concept1: str, concept2: str
    ) -> float:
        c1 = self.resolve(concept1) or concept1
        c2 = self.resolve(concept2) or concept2
        if c1 == c2:
            return 1.0
        if (
            c2 in self.ontology.get_hypernyms(c1)
            or c1 in self.ontology.get_hypernyms(c2)
        ):
            return 0.9
        if (
            c2 in self.ontology.get_hyponyms(c1)
            or c1 in self.ontology.get_hyponyms(c2)
        ):
            return 0.9
        try:
            with torch.no_grad():
                emb1 = self.embed_model.encode(
                    c1, show_progress_bar=False, convert_to_numpy=True
                )
                emb2 = self.embed_model.encode(
                    c2, show_progress_bar=False, convert_to_numpy=True
                )
            return float(cosine_similarity([emb1], [emb2])[0][0])
        except Exception:
            return 0.0


# ============================================================================
# ENHANCED CONCEPT EXTRACTOR (Li‑ion Battery)
# ============================================================================
class EnhancedConceptExtractor:
    def __init__(
        self,
        ontology: DomainOntology,
        resolver: AdvancedConceptResolver,
        store_contexts: bool = False,
        store_documents: bool = True,
    ) -> None:
        self.ontology = ontology
        self.resolver = resolver
        self.concept_frequencies: Dict[str, int] = defaultdict(int)
        self.store_contexts = store_contexts
        self.store_documents = store_documents
        self.concept_contexts: Dict[str, List[str]] = defaultdict(list)
        self.document_concepts: Dict[int, List[str]] = defaultdict(list)
        self._build_extraction_patterns()
        all_keywords = self._get_all_keywords()
        if all_keywords:
            sorted_keywords = sorted(all_keywords, key=len, reverse=True)[:500]
            pattern = r'\b(' + '|'.join(
                re.escape(k) for k in sorted_keywords
            ) + r')\b'
            self._keyword_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self._keyword_regex = None

    def _build_extraction_patterns(self) -> None:
        # Li-ion specific patterns
        self.material_patterns = [
            r'\bnmc811\b', r'\blini0\.8mn0\.1co0\.1o2\b', r'\bnmc622\b', r'\bnmc532\b', r'\bnmc333\b',
            r'\blfp\b', r'\blifepo4\b', r'\blco\b', r'\blicoo2\b', r'\bnca\b', r'\blinicoalo2\b',
            r'\bgraphite\b', r'\bsilicon\s+anode\b', r'\bsi\s+anode\b', r'\bsiox\b', r'\blithium\s+metal\b',
            r'\blto\b', r'\bli4ti5o12\b', r'\bliquid\s+electrolyte\b', r'\bsolid\s+electrolyte\b',
            r'\bfec\b', r'\bvc\b', r'\bvinylene\s+carbonate\b', r'\bfluoroethylene\s+carbonate\b'
        ]
        self.process_patterns = [
            r'\bcalendering\b', r'\bmechanical\s+pressing\b', r'\bslot[-\s]die\s+coating\b',
            r'\bslurry\s+coating\b', r'\bdry\s+electrode\b', r'\bsolvent[-\s]free\b',
            r'\bdoping\b', r'\bsurface\s+coating\b', r'\bprotective\s+coating\b',
            r'\bprelithiation\b', r'\blithium\s+pre[-\s]doping\b', r'\bformation\s+cycling\b'
        ]
        self.property_patterns = [
            r'\benergy\s+density\b', r'\bspecific\s+energy\b', r'\bwh/kg\b',
            r'\bpower\s+density\b', r'\bspecific\s+power\b', r'\bw/kg\b',
            r'\bspecific\s+capacity\b', r'\bmah/g\b', r'\bcapacity\b',
            r'\bcycle\s+life\b', r'\bcycling\s+stability\b', r'\bcoulombic\s+efficiency\b',
            r'\brate\s+capability\b', r'\bfast\s+charging\b', r'\bthermal\s+stability\b',
            r'\bionic\s+conductivity\b', r'\belectronic\s+conductivity\b',
            r'\belectrode\s+density\b', r'\bporosity\b', r'\bvolumetric\s+energy\s+density\b'
        ]
        self.phenomena_patterns = [
            r'\bcapacity\s+fade\b', r'\bcapacity\s+loss\b', r'\bdegradation\b',
            r'\bsei\s+formation\b', r'\bsolid\s+electrolyte\s+interphase\b',
            r'\bdendrite\s+growth\b', r'\blithium\s+whiskers\b',
            r'\bthermal\s+runaway\b', r'\bimpedance\s+growth\b',
            r'\btransition\s+metal\s+dissolution\b', r'\bcation\s+mixing\b',
            r'\bparticle\s+cracking\b', r'\bparticle\s+fracture\b'
        ]
        self.param_patterns = [
            r'\belectrode\s+thickness\b', r'\bcoating\s+thickness\b', r'\bt_e\b',
            r'\bn/p\s+ratio\b', r'\bnegative/positive\s+capacity\s+ratio\b',
            r'\bc[-\s]rate\b', r'\bcurrent\s+density\b',
            r'\btemperature\b', r'\bpressure\b', r'\bstack\s+pressure\b'
        ]
        self.method_patterns = [
            r'\beis\b', r'\belectrochemical\s+impedance\s+spectroscopy\b',
            r'\bdq/dv\b', r'\bdifferential\s+voltage\s+analysis\b',
            r'\bgalvanostatic\s+cycling\b', r'\bconstant\s+current\s+cycling\b',
            r'\bcoin\s+cell\b', r'\bcr2032\b', r'\bpouch\s+cell\b'
        ]

        self.all_patterns = (
            self.material_patterns + self.process_patterns + self.property_patterns +
            self.phenomena_patterns + self.param_patterns + self.method_patterns
        )
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.all_patterns
        ]
        self.compiled_cause_patterns = [
            re.compile(r'\b(increase|decrease|enhance|reduce)\w*\s+(?:in|of)\s+([\w\s-]+?)\s+(?:lead[s]?|result[s]?|cause[s]?)\s+(?:to|in)?\s+([\w\s-]+?)\b', re.I),
        ]

    @timed
    def extract_from_text(self, text: str, doc_id: int = 0, allowed_concepts: Optional[Set[str]] = None) -> List[str]:
        concepts: Set[str] = set()
        text_lower = text.lower()

        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = (
                        match[0] if match[0]
                        else (match[1] if len(match) > 1 else match[0])
                    )
                concept = match.lower().strip()
                if len(concept) > 3:
                    canonical = self.resolver.resolve(concept, context=text[:200])
                    if canonical:
                        if allowed_concepts is not None and canonical not in allowed_concepts:
                            continue
                        concepts.add(canonical)
                    else:
                        if allowed_concepts is not None:
                            continue
                        concepts.add(concept)

        context_concepts = self._extract_from_context_windows(text)
        if allowed_concepts is not None:
            context_concepts = {c for c in context_concepts if c in allowed_concepts}
        concepts.update(context_concepts)

        raw_concepts = set()
        for c in concepts:
            if c not in self.ontology.concepts and not self.resolver.resolve(c):
                raw_concepts.add(c)
        if raw_concepts:
            raw_list = list(raw_concepts)[:50]
            resolved_map = self.resolver.resolve_batch(raw_list, context="")
            for raw, canonical in resolved_map.items():
                if canonical:
                    if allowed_concepts is not None and canonical not in allowed_concepts:
                        continue
                    concepts.add(canonical)
                else:
                    if allowed_concepts is not None:
                        continue
                    concepts.add(raw)

        for concept in concepts:
            self.concept_frequencies[concept] += 1
            if self.store_contexts:
                self.concept_contexts[concept].append(text[:200])
        if self.store_documents:
            self.document_concepts[doc_id] = list(concepts)
        return list(concepts)

    def _extract_from_context_windows(
        self, text: str, window_size: int = 100
    ) -> Set[str]:
        if not self._keyword_regex:
            return set()
        candidate_phrases: Set[str] = set()
        text_lower = text.lower()
        match_count = 0
        for match in self._keyword_regex.finditer(text_lower):
            if match_count > 20:
                break
            match_count += 1
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            local_context = text_lower[start:end]
            phrases = re.findall(
                r'\b([a-z]+(?:[-\s][a-z]+){1,3})\b', local_context
            )
            for phrase in phrases:
                if 5 <= len(phrase) <= 40:
                    canonical = self.resolver.resolve(phrase, context=local_context)
                    if canonical:
                        candidate_phrases.add(canonical)
        return candidate_phrases

    def _get_all_keywords(self) -> Set[str]:
        keywords: Set[str] = set()
        for canonical, node in self.ontology.concepts.items():
            keywords.add(canonical)
            keywords.update(node.synonyms)
        return keywords

    def extract_relationships(self, text: str) -> List[Relationship]:
        relationships: List[Relationship] = []
        for pattern in self.compiled_cause_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) >= 2:
                    source = (
                        match[0] if isinstance(match[0], str) else match[1]
                    )
                    target = (
                        match[-1] if isinstance(match[-1], str) else match[0]
                    )
                    source_canon = self.resolver.resolve(source, context=text[:200])
                    target_canon = self.resolver.resolve(target, context=text[:200])
                    if (
                        source_canon and target_canon
                        and source_canon != target_canon
                    ):
                        rel = Relationship(
                            source=source_canon,
                            target=target_canon,
                            rel_type=RelationshipType.CAUSES,
                            confidence=0.7,
                            evidence=text[:150],
                        )
                        relationships.append(rel)
        return relationships

    def get_concept_frequencies(self) -> Dict[str, int]:
        return dict(self.concept_frequencies)

    def get_concept_contexts(self, concept: str) -> List[str]:
        return self.concept_contexts.get(concept, [])

    def get_document_concepts(self, doc_id: int) -> List[str]:
        return self.document_concepts.get(doc_id, [])


# ============================================================================
# REASONING-ENHANCED GRAPH BUILDER
# ============================================================================
class ReasoningEnhancedGraphBuilder:
    def __init__(
        self, ontology: DomainOntology, extractor: EnhancedConceptExtractor
    ) -> None:
        self.ontology = ontology
        self.extractor = extractor
        self.reasoning_paths: List[List[str]] = []
        self.inferred_edges: Set[Tuple[str, str]] = set()

    @timed
    def build_graph(
        self,
        all_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(3000)
        nx_graph = nx.Graph()

        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            freq = self.extractor.concept_frequencies.get(c, 0)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=freq,
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )

        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in all_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1

        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=count,
                cooccurrence=count,
                semantic=0,
                edge_type='cooccurrence',
                inferred=False,
            )

        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(nx_graph, valid_concepts, embed_model, config)

        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
            self._add_cause_effect_edges(nx_graph)
            self._add_hierarchical_edges(nx_graph, valid_concepts)

        self._compute_final_weights(nx_graph, config)
        return nx_graph

    def _add_semantic_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str],
        embed_model, config: Dict,
    ) -> None:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts,
                    show_progress_bar=False,
                    batch_size=64,
                    convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh:
                        if nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3:
                            nx_graph.add_edge(
                                c1, c2,
                                weight=sim * 2,
                                cooccurrence=0,
                                semantic=sim,
                                edge_type='semantic',
                                inferred=False,
                            )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")

    def _add_inferred_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for rel in self.ontology.relationships:
            if rel.source in valid_concepts and rel.target in valid_concepts:
                if not nx_graph.has_edge(rel.source, rel.target):
                    nx_graph.add_edge(
                        rel.source, rel.target,
                        weight=rel.confidence * 2,
                        cooccurrence=0,
                        semantic=rel.confidence,
                        edge_type=rel.rel_type.value,
                        inferred=True,
                        confidence=rel.confidence,
                    )
                    self.inferred_edges.add((rel.source, rel.target))
        self._infer_cross_domain_bridges(nx_graph, valid_concepts)

    def _infer_cross_domain_bridges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        material_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.MATERIAL
        ]
        property_nodes = [
            c for c in valid_concepts
            if self.ontology.get_concept_type(c) == ConceptType.PROPERTY
        ]
        for mat in material_nodes:
            for prop in property_nodes:
                if not nx_graph.has_edge(mat, prop):
                    paths = self.ontology.infer_path(mat, prop, max_depth=2)
                    if paths:
                        avg_confidence = 0.6
                        nx_graph.add_edge(
                            mat, prop,
                            weight=avg_confidence,
                            cooccurrence=0,
                            semantic=avg_confidence,
                            edge_type='bridge',
                            inferred=True,
                            path=" -> ".join(paths[0]),
                        )
                        self.inferred_edges.add((mat, prop))
                        self.reasoning_paths.append(paths[0])

    def _add_cause_effect_edges(self, nx_graph: nx.Graph) -> None:
        pass

    def _add_hierarchical_edges(
        self, nx_graph: nx.Graph, valid_concepts: List[str]
    ) -> None:
        for concept in valid_concepts:
            if concept not in self.ontology.concepts:
                continue
            node = self.ontology.concepts[concept]
            for hypernym in node.hypernyms:
                if (
                    hypernym in valid_concepts
                    and not nx_graph.has_edge(concept, hypernym)
                ):
                    nx_graph.add_edge(
                        concept, hypernym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hypernym', inferred=True,
                    )
            for hyponym in node.hyponyms:
                if (
                    hyponym in valid_concepts
                    and not nx_graph.has_edge(concept, hyponym)
                ):
                    nx_graph.add_edge(
                        concept, hyponym,
                        weight=1.0, cooccurrence=0, semantic=0.95,
                        edge_type='hyponym', inferred=True,
                    )

    def _compute_final_weights(
        self, nx_graph: nx.Graph, config: Dict
    ) -> None:
        cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.7)
        sem_weight = config.get("SEMANTIC_WEIGHT", 0.2)
        inf_weight = config.get("INFERENCE_WEIGHT", 0.1)
        for u, v, data in nx_graph.edges(data=True):
            cooc = data.get('cooccurrence', 0)
            sem = data.get('semantic', 0)
            inf = 1.0 if data.get('inferred', False) else 0
            conf = data.get('confidence', 0.5)
            data['weight'] = (
                cooc_weight * cooc
                + sem_weight * sem
                + inf_weight * inf * conf
            )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def build_query_whitelist(st_session):
    if not st_session.get('query_focused_build', False):
        return None
    analysis = st_session.get('last_query_analysis')
    if analysis is None:
        st.warning("No query analysis available – falling back to full graph.")
        return None
    whitelist = set(analysis.explicitly_mentioned)
    whitelist.update(analysis.inferred_concepts)
    whitelist.update(st_session.get('last_query_dynamic_concepts', set()))
    whitelist.update(st_session.get('last_query_bridge_concepts', {}).keys())
    return whitelist


def get_adaptive_config(num_abstracts: int) -> Dict[str, Any]:
    if num_abstracts <= 50:
        return {
            "MIN_CONCEPT_FREQ": 2, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 1, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.72, "COOCCURRENCE_WEIGHT": 0.5,
            "SEMANTIC_WEIGHT": 0.5, "CLUSTER_SIMILARITY": 0.75,
            "TOP_N_CONCEPTS": 200, "MAX_CONCEPT_LENGTH": 6,
            "INFERENCE_WEIGHT": 0.1,
        }
    elif num_abstracts <= 500:
        return {
            "MIN_CONCEPT_FREQ": 3, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 2, "USE_SEMANTIC_CLUSTERING": True,
            "SIMILARITY_THRESHOLD": 0.78, "COOCCURRENCE_WEIGHT": 0.6,
            "SEMANTIC_WEIGHT": 0.3, "CLUSTER_SIMILARITY": 0.72,
            "TOP_N_CONCEPTS": 500, "MAX_CONCEPT_LENGTH": 8,
            "INFERENCE_WEIGHT": 0.1,
        }
    else:
        return {
            "MIN_CONCEPT_FREQ": 5, "MIN_CONCEPT_LENGTH_WORDS": 2,
            "MIN_DEGREE": 3, "USE_SEMANTIC_CLUSTERING": False,
            "SIMILARITY_THRESHOLD": 0.85, "COOCCURRENCE_WEIGHT": 0.7,
            "SEMANTIC_WEIGHT": 0.2, "CLUSTER_SIMILARITY": 0.68,
            "TOP_N_CONCEPTS": 1000, "MAX_CONCEPT_LENGTH": 10,
            "INFERENCE_WEIGHT": 0.1,
        }


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = get_device()
    try:
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
    except Exception as e:
        st.error(f"Embedding model error: {e}")
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )


# ============================================================================
# Li‑ion Battery Keywords
# ============================================================================
MATERIAL_KEYWORDS = [
    "nmc811", "lini0.8mn0.1co0.1o2", "nmc622", "nmc532", "nmc333",
    "lfp", "lifepo4", "lco", "licoo2", "nca", "linicoalo2",
    "graphite", "silicon anode", "si anode", "siox", "lithium metal",
    "lto", "li4ti5o12", "liquid electrolyte", "solid electrolyte",
    "fec", "vc", "fluoroethylene carbonate", "vinylene carbonate"
]
PROPERTY_KEYWORDS = [
    "energy density", "specific energy", "wh/kg", "power density",
    "specific power", "w/kg", "specific capacity", "mah/g",
    "cycle life", "cycling stability", "coulombic efficiency",
    "rate capability", "fast charging", "thermal stability",
    "ionic conductivity", "electronic conductivity", "electrode density",
    "porosity", "volumetric energy density"
]
PHENOMENON_KEYWORDS = [
    "capacity fade", "capacity loss", "degradation", "sei formation",
    "solid electrolyte interphase", "dendrite growth", "lithium whiskers",
    "thermal runaway", "impedance growth", "transition metal dissolution",
    "cation mixing", "particle cracking", "particle fracture"
]
PROCESS_KEYWORDS = [
    "calendering", "mechanical pressing", "slot-die coating",
    "slurry coating", "dry electrode", "solvent-free", "doping",
    "surface coating", "protective coating", "prelithiation",
    "lithium pre-doping", "formation cycling"
]
PARAM_KEYWORDS = [
    "electrode thickness", "coating thickness", "n/p ratio",
    "negative/positive capacity ratio", "c-rate", "current density",
    "temperature", "pressure", "stack pressure"
]
METHOD_KEYWORDS = [
    "eis", "electrochemical impedance spectroscopy", "dq/dv",
    "differential voltage analysis", "galvanostatic cycling",
    "constant current cycling", "coin cell", "cr2032", "pouch cell"
]

ALL_DOMAIN_KEYWORDS = (
    MATERIAL_KEYWORDS + PROPERTY_KEYWORDS + PHENOMENON_KEYWORDS +
    PROCESS_KEYWORDS + PARAM_KEYWORDS + METHOD_KEYWORDS
)

LIB_PATTERNS = [
    r'\bnmc811\b', r'\bnmc622\b', r'\bnmc532\b', r'\bnmc333\b',
    r'\blfp\b', r'\blco\b', r'\bnca\b',
    r'\bgraphite\b', r'\bsilicon\s+anode\b', r'\bsi\s+anode\b', r'\bsiox\b',
    r'\blithium\s+metal\b', r'\blto\b',
    r'\bliquid\s+electrolyte\b', r'\bsolid\s+electrolyte\b',
    r'\bfec\b', r'\bvc\b',
    r'\bcalendering\b', r'\bslot[-\s]die\s+coating\b',
    r'\bdry\s+electrode\b', r'\bdoping\b', r'\bsurface\s+coating\b',
    r'\bprelithiation\b', r'\bformation\s+cycling\b',
    r'\benergy\s+density\b', r'\bspecific\s+capacity\b', r'\bcycle\s+life\b',
    r'\bcoulombic\s+efficiency\b', r'\brate\s+capability\b',
    r'\bthermal\s+stability\b', r'\bionic\s+conductivity\b',
    r'\belectrode\s+density\b', r'\bporosity\b',
    r'\bcapacity\s+fade\b', r'\bsei\s+formation\b',
    r'\bdendrite\s+growth\b', r'\bthermal\s+runaway\b',
    r'\bimpedance\s+growth\b', r'\btransition\s+metal\s+dissolution\b',
    r'\bcation\s+mixing\b', r'\bparticle\s+cracking\b',
    r'\belectrode\s+thickness\b', r'\bn/p\s+ratio\b', r'\bc[-\s]rate\b',
    r'\btemperature\b', r'\bpressure\b',
    r'\beis\b', r'\bdq/dv\b', r'\bgalvanostatic\s+cycling\b',
    r'\bcoin\s+cell\b', r'\bpouch\s+cell\b'
]

LIB_DESCRIPTOR_MAPPING = {
    r'nmc811|nmc622|nmc532|nmc333|lfp|lco|nca|graphite|silicon|siox|lithium metal|lto': 'material',
    r'calendering|slot-die coating|dry electrode|doping|surface coating|prelithiation|formation cycling': 'process',
    r'energy density|specific capacity|cycle life|coulombic efficiency|rate capability|thermal stability|ionic conductivity|electronic conductivity|electrode density|porosity|volumetric energy density': 'property',
    r'capacity fade|sei formation|dendrite growth|thermal runaway|impedance growth|transition metal dissolution|cation mixing|particle cracking': 'phenomenon',
    r'electrode thickness|n/p ratio|c-rate|temperature|pressure': 'parameter',
    r'eis|dq/dv|galvanostatic cycling|coin cell|pouch cell': 'method',
    r'general': 'general'
}

def normalize_lib_concept(concept: str) -> str:
    concept = concept.lower().strip()
    mapping = {
        r'nmc811|lni0\.8mn0\.1co0\.1o2': 'nmc811',
        r'nmc622': 'nmc622',
        r'nmc532': 'nmc532',
        r'nmc333': 'nmc333',
        r'lfp|lifepo4': 'lfp',
        r'lco|licoo2': 'lco',
        r'nca|linicoalo2': 'nca',
        r'graphite': 'graphite',
        r'silicon\s+anode|si\s+anode': 'silicon',
        r'siox|silicon\s+oxide': 'siox',
        r'lithium\s+metal|li\s+metal': 'lithium_metal',
        r'lto|li4ti5o12': 'lto',
        r'liquid\s+electrolyte': 'liquid_electrolyte',
        r'solid\s+electrolyte': 'solid_electrolyte',
        r'fec|fluoroethylene\s+carbonate': 'fec',
        r'vc|vinylene\s+carbonate': 'vc',
        r'calendering|mechanical\s+pressing': 'calendering',
        r'slot[-\s]die\s+coating|slurry\s+coating': 'slot_die_coating',
        r'dry\s+electrode|solvent[-\s]free': 'dry_electrode_processing',
        r'doping': 'doping',
        r'surface\s+coating|protective\s+coating': 'surface_coating',
        r'prelithiation|lithium\s+pre[-\s]doping': 'prelithiation',
        r'formation\s+cycling|initial\s+cycling': 'formation',
        r'energy\s+density|specific\s+energy': 'energy_density',
        r'power\s+density|specific\s+power': 'power_density',
        r'specific\s+capacity|capacity': 'specific_capacity',
        r'cycle\s+life|cycling\s+stability': 'cycle_life',
        r'coulombic\s+efficiency': 'coulombic_efficiency',
        r'rate\s+capability|fast\s+charging': 'rate_capability',
        r'thermal\s+stability|thermal\s+safety': 'thermal_stability',
        r'ionic\s+conductivity': 'ionic_conductivity',
        r'electronic\s+conductivity': 'electronic_conductivity',
        r'electrode\s+density|compaction\s+density': 'electrode_density',
        r'porosity': 'porosity',
        r'volumetric\s+energy\s+density': 'volumetric_energy_density',
        r'capacity\s+fade|capacity\s+loss|degradation': 'capacity_fade',
        r'sei\s+formation|solid\s+electrolyte\s+interphase': 'sei_formation',
        r'dendrite\s+growth|lithium\s+whiskers': 'dendrite_growth',
        r'thermal\s+runaway': 'thermal_runaway',
        r'impedance\s+growth|resistance\s+increase': 'impedance_growth',
        r'transition\s+metal\s+dissolution': 'transition_metal_dissolution',
        r'cation\s+mixing|cation\s+disorder': 'cation_mixing',
        r'particle\s+cracking|particle\s+fracture': 'particle_cracking',
        r'electrode\s+thickness|coating\s+thickness': 'electrode_thickness',
        r'n/p\s+ratio|negative/positive\s+capacity\s+ratio': 'np_ratio',
        r'c[-\s]rate|current\s+density': 'c_rate',
        r'temperature': 'temperature',
        r'pressure|stack\s+pressure': 'pressure',
        r'eis|electrochemical\s+impedance\s+spectroscopy': 'eis',
        r'dq/dv|differential\s+voltage\s+analysis': 'dqdv',
        r'galvanostatic\s+cycling|constant\s+current\s+cycling': 'galvanostatic_cycling',
        r'coin\s+cell|cr2032': 'coin_cell',
        r'pouch\s+cell': 'pouch_cell'
    }
    for pattern, canonical in mapping.items():
        if re.search(pattern, concept, re.I):
            return canonical
    return concept

def is_valid_lib_concept(concept: str) -> bool:
    concept_lower = concept.lower()
    has_domain = any(kw.lower() in concept_lower for kw in ALL_DOMAIN_KEYWORDS)
    has_pattern = any(re.search(p, concept, re.I) for p in LIB_PATTERNS)
    generic = {
        'study', 'analysis', 'effect', 'role', 'investigation', 'research',
        'method', 'approach', 'paper', 'work', 'using', 'based', 'novel',
        'material', 'system', 'sample', 'specimen', 'structure', 'surface'
    }
    has_generic = any(term in concept_lower.split() for term in generic)
    words = concept.split()
    if len(words) < 2 or len(words) > 10:
        return False
    return (has_domain or has_pattern) and not has_generic

def extract_concepts_from_text(text: str) -> List[str]:
    concepts: Set[str] = set()
    text_lower = text.lower()
    for pattern in LIB_PATTERNS:
        matches = re.findall(pattern, text, re.I)
        for m in matches:
            concept = m.lower().strip().rstrip('.').rstrip(',')
            if len(concept.split()) >= 1 and len(concept) > 3:
                concepts.add(concept)
    noun_pattern = (
        r'\b(?:[a-z]+(?:[-\s]?[a-z]+){0,2}[-\s]?)?'
        r'(?:nmc|lfp|graphite|silicon|capacity|energy|cycle|coulombic|rate|thermal|ionic|electronic|porosity|fade|sei|dendrite|impedance|thickness|ratio|temperature|pressure)\b'
    )
    matches = re.findall(noun_pattern, text, re.I)
    for m in matches:
        concept = m.lower().strip()
        if is_valid_lib_concept(concept):
            concepts.add(concept)
    for keyword in ALL_DOMAIN_KEYWORDS:
        for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text_lower[start:end]
            context_phrases = re.findall(
                r'\b([a-z]+(?:\s+[a-z]+){1,3})\s+'
                r'(?:of|for|in|with|using|via|through|by|to|and|or)\s+'
                + re.escape(keyword) + r'\b',
                context,
            )
            for phrase in context_phrases:
                concept = f"{phrase.strip()} {keyword}"
                if is_valid_lib_concept(concept):
                    concepts.add(concept)
    return list(concepts)


def extract_concepts_from_abstracts(
    df: pd.DataFrame, text_columns: List[str]
) -> Tuple[List[List[str]], List[Dict]]:
    all_concepts: List[List[str]] = []
    all_metrics: List[Dict] = []
    for idx, row in df.iterrows():
        combined_text = ""
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                combined_text += " " + str(row[col])
        metrics: Dict[str, Any] = {}
        # Extract metrics for Li-ion literature
        capacity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g)', combined_text, re.I)
        if capacity_matches:
            metrics['specific_capacity_mAh_g'] = [float(m) for m in capacity_matches]
        energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg)', combined_text, re.I)
        if energy_matches:
            metrics['energy_density_Wh_kg'] = [float(m) for m in energy_matches]
        cycle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:cycles)', combined_text, re.I)
        if cycle_matches:
            metrics['cycle_life_cycles'] = [float(m) for m in cycle_matches]
        impedance_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ohm|Ω)', combined_text, re.I)
        if impedance_matches:
            metrics['impedance_Ohm'] = [float(m) for m in impedance_matches]
        thickness_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|um)', combined_text, re.I)
        if thickness_matches:
            metrics['electrode_thickness_um'] = [float(m) for m in thickness_matches]
        temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', combined_text, re.I)
        if temp_matches:
            metrics['temperature_C'] = [float(m) for m in temp_matches]
        all_metrics.append(metrics)
        concepts = extract_concepts_from_text(combined_text)
        normalized = [normalize_lib_concept(c) for c in concepts]
        all_concepts.append(normalized)
    return all_concepts, all_metrics


def cluster_similar_concepts(
    valid_concepts: List[str], embed_model, similarity_threshold: float = 0.75
) -> Tuple[List[str], Dict[str, str]]:
    if len(valid_concepts) < 5:
        return valid_concepts, {c: c for c in valid_concepts}
    try:
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts,
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            linkage='average',
            metric='cosine',
        ).fit(embeddings)
        cluster_members: Dict[int, List[str]] = defaultdict(list)
        concept_to_cluster: Dict[str, int] = {}
        for idx, label in enumerate(clustering.labels_):
            concept = valid_concepts[idx]
            cluster_members[label].append(concept)
            concept_to_cluster[concept] = label
        cluster_representatives: Dict[int, str] = {}
        for label, members in cluster_members.items():
            def score(m):
                domain_hits = sum(
                    1 for kw in ALL_DOMAIN_KEYWORDS if kw.lower() in m.lower()
                )
                return (domain_hits, -len(m))
            representative = max(members, key=score)
            cluster_representatives[label] = representative
        final_mapping = {
            c: cluster_representatives[label]
            for c, label in concept_to_cluster.items()
        }
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()
        return list(cluster_representatives.values()), final_mapping
    except Exception as e:
        st.warning(f"Semantic clustering skipped: {e}")
        return valid_concepts, {c: c for c in valid_concepts}


def normalize_and_filter_concepts(
    all_concepts: List[List[str]], config: Dict
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, List[int]]]:
    concept_counts: Dict[str, int] = defaultdict(int)
    concept_abstract_map: Dict[str, List[int]] = defaultdict(list)
    for doc_idx, concepts in enumerate(all_concepts):
        seen_in_doc: Set[str] = set()
        for c in concepts:
            if c not in seen_in_doc and is_valid_lib_concept(c):
                concept_counts[c] += 1
                concept_abstract_map[c].append(doc_idx)
                seen_in_doc.add(c)
    min_freq = config.get("MIN_CONCEPT_FREQ", 5)
    min_words = config.get("MIN_CONCEPT_LENGTH_WORDS", 2)
    max_words = config.get("MAX_CONCEPT_LENGTH", 10)
    valid_concepts = [
        c for c, cnt in concept_counts.items()
        if cnt >= min_freq and min_words <= len(c.split()) <= max_words
    ]
    if config.get("USE_SEMANTIC_CLUSTERING", False) and len(valid_concepts) > 50:
        try:
            embed_model = load_embedding_model()
            valid_concepts, concept_to_cluster = cluster_similar_concepts(
                valid_concepts, embed_model,
                similarity_threshold=config.get("CLUSTER_SIMILARITY", 0.72),
            )
            new_abstract_map: Dict[str, List[int]] = defaultdict(list)
            for orig_concept, docs in concept_abstract_map.items():
                clustered = concept_to_cluster.get(orig_concept, orig_concept)
                if clustered in valid_concepts:
                    new_abstract_map[clustered].extend(docs)
            concept_abstract_map = new_abstract_map
        except Exception as e:
            st.warning(f"Semantic clustering skipped: {e}")
    valid_concepts = sorted(
        valid_concepts, key=lambda c: concept_counts[c], reverse=True
    )
    top_n = config.get("TOP_N_CONCEPTS", 1000)
    if len(valid_concepts) > top_n:
        valid_concepts = valid_concepts[:top_n]
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return valid_concepts, concept_to_id, id_to_concept, concept_abstract_map


def abstract_concepts_to_categories(concepts: List[str]) -> Dict[str, str]:
    concept_to_abstract: Dict[str, str] = {}
    for concept in concepts:
        matched = False
        for pattern, category in LIB_DESCRIPTOR_MAPPING.items():
            if re.search(pattern, concept, re.I):
                concept_to_abstract[concept] = category
                matched = True
                break
        if not matched:
            if any(re.search(p, concept, re.I) for p in [r'nmc', r'lfp', r'graphite', r'silicon']):
                concept_to_abstract[concept] = 'material'
            elif any(re.search(p, concept, re.I) for p in [r'calendering', r'coating', r'doping', r'formation']):
                concept_to_abstract[concept] = 'process'
            elif any(re.search(p, concept, re.I) for p in [r'energy', r'capacity', r'cycle', r'rate', r'thermal']):
                concept_to_abstract[concept] = 'property'
            elif any(re.search(p, concept, re.I) for p in [r'fade', r'sei', r'dendrite', r'impedance']):
                concept_to_abstract[concept] = 'phenomenon'
            elif any(re.search(p, concept, re.I) for p in [r'thickness', r'ratio', r'temperature']):
                concept_to_abstract[concept] = 'parameter'
            elif any(re.search(p, concept, re.I) for p in [r'eis', r'dq/dv', r'cycling', r'cell']):
                concept_to_abstract[concept] = 'method'
            else:
                concept_to_abstract[concept] = 'general'
    return concept_to_abstract


# ============================================================================
# CONCEPT DISTILLATION (Memory-safe)
# ============================================================================
def compute_concept_distillation(
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    all_texts: Union[List[str], Dict[int, str]],
    max_docs_per_concept: int = 30,
) -> pd.DataFrame:
    """Memory-safe concept distillation (v6.1 rewrite)."""
    distill_data: List[Dict[str, Any]] = []
    doc_corpus: List[str] = []

    texts_is_dict = isinstance(all_texts, dict)
    n_texts = len(all_texts)

    for c in valid_concepts:
        doc_indices = concept_abstract_map.get(c, [])
        if max_docs_per_concept and len(doc_indices) > max_docs_per_concept:
            doc_indices = doc_indices[:max_docs_per_concept]
        if texts_is_dict:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if i in all_texts
            ])
        else:
            doc_text = " ".join([
                all_texts[i] for i in doc_indices
                if isinstance(i, int) and 0 <= i < n_texts
            ])
        doc_corpus.append(doc_text)

    tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        stop_words='english', max_features=2000,
    )
    try:
        if any(doc_corpus) and any(t.strip() for t in doc_corpus):
            tfidf_matrix = tfidf.fit_transform(doc_corpus)
            tfidf_scores = tfidf_matrix.max(axis=1).A1
            del tfidf_matrix
        else:
            tfidf_scores = np.ones(len(valid_concepts))
    except Exception:
        tfidf_scores = np.ones(len(valid_concepts))
    gc.collect()

    embed_model = load_embedding_model()

    for i, c in enumerate(valid_concepts):
        freq = len(concept_abstract_map.get(c, []))
        semantic_density = float(tfidf_scores[i])
        coherence = 0.0
        if freq > 1 and doc_corpus[i].strip():
            try:
                words = doc_corpus[i].split()[:20]
                with torch.no_grad():
                    concept_embeddings = embed_model.encode(
                        words, show_progress_bar=False,
                        batch_size=16, convert_to_numpy=True,
                    )
                if len(concept_embeddings) > 1:
                    sim_matrix = cosine_similarity(concept_embeddings)
                    coherence = float(np.mean(
                        sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                    ))
                    del sim_matrix
                del concept_embeddings, words
                gc.collect()
                if torch.cuda.is_available():
                    maybe_empty_cache()
            except Exception:
                coherence = 0.0
        distill_data.append({
            "concept": c,
            "frequency": freq,
            "tfidf_weight": semantic_density,
            "semantic_density": semantic_density,
            "coherence_score": float(coherence),
            "distillation_efficiency": float(
                semantic_density * np.log1p(freq) * (0.5 + 0.5 * coherence)
            ),
        })

    del doc_corpus
    gc.collect()
    return pd.DataFrame(distill_data).sort_values(
        "distillation_efficiency", ascending=False
    )


# ============================================================================
# LEGACY GRAPH CONSTRUCTION (FALLBACK)
# ============================================================================
def build_hybrid_graph(
    all_concepts: List[List[str]],
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    embed_model=None,
    config: Dict = None,
    ontology: DomainOntology = None,
) -> nx.Graph:
    if config is None:
        config = get_adaptive_config(3000)
    nx_graph = nx.Graph()
    for c in valid_concepts:
        concept_type = ontology.get_concept_type(c).value if ontology else 'general'
        definition = ontology.get_definition(c) if ontology else ''
        nx_graph.add_node(
            c, frequency=0, concept_type=concept_type, definition=definition,
        )
    for concepts in all_concepts:
        valid_in_doc = [c for c in concepts if c in concept_to_id]
        for i in range(len(valid_in_doc)):
            for j in range(i + 1, len(valid_in_doc)):
                u, v = valid_in_doc[i], valid_in_doc[j]
                if nx_graph.has_edge(u, v):
                    nx_graph[u][v]['weight'] += 1
                    nx_graph[u][v]['cooccurrence'] += 1
                else:
                    nx_graph.add_edge(
                        u, v, weight=1, cooccurrence=1, semantic=0,
                        edge_type='cooccurrence',
                    )
                nx_graph.nodes[u]['frequency'] = (
                    nx_graph.nodes[u].get('frequency', 0) + 1
                )
                nx_graph.nodes[v]['frequency'] = (
                    nx_graph.nodes[v].get('frequency', 0) + 1
                )
    if embed_model and len(valid_concepts) >= 10:
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=64, convert_to_numpy=True,
                )
            sim_matrix = cosine_similarity(embeddings)
            sim_thresh = config.get("SIMILARITY_THRESHOLD", 0.85)
            for i, c1 in enumerate(valid_concepts):
                for j, c2 in enumerate(valid_concepts[i + 1:], start=i + 1):
                    if c1 == c2 or nx_graph.has_edge(c1, c2):
                        continue
                    sim = sim_matrix[i][j]
                    if sim > sim_thresh and (
                        nx_graph.degree(c1) < 3 or nx_graph.degree(c2) < 3
                    ):
                        nx_graph.add_edge(
                            c1, c2, weight=sim * 2, cooccurrence=0,
                            semantic=sim, edge_type='semantic',
                        )
            del embeddings, sim_matrix
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_cache()
        except Exception as e:
            st.warning(f"Semantic edge addition skipped: {e}")
    cooc_weight = config.get("COOCCURRENCE_WEIGHT", 0.9)
    sem_weight = config.get("SEMANTIC_WEIGHT", 0.1)
    for u, v, data in nx_graph.edges(data=True):
        cooc = data.get('cooccurrence', 0)
        sem = data.get('semantic', 0)
        data['weight'] = cooc_weight * cooc + sem_weight * sem
    return nx_graph


def sample_edges_for_training(
    nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_to_id: Dict[str, int],
    config: Dict = None,
    memory_safe: bool = False,
) -> Tuple[List[Tuple], List[Tuple]]:
    pos_pairs = [(concept_to_id[u], concept_to_id[v]) for u, v in nx_graph.edges()]
    neg_pairs: List[Tuple[int, int]] = []
    n_nodes = len(valid_concepts)
    if n_nodes < 3:
        return pos_pairs, neg_pairs

    max_possible_negs = n_nodes * (n_nodes - 1) // 2 - nx_graph.number_of_edges()
    if max_possible_negs <= 0:
        return pos_pairs, neg_pairs

    if memory_safe:
        target_negs = min(len(pos_pairs) * 2 if pos_pairs else 30, 2000)
    else:
        target_negs = min(len(pos_pairs) * 3 if pos_pairs else 30, 5000)

    target_negs = min(target_negs, max_possible_negs)

    attempts = 0
    max_attempts = 50000
    if memory_safe:
        path_lengths = {}
    else:
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
            neg_pairs.append((int(u_idx), int(v_idx)))
        elif dist == 999 and np.random.rand() < 0.1:
            neg_pairs.append((int(u_idx), int(v_idx)))
        attempts += 1

    attempts = 0
    while len(neg_pairs) < target_negs and attempts < max_attempts:
        u_idx, v_idx = np.random.choice(n_nodes, 2, replace=False)
        if not nx_graph.has_edge(valid_concepts[u_idx], valid_concepts[v_idx]):
            neg_pairs.append((int(u_idx), int(v_idx)))
        attempts += 1

    return pos_pairs, neg_pairs


class SparseGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, adj_indices, adj_values, num_nodes, h,
        pos_u, pos_v, neg_u, neg_v,
    ):
        A = sparse.FloatTensor(
            adj_indices, adj_values, torch.Size([num_nodes, num_nodes])
        ).to(h.device)
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1)
        deg_inv = 1.0 / deg
        h1 = F.relu(
            self.lin1(torch.sparse.mm(A, h) * deg_inv.unsqueeze(1))
        )
        h2 = self.lin2(torch.sparse.mm(A, h1) * deg_inv.unsqueeze(1))
        pos_scores = self.decoder(
            torch.cat([h2[pos_u], h2[pos_v]], dim=1)
        ).squeeze(1)
        neg_scores = self.decoder(
            torch.cat([h2[neg_u], h2[neg_v]], dim=1)
        ).squeeze(1)
        return pos_scores, neg_scores, h2


def train_gnn(
    node_features, nx_graph, concept_to_id, pos_pairs, neg_pairs,
    progress_callback=None, epochs: int = 50, lr: float = 1e-3,
):
    target_device = torch.device("cpu")

    num_nodes = len(concept_to_id)
    in_dim = node_features.shape[1] if node_features.numel() > 0 else 384

    if not pos_pairs:
        nodes = list(concept_to_id.values())
        if len(nodes) >= 2:
            pos_pairs = [(nodes[0], nodes[1])]
        else:
            raise ValueError("Cannot train GNN with fewer than 2 concepts")

    unique_edges = {(min(u, v), max(u, v)) for u, v in pos_pairs}
    src_adj = torch.tensor([u for u, v in unique_edges], dtype=torch.long, device=target_device)
    dst_adj = torch.tensor([v for u, v in unique_edges], dtype=torch.long, device=target_device)
    adj_indices = torch.stack([src_adj, dst_adj], dim=0)
    adj_values = torch.ones(adj_indices.shape[1], dtype=torch.float32, device=target_device)

    node_features = node_features.to(target_device)

    pos_u = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=target_device)
    pos_v = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=target_device)

    neg_u = torch.tensor([n[0] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else pos_u[:1]
    neg_v = torch.tensor([n[1] for n in neg_pairs], dtype=torch.long, device=target_device) if neg_pairs else pos_v[:1]

    model = SparseGraphSAGE(in_dim=in_dim, hidden_dim=128).to(target_device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pos_out, neg_out, _ = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u, pos_v, neg_u, neg_v,
        )

        pos_loss = criterion(pos_out, torch.ones_like(pos_out))
        neg_loss = criterion(neg_out, torch.zeros_like(neg_out)) if neg_pairs else torch.tensor(0.0, device=target_device)
        loss = 0.5 * (pos_loss + neg_loss)

        loss.backward()
        optimizer.step()
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, loss.item())

    model.eval()
    with torch.no_grad():
        _, _, final_embeddings = model(
            adj_indices, adj_values, num_nodes, node_features,
            pos_u[:1], pos_v[:1], neg_u[:1], neg_v[:1],
        )
    return model, final_embeddings.cpu(), adj_indices.cpu(), adj_values.cpu()


# ============================================================================
# RESEARCH DIRECTION SCORING
# ============================================================================
def compute_research_direction_scores(
    model, node_features, final_emb, nx_graph,
    valid_concepts, concept_properties, ridge,
    embed_model, n_samples: int = 5000,
) -> pd.DataFrame:
    n_concepts = len(valid_concepts)
    if n_concepts < 3:
        return pd.DataFrame()
    u_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    v_ids = np.random.randint(
        n_concepts, size=min(n_samples, n_concepts * 5)
    )
    candidate_pairs: List[Tuple[int, int, str, str]] = []
    for u_idx, v_idx in zip(u_ids, v_ids):
        if u_idx == v_idx:
            continue
        u_c, v_c = valid_concepts[u_idx], valid_concepts[v_idx]
        if nx_graph.has_edge(u_c, v_c):
            continue
        candidate_pairs.append((int(u_idx), int(v_idx), u_c, v_c))
    if not candidate_pairs:
        return pd.DataFrame()
    u_tensor = torch.tensor([p[0] for p in candidate_pairs], dtype=torch.long)
    v_tensor = torch.tensor([p[1] for p in candidate_pairs], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pair_features = torch.cat(
            [final_emb[u_tensor], final_emb[v_tensor]], dim=1
        )
        gnn_logits = model.decoder(pair_features).squeeze(1)
        gnn_scores = torch.sigmoid(gnn_logits).numpy()
    with torch.no_grad():
        emb_np = embed_model.encode(
            valid_concepts, show_progress_bar=False,
            batch_size=64, convert_to_numpy=True,
        )
    cos_sims = np.sum(
        emb_np[u_tensor.numpy()] * emb_np[v_tensor.numpy()], axis=1
    )
    results: List[Dict[str, Any]] = []
    for i, (u_idx, v_idx, u_c, v_c) in enumerate(candidate_pairs):
        p_u = concept_properties.get(u_c, 0)
        p_v = concept_properties.get(v_c, 0)
        expected_improvement = 0
        if ridge is not None and (p_u > 0 or p_v > 0):
            try:
                expected_improvement = float(
                    ridge.predict([[p_u, p_v, 1.0]])[0]
                )
            except Exception:
                expected_improvement = max(p_u, p_v) * 1.05
        semantic_novelty = 1.0 - cos_sims[i]
        feasibility = (
            np.exp(-0.5 * semantic_novelty)
            * (1.0 if (p_u > 0 or p_v > 0) else 0.6)
        )
        alpha = {'gnn': 0.4, 'novelty': 0.3, 'gain': 0.2, 'feas': -0.1}
        norm_gain = (
            np.clip((expected_improvement - 50) / 200, 0, 1)
            if expected_improvement > 0 else 0
        )
        D_uv = (
            alpha['gnn'] * gnn_scores[i]
            + alpha['novelty'] * semantic_novelty
            + alpha['gain'] * norm_gain
            + alpha['feas'] * (1.0 - feasibility)
        )
        results.append({
            'concept_u': u_c, 'concept_v': v_c,
            'gnn_affinity': float(gnn_scores[i]),
            'semantic_novelty': float(semantic_novelty),
            'expected_property_gain': expected_improvement,
            'feasibility_score': float(feasibility),
            'composite_score': float(D_uv),
        })
    df = pd.DataFrame(results).sort_values('composite_score', ascending=False)
    del emb_np
    gc.collect()
    if torch.cuda.is_available():
        maybe_empty_cache()
    return df.head(min(100, len(df)))


# ============================================================================
# MATHEMATICAL VALIDATION
# ============================================================================
def validate_graph_metrics(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
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
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        if len(valid_concepts) >= 3:
            labels = np.zeros(len(valid_concepts))
            for i, c in enumerate(valid_concepts):
                for idx, comm in enumerate(
                    partition if 'partition' in locals() else [[]]
                ):
                    if c in comm:
                        labels[i] = idx
                        break
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
        else:
            metrics["silhouette_score"] = 0.0
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()
    except Exception:
        metrics["silhouette_score"] = 0.0
    weights = [d.get('weight', 1) for _, _, d in nx_graph.edges(data=True)]
    if len(weights) > 10:
        p_values = []
        for w in weights[:50]:
            permuted = np.random.permutation(weights)
            p_values.append(np.sum(permuted >= w) / len(weights))
        metrics["edge_significance_p_mean"] = float(np.mean(p_values))
        metrics["edge_significant_count"] = int(
            sum(1 for p in p_values if p < 0.05)
        )
    else:
        metrics["edge_significance_p_mean"] = 1.0
        metrics["edge_significant_count"] = 0
    try:
        metrics["avg_betweenness"] = np.mean(
            list(nx.betweenness_centrality(nx_graph).values())
        )
        metrics["avg_closeness"] = np.mean(
            list(nx.closeness_centrality(nx_graph).values())
        )
    except Exception:
        pass
    return metrics


@st.cache_data(ttl=3600)
def compute_bootstrap_ci(
    scores: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05
) -> Tuple[float, float, float]:
    if len(scores) < 2:
        return float(np.mean(scores)), 0.0, 0.0
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(float(np.mean(sample)))
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(scores)), ci_low, ci_high


# ============================================================================
# ADVANCED ANALYTICS (CACHED)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def detect_keyword_bursts(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    burst_threshold: float = 2.0,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 3:
        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])
    year_range = sorted(years.unique())
    burst_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 5:
            continue
        concept_years: List[int] = []
        for idx in doc_indices:
            if (
                idx < len(df_filtered)
                and pd.notna(df_filtered.iloc[idx].get("Year"))
            ):
                concept_years.append(int(df_filtered.iloc[idx]["Year"]))
        if len(concept_years) < 3:
            continue
        year_counts = Counter(concept_years)
        counts = [year_counts.get(y, 0) for y in year_range]
        if len(counts) < 3:
            continue
        window = max(2, len(counts) // 5)
        moving_avg = pd.Series(counts).rolling(
            window=window, min_periods=1
        ).mean()
        burst_scores: List[float] = []
        for i in range(window, len(counts)):
            if moving_avg.iloc[i - 1] > 0:
                ratio = counts[i] / max(moving_avg.iloc[i - 1], 0.1)
                burst_scores.append(float(ratio))
        if burst_scores:
            max_burst = max(burst_scores)
            burst_year = year_range[window + burst_scores.index(max_burst)]
            if max_burst >= burst_threshold:
                burst_data.append({
                    "concept": concept,
                    "burst_score": round(max_burst, 2),
                    "burst_year": burst_year,
                    "total_mentions": len(concept_years),
                    "year_range": f"{min(concept_years)}-{max(concept_years)}",
                })
    if not burst_data:
        return pd.DataFrame(columns=["concept", "burst_score", "burst_year", "total_mentions", "year_range"])
    return pd.DataFrame(burst_data).sort_values(
        "burst_score", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_semantic_drift(
    df_filtered: pd.DataFrame,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
    text_columns: List[str],
    early_fraction: float = 0.3,
    late_fraction: float = 0.3,
) -> pd.DataFrame:
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])
    years = df_filtered["Year"].dropna().astype(int)
    if len(years.unique()) < 4:
        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])
    embed_model = load_embedding_model()
    sorted_years = sorted(years.unique())
    n_years = len(sorted_years)
    early_cutoff = sorted_years[int(n_years * early_fraction)]
    late_cutoff = sorted_years[int(n_years * (1 - late_fraction))]
    drift_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        doc_indices = concept_abstract_map.get(concept, [])
        if len(doc_indices) < 10:
            continue
        early_texts: List[str] = []
        late_texts: List[str] = []
        for idx in doc_indices:
            if idx >= len(df_filtered):
                continue
            row = df_filtered.iloc[idx]
            year = row.get("Year")
            if pd.isna(year):
                continue
            year = int(year)
            text = " ".join([
                str(row.get(col, ""))
                for col in text_columns if pd.notna(row.get(col))
            ])
            if year <= early_cutoff:
                early_texts.append(text)
            elif year >= late_cutoff:
                late_texts.append(text)
        if len(early_texts) < 3 or len(late_texts) < 3:
            continue
        try:
            with torch.no_grad():
                early_emb = embed_model.encode(
                    early_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
                late_emb = embed_model.encode(
                    late_texts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            early_centroid = np.mean(early_emb, axis=0)
            late_centroid = np.mean(late_emb, axis=0)
            drift = 1.0 - cosine_similarity(
                [early_centroid], [late_centroid]
            )[0][0]
            drift_data.append({
                "concept": concept,
                "semantic_drift": round(float(drift), 4),
                "early_papers": len(early_texts),
                "late_papers": len(late_texts),
                "early_period": f"{sorted_years[0]}-{early_cutoff}",
                "late_period": f"{late_cutoff}-{sorted_years[-1]}",
            })
            del early_emb, late_emb
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_cache()
        except Exception:
            continue
    if not drift_data:
        return pd.DataFrame(columns=["concept", "semantic_drift", "early_papers", "late_papers", "early_period", "late_period"])
    return pd.DataFrame(drift_data).sort_values(
        "semantic_drift", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def build_concept_genealogy(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    try:
        pagerank = nx.pagerank(_nx_graph, weight='weight')
    except Exception:
        pagerank = {n: 1.0 for n in _nx_graph.nodes()}
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    genealogy_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        pr = pagerank.get(concept, 0)
        bc = betweenness.get(concept, 0)
        freq = len(concept_abstract_map.get(concept, []))
        degree = _nx_graph.degree(concept)
        if (
            pr > np.percentile(list(pagerank.values()), 75)
            and degree > np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 75
            )
        ):
            generation = "Foundational (Parent)"
        elif (
            pr < np.percentile(list(pagerank.values()), 25)
            and degree < np.percentile(
                [_nx_graph.degree(n) for n in _nx_graph.nodes()], 25
            )
        ):
            generation = "Emerging (Child)"
        else:
            generation = "Intermediate"
        genealogy_data.append({
            "concept": concept,
            "pagerank": round(pr, 5),
            "betweenness": round(bc, 5),
            "frequency": freq,
            "degree": degree,
            "generation": generation,
        })
    if not genealogy_data:
        return pd.DataFrame(columns=["concept", "pagerank", "betweenness", "frequency", "degree", "generation"])
    return pd.DataFrame(genealogy_data).sort_values(
        "pagerank", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def detect_cross_domain_bridges(
    _nx_graph: nx.Graph,
    valid_concepts: List[str],
    concept_abstract_map: Dict[str, List[int]],
) -> pd.DataFrame:
    if _nx_graph.number_of_nodes() < 5:
        return pd.DataFrame()
    category_map = abstract_concepts_to_categories(valid_concepts)
    try:
        betweenness = nx.betweenness_centrality(_nx_graph, weight='weight')
    except Exception:
        betweenness = {n: 0.0 for n in _nx_graph.nodes()}
    bridge_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        if concept not in _nx_graph:
            continue
        neighbors = list(_nx_graph.neighbors(concept))
        if len(neighbors) < 2:
            continue
        own_cat = category_map.get(concept, 'general')
        neighbor_cats = [category_map.get(n, 'general') for n in neighbors]
        unique_cats = set(neighbor_cats)
        if len(unique_cats) < 2:
            continue
        bridge_score = betweenness.get(concept, 0) * len(unique_cats)
        bridge_data.append({
            "concept": concept,
            "bridge_score": round(bridge_score, 4),
            "betweenness": round(betweenness.get(concept, 0), 4),
            "connected_categories": len(unique_cats),
            "categories": ", ".join(sorted(unique_cats)),
            "degree": len(neighbors),
            "own_category": own_cat,
        })
    if not bridge_data:
        return pd.DataFrame(columns=["concept", "bridge_score", "betweenness", "connected_categories", "categories", "degree", "own_category"])
    return pd.DataFrame(bridge_data).sort_values(
        "bridge_score", ascending=False
    )


@st.cache_data(ttl=3600, show_spinner=False)
def analyze_network_motifs(_nx_graph: nx.Graph) -> Dict[str, Any]:
    if _nx_graph.number_of_nodes() < 3:
        return {}
    motifs: Dict[str, Any] = {}
    try:
        triangles = nx.triangles(_nx_graph)
        motifs["total_triangles"] = sum(triangles.values()) // 3
        motifs["avg_triangles_per_node"] = round(
            np.mean(list(triangles.values())), 2
        )
        motifs["nodes_in_triangles"] = sum(
            1 for v in triangles.values() if v > 0
        )
    except Exception:
        motifs["total_triangles"] = 0
    try:
        cliques = list(nx.find_cliques(_nx_graph))
        clique_sizes = [len(c) for c in cliques]
        motifs["total_cliques"] = len(cliques)
        motifs["max_clique_size"] = max(clique_sizes) if clique_sizes else 0
        motifs["avg_clique_size"] = (
            round(np.mean(clique_sizes), 2) if clique_sizes else 0
        )
        motifs["4cliques"] = sum(1 for c in clique_sizes if c >= 4)
    except Exception:
        motifs["total_cliques"] = 0
    try:
        clustering = nx.clustering(_nx_graph)
        stars: List[Tuple[str, int, float]] = []
        for node in _nx_graph.nodes():
            deg = _nx_graph.degree(node)
            clust = clustering.get(node, 0)
            if deg >= 5 and clust < 0.2:
                stars.append((node, deg, clust))
        stars.sort(key=lambda x: x[1], reverse=True)
        motifs["star_motifs"] = len(stars)
        motifs["top_stars"] = stars[:10]
    except Exception:
        motifs["star_motifs"] = 0
    return motifs


# ============================================================================
# CENTRALITY & DEGREE DISTRIBUTION
# ============================================================================
def compute_centrality_comparison(
    nx_graph: nx.Graph, valid_concepts: List[str]
) -> pd.DataFrame:
    if nx_graph.number_of_nodes() < 3:
        return pd.DataFrame()
    centrality_data: List[Dict[str, Any]] = []
    try:
        degree_c = dict(nx_graph.degree())
        betweenness_c = nx.betweenness_centrality(nx_graph, weight='weight')
        closeness_c = nx.closeness_centrality(nx_graph)
        eigenvector_c = nx.eigenvector_centrality(
            nx_graph, weight='weight', max_iter=1000
        )
        pagerank_c = nx.pagerank(nx_graph, weight='weight')
        for concept in valid_concepts:
            if concept not in nx_graph:
                continue
            centrality_data.append({
                "concept": concept,
                "degree": degree_c.get(concept, 0),
                "betweenness": round(betweenness_c.get(concept, 0), 5),
                "closeness": round(closeness_c.get(concept, 0), 5),
                "eigenvector": round(eigenvector_c.get(concept, 0), 5),
                "pagerank": round(pagerank_c.get(concept, 0), 5),
            })
    except Exception as e:
        st.warning(f"Centrality computation error: {e}")
    return pd.DataFrame(centrality_data)


def plot_degree_distribution(
    nx_graph: nx.Graph, theme: Dict = None
) -> go.Figure:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    degrees = [d for n, d in nx_graph.degree()]
    if len(degrees) < 3:
        return go.Figure()
    degree_counts = Counter(degrees)
    x = sorted(degree_counts.keys())
    y = [degree_counts[k] for k in x]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Degree Distribution',
        marker=dict(size=10, color=theme.get('highlight_bg', '#ff6b6b')),
    ))
    fig.update_layout(
        title="Degree Distribution (Log-Log)",
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Degree (k)", yaxis_title="Frequency P(k)",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    return fig


# ============================================================================
# PUBLICATION-READY EXPORTS
# ============================================================================
def export_publication_figure(
    nx_graph, valid_concepts, concept_abstract_map,
    cmap_name="viridis", dpi=300, figsize=(14, 12),
    filename="lib_graph_pub.png",
) -> bytes:
    try:
        pos = nx.spring_layout(nx_graph, seed=42, k=2.5, iterations=200)
        plt.figure(figsize=figsize, dpi=dpi)
        node_colors = [get_lib_category_color(n) for n in nx_graph.nodes()]
        node_sizes = [
            max(100, min(800, len(concept_abstract_map.get(n, [])) * 20 + 50))
            for n in nx_graph.nodes()
        ]
        nx.draw(
            nx_graph, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color='lightgray',
            node_size=node_sizes,
            font_size=6,
            font_weight='bold',
            edgecolors='white',
            linewidths=1.5,
            width=0.5,
            alpha=0.9,
        )
        plt.title(
            "Lithium‑Ion Battery Concept Graph",
            fontsize=14, fontweight='bold', pad=20,
        )
        buf = io.BytesIO()
        plt.savefig(
            buf, format='png', dpi=dpi, bbox_inches='tight',
            facecolor='white', edgecolor='none',
        )
        buf.seek(0)
        plt.close()
        return buf.read()
    except Exception as e:
        st.error(f"Publication figure export failed: {e}")
        return b''


def generate_analysis_report(
    nx_graph, valid_concepts, concept_abstract_map,
    top_scores, distill_df, burst_df, drift_df,
    genealogy_df, bridge_df, motifs, val_metrics,
    df_filtered,
) -> str:
    report: List[str] = []
    report.append("# Lithium‑Ion Battery Concept Graph Analysis Report")
    report.append(
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"
    )
    report.append("## 1. Dataset Overview")
    report.append(f"- **Total Records**: {len(df_filtered)}")
    if 'Year' in df_filtered.columns:
        years = df_filtered['Year'].dropna()
        report.append(
            f"- **Year Range**: {int(years.min())} - {int(years.max())}"
        )
    report.append(f"- **Total Concepts**: {len(valid_concepts)}")
    report.append(f"- **Total Edges**: {nx_graph.number_of_edges()}")
    report.append(f"- **Graph Density**: {nx.density(nx_graph):.4f}")
    report.append("")
    report.append("## 2. Top Concepts by Frequency")
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:20]
    for i, c in enumerate(top_concepts, 1):
        freq = len(concept_abstract_map.get(c, []))
        deg = nx_graph.degree(c)
        report.append(f"{i}. **{c}** - Freq: {freq}, Degree: {deg}")
    report.append("")
    report.append("## 3. Concept Distillation Efficiency (Top 15)")
    if not distill_df.empty:
        for _, row in distill_df.head(15).iterrows():
            report.append(
                f"- **{row['concept']}**: Efficiency="
                f"{row['distillation_efficiency']:.3f}, "
                f"Freq={row['frequency']}, "
                f"Coherence={row['coherence_score']:.3f}"
            )
    report.append("")
    report.append("## 4. Research Direction Recommendations (Top 10)")
    if not top_scores.empty:
        for i, (_, row) in enumerate(top_scores.head(10).iterrows(), 1):
            report.append(
                f"{i}. **{row['concept_u']}** + **{row['concept_v']}** - "
                f"Composite Score: {row['composite_score']:.3f}"
            )
    report.append("")
    report.append("## 5. Keyword Burst Detection")
    if not burst_df.empty:
        for _, row in burst_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Burst Score={row['burst_score']:.2f} "
                f"(Year {row['burst_year']})"
            )
    else:
        report.append("No significant keyword bursts detected.")
    report.append("")
    report.append("## 6. Semantic Drift Detection")
    if not drift_df.empty:
        for _, row in drift_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Drift={row['semantic_drift']:.4f} "
                f"({row['early_period']} -> {row['late_period']})"
            )
    else:
        report.append("No significant semantic drift detected.")
    report.append("")
    report.append("## 7. Cross-Domain Bridge Concepts")
    if not bridge_df.empty:
        for _, row in bridge_df.head(10).iterrows():
            report.append(
                f"- **{row['concept']}**: Bridge Score={row['bridge_score']:.4f}, "
                f"Connects {row['connected_categories']} categories"
            )
    else:
        report.append("No cross-domain bridges detected.")
    report.append("")
    report.append("## 8. Network Motif Analysis")
    report.append(f"- Total Triangles: {motifs.get('total_triangles', 0)}")
    report.append(f"- Total Cliques: {motifs.get('total_cliques', 0)}")
    report.append(f"- Max Clique Size: {motifs.get('max_clique_size', 0)}")
    report.append(f"- Star Motifs: {motifs.get('star_motifs', 0)}")
    report.append("")
    report.append("## 9. Graph Validation Metrics")
    report.append(f"- Modularity: {val_metrics.get('modularity', 0):.3f}")
    report.append(
        f"- Silhouette Score: {val_metrics.get('silhouette_score', 0):.3f}"
    )
    report.append(f"- Number of Communities: {val_metrics.get('n_communities', 0)}")
    report.append(f"- Avg Betweenness: {val_metrics.get('avg_betweenness', 0):.3f}")
    report.append("")
    report.append("---")
    report.append("*Report generated by Lithium‑Ion Battery Concept Graph v7.0*")
    return "\n".join(report)


# ============================================================================
# GRAPH EDIT HISTORY (AgNPs pattern: max_history=20)
# ============================================================================
class GraphEditHistory:
    def __init__(self, max_history: int = 20) -> None:
        self.history: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)
        self._snapshot_counter = 0

    def save_snapshot(
        self, nx_graph, valid_concepts, concept_to_id,
        id_to_concept, concept_abstract_map,
    ) -> int:
        snapshot = {
            'id': self._snapshot_counter,
            'nx_graph': copy.copy(nx_graph),
            'valid_concepts': list(valid_concepts),
            'concept_to_id': dict(concept_to_id),
            'id_to_concept': dict(id_to_concept),
            'concept_abstract_map': {
                k: list(v) for k, v in concept_abstract_map.items()
            },
            'timestamp': datetime.now().isoformat(),
        }
        self.history.append(snapshot)
        self._snapshot_counter += 1
        self.redo_stack.clear()
        return snapshot['id']

    def undo(self) -> Optional[Dict]:
        if len(self.history) < 2:
            return None
        current = self.history.pop()
        self.redo_stack.append(current)
        previous = self.history[-1]
        return previous

    def redo(self) -> Optional[Dict]:
        if not self.redo_stack:
            return None
        snapshot = self.redo_stack.pop()
        self.history.append(snapshot)
        return snapshot

    def can_undo(self) -> bool:
        return len(self.history) >= 2

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def get_history_summary(self) -> List[str]:
        return [
            f"Snapshot {s['id']} @ {s['timestamp']}" for s in self.history
        ]


# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEME_PRESETS = {
    "Bright (Default)": {
        "bg": "#ffffff", "font": "#1e293b",
        "tooltip_bg": "rgba(255,255,255,0.95)",
        "tooltip_border": "#cbd5e1", "tooltip_text": "#1e293b",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.45)",
        "edge_semantic": "rgba(251, 146, 60, 0.40)",
        "edge_bridge": "rgba(250, 204, 21, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(34, 197, 94, 0.45)",
        "edge_unknown": "rgba(148, 163, 184, 0.30)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.15)",
        "plotly_bg": "#ffffff", "plotly_paper": "#ffffff",
        "grid_color": "#e2e8f0", "axis_color": "#64748b",
    },
    "Dark": {
        "bg": "#0f172a", "font": "#e2e8f0",
        "tooltip_bg": "rgba(15, 23, 42, 0.95)",
        "tooltip_border": "#334155", "tooltip_text": "#e2e8f0",
        "edge_cooccurrence": "rgba(56, 189, 248, 0.55)",
        "edge_semantic": "rgba(251, 146, 60, 0.50)",
        "edge_bridge": "rgba(250, 204, 21, 0.65)",
        "edge_inferred": "rgba(139, 92, 246, 0.60)",
        "edge_cause": "rgba(239, 68, 68, 0.65)",
        "edge_hypernym": "rgba(34, 197, 94, 0.55)",
        "edge_unknown": "rgba(148, 163, 184, 0.40)",
        "node_border": "#f8fafc", "highlight_bg": "#ff6b6b",
        "hover_bg": "#ffd93d",
        "shadow_color": "rgba(0,0,0,0.6)",
        "plotly_bg": "#0f172a", "plotly_paper": "#0f172a",
        "grid_color": "#1e293b", "axis_color": "#94a3b8",
    },
    "Midnight": {
        "bg": "#020617", "font": "#f1f5f9",
        "tooltip_bg": "rgba(2, 6, 23, 0.97)",
        "tooltip_border": "#1e293b", "tooltip_text": "#f1f5f9",
        "edge_cooccurrence": "rgba(99, 102, 241, 0.55)",
        "edge_semantic": "rgba(236, 72, 153, 0.50)",
        "edge_bridge": "rgba(34, 211, 238, 0.65)",
        "edge_inferred": "rgba(168, 85, 247, 0.60)",
        "edge_cause": "rgba(244, 63, 94, 0.65)",
        "edge_hypernym": "rgba(52, 211, 153, 0.55)",
        "edge_unknown": "rgba(71, 85, 105, 0.40)",
        "node_border": "#e2e8f0", "highlight_bg": "#f43f5e",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(0,0,0,0.7)",
        "plotly_bg": "#020617", "plotly_paper": "#020617",
        "grid_color": "#0f172a", "axis_color": "#64748b",
    },
    "Warm": {
        "bg": "#fff7ed", "font": "#431407",
        "tooltip_bg": "rgba(255, 247, 237, 0.97)",
        "tooltip_border": "#fdba74", "tooltip_text": "#431407",
        "edge_cooccurrence": "rgba(234, 88, 12, 0.45)",
        "edge_semantic": "rgba(180, 83, 9, 0.40)",
        "edge_bridge": "rgba(202, 138, 4, 0.55)",
        "edge_inferred": "rgba(147, 51, 234, 0.50)",
        "edge_cause": "rgba(220, 38, 38, 0.55)",
        "edge_hypernym": "rgba(22, 163, 74, 0.45)",
        "edge_unknown": "rgba(120, 53, 15, 0.25)",
        "node_border": "#fff7ed", "highlight_bg": "#dc2626",
        "hover_bg": "#f59e0b",
        "shadow_color": "rgba(124, 45, 18, 0.15)",
        "plotly_bg": "#fff7ed", "plotly_paper": "#fff7ed",
        "grid_color": "#fed7aa", "axis_color": "#9a3412",
    },
    "Forest": {
        "bg": "#f0fdf4", "font": "#052e16",
        "tooltip_bg": "rgba(240, 253, 244, 0.97)",
        "tooltip_border": "#86efac", "tooltip_text": "#052e16",
        "edge_cooccurrence": "rgba(22, 163, 74, 0.45)",
        "edge_semantic": "rgba(5, 150, 105, 0.40)",
        "edge_bridge": "rgba(234, 179, 8, 0.55)",
        "edge_inferred": "rgba(139, 92, 246, 0.50)",
        "edge_cause": "rgba(239, 68, 68, 0.55)",
        "edge_hypernym": "rgba(21, 128, 61, 0.45)",
        "edge_unknown": "rgba(20, 83, 45, 0.25)",
        "node_border": "#f0fdf4", "highlight_bg": "#15803d",
        "hover_bg": "#84cc16",
        "shadow_color": "rgba(20, 83, 45, 0.15)",
        "plotly_bg": "#f0fdf4", "plotly_paper": "#f0fdf4",
        "grid_color": "#bbf7d0", "axis_color": "#166534",
    },
    "Ocean": {
        "bg": "#ecfeff", "font": "#083344",
        "tooltip_bg": "rgba(236, 254, 255, 0.97)",
        "tooltip_border": "#67e8f9", "tooltip_text": "#083344",
        "edge_cooccurrence": "rgba(6, 182, 212, 0.45)",
        "edge_semantic": "rgba(14, 165, 233, 0.40)",
        "edge_bridge": "rgba(99, 102, 241, 0.55)",
        "edge_inferred": "rgba(168, 85, 247, 0.50)",
        "edge_cause": "rgba(244, 63, 94, 0.55)",
        "edge_hypernym": "rgba(13, 148, 136, 0.45)",
        "edge_unknown": "rgba(21, 94, 117, 0.25)",
        "node_border": "#ecfeff", "highlight_bg": "#0ea5e9",
        "hover_bg": "#22d3ee",
        "shadow_color": "rgba(8, 51, 68, 0.15)",
        "plotly_bg": "#ecfeff", "plotly_paper": "#ecfeff",
        "grid_color": "#a5f3fc", "axis_color": "#0e7490",
    },
}

PHYSICS_PRESETS = {
    "Stable (Default)": {
        "damping": 0.55, "gravity": -2500, "spring_length": 140,
        "spring_strength": 0.05, "central_gravity": 0.25,
        "stabilization": 2500,
    },
    "Fluid": {
        "damping": 0.25, "gravity": -1800, "spring_length": 120,
        "spring_strength": 0.05, "central_gravity": 0.30,
        "stabilization": 1500,
    },
    "Tight": {
        "damping": 0.70, "gravity": -4000, "spring_length": 80,
        "spring_strength": 0.08, "central_gravity": 0.20,
        "stabilization": 3000,
    },
    "Off": {
        "damping": 0.99, "gravity": 0, "spring_length": 200,
        "spring_strength": 0.0, "central_gravity": 0.0,
        "stabilization": 0,
    },
}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def get_lib_category_color(concept: str, cmap_colors: Optional[List[str]] = None) -> str:
    if cmap_colors:
        return cmap_colors[hash(concept) % len(cmap_colors)]
    concept_lower = concept.lower()
    category = 'general'
    for pattern, cat in LIB_DESCRIPTOR_MAPPING.items():
        if re.search(pattern, concept_lower):
            category = cat
            break
    color_map = {
        'material': '#E74C3C',
        'process': '#3498DB',
        'property': '#2ECC71',
        'phenomenon': '#F39C12',
        'method': '#9B59B6',
        'parameter': '#1ABC9C',
        'general': '#95A5A6'
    }
    return color_map.get(category, '#95A5A6')


# ============================================================================
# PYVIS RENDERER — colored edges + hierarchy labels
# ============================================================================

_NODE_TYPE_COLORS = {
    ConceptType.MATERIAL:       "#E74C3C",
    ConceptType.PROCESS:        "#3498DB",
    ConceptType.PROPERTY:       "#2ECC71",
    ConceptType.PHENOMENON:     "#F39C12",
    ConceptType.METHOD:         "#9B59B6",
    ConceptType.PARAMETER:      "#1ABC9C",
    ConceptType.MICROSTRUCTURE: "#E67E22",
    ConceptType.MODEL:          "#2980B9",
    ConceptType.GENERAL:        "#95A5A6",
}


def render_pyvis_graph(
    nx_graph, concept_abstract_map, physics_enabled=True,
    cmap_name="viridis", top_n_nodes=0, theme=None, physics_preset=None,
    show_edge_weights=False, edge_label_mode="hover",
    node_label_size=12, node_label_position="center",
    node_font_face="Inter, Segoe UI, Roboto, sans-serif",
    edge_label_size=10, edge_label_color=None,
    edge_label_position="middle",
    use_abbreviated_labels=False, max_label_length=15,
    enable_node_highlight=True, show_definitions=True, ontology=None,
    edge_lightness=0.6, edge_color_mode="theme",
    custom_edge_color="#AAAAAA", tooltip_font_size=13,
    node_legend_font_size=13,
    label_mode=NodeLabelMode.FULL_NAME,
    external_label_text="",
    external_font_size=14,
    external_font_color="#333333",
    external_label_align="left",
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if physics_preset is None:
        physics_preset = PHYSICS_PRESETS["Stable (Default)"]

    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree(weight='weight'))
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()

    cmap_colors = get_colormap_colors(cmap_name, max(1, len(nx_graph.nodes())))
    
    net = Network(height="780px", width="100%", bgcolor=theme['bg'], font_color=theme['font'],
                  select_menu=True, notebook=False, cdn_resources='remote')

    if physics_enabled and physics_preset.get("gravity", 0) != 0:
        net.set_options(f"""
        var options = {{
            "physics": {{
                "enabled": true, "solver": "barnesHut",
                "barnesHut": {{
                    "gravitationalConstant": {physics_preset['gravity']},
                    "centralGravity": {physics_preset['central_gravity']},
                    "springLength": {physics_preset['spring_length']},
                    "springConstant": {physics_preset['spring_strength']},
                    "damping": {physics_preset['damping']}, "overlap": 0.15
                }},
                "stabilization": {{ "enabled": true, "iterations": 500, "updateInterval": 50, "onlyDynamicEdges": true, "fit": true }}
            }},
            "interaction": {{ "hover": true, "tooltipDelay": 180, "hideEdgesOnDrag": false, "zoomView": true, "dragView": true }}
        }}
        """)
    else:
        net.set_options("""var options = { "physics": { "enabled": false }, "interaction": { "hover": true, "dragNodes": true, "dragView": true, "zoomView": true } }""")

    label_map = {}
    audit_rows = []
    n_counter = 1
    used_rel_types = {}

    for i, node in enumerate(nx_graph.nodes()):
        freq = len(concept_abstract_map.get(node, []))
        size = int(np.clip(8 + freq * 1.2, 8, 40))
        color = get_lib_category_color(node, cmap_colors)
        degree = int(nx_graph.degree(node))
        
        original_label = node
        _custom_map = st.session_state.get('custom_label_map', {}) or {}

        full_display = (get_hierarchy_label(node, "arrow")
                        if node in _HIERARCHY_PARENTS
                        else node.replace("_", " ").title())

        if label_mode == NodeLabelMode.FULL_NAME:
            label = full_display
        elif label_mode == NodeLabelMode.ANNOTATION:
            _prefix = st.session_state.get('annot_prefix', 'N') or 'N'
            label = f"{_prefix}{n_counter}"
            label_map[label] = original_label
            n_counter += 1
        elif label_mode == NodeLabelMode.CUSTOM_BLANK:
            _user_text = (_custom_map.get(node) or external_label_text or "")
            label = _user_text if _user_text.strip() else " "

        node_shape = 'circle'

        if label_mode == NodeLabelMode.FULL_NAME:
            font_dict = {
                'color': '#ffffff',
                'size': max(8, min(int(node_label_size), 60)),
                'face': node_font_face, 'bold': True,
                'align': node_label_position, 'strokeWidth': 0,
            }
        elif label_mode == NodeLabelMode.ANNOTATION:
            font_dict = {
                'color': '#ffffff',
                'size': max(10, min(int(node_label_size), 60)),
                'face': node_font_face, 'bold': True,
                'align': 'center', 'strokeWidth': 0,
            }
        elif label_mode == NodeLabelMode.CUSTOM_BLANK:
            _is_blank = (len(label.strip()) == 0)
            font_dict = {
                'color': 'rgba(0,0,0,0)' if _is_blank else external_font_color,
                'size': 0 if _is_blank else max(8, min(int(external_font_size), 60)),
                'face': node_font_face,
                'bold': not _is_blank,
                'align': external_label_align,
                'strokeWidth': 0,
                'vadjust': 0,
                'multi': False,
            }

        audit_rows.append((node, label,
                           font_dict.get('size'), font_dict.get('align', 'center')))

        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        definition = nx_graph.nodes[node].get('definition', '')
        _def_display = ""
        if show_definitions and definition:
            _def_display = definition[:180] + "..." if len(definition) > 180 else definition
        _full_label_display = ""
        if label_mode == NodeLabelMode.ANNOTATION:
            _full_label_display = full_display
        elif label_mode == NodeLabelMode.CUSTOM_BLANK and not label.strip():
            _full_label_display = full_display
        tooltip_content = (
            f"{node}\n"
            f"Type: {concept_type}\n"
            f"Degree: {degree}\n"
            f"Frequency: {freq}"
            + (f"\nDefinition: {_def_display}" if _def_display else "")
            + (f"\nFull Label: {_full_label_display}" if _full_label_display else "")
        )
        
        net.add_node(node, label=label, size=size,
                     color={'background': color, 'border': theme['node_border'],
                            'highlight': {'background': theme['highlight_bg'], 'border': '#ffffff'},
                            'hover': {'background': theme['hover_bg'], 'border': '#ffffff'}},
                     font=font_dict, title=tooltip_content,
                     borderWidth=2, borderWidthSelected=3,
                     shadow={'enabled': True, 'color': theme['shadow_color'],
                             'size': 12, 'x': 4, 'y': 4},
                     shape=node_shape, mass=max(1, 1 + freq * 0.05))

    all_weights = [nx_graph[u][v].get('weight', 1) for u, v in nx_graph.edges()]
    weight_threshold = float(np.percentile(all_weights, 80)) if all_weights else 0.0

    for u, v in nx_graph.edges():
        w = float(nx_graph[u][v].get('weight', 1))
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        rel_type = RelationshipType.SEMANTIC
        if edge_type != 'unknown':
            try: rel_type = RelationshipType(edge_type)
            except ValueError: pass

        if edge_color_mode == "theme":
            base_color = theme['edge_unknown'] if edge_type == 'unknown' else get_edge_color(rel_type)
            if edge_lightness > 0:
                base_color = lighten_hex_color(base_color, edge_lightness)
        elif edge_color_mode == "uniform_grey":
            base_color = lighten_hex_color("#808080", edge_lightness)
        else:
            base_color = lighten_hex_color(custom_edge_color, edge_lightness)

        width = float(get_edge_width(rel_type) * (0.5 + 0.5 * w))
        style = get_edge_style(rel_type)
        dashes = True if style == "dashed" or is_inferred else False

        edge_kwargs = dict(
            value=float(np.clip(w, 0.5, 5)), width=width,
            color={'color': base_color, 'highlight': theme['highlight_bg'], 'hover': theme['hover_bg'], 'opacity': 0.85},
            smooth={"type": "dynamic"},
            title=f"Weight: {w:.2f}\nType: {edge_type}\nInferred: {is_inferred}",
            dashes=dashes
        )
        if edge_label_mode == "all" or (edge_label_mode == "threshold" and w >= weight_threshold):
            edge_kwargs['label'] = f"{w:.1f}"
            edge_kwargs['font'] = {'color': edge_label_color or theme['font'], 'size': int(edge_label_size),
                                   'background': theme['tooltip_bg'], 'strokeWidth': 2, 'strokeColor': theme['node_border'],
                                   'align': edge_label_position, 'face': node_font_face}
        net.add_edge(u, v, **edge_kwargs)
        if rel_type not in used_rel_types:
            used_rel_types[rel_type] = rel_type.value.replace("_", " ").title()

    if used_rel_types:
        legend_rows = []
        for rt, human in sorted(used_rel_types.items(), key=lambda x: x[1]):
            c = get_edge_color(rt)
            if edge_color_mode == "theme":
                c = lighten_hex_color(c, edge_lightness) if edge_lightness > 0 else c
            elif edge_color_mode == "uniform_grey":
                c = lighten_hex_color("#808080", edge_lightness)
            else:
                c = lighten_hex_color(custom_edge_color, edge_lightness)
            w_leg = get_edge_width(rt)
            s_leg = get_edge_style(rt)
            border = 'border: 1px dashed #888;' if s_leg == "dashed" else 'border: 1px solid transparent;'
            legend_rows.append(f'<tr><td style="padding:2px 6px;"><span style="display:inline-block;width:{int(20*w_leg)}px;height:3px;background:{c};vertical-align:middle;{border}"></span></td><td style="padding:2px 6px;color:#ccc;font-size:11px;">{human}</td></tr>')
        legend_html = f'<div style="background:#0d0d1a;border-radius:8px;padding:12px 16px;margin-top:8px;max-height:280px;overflow-y:auto;"><div style="color:#fff;font-size:13px;font-weight:bold;margin-bottom:6px;">Edge Colors ({len(used_rel_types)} types)</div><table style="border-collapse:collapse;">{"".join(legend_rows)}</table></div>'
        net.add_node("__legend__", label="", shape="dot", size=0, color="rgba(0,0,0,0)", fixed=True, x=-500, y=-500, physics=False, title=legend_html)

    try:
        html_content = net.generate_html(notebook=False)
    except Exception as e:
        st.error(f"PyVis HTML generation failed: {e}")
        return

    if use_abbreviated_labels and label_map:
        label_map_json = json.dumps(label_map)
        label_map_div = f'<div id="hea-label-map-data" style="display:none;">{label_map_json}</div>'
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', label_map_div + '</body>', 1)
        else:
            html_content += label_map_div

    custom_css = f"""
    <style>
    body {{ background: {theme['bg']}; margin: 0; padding: 0; font-family: '{node_font_face}', sans-serif; }}
    #mynetwork {{ border-radius: 16px; box-shadow: 0 12px 48px {theme['shadow_color']}; outline: none; }}
    
    div.vis-tooltip {{
        max-width: 540px !important;
        width: auto !important;
        max-height: 280px !important;
        height: auto !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        z-index: 10000 !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.45 !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    }}
    div.vis-tooltip > div {{
        max-width: 520px !important;
        width: auto !important;
        max-height: 260px !important;
        overflow: auto !important;
        white-space: pre-wrap !important;
    }}
    .hea-legend {{ font-size: {node_legend_font_size}px !important; }}
    
    #edge-info-panel > div:first-child > div:first-child {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: break-word !important;
    }}
    </style>
    """

    if '</head>' in html_content:
        html_content = html_content.replace('</head>', custom_css + '</head>', 1)
    elif '<head>' in html_content:
        html_content = html_content.replace('<head>', '<head>' + custom_css, 1)
    else:
        html_content = custom_css + html_content

    if 'div.vis-tooltip' not in html_content:
        st.warning("Tooltip CSS injection failed — tooltips may render with default (clipped) styling.")

    if enable_node_highlight:
        highlight_js = r"""
        <script>
        (function() {
            var checkExist = setInterval(function() {
                if (typeof network !== 'undefined' && network !== null && network.body && network.body.data) {
                    clearInterval(checkExist);
                    var nodesDS = network.body.data.nodes;
                    var edgesDS = network.body.data.edges;
                    var savedNodeColors = {};
                    var activeNodeId = null;
                    var labelMode = 'short';
                    var labelMap = {};
                    
                    (function initLabelMap() {
                        var hidden = document.getElementById('hea-label-map-data');
                        if (hidden && hidden.textContent) { try { labelMap = JSON.parse(hidden.textContent); } catch(e) {} }
                    })();

                    function resetAll() {
                        var nodeRestores = [];
                        for (var nid in savedNodeColors) { nodeRestores.push({id: nid, color: savedNodeColors[nid]}); }
                        if (nodeRestores.length > 0) nodesDS.update(nodeRestores);
                        savedNodeColors = {}; activeNodeId = null;
                        var panel = document.getElementById('edge-info-panel'); if (panel) panel.style.display = 'none';
                    }

                    function resolveFullName(shortOrId) {
                        if (labelMap && labelMap[shortOrId]) return labelMap[shortOrId];
                        return shortOrId;
                    }

                    function formatEdgeRow(e, idx, mode) {
                        var typeColor = e.inferred ? '#8b5cf6' : '#0ea5e9';
                        var badge = e.inferred ? ' <span style="background:#8b5cf6;color:white;padding:1px 4px;border-radius:3px;font-size:9px;">INFERRED</span>' : '';
                        var typeBadge = '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:600;">' + e.type + '</span>';
                        var fromName = (mode === 'short') ? e.from : resolveFullName(e.from);
                        var toName = (mode === 'short') ? e.to : resolveFullName(e.to);
                        return '<div style="padding:8px 10px;margin:4px 0;background:rgba(248,250,252,0.9);border-left:4px solid ' + typeColor + ';border-radius:6px;font-size:12px;">' +
                            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap;">' +
                            '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;word-break:break-word;">' + fromName + '</span>' +
                            '<span style="color:#94a3b8;font-size:13px;">↔</span>' +
                            '<span style="font-family:monospace;font-size:11px;color:#1e293b;font-weight:600;word-break:break-word;">' + toName + '</span></div>' +
                            '<div style="display:flex;align-items:center;gap:8px;padding-left:10px;">' +
                            '<span style="background:#0ea5e9;color:white;font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700;">W: ' + e.weight + '</span>' +
                            typeBadge + badge + '</div></div>';
                    }

                    function showEdgeInfoPanel(nodeId, connectedEdges) {
                        var panel = document.getElementById('edge-info-panel');
                        if (!panel) { panel = document.createElement('div'); panel.id = 'edge-info-panel'; document.body.appendChild(panel); }
                        panel.style.cssText = 'position:fixed;top:90px;right:20px;width:400px;max-height:calc(100vh - 110px);overflow-y:auto;z-index:9990;' +
                            'background:rgba(255,255,255,0.95);border:1px solid rgba(255,215,0,0.6);border-radius:16px;padding:0;' +
                            'font-family:Inter,Segoe UI,Roboto,sans-serif;box-shadow:0 20px 60px rgba(0,0,0,0.15);backdrop-filter:blur(20px);';

                        var nodeData = nodesDS.get(nodeId);
                        
                        var nodeName = nodeId; 
                        var nodeDefinition = ""; var nodeType = ""; var nodeFreq = ""; var nodeDegree = "";
                        
                        if (nodeData && nodeData.title) {
                            var tooltipText = nodeData.title;
                            var defMatch = tooltipText.match(/Definition:\s*(.+)/i); if (defMatch && defMatch[1]) { nodeDefinition = defMatch[1].trim(); }
                            var typeMatch = tooltipText.match(/Type:\s*(\w+)/i); if (typeMatch && typeMatch[1]) { nodeType = typeMatch[1].trim(); }
                            var freqMatch = tooltipText.match(/Frequency:\s*(\d+)/i); if (freqMatch && freqMatch[1]) { nodeFreq = freqMatch[1].trim(); }
                            var degMatch = tooltipText.match(/Degree:\s*(\d+)/i); if (degMatch && degMatch[1]) { nodeDegree = degMatch[1].trim(); }
                        }

                        var html = '<div style="padding:16px 20px;background:linear-gradient(135deg,rgba(255,215,0,0.15),rgba(255,183,77,0.1));border-radius:16px 16px 0 0;border-bottom:2px solid rgba(255,215,0,0.4);">';
                        html += '<div style="font-size:18px;font-weight:800;color:#1e293b;margin-bottom:8px;word-break:break-word;white-space:normal;overflow:visible;">🔬 ' + nodeName + '</div>';
                        html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
                        if (nodeType) html += '<span style="background:rgba(14,165,233,0.1);color:#0ea5e9;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">' + nodeType + '</span>';
                        if (nodeDegree) html += '<span style="background:rgba(168,85,247,0.1);color:#a855f7;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Deg: ' + nodeDegree + '</span>';
                        if (nodeFreq) html += '<span style="background:rgba(34,197,94,0.1);color:#22c55e;font-size:10px;padding:3px 8px;border-radius:10px;font-weight:600;">Freq: ' + nodeFreq + '</span>';
                        html += '</div></div>';
                        
                        if (nodeDefinition) {
                            html += '<div style="padding:12px 20px;background:rgba(251,191,36,0.06);border-bottom:1px solid rgba(0,0,0,0.04);">';
                            html += '<div style="font-size:10px;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-bottom:4px;">📖 Definition</div>';
                            html += '<div style="font-size:12px;color:#475569;font-style:italic;line-height:1.4;word-break:break-word;">' + nodeDefinition + '</div></div>';
                        }
                        
                        html += '<div style="padding:10px 20px;background:rgba(248,250,252,0.8);border-bottom:1px solid rgba(0,0,0,0.04);display:flex;align-items:center;gap:10px;">';
                        html += '<span style="font-size:10px;color:#94a3b8;font-weight:600;">Label Mode</span>';
                        html += '<button id="btn-short" onclick="window._heaSetLabelMode(\'short\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:#D32F2F;color:white;">Short</button>';
                        html += '<button id="btn-full" onclick="window._heaSetLabelMode(\'full\')" style="padding:4px 10px;border:none;border-radius:6px;font-size:10px;font-weight:700;cursor:pointer;background:transparent;color:#64748b;">Full</button>';
                        html += '</div>';
                        
                        html += '<div id="edges-container" style="padding:12px 16px 16px;">';
                        var edgeList = [];
                        connectedEdges.forEach(function(eId) {
                            var e = edgesDS.get(eId); if (!e) return;
                            var fromNode = nodesDS.get(e.from); var toNode = nodesDS.get(e.to);
                            var fromLabel = fromNode ? (fromNode.label || e.from) : e.from;
                            var toLabel = toNode ? (toNode.label || e.to) : e.to;
                            var w = (typeof e.value === 'number') ? e.value : (e.width || 1);
                            var edgeType = 'unknown', isInferred = false;
                            if (e.title) {
                                var _txt = e.title;
                                var m = _txt.match(/Type:\s*(\w+)/); if (m) edgeType = m[1];
                                if (_txt.indexOf('Inferred: true') !== -1) isInferred = true;
                            }
                            edgeList.push({from: fromLabel, to: toLabel, weight: (typeof w === 'number') ? w.toFixed(2) : String(w), type: edgeType, inferred: isInferred});
                        });
                        edgeList.sort(function(a,b){ return parseFloat(b.weight)-parseFloat(a.weight); });
                        edgeList.forEach(function(e, idx){ html += formatEdgeRow(e, idx, labelMode); });
                        html += '</div>';
                        
                        panel.innerHTML = html; panel.style.display = 'block'; panel._edgeList = edgeList;
                        window._heaSetLabelMode = function(mode) {
                            labelMode = mode; var p = document.getElementById('edge-info-panel');
                            if (!p || !p._edgeList) return;
                            var btnShort = document.getElementById('btn-short'); var btnFull = document.getElementById('btn-full');
                            if (mode === 'short') { btnShort.style.background = '#D32F2F'; btnShort.style.color = 'white'; btnFull.style.background = 'transparent'; btnFull.style.color = '#64748b'; }
                            else { btnFull.style.background = '#D32F2F'; btnFull.style.color = 'white'; btnShort.style.background = 'transparent'; btnShort.style.color = '#64748b'; }
                            var container = document.getElementById('edges-container');
                            if (container) { var newHtml = ''; p._edgeList.forEach(function(e, idx){ newHtml += formatEdgeRow(e, idx, mode); }); container.innerHTML = newHtml; }
                        };
                    }

                    network.on("selectNode", function(params) {
                        var nodeId = params.nodes[0];
                        if (nodeId === "__legend__") { network.unselectAll(); return; }
                        if (activeNodeId !== null && activeNodeId !== nodeId) resetAll();
                        activeNodeId = nodeId;
                        var connectedEdges = network.getConnectedEdges(nodeId);
                        var connectedNodes = network.getConnectedNodes(nodeId);
                        var nodeUpdates = [];
                        connectedNodes.forEach(function(nId) {
                            var n = nodesDS.get(nId);
                            if (n && !savedNodeColors[nId]) {
                                savedNodeColors[nId] = JSON.parse(JSON.stringify(n.color));
                                var newColor = JSON.parse(JSON.stringify(n.color));
                                if (typeof newColor === 'string') newColor = {background: newColor, border: '#FFD700'}; else newColor.border = '#FFD700';
                                nodeUpdates.push({id: nId, color: newColor, shadow: {enabled: true, color: 'rgba(255,215,0,0.5)', size: 15, x: 0, y: 0}});
                            }
                        });
                        if (nodeUpdates.length > 0) nodesDS.update(nodeUpdates);
                        showEdgeInfoPanel(nodeId, connectedEdges);
                    });
                    network.on("deselectNode", function(){ resetAll(); });
                    network.on("click", function(params){ if (params.nodes.length === 0 && activeNodeId !== null) resetAll(); });
                }
            }, 250);

            setTimeout(function() { clearInterval(checkExist); }, 15000);
        })();
        </script>
        """
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', highlight_js + '</body>', 1)
        else:
            html_content += highlight_js

    st.session_state['_label_audit'] = {'mode': label_mode.value, 'rows': audit_rows[:8]}
    with st.expander("🔬 Label Engine audit (what vis.js actually received)"):
        _au = st.session_state.get('_label_audit', {})
        st.write(f"mode = `{_au.get('mode')}`")
        if _au.get('rows'):
            st.table(pd.DataFrame(_au.get('rows', []),
                                  columns=['node', 'label', 'font.size', 'font.align']))
        else:
            st.info("No audit data – build the graph first.")

    st.components.v1.html(html_content, height=950, scrolling=True)

    try:
        html_bytes = html_content.encode('utf-8')
        st.download_button(
            "📥 Download Interactive Graph (HTML)",
            data=html_bytes,
            file_name="lib_concept_graph.html",
            mime="text/html"
        )
        del html_content, html_bytes
        gc.collect()
    except Exception as e:
        st.error(f"Download preparation failed: {e}")

    if label_mode == NodeLabelMode.ANNOTATION and label_map:
        st.markdown("---")
        st.markdown("### 🗺️ Annotation Legend  (N# → Concept Name)")
        sorted_legend = sorted(label_map.items(), key=lambda x: int(x[0][1:]))
        cols = st.columns(4)
        for i, (short, full) in enumerate(sorted_legend):
            with cols[i % 4]:
                st.markdown(f"""<div class='hea-legend' style='padding:8px; border-radius:6px; background-color:{theme.get('tooltip_bg', '#f8fafc')}; border-left:4px solid {theme.get('highlight_bg', '#ff6b6b')}; margin-bottom:6px;'>
<b style='color:{theme.get('highlight_bg', '#ff6b6b')}; font-size:{node_legend_font_size+1}px;'>{short}</b>: <span style='font-size:{node_legend_font_size}px; color:{theme.get('font', '#1e293b')}; word-break:break-word;'>{full}</span></div>""", unsafe_allow_html=True)

def render_graph_plotly_2d(
    nx_graph, concept_abstract_map, cmap_name="viridis",
    custom_labels=None, top_n_nodes=0, node_label_size=10,
    theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos = nx.spring_layout(nx_graph, k=1.5, iterations=50, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_hover: List[Optional[str]] = []
    for u, v in nx_graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        w = nx_graph[u][v].get('weight', 1)
        edge_type = nx_graph[u][v].get('edge_type', 'unknown')
        is_inferred = nx_graph[u][v].get('inferred', False)
        edge_hover.extend([
            (
                f"<b>{u} + {v}</b><br>"
                f"Weight: {w:.2f}<br>"
                f"Type: {edge_type}<br>"
                f"Inferred: {is_inferred}"
            )
        ] * 2 + [None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1, color=theme['edge_unknown']),
        hoverinfo='text', hovertext=edge_hover, name='Connections',
    )
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_size: List[int] = []
    node_color: List[str] = []
    node_labels: List[str] = []
    for i, node in enumerate(nx_graph.nodes()):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(
            f"{node}<br>Type: {concept_type}<br>"
            f"Degree: {deg}<br>Frequency: {freq}"
        )
        node_size.append(max(8, min(35, deg * 2.5 + 10)))
        node_color.append(cmap_colors[i])
        node_labels.append(
            custom_labels.get(node, node) if custom_labels else node
        )
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(
            size=node_size, color=node_color,
            line=dict(width=2, color=theme['node_border']),
        ),
        text=node_labels, textposition="bottom center",
        textfont=dict(size=node_label_size, color=theme['font']),
        hovertext=node_text, hoverinfo='text', name='Concepts',
    )
    fig_data = [edge_trace, node_trace]
    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig_data.append(go.Scatter(
                x=[mid_x], y=[mid_y], mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=8, color=theme['font']),
                hoverinfo='skip', showlegend=False,
            ))
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            plot_bgcolor=theme['plotly_bg'],
            paper_bgcolor=theme['plotly_paper'],
            font=dict(color=theme['font']),
            xaxis=dict(
                showgrid=True, gridcolor=theme['grid_color'],
                zeroline=False, showticklabels=False,
                linecolor=theme['axis_color'],
            ),
            yaxis=dict(
                showgrid=True, gridcolor=theme['grid_color'],
                zeroline=False, showticklabels=False,
                linecolor=theme['axis_color'],
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_plotly_3d(
    nx_graph, concept_abstract_map, cmap_name="viridis",
    top_n_nodes=0, theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("3D view requires >=3 nodes.")
        return
    if top_n_nodes > 0 and len(nx_graph.nodes()) > top_n_nodes:
        degrees = dict(nx_graph.degree())
        top_nodes = sorted(
            degrees.keys(), key=lambda x: degrees[x], reverse=True
        )[:top_n_nodes]
        nx_graph = nx_graph.subgraph(top_nodes).copy()
    pos_3d = nx.spring_layout(nx_graph, dim=3, seed=42)
    cmap_colors = get_colormap_colors(cmap_name, len(nx_graph.nodes()))
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    edge_z: List[Optional[float]] = []
    for u, v in nx_graph.edges():
        x0, y0, z0 = pos_3d[u]
        x1, y1, z1 = pos_3d[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(width=2, color=theme['edge_unknown']),
        hoverinfo='skip',
    )
    node_x: List[float] = []
    node_y: List[float] = []
    node_z: List[float] = []
    node_text: List[str] = []
    node_size: List[int] = []
    node_color: List[str] = []
    node_labels: List[str] = []
    for i, node in enumerate(nx_graph.nodes()):
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        deg = nx_graph.degree(node)
        freq = len(concept_abstract_map.get(node, []))
        concept_type = nx_graph.nodes[node].get('concept_type', 'general')
        node_text.append(
            f"{node}<br>Type: {concept_type}<br>"
            f"Degree: {deg}<br>Frequency: {freq}"
        )
        node_size.append(max(6, min(25, deg * 2 + 8)))
        node_color.append(cmap_colors[i])
        node_labels.append(node)
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text',
        marker=dict(size=node_size, color=node_color, opacity=0.9),
        text=node_labels, textposition="top center",
        textfont=dict(size=8, color=theme['font']),
        hovertext=node_text, hoverinfo='text',
    )
    fig_data = [edge_trace, node_trace]
    if show_edge_weights:
        for u, v in nx_graph.edges():
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            w = nx_graph[u][v].get('weight', 1)
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            mid_z = (z0 + z1) / 2
            fig_data.append(go.Scatter3d(
                x=[mid_x], y=[mid_y], z=[mid_z], mode='text',
                text=[f"{w:.1f}"],
                textfont=dict(size=7, color=theme['font']),
                hoverinfo='skip', showlegend=False,
            ))
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
                yaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
                zaxis=dict(
                    showbackground=False,
                    gridcolor=theme['grid_color'],
                    linecolor=theme['axis_color'],
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False,
            paper_bgcolor=theme['plotly_paper'],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_graph_fallback(
    nx_graph, concept_abstract_map, theme=None, show_edge_weights=False,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    st.markdown(f"### Graph Summary (Text View)")
    st.markdown(f"- **Nodes**: {len(nx_graph.nodes())}")
    st.markdown(f"- **Edges**: {len(nx_graph.edges())}")
    if len(nx_graph.edges()) > 0:
        edge_list = [
            (
                u, v,
                nx_graph[u][v].get('weight', 1),
                nx_graph[u][v].get('edge_type', 'unknown'),
                nx_graph[u][v].get('inferred', False),
            )
            for u, v in nx_graph.edges()
        ]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        st.markdown("**Top 20 Strongest Connections:**")
        for i, (u, v, w, etype, inferred) in enumerate(edge_list[:20], 1):
            inferred_badge = (
                "<span style='background:#8b5cf6;color:white;"
                "padding:1px 5px;border-radius:4px;font-size:11px;'>"
                "INFERRED</span>"
                if inferred else ""
            )
            st.markdown(
                f"{i}. `{u}` + `{v}` {inferred_badge} "
                f"(weight: {w:.2f}, type: {etype})",
                unsafe_allow_html=True,
            )
    if len(concept_abstract_map) > 0:
        freq_data = [
            (c, len(concept_abstract_map.get(c, [])))
            for c in nx_graph.nodes()
        ]
        freq_data.sort(key=lambda x: x[1], reverse=True)
        st.markdown("**Top Concepts by Frequency:**")
        st.dataframe(
            pd.DataFrame(
                freq_data[:15], columns=["Concept", "Abstract Count"]
            ),
            use_container_width=True,
        )


# ============================================================================
# SUNBURST & RADAR CHARTS
# ============================================================================

_SUNBURST_CATEGORY_COLORS = {
    "Cathode Materials":        "#E74C3C",
    "Anode Materials":          "#3498DB",
    "Electrolytes":             "#2ECC71",
    "Manufacturing Processes":  "#F39C12",
    "Energy Metrics":           "#1ABC9C",
    "Capacity Metrics":         "#9B59B6",
    "Kinetic Metrics":          "#E67E22",
    "Safety Metrics":           "#FF6B6B",
    "Structural Metrics":       "#3498DB",
    "Degradation Phenomena":    "#F1C40F",
    "Design Parameters":        "#2980B9",
    "Characterization Methods": "#8E44AD",
    "Testing Methods":          "#16A085",
}


def build_category_hierarchy(
    valid_concepts: List[str],
    concept_abstract_map: Dict,
    top_n_per_category: int = 40,
) -> Tuple[List, List, List]:
    category_map = abstract_concepts_to_categories(valid_concepts)
    all_category_names = set(category_map.values())

    hierarchy: Dict[str, Dict] = {}
    for cat in all_category_names:
        hierarchy[cat] = {"children": [], "count": 0}

    for concept in valid_concepts:
        category = category_map.get(concept, 'general')
        freq = len(concept_abstract_map.get(concept, []))

        if concept in all_category_names:
            hierarchy.setdefault(category, {"children": [], "count": 0})
            hierarchy[category]["count"] += freq
            continue

        hierarchy.setdefault(category, {"children": [], "count": 0})
        hierarchy[category]["children"].append((concept, freq))
        hierarchy[category]["count"] += freq

    labels: List[str] = []
    parents: List[str] = []
    values: List[int] = []

    root_label = "Lithium‑Ion Batteries"
    total = sum(h["count"] for h in hierarchy.values())
    labels.append(root_label)
    parents.append("")
    values.append(total)

    for category, data in sorted(hierarchy.items()):
        children = data["children"]
        children.sort(key=lambda x: x[1], reverse=True)

        if top_n_per_category > 0 and len(children) > top_n_per_category:
            children = children[:top_n_per_category]

        cat_child_sum = sum(freq for _, freq in children)
        cat_display = category.replace('_', ' ').title()

        labels.append(cat_display)
        parents.append(root_label)
        values.append(cat_child_sum if cat_child_sum > 0 else data["count"])

        for concept, freq in children:
            if concept in all_category_names:
                continue
            concept_display = concept.replace('_', ' ').title()

            labels.append(concept_display)
            parents.append(cat_display)
            values.append(max(freq, 1))

    return labels, parents, values


def render_sunburst_chart(
    labels, parents, values, cmap_name="viridis",
    label_size=20, width=900, height=700,
    theme=None, branchvalues="total",
    show_labels=True, show_values=False,
    hover_info="all", color_continuous_scale=None,
    font_family="Arial, sans-serif",
    legend_font_size=12,
) -> None:
    if not labels or len(labels) < 2:
        st.info("Not enough categories for sunburst chart.")
        return
    if len(labels) != len(parents) or len(labels) != len(values):
        st.error("Sunburst data mismatch.")
        return

    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]

    parent_map = {labels[i]: parents[i] for i in range(len(labels))}

    def get_depth(label, visited=None):
        if visited is None: visited = set()
        if label in visited: return 0
        visited.add(label)
        p = parent_map.get(label, "")
        if p == "": return 0
        return 1 + get_depth(p, visited)

    depths = [get_depth(l) for l in labels]
    SYMBOL_LIBRARY = ['✦', '★', '●', '■', '▲', '◆', '⬟', '⬢', '◉', '◈', '◇', '○', '□', '△', '◊']
    node_symbols = {}
    for i, lab in enumerate(labels):
        d = depths[i]
        p = parents[i]
        if d == 0:
            node_symbols[lab] = SYMBOL_LIBRARY[0]
        else:
            siblings = [labels[j] for j in range(len(labels)) if parents[j] == p and depths[j] == d]
            sym_idx = siblings.index(lab) if lab in siblings else 0
            node_symbols[lab] = SYMBOL_LIBRARY[(d + sym_idx) % len(SYMBOL_LIBRARY)]

    display_labels = []
    for i, lab in enumerate(labels):
        if show_labels:
            chain = []
            current = lab
            visited = set()
            while current != "" and current not in visited:
                visited.add(current)
                if current in node_symbols: chain.insert(0, node_symbols[current])
                current = parent_map.get(current, "")
            combo = "".join(chain[-3:]) if len(chain) > 3 else "".join(chain)
            display_labels.append(combo)
        else:
            display_labels.append(lab)

    unique_ids: List[str] = []
    seen: Dict[str, int] = {}
    for i, lab in enumerate(labels):
        base = f"{lab}_d{depths[i]}"
        if base in seen:
            unique_ids.append(f"{base}_{seen[base]}")
            seen[base] += 1
        else:
            unique_ids.append(base)
            seen[base] = 1

    parent_ids: List[str] = []
    for p in parents:
        if p == "":
            parent_ids.append("")
        else:
            found = False
            for i, lab in enumerate(labels):
                if lab == p:
                    parent_ids.append(unique_ids[i])
                    found = True
                    break
            if not found:
                parent_ids.append("")

    n_nodes = len(labels)
    cmap_to_use = color_continuous_scale or cmap_name or "Spectral"
    plot_colors: List[str] = []

    color_success = False
    try:
        cmap_obj = plt.cm.get_cmap(cmap_to_use)
        t_vals = np.linspace(0.05, 0.95, n_nodes)
        rgbas = [cmap_obj(t) for t in t_vals]
        plot_colors = [matplotlib.colors.to_hex(rgba) for rgba in rgbas]
        color_success = True
    except Exception:
        pass

    if not color_success:
        try:
            if hasattr(px.colors.sequential, cmap_to_use):
                px_scale = getattr(px.colors.sequential, cmap_to_use)
                plot_colors = [
                    px_scale[int(i * len(px_scale) / n_nodes) % len(px_scale)]
                    for i in range(n_nodes)
                ]
                color_success = True
        except Exception:
            pass

    if not color_success:
        try:
            from plotly.express import colors as px_colors
            qual_palettes = [
                px_colors.qualitative.Bold,
                px_colors.qualitative.Vivid,
                px_colors.qualitative.Safe,
                px_colors.qualitative.Pastel,
                px_colors.qualitative.Dark24,
                px_colors.qualitative.Light24,
            ]
            long_palette: List[str] = []
            for pal in qual_palettes:
                long_palette.extend(pal)
            plot_colors = [
                long_palette[i % len(long_palette)] for i in range(n_nodes)
            ]
            color_success = True
        except Exception:
            pass

    if not color_success:
        try:
            cmap_obj = plt.cm.get_cmap("tab20")
            plot_colors = [
                matplotlib.colors.to_hex(cmap_obj(i % 20 / 20))
                for i in range(n_nodes)
            ]
        except Exception:
            plot_colors = ["#ff6b6b"] * n_nodes

    sunburst_colors = plot_colors.copy()
    for i in range(len(labels)):
        if depths[i] == 0:
            sunburst_colors[i] = theme.get("plotly_paper", "#f8f9fa")

    bv = branchvalues if branchvalues in ["total", "remainder"] else "total"
    textinfo = 'label+value' if show_labels and show_values else 'label' if show_labels else 'value' if show_values else 'none'

    fig = go.Figure(go.Sunburst(
        ids=unique_ids,
        labels=display_labels,
        parents=parent_ids,
        values=values,
        customdata=labels,
        branchvalues=bv,
        marker=dict(colors=sunburst_colors, line=dict(width=0.5, color="rgba(255,255,255,0.25)")),
        textinfo=textinfo,
        hovertemplate='<b>%{customdata}</b><br>Value: %{value}<extra></extra>' if hover_info == "all" else '<b>%{customdata}</b><extra></extra>' if hover_info == "minimal" else '<extra></extra>',
        insidetextorientation="radial",
        textfont=dict(size=int(label_size), family=font_family, color="white")
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font=dict(color=theme.get("font", "#000000"), family=font_family),
        width=int(width), height=int(height),
        title=dict(text=f"<b>Hierarchical Concept Map</b><br><sup>★ Parent | ★□ Child | ★□◆ Grandchild — Hover for names</sup>", font=dict(size=16, family=font_family))
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get('sunburst_show_legend', True):
        st.markdown("### 📊 Symbol-to-Label Legend")
        legend_entries = [{'symbol': display_labels[i], 'label': labels[i], 'depth': depths[i], 'color': plot_colors[i], 'value': values[i]} for i in range(len(labels))]
        legend_entries.sort(key=lambda x: (x['depth'], -x['value']))
        for d in sorted(set([e['depth'] for e in legend_entries])):
            st.markdown(f"**{'Root' if d == 0 else 'Category' if d == 1 else 'Concept'}**")
            entries = [e for e in legend_entries if e['depth'] == d]
            cols = st.columns(min(4, max(1, len(entries))))
            for i, entry in enumerate(entries):
                with cols[i % len(cols)]:
                    st.markdown(f"""<div style='padding:8px; border-radius:6px; background-color:{entry['color']}22; border-left:4px solid {entry['color']}; margin-bottom:6px; font-size:{legend_font_size}px;'>
                    <span style='font-size:{legend_font_size+4}px; color:{entry['color']}; margin-right:6px;'>{entry['symbol']}</span>
                    <span style='font-size:{legend_font_size}px; color:{theme.get("font", "#333")}; font-weight:500;'>{entry['label']}</span>
                    <span style='font-size:{legend_font_size-1}px; color:#666; float:right;'>({entry['value']:.0f})</span></div>""", unsafe_allow_html=True)


def render_radar_chart(
    distill_df, top_k=15, cmap_name="viridis", theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if distill_df.empty or top_k == 0:
        st.info("No data available for radar chart.")
        return
    df = distill_df.head(top_k).copy()
    if df.empty:
        return
    metrics = [
        'frequency', 'tfidf_weight', 'semantic_density', 'coherence_score',
    ]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        st.info("No metric columns available for radar chart.")
        return
    for m in available_metrics:
        max_val = df[m].max()
        if max_val > 0:
            df[f'{m}_norm'] = df[m] / max_val
        else:
            df[f'{m}_norm'] = 0
    fig = go.Figure()
    plot_df = df.head(min(top_k, 10))
    for i, row in plot_df.iterrows():
        values = [row[f'{m}_norm'] for m in available_metrics]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=row['concept'][:25],
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        showlegend=True,
        title=f"Concept Radar Chart (Top {min(top_k, 10)})",
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tsne_projection(
    valid_concepts: List[str], concept_abstract_map: Dict[str, List[int]],
    embed_model, theme: Dict = None, n_components: int = 2,
    perplexity: int = 30,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 10:
        st.info("Need at least 10 concepts for t-SNE projection.")
        return
    try:
        with torch.no_grad():
            embeddings = embed_model.encode(
                valid_concepts, show_progress_bar=False,
                batch_size=64, convert_to_numpy=True,
            )
        actual_perplexity = min(perplexity, len(valid_concepts) - 1)
        tsne = TSNE(
            n_components=n_components, random_state=42,
            perplexity=actual_perplexity,
        )
        coords = tsne.fit_transform(embeddings)
        category_map = abstract_concepts_to_categories(valid_concepts)
        categories = [category_map.get(c, 'general') for c in valid_concepts]
        freqs = [len(concept_abstract_map.get(c, [])) for c in valid_concepts]
        if n_components == 2:
            fig = px.scatter(
                x=coords[:, 0], y=coords[:, 1],
                color=categories, size=freqs,
                hover_name=valid_concepts,
                title="t-SNE Projection of Concept Embeddings",
                labels={'color': 'Category', 'size': 'Frequency'},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
        else:
            fig = px.scatter_3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                color=categories, size=freqs,
                hover_name=valid_concepts,
                title="3D t-SNE Projection of Concept Embeddings",
                labels={'color': 'Category', 'size': 'Frequency'},
            )
        fig.update_layout(
            paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
            font_color=theme.get("font", "#000000"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        del embeddings, coords
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()
    except Exception as e:
        st.error(f"t-SNE projection failed: {e}")


def render_community_detection(
    nx_graph, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(nx_graph.nodes()) < 3:
        st.info("Need at least 3 nodes for community detection.")
        return
    try:
        from networkx.algorithms import community
        communities = list(community.greedy_modularity_communities(nx_graph))
        node_to_comm: Dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        pos = nx.spring_layout(nx_graph, seed=42)
        cmap_colors = get_colormap_colors(
            "tab20", max(len(communities), 1)
        )
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        for u, v in nx_graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(width=0.8, color=theme['edge_unknown']),
            hoverinfo='none',
        )
        node_traces: List[go.Scatter] = []
        for i, comm in enumerate(communities):
            comm_nodes = list(comm)
            node_x: List[float] = []
            node_y: List[float] = []
            node_text: List[str] = []
            node_size: List[int] = []
            for node in comm_nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                deg = nx_graph.degree(node)
                freq = len(concept_abstract_map.get(node, []))
                node_text.append(
                    f"{node}<br>Community {i}<br>"
                    f"Degree: {deg}<br>Freq: {freq}"
                )
                node_size.append(max(10, min(30, deg * 2 + 8)))
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=cmap_colors[i % len(cmap_colors)],
                    line=dict(width=1.5, color='white'),
                ),
                text=comm_nodes, textposition="bottom center",
                textfont=dict(size=8, color=theme['font']),
                hovertext=node_text, hoverinfo='text',
                name=f"Community {i} ({len(comm_nodes)})",
            )
            node_traces.append(node_trace)
        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                showlegend=True, hovermode='closest',
                title=f"Community Detection ({len(communities)} communities)",
                margin=dict(b=0, l=0, r=0, t=40),
                plot_bgcolor=theme['plotly_bg'],
                paper_bgcolor=theme['plotly_paper'],
                font=dict(color=theme['font']),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        comm_data: List[Dict[str, Any]] = []
        for i, comm in enumerate(communities):
            comm_data.append({
                "Community": i,
                "Size": len(comm),
                "Top Concepts": ", ".join(
                    sorted(
                        comm,
                        key=lambda c: len(concept_abstract_map.get(c, [])),
                        reverse=True,
                    )[:5]
                ),
            })
        st.dataframe(pd.DataFrame(comm_data), use_container_width=True)
    except Exception as e:
        st.warning(f"Community detection failed: {e}")


def render_concept_growth(
    df_filtered, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No 'Year' data available for growth analysis.")
        return
    years = df_filtered["Year"].dropna().astype(int)
    if len(years) == 0:
        st.info("No valid year data found.")
        return
    mid_year = int(years.median())
    early_df = df_filtered[df_filtered["Year"] <= mid_year]
    recent_df = df_filtered[df_filtered["Year"] > mid_year]
    if len(early_df) == 0 or len(recent_df) == 0:
        st.info("Need data from both early and recent periods.")
        return
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:15]
    growth_data: List[Dict[str, Any]] = []
    for concept in top_concepts:
        early_count = 0
        recent_count = 0
        for idx, row in early_df.iterrows():
            text = " ".join([
                str(row[col]) for col in df_filtered.columns
                if pd.notna(row[col])
            ])
            early_count += len(re.findall(
                r'\b' + re.escape(concept) + r'\b', text, re.I
            ))
        for idx, row in recent_df.iterrows():
            text = " ".join([
                str(row[col]) for col in df_filtered.columns
                if pd.notna(row[col])
            ])
            recent_count += len(re.findall(
                r'\b' + re.escape(concept) + r'\b', text, re.I
            ))
        growth_rate = (
            ((recent_count - early_count) / max(early_count, 1)) * 100
            if early_count > 0 else 0
        )
        growth_data.append({
            "Concept": concept,
            "Early Count": early_count,
            "Recent Count": recent_count,
            "Growth Rate (%)": growth_rate,
        })
    growth_df = pd.DataFrame(growth_data).sort_values(
        "Growth Rate (%)", ascending=False
    )
    fig = px.bar(
        growth_df, x="Concept", y="Growth Rate (%)",
        color="Growth Rate (%)", color_continuous_scale="RdYlGn",
        title=(
            f"Concept Growth Rate "
            f"(Early <={mid_year} vs Recent >{mid_year})"
        ),
        labels={"Growth Rate (%)": "Growth Rate (%)"},
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(growth_df, use_container_width=True)


def render_bubble_chart(
    nx_graph, valid_concepts, concept_abstract_map, distill_df, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if len(valid_concepts) < 3:
        st.info("Need at least 3 concepts for bubble chart.")
        return
    category_map = abstract_concepts_to_categories(valid_concepts)
    bubble_data: List[Dict[str, Any]] = []
    for concept in valid_concepts:
        degree = nx_graph.degree(concept) if concept in nx_graph else 0
        freq = len(concept_abstract_map.get(concept, []))
        efficiency = distill_df[
            distill_df['concept'] == concept
        ]['distillation_efficiency'].values
        efficiency = (
            float(efficiency[0]) if len(efficiency) > 0 else 0.0
        )
        category = category_map.get(concept, 'general')
        bubble_data.append({
            "Concept": concept, "Degree": degree,
            "Frequency": freq,
            "Distillation Efficiency": efficiency,
            "Category": category,
        })
    bubble_df = pd.DataFrame(bubble_data)
    fig = px.scatter(
        bubble_df, x="Degree", y="Frequency",
        size="Distillation Efficiency", color="Category",
        hover_data=["Concept"],
        title="Concept Importance Bubble Chart",
        size_max=50,
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# INTERACTIVE GRAPH EDITING (WITH UNDO/REDO)
# ============================================================================
def apply_graph_edits(
    nx_graph, valid_concepts, concept_to_id, id_to_concept,
    concept_abstract_map,
    nodes_to_remove=None, nodes_to_merge=None, merge_name=None,
    new_edge=None, new_edge_weight=1.0, min_degree=0, min_freq=0,
):
    edited = False
    if nodes_to_remove:
        for node in nodes_to_remove:
            if node in nx_graph:
                nx_graph.remove_node(node)
                edited = True
        valid_concepts = [
            c for c in valid_concepts if c not in nodes_to_remove
        ]
        for node in nodes_to_remove:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
    if nodes_to_merge and merge_name and len(nodes_to_merge) >= 2:
        merged_edges: Dict[str, Dict[str, Any]] = {}
        merged_freq = 0
        merged_abstracts: Set[int] = set()
        for node in nodes_to_merge:
            if node in nx_graph:
                for neighbor in list(nx_graph.neighbors(node)):
                    if neighbor not in nodes_to_merge:
                        w = nx_graph[node][neighbor].get('weight', 1)
                        cooc = nx_graph[node][neighbor].get('cooccurrence', 0)
                        sem = nx_graph[node][neighbor].get('semantic', 0)
                        etype = nx_graph[node][neighbor].get('edge_type', 'unknown')
                        if neighbor in merged_edges:
                            merged_edges[neighbor]['weight'] += w
                            merged_edges[neighbor]['cooccurrence'] += cooc
                            merged_edges[neighbor]['semantic'] += sem
                        else:
                            merged_edges[neighbor] = {
                                'weight': w, 'cooccurrence': cooc,
                                'semantic': sem, 'edge_type': etype,
                            }
                merged_freq += nx_graph.nodes[node].get('frequency', 0)
                if node in concept_abstract_map:
                    merged_abstracts.update(concept_abstract_map[node])
                nx_graph.remove_node(node)
        nx_graph.add_node(merge_name, frequency=merged_freq)
        for neighbor, edge_data in merged_edges.items():
            nx_graph.add_edge(merge_name, neighbor, **edge_data)
        concept_abstract_map[merge_name] = list(merged_abstracts)
        valid_concepts = [
            c for c in valid_concepts if c not in nodes_to_merge
        ]
        if merge_name not in valid_concepts:
            valid_concepts.append(merge_name)
        for node in nodes_to_merge:
            if node in concept_abstract_map and node != merge_name:
                del concept_abstract_map[node]
        edited = True
    if new_edge and len(new_edge) == 2:
        u, v = new_edge
        if (
            u in nx_graph and v in nx_graph
            and not nx_graph.has_edge(u, v)
        ):
            nx_graph.add_edge(
                u, v, weight=new_edge_weight,
                cooccurrence=0, semantic=0, edge_type='manual',
            )
            edited = True
    if min_degree > 0:
        low_degree = [
            n for n in nx_graph.nodes() if nx_graph.degree(n) < min_degree
        ]
        for node in low_degree:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c not in low_degree]
        for node in low_degree:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
        edited = True
    if min_freq > 0:
        low_freq = [
            n for n in nx_graph.nodes()
            if nx_graph.nodes[n].get('frequency', 0) < min_freq
        ]
        for node in low_freq:
            nx_graph.remove_node(node)
        valid_concepts = [c for c in valid_concepts if c not in low_freq]
        for node in low_freq:
            if node in concept_abstract_map:
                del concept_abstract_map[node]
        edited = True
    valid_concepts = sorted(set(valid_concepts))
    concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
    id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
    return (
        nx_graph, valid_concepts, concept_to_id,
        id_to_concept, concept_abstract_map, edited,
    )


# ============================================================================
# GRAPH METRICS DASHBOARD
# ============================================================================
def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    if G.number_of_nodes() == 0:
        return {}
    metrics: Dict[str, Any] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
        "connected_components": nx.number_connected_components(G),
        "avg_clustering": (
            nx.average_clustering(G) if G.number_of_nodes() > 2 else 0
        ),
    }
    try:
        bc = nx.betweenness_centrality(
            G, normalized=True, k=min(100, G.number_of_nodes())
        )
        top_bridges = sorted(
            bc.items(), key=lambda x: x[1], reverse=True
        )[:10]
        metrics["top_bridges"] = top_bridges
        metrics["avg_betweenness"] = np.mean(list(bc.values()))
    except Exception:
        metrics["top_bridges"] = []
    return metrics


def display_metric_dashboard(metrics: Dict, theme=None) -> None:
    if not metrics:
        st.warning("No graph metrics available.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", metrics["nodes"])
    col2.metric("Edges", metrics["edges"])
    col3.metric("Density", f"{metrics['density']:.3f}")
    col4.metric("Avg Degree", f"{metrics['avg_degree']:.2f}")
    col5, col6, col7 = st.columns(3)
    col5.metric("Clustering", f"{metrics['clustering']:.3f}")
    col6.metric("Components", metrics["connected_components"])
    col7.metric(
        "Avg Betweenness", f"{metrics.get('avg_betweenness', 0):.3f}"
    )
    if metrics.get("top_bridges"):
        st.markdown("**Top Bridge Concepts (High Betweenness)**")
        bridge_df = pd.DataFrame(
            metrics["top_bridges"], columns=["Concept", "Bridge Score"]
        )
        st.dataframe(bridge_df, use_container_width=True)


# ============================================================================
# EXTRA VISUALIZATIONS
# ============================================================================
def render_concept_timeline(
    df_filtered, valid_concepts, concept_abstract_map, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    if "Year" not in df_filtered.columns or df_filtered["Year"].isna().all():
        st.info("No 'Year' data available for timeline visualization.")
        return
    years = df_filtered["Year"].dropna().astype(int)
    if len(years) == 0:
        st.info("No valid year data found.")
        return
    year_range = sorted(years.unique())
    if len(year_range) < 2:
        st.info("Need at least 2 different years for timeline.")
        return
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:10]
    timeline_data: List[Dict[str, Any]] = []
    for year in year_range:
        year_mask = df_filtered["Year"] == year
        year_df = df_filtered[year_mask]
        year_text = ""
        for idx, row in year_df.iterrows():
            for col in df_filtered.columns:
                if pd.notna(row[col]):
                    year_text += " " + str(row[col])
        for concept in top_concepts:
            count = len(re.findall(
                r'\b' + re.escape(concept) + r'\b', year_text, re.I
            ))
            timeline_data.append({
                "Year": year, "Concept": concept, "Count": count,
            })
    if not timeline_data:
        st.info("No timeline data to display.")
        return
    timeline_df = pd.DataFrame(timeline_data)
    fig = px.line(
        timeline_df, x="Year", y="Count", color="Concept",
        title="Concept Frequency Over Time",
        labels={"Count": "Mentions", "Year": "Publication Year"},
        template=(
            "plotly_white" if theme == THEME_PRESETS["Bright (Default)"]
            else "plotly_dark"
        ),
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cooccurrence_heatmap(
    nx_graph, valid_concepts, concept_abstract_map,
    top_n=30, theme=None,
) -> None:
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    top_concepts = sorted(
        valid_concepts,
        key=lambda c: len(concept_abstract_map.get(c, [])),
        reverse=True,
    )[:top_n]
    if len(top_concepts) < 3:
        st.info("Need at least 3 concepts for heatmap.")
        return
    n = len(top_concepts)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(top_concepts):
        for j, c2 in enumerate(top_concepts):
            if i == j:
                matrix[i][j] = len(concept_abstract_map.get(c1, []))
            elif nx_graph.has_edge(c1, c2):
                matrix[i][j] = nx_graph[c1][c2].get('cooccurrence', 0)
    fig = px.imshow(
        matrix, x=top_concepts, y=top_concepts,
        labels=dict(x="Concept", y="Concept", color="Co-occurrence"),
        title=f"Co-occurrence Heatmap (Top {n} Concepts)",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        font_color=theme.get("font", "#000000"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def export_graph(
    nx_graph, concept_abstract_map, export_format: str,
    include_metadata: bool = True,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    if export_format == "GraphML":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '7.0'
                nx_graph.graph['tool'] = 'LIB-ConceptGraph'
            try:
                nx.write_graphml_lxml(nx_graph, "lib_graph.graphml")
            except Exception:
                nx.write_graphml(nx_graph, "lib_graph.graphml")
            with open("lib_graph.graphml", "rb") as f:
                return f.read(), "application/graphml+xml", "lib_graph.graphml"
        except Exception as e:
            st.error(f"GraphML export failed: {e}")
            return None, None, None
    elif export_format == "JSON (Full Metadata)":
        data = nx.node_link_data(nx_graph)
        if include_metadata:
            data['metadata'] = {
                'created': datetime.now().isoformat(),
                'version': '7.0',
                'tool': 'LIB-ConceptGraph',
                'node_count': len(nx_graph.nodes()),
                'edge_count': len(nx_graph.edges()),
                'inferred_edges': sum(
                    1 for u, v, d in nx_graph.edges(data=True)
                    if d.get('inferred', False)
                ),
                'categories': list(set(
                    abstract_concepts_to_categories(
                        list(nx_graph.nodes())
                    ).values()
                )),
            }
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "lib_graph_full.json"
    elif export_format == "JSON (Compact)":
        data = nx.node_link_data(nx_graph)
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8'), "application/json", "lib_graph.json"
    elif export_format == "CSV (Edges + Metadata)":
        edge_data: List[Dict[str, Any]] = []
        for u, v, data in nx_graph.edges(data=True):
            row = {
                "source": u, "target": v,
                "weight": data.get('weight', 1),
                "cooccurrence": data.get('cooccurrence', 0),
                "semantic_similarity": data.get('semantic', 0),
                "edge_type": data.get('edge_type', 'unknown'),
                "inferred": data.get('inferred', False),
                "confidence": data.get('confidence', 1.0),
                "path": data.get('path', ''),
            }
            edge_data.append(row)
        csv_df = pd.DataFrame(edge_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "lib_edges_enhanced.csv"
    elif export_format == "CSV (Nodes + Metadata)":
        node_data: List[Dict[str, Any]] = []
        for node in nx_graph.nodes():
            row = {
                "concept": node,
                "frequency": len(concept_abstract_map.get(node, [])),
                "degree": nx_graph.degree(node),
                "concept_type": nx_graph.nodes[node].get('concept_type', 'general'),
                "definition": nx_graph.nodes[node].get('definition', ''),
                "category": abstract_concepts_to_categories([node]).get(node, 'general'),
            }
            row.update({
                k: v for k, v in nx_graph.nodes[node].items()
                if isinstance(v, (str, int, float, bool))
            })
            node_data.append(row)
        csv_df = pd.DataFrame(node_data)
        return csv_df.to_csv(index=False).encode('utf-8'), "text/csv", "lib_nodes_enhanced.csv"
    elif export_format == "PNG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=300)
            node_colors = [
                get_lib_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='png', dpi=300,
                bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/png", "lib_graph.png"
        except Exception as e:
            st.error(f"PNG export failed: {e}")
            return None, None, None
    elif export_format == "SVG":
        try:
            pos = nx.spring_layout(nx_graph, seed=42)
            plt.figure(figsize=(14, 12), dpi=150)
            node_colors = [
                get_lib_category_color(n) for n in nx_graph.nodes()
            ]
            nx.draw(
                nx_graph, pos, with_labels=True,
                node_color=node_colors, edge_color='gray',
                node_size=400, font_size=7, font_weight='bold',
                edgecolors='white', linewidths=1,
            )
            buf = io.BytesIO()
            plt.savefig(
                buf, format='svg', bbox_inches='tight', facecolor='white',
            )
            buf.seek(0)
            plt.close()
            return buf.read(), "image/svg+xml", "lib_graph.svg"
        except Exception as e:
            st.error(f"SVG export failed: {e}")
            return None, None, None
    elif export_format == "GEXF":
        try:
            if include_metadata:
                nx_graph.graph['created'] = datetime.now().isoformat()
                nx_graph.graph['version'] = '7.0'
            nx.write_gexf(nx_graph, "lib_graph.gexf")
            with open("lib_graph.gexf", "rb") as f:
                return f.read(), "application/xml", "lib_graph.gexf"
        except Exception as e:
            st.error(f"GEXF export failed: {e}")
            return None, None, None
    return None, None, None


# ============================================================================
# REASONING DASHBOARD
# ============================================================================
def render_reasoning_dashboard(
    nx_graph, valid_concepts, ontology, extractor,
) -> None:
    st.subheader("🔍 Ontology-Based Reasoning Insights")
    type_counts: Dict[str, int] = defaultdict(int)
    for c in valid_concepts:
        if c in ontology.concepts:
            type_counts[ontology.concepts[c].concept_type.value] += 1
        else:
            type_counts["unknown"] += 1
    fig = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title="Concept Type Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)
    inferred_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if d.get('inferred', False)
    ]
    observed_edges = [
        (u, v) for u, v, d in nx_graph.edges(data=True)
        if not d.get('inferred', False)
    ]
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed Edges", len(observed_edges))
    col2.metric("Inferred Edges", len(inferred_edges))
    col3.metric(
        "Inference Ratio",
        f"{len(inferred_edges) / max(len(observed_edges), 1):.2f}",
    )
    rel_types: Dict[str, int] = defaultdict(int)
    for u, v, d in nx_graph.edges(data=True):
        rel_types[d.get('edge_type', 'unknown')] += 1
    if rel_types:
        rel_df = pd.DataFrame(
            [(k, v) for k, v in rel_types.items()],
            columns=['Relationship Type', 'Count'],
        )
        rel_df = rel_df.sort_values('Count', ascending=False)
        st.dataframe(rel_df, use_container_width=True)
        fig = px.bar(
            rel_df, x='Relationship Type', y='Count',
            title="Edge Type Distribution",
            color='Relationship Type',
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔗 Inferred Material-Property Chains")
    material_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.MATERIAL
    ]
    property_nodes = [
        c for c in valid_concepts
        if c in ontology.concepts
        and ontology.concepts[c].concept_type == ConceptType.PROPERTY
    ]
    chains_found: List[Dict[str, Any]] = []
    for mat in material_nodes[:5]:
        for prop in property_nodes[:5]:
            paths = ontology.infer_path(mat, prop, max_depth=3)
            if paths:
                chains_found.append({
                    "Material": mat,
                    "Property": prop,
                    "Path Length": len(paths[0]),
                    "Path": " → ".join(paths[0]),
                })
    if chains_found:
        st.dataframe(pd.DataFrame(chains_found), use_container_width=True)
    else:
        st.info(
            "No direct inference chains found. "
            "Build graph with more concepts."
        )
    st.subheader("📚 Synonym Resolution Examples")
    synonym_examples = [
        ("nmc811", "nmc811"),
        ("lifepo4", "lfp"),
        ("silicon anode", "silicon"),
        ("electrode thickness", "electrode_thickness"),
    ]
    syn_data: List[Dict[str, Any]] = []
    for original, expected in synonym_examples:
        resolved = ontology.resolve_concept(original)
        syn_data.append({
            "Original": original,
            "Expected": expected,
            "Resolved": resolved,
            "Match": (
                "✅" if resolved == expected
                else ("⚠️" if resolved else "❌")
            ),
        })
    st.dataframe(pd.DataFrame(syn_data), use_container_width=True)
    st.subheader("🏛️ Concept Hierarchy")
    hierarchy_data: List[Dict[str, str]] = []
    for concept in valid_concepts[:20]:
        if concept in ontology.concepts:
            node = ontology.concepts[concept]
            if node.hypernyms:
                for hyp in node.hypernyms:
                    hierarchy_data.append({
                        "Child": concept, "Parent": hyp,
                        "Relation": "is-a",
                    })
            if node.hyponyms:
                for hyp in node.hyponyms:
                    if hyp in valid_concepts:
                        hierarchy_data.append({
                            "Parent": concept, "Child": hyp,
                            "Relation": "has-subtype",
                        })
    if hierarchy_data:
        st.dataframe(
            pd.DataFrame(hierarchy_data), use_container_width=True,
        )
    else:
        st.info(
            "No hierarchical relationships found in current concept set."
        )


# ============================================================================
# BATCH PROCESSING MODE v6.0 (Streamlit Cloud ≤ 1 GB RAM)
# ============================================================================
def get_memory_usage_mb() -> float:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return 0.0


def split_into_batches(
    df: pd.DataFrame, batch_size: int
) -> Iterator[Tuple[int, pd.DataFrame]]:
    total_batches = math.ceil(len(df) / batch_size)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        yield start_idx, df.iloc[start_idx:end_idx]


def merge_graphs(existing_graph: nx.Graph, new_graph: nx.Graph) -> nx.Graph:
    merged = existing_graph
    for node, data in new_graph.nodes(data=True):
        if node in merged:
            merged.nodes[node]["frequency"] = (
                merged.nodes[node].get("frequency", 0)
                + data.get("frequency", 0)
            )
            for attr in ("concept_type", "definition"):
                if not merged.nodes[node].get(attr) and data.get(attr):
                    merged.nodes[node][attr] = data[attr]
        else:
            merged.add_node(node, **data)
    for u, v, data in new_graph.edges(data=True):
        if merged.has_edge(u, v):
            ed = merged[u][v]
            ed["cooccurrence"] = (
                ed.get("cooccurrence", 0) + data.get("cooccurrence", 0)
            )
            ed["semantic"] = max(
                ed.get("semantic", 0) or 0, data.get("semantic", 0) or 0
            )
            ed["inferred"] = bool(ed.get("inferred", False)) or bool(
                data.get("inferred", False)
            )
            if data.get("confidence") is not None:
                ed["confidence"] = max(
                    ed.get("confidence", 0), data["confidence"]
                )
            if data.get("path") and not ed.get("path"):
                ed["path"] = data["path"]
            if (
                ed.get("edge_type", "cooccurrence") == "cooccurrence"
                and data.get("edge_type") not in (None, "cooccurrence")
            ):
                ed["edge_type"] = data["edge_type"]
        else:
            merged.add_edge(u, v, **data)
    return merged


def recompute_edge_weights(nx_graph: nx.Graph, config: Dict) -> None:
    cooc_w = config.get("COOCCURRENCE_WEIGHT", 0.7)
    sem_w = config.get("SEMANTIC_WEIGHT", 0.2)
    inf_w = config.get("INFERENCE_WEIGHT", 0.1)
    for _, _, data in nx_graph.edges(data=True):
        cooc = data.get("cooccurrence", 0)
        sem = data.get("semantic", 0) or 0
        inf = 1.0 if data.get("inferred", False) else 0.0
        conf = data.get("confidence", 0.5)
        data["weight"] = cooc_w * cooc + sem_w * sem + inf_w * inf * conf


def extract_doc_metrics(text: str) -> Dict[str, Any]:
    """Regex metric extraction for Li-ion battery literature."""
    metrics: Dict[str, Any] = {}
    
    capacity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:mah/g)', text, re.I)
    if capacity_matches:
        metrics['specific_capacity_mAh_g'] = [float(m) for m in capacity_matches]
    
    energy_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:wh/kg)', text, re.I)
    if energy_matches:
        metrics['energy_density_Wh_kg'] = [float(m) for m in energy_matches]
    
    cycle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:cycles)', text, re.I)
    if cycle_matches:
        metrics['cycle_life_cycles'] = [float(m) for m in cycle_matches]
    
    impedance_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:ohm|Ω)', text, re.I)
    if impedance_matches:
        metrics['impedance_Ohm'] = [float(m) for m in impedance_matches]
    
    thickness_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:μm|um)', text, re.I)
    if thickness_matches:
        metrics['electrode_thickness_um'] = [float(m) for m in thickness_matches]
    
    temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I)
    if temp_matches:
        metrics['temperature_C'] = [float(m) for m in temp_matches]
    
    return metrics


class IncrementalGraphBuilder(ReasoningEnhancedGraphBuilder):
    @timed
    def build_batch_graph(
        self,
        batch_concepts: List[List[str]],
        valid_concepts: List[str],
        concept_to_id: Dict[str, int],
        batch_doc_freq: Dict[str, int],
        embed_model=None,
        config: Dict = None,
    ) -> nx.Graph:
        if config is None:
            config = get_adaptive_config(1000)
        nx_graph = nx.Graph()
        for c in valid_concepts:
            concept_type = self.ontology.get_concept_type(c)
            definition = self.ontology.get_definition(c)
            nx_graph.add_node(
                c,
                frequency=batch_doc_freq.get(c, 0),
                concept_type=concept_type.value,
                definition=definition,
                degree=0,
            )
        cooccurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)
        for concepts in batch_concepts:
            valid_in_doc = [c for c in concepts if c in concept_to_id]
            for i in range(len(valid_in_doc)):
                for j in range(i + 1, len(valid_in_doc)):
                    u, v = valid_in_doc[i], valid_in_doc[j]
                    if u != v:
                        key = tuple(sorted([u, v]))
                        cooccurrence_map[key] += 1
        for (u, v), count in cooccurrence_map.items():
            nx_graph.add_edge(
                u, v,
                weight=float(count),
                cooccurrence=count,
                semantic=0.0,
                edge_type='cooccurrence',
                inferred=False,
            )
        if embed_model and len(valid_concepts) >= 10:
            self._add_semantic_edges(
                nx_graph, valid_concepts, embed_model, config
            )
        if st.session_state.get('use_inference', True):
            self._add_inferred_edges(nx_graph, valid_concepts)
        self._add_hierarchical_edges(nx_graph, valid_concepts)
        self._compute_final_weights(nx_graph, config)
        return nx_graph


def reset_batch_state(clear_analysis: bool = False) -> None:
    st.session_state.batch_state = None
    st.session_state.pop("batch_trigger", None)
    if clear_analysis:
        st.session_state.analysis_data = None
        st.session_state.burst_df = None
        st.session_state.drift_df = None
        st.session_state.genealogy_df = None
        st.session_state.bridge_df = None
        st.session_state.motifs = {}
        st.session_state.edit_history = GraphEditHistory()
    gc.collect()
    if torch.cuda.is_available():
        maybe_empty_cache()


def render_batch_processing_controls() -> None:
    st.markdown("---")
    st.subheader("📦 Batch Processing (≤1 GB RAM)")
    st.toggle(
        "Enable batch processing",
        key="batch_mode",
        help=(
            "Process documents in small batches with incremental graph "
            "merging and memory cleanup after each batch. Recommended for "
            "Streamlit Cloud free tier (1 GB RAM)."
        ),
    )
    if not st.session_state.get("batch_mode", False):
        return
    st.slider(
        "Batch size (documents)", 100, 2000, 1000, 100,
        key="batch_size",
        help="Smaller batches = lower peak memory but more merge steps.",
    )
    st.slider(
        "GNN epochs (final training)", 10, 50, 40, 5,
        key="batch_gnn_epochs",
        help="GNN is trained ONCE on the final merged graph.",
    )
    bs = st.session_state.get("batch_state")
    if bs:
        total = max(bs.get("total_batches", 1), 1)
        done = bs.get("next_batch", 0)
        st.progress(done / total)
        st.caption(
            f"Batch {done}/{total} • "
            f"{bs.get('docs_processed', len(bs.get('all_texts', {})))} "
            f"docs processed • "
            f"{len(bs.get('all_texts', {}))} texts cached"
        )
    col_next, col_all = st.columns(2)
    with col_next:
        if st.button(
            "▶️ Next batch", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "next"
    with col_all:
        if st.button(
            "⏩ All remaining", use_container_width=True,
            disabled=bool(bs and bs.get("done")),
        ):
            st.session_state["batch_trigger"] = "all"
    if bs:
        if st.button("🗑️ Reset batch state", use_container_width=True):
            reset_batch_state(clear_analysis=True)
            st.success("Batch state cleared!")
            st.rerun()
    else:
        st.caption(
            "Click 🚀 Build Concept Graph (or ▶️ Next batch) to start."
        )


BATCH_TEXT_STORE_CAP = 4000


def run_batch_analysis(
    df_filtered: pd.DataFrame,
    selected_text_cols: List[str],
    ontology: DomainOntology,
    run_mode: str = "all",
) -> None:
    overall_start = time.perf_counter()
    if 'qa_factory' in st.session_state:
        factory = st.session_state.qa_factory
        for analyzer in factory._local_cache.values():
            if hasattr(analyzer, 'unload_model'):
                analyzer.unload_model()
        factory._local_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        maybe_empty_cache()

    try:
        cpu_count = os.cpu_count() or 2
        torch.set_num_threads(min(4, max(2, cpu_count // 2)))
    except Exception:
        pass
    batch_size = int(st.session_state.get("batch_size", 1000))
    total_docs = len(df_filtered)
    if total_docs == 0:
        st.error("No documents to process.")
        return
    total_batches = math.ceil(total_docs / batch_size)

    data_hash = hashlib.md5(
        (
            f"{total_docs}|{'|'.join(selected_text_cols)}|"
            f"{df_filtered.index.min()}|{df_filtered.index.max()}"
        ).encode("utf-8")
    ).hexdigest()

    bs = st.session_state.get("batch_state")
    if bs is not None and (
        bs.get("data_hash") != data_hash
        or bs.get("batch_size") != batch_size
    ):
        st.info("Dataset or batch size changed — resetting batch state.")
        reset_batch_state(clear_analysis=False)
        bs = None
    if bs is None:
        bs = {
            "data_hash": data_hash,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "next_batch": 0,
            "all_concepts": [],
            "all_metrics": [],
            "all_texts": {},
            "valid_doc_indices": set(),
            "docs_processed": 0,
            "concept_freq": defaultdict(int),
            "concept_abstract_map": defaultdict(list),
            "merged_graph": None,
            "extractor": None,
            "resolver": None,
            "builder": None,
            "done": False,
        }
        st.session_state.batch_state = bs

    if bs["done"]:
        st.success("✅ All batches already processed — see results below.")
        return

    _query_whitelist = st.session_state.get('last_query_whitelist', None)
    _is_query_focused = (
        st.session_state.get('query_focused_build', False)
        and _query_whitelist is not None
        and len(_query_whitelist) > 0
    )

    config = get_adaptive_config(total_docs)
    config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
    config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
    config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
    config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
    config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
    config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)

    if _is_query_focused:
        wl_size = len(_query_whitelist)
        if wl_size <= 15:
            config["MIN_CONCEPT_FREQ"] = 1
        elif wl_size <= 50:
            config["MIN_CONCEPT_FREQ"] = 2
        else:
            config["MIN_CONCEPT_FREQ"] = min(config["MIN_CONCEPT_FREQ"], 3)
        config["USE_SEMANTIC_CLUSTERING"] = False
        st.info(
            f"🎯 Query-focused batch mode: {wl_size} whitelisted concepts. "
            f"MIN_CONCEPT_FREQ lowered to {config['MIN_CONCEPT_FREQ']}."
        )

    use_ontology = st.session_state.get('use_ontology', True)
    embed_model = load_embedding_model()

    if use_ontology and bs["extractor"] is None:
        with st.spinner("Initializing ontology resolver (one-time)..."):
            resolver = AdvancedConceptResolver(
                ontology, embed_model, cache_max=2000,
            )
            extractor = EnhancedConceptExtractor(
                ontology, resolver,
                store_contexts=False, store_documents=False,
            )
            builder = IncrementalGraphBuilder(ontology, extractor)
            bs["resolver"] = resolver
            bs["extractor"] = extractor
            bs["builder"] = builder
            st.session_state.resolver = resolver
            st.session_state.extractor = extractor
        gc.collect()

    pending = list(range(bs["next_batch"], total_batches))
    if run_mode == "next":
        pending = pending[:1]
    if not pending:
        st.success("✅ Nothing left to process.")
        return

    progress_bar = st.progress(0.0)
    status = st.status("📦 Batch processing running...", expanded=True)

    def _process_one_batch(batch_num: int) -> None:
        start = batch_num * batch_size
        end = min(start + batch_size, total_docs)
        batch_df = df_filtered.iloc[start:end]
        n_this = len(batch_df)
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        with status:
            st.write(
                f"📦 Batch {batch_num + 1}/{total_batches} — "
                f"docs {start}–{end - 1} ({n_this} docs)"
            )
        batch_concepts: List[List[str]] = []
        batch_metrics: List[Dict] = []
        batch_doc_freq: Dict[str, int] = defaultdict(int)
        extractor = bs["extractor"]
        whitelist = st.session_state.get('last_query_whitelist', None)

        for local_i, (_, row) in enumerate(batch_df.iterrows()):
            text = " ".join([
                str(row[col]) for col in selected_text_cols
                if col in row and pd.notna(row[col])
            ])
            if use_ontology and extractor is not None:
                concepts = extractor.extract_from_text(
                    text, start + local_i,
                    allowed_concepts=whitelist
                )
            else:
                concepts = extract_concepts_from_text(text)
            batch_concepts.append(concepts)
            batch_metrics.append(extract_doc_metrics(text))
            unique_concepts = set(concepts)
            for c in unique_concepts:
                batch_doc_freq[c] += 1
                bs["concept_freq"][c] += 1
                bs["concept_abstract_map"][c].append(start + local_i)
            has_valid = any(
                bs["concept_freq"].get(c, 0) >= min_freq
                for c in unique_concepts
            )
            if has_valid:
                bs["all_texts"][start + local_i] = (
                    text[:BATCH_TEXT_STORE_CAP]
                )
                bs["valid_doc_indices"].add(start + local_i)
            bs["docs_processed"] += 1
            del text
            if (local_i + 1) % 100 == 0 or (local_i + 1) == n_this:
                frac = (batch_num + (local_i + 1) / n_this) / total_batches
                progress_bar.progress(min(0.90 * frac, 0.90))
                with status:
                    st.write(f"  … {local_i + 1}/{n_this} docs extracted")

        bs["all_concepts"].extend(batch_concepts)
        bs["all_metrics"].extend(batch_metrics)

        if _is_query_focused and _query_whitelist:
            batch_unique_global = set()
            for cs in batch_concepts:
                batch_unique_global.update(cs)
            _hits = batch_unique_global & _query_whitelist
            with status:
                st.write(
                    f"  🎯 Whitelist matches this batch: "
                    f"{len(_hits)}/{len(_query_whitelist)} "
                    f"({', '.join(sorted(_hits)[:6])}{'...' if len(_hits) > 6 else ''})"
                )

        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        batch_unique: Set[str] = set()
        for cs in batch_concepts:
            batch_unique.update(cs)
        batch_valid = [
            c for c in batch_unique
            if bs["concept_freq"].get(c, 0) >= min_freq
        ]
        batch_valid.sort(
            key=lambda c: bs["concept_freq"][c], reverse=True
        )
        batch_valid = batch_valid[:top_n]
        concept_to_id_batch = {c: i for i, c in enumerate(batch_valid)}

        if use_ontology and bs["builder"] is not None:
            batch_graph = bs["builder"].build_batch_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                batch_doc_freq, embed_model, config,
            )
        else:
            batch_graph = build_hybrid_graph(
                batch_concepts, batch_valid, concept_to_id_batch,
                embed_model, config, ontology,
            )

        if bs["merged_graph"] is None:
            bs["merged_graph"] = batch_graph
        else:
            bs["merged_graph"] = merge_graphs(bs["merged_graph"], batch_graph)
        recompute_edge_weights(bs["merged_graph"], config)
        bs["next_batch"] = batch_num + 1

        bs["all_concepts"] = []
        bs["all_metrics"] = []

        g = bs["merged_graph"]
        with status:
            st.write(
                f"✅ Batch {batch_num + 1} done — cumulative graph: "
                f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
                f"| peak RSS ≈ {get_memory_usage_mb():.0f} MB"
            )
        del batch_concepts, batch_metrics, batch_doc_freq
        del batch_graph, batch_df
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()

    def _finalize() -> None:
        merged = bs["merged_graph"]
        if merged is None or merged.number_of_nodes() == 0:
            st.error("No graph could be built from the processed batches.")
            return
        min_freq = config.get("MIN_CONCEPT_FREQ", 2)
        top_n = config.get("TOP_N_CONCEPTS", 1000)
        with status:
            st.write("🧩 Finalizing — selecting top concepts...")

        _wl = st.session_state.get('last_query_whitelist', set())
        _is_qf = st.session_state.get('query_focused_build', False)

        valid_concepts = [
            c for c, f in bs["concept_freq"].items()
            if f >= min_freq or (_is_qf and c in _wl)
        ]
        valid_concepts.sort(
            key=lambda c: (
                1 if c in _wl and _is_qf else 0,
                len(bs["concept_abstract_map"].get(c, [])),
            ),
            reverse=True,
        )
        valid_concepts = valid_concepts[:top_n]

        if _is_qf and _wl:
            for c in _wl:
                if c not in valid_concepts and c in bs["concept_freq"]:
                    valid_concepts.append(c)

        min_required = 3 if _is_qf else 5
        if len(valid_concepts) < min_required:
            st.error(
                f"Too few concepts extracted ({len(valid_concepts)}). "
                f"Whitelist hits: {len([c for c in _wl if c in bs['concept_freq']])}/"
                f"{len(_wl)}. Try lowering frequency thresholds."
            )
            return
        valid_set = set(valid_concepts)
        drop_nodes = [n for n in merged.nodes() if n not in valid_set]
        merged.remove_nodes_from(drop_nodes)
        del drop_nodes
        concept_to_id = {c: i for i, c in enumerate(valid_concepts)}
        id_to_concept = {i: c for i, c in enumerate(valid_concepts)}
        concept_abstract_map = {
            c: bs["concept_abstract_map"][c] for c in valid_concepts
        }
        progress_bar.progress(0.90)

        with status:
            st.write("🔢 Generating node embeddings...")
        try:
            with torch.no_grad():
                embeddings = embed_model.encode(
                    valid_concepts, show_progress_bar=False,
                    batch_size=32, convert_to_numpy=True,
                )
            node_features = torch.tensor(embeddings, dtype=torch.float32)
            del embeddings
        except Exception:
            node_features = torch.randn(len(valid_concepts), 384)
        gc.collect()

        with status:
            st.write("🧠 Training GraphSAGE (final, once)...")
        pos_pairs, neg_pairs = sample_edges_for_training(
            merged, valid_concepts, concept_to_id, config, memory_safe=True,
        )
        epochs = int(st.session_state.get("batch_gnn_epochs", 40))

        def _gnn_progress(epoch, loss):
            frac = 0.90 + (epoch / max(epochs, 1)) * 0.05
            progress_bar.progress(min(frac, 0.95))
            if epoch % 10 == 0:
                with status:
                    st.write(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

        gnn_model, final_emb, adj_indices, adj_values = train_gnn(
            node_features, merged, concept_to_id,
            pos_pairs, neg_pairs, _gnn_progress, epochs=epochs,
        )
        del pos_pairs, neg_pairs, adj_indices, adj_values
        gc.collect()

        with status:
            st.write("🎯 Scoring research directions...")
        concept_properties: Dict[str, float] = {}
        all_metrics = bs["all_metrics"]
        for concept in valid_concepts:
            values: List[float] = []
            for idx in concept_abstract_map.get(concept, []):
                if idx < len(all_metrics):
                    for metric_values in all_metrics[idx].values():
                        values.extend(metric_values)
            concept_properties[concept] = (
                float(np.median(values)) if values else 0.0
            )
        X_feat: List[List[float]] = []
        y_target: List[float] = []
        for u, v in merged.edges():
            pu = concept_properties.get(u, 0)
            pv = concept_properties.get(v, 0)
            w = merged[u][v].get('weight', 1)
            X_feat.append([pu, pv, w])
            y_target.append(
                max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
            )
        ridge = None
        if len(X_feat) > 5:
            ridge = Ridge(alpha=1.0).fit(
                np.array(X_feat), np.array(y_target)
            )
        top_scores = compute_research_direction_scores(
            gnn_model, node_features, final_emb, merged,
            valid_concepts, concept_properties, ridge, embed_model,
        )
        del X_feat, y_target, node_features
        gc.collect()

        with status:
            st.write("🧪 Distillation + advanced analytics...")
        distill_df = compute_concept_distillation(
            valid_concepts, concept_abstract_map, bs["all_texts"],
            max_docs_per_concept=30,
        )
        burst_df = None
        drift_df = None
        genealogy_df = None
        bridge_df = None
        motifs: Dict[str, Any] = {}
        try:
            burst_df = detect_keyword_bursts(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            drift_df = detect_semantic_drift(
                df_filtered, valid_concepts,
                concept_abstract_map, selected_text_cols,
            )
            genealogy_df = build_concept_genealogy(
                merged, valid_concepts, concept_abstract_map,
            )
            bridge_df = detect_cross_domain_bridges(
                merged, valid_concepts, concept_abstract_map,
            )
            motifs = analyze_network_motifs(merged)
        except Exception as e:
            st.warning(f"Some analytics skipped: {e}")
        st.session_state.burst_df = burst_df
        st.session_state.drift_df = drift_df
        st.session_state.genealogy_df = genealogy_df
        st.session_state.bridge_df = bridge_df
        st.session_state.motifs = motifs
        gc.collect()

        analysis_data = {
            "valid_concepts": valid_concepts,
            "concept_to_id": concept_to_id,
            "id_to_concept": id_to_concept,
            "concept_abstract_map": concept_abstract_map,
            "nx_graph": merged,
            "concept_properties": concept_properties,
            "ridge": ridge,
            "top_scores": top_scores,
            "distill_df": distill_df,
            "gnn_model": gnn_model,
            "final_emb": final_emb,
            "embed_model": embed_model,
            "all_metrics": bs["all_metrics"],
            "all_texts": bs["all_texts"],
            "config": config,
            "df_filtered": df_filtered,
            "selected_text_cols": selected_text_cols,
            "batch_info": {
                "mode": "batch",
                "batch_size": batch_size,
                "total_batches": total_batches,
                "total_docs": total_docs,
            },
        }
        if use_ontology:
            analysis_data.update({
                "ontology": ontology,
                "resolver": bs["resolver"],
                "extractor": bs["extractor"],
                "graph_builder": bs["builder"],
                "reasoning_paths": (
                    bs["builder"].reasoning_paths if bs["builder"] else []
                ),
            })
        st.session_state.analysis_data = analysis_data
        st.session_state.edit_history = GraphEditHistory()
        st.session_state.edit_history.save_snapshot(
            merged, valid_concepts, concept_to_id,
            id_to_concept, concept_abstract_map,
        )
        bs["all_concepts"] = []
        bs["all_metrics"] = []
        bs["valid_doc_indices"] = set()
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()
        bs["done"] = True

    try:
        for b in pending:
            _process_one_batch(b)
        if bs["next_batch"] >= total_batches:
            with status:
                st.write("🏁 All batches processed — finalizing...")
            _finalize()
            total_time = time.perf_counter() - overall_start
            progress_bar.progress(1.0)
            status.update(
                label=(
                    f"Batch analysis complete! ({total_time:.1f}s, "
                    f"peak RSS ≈ {get_memory_usage_mb():.0f} MB)"
                ),
                state="complete", expanded=False,
            )
            st.success(
                f"✅ All {total_batches} batches processed in "
                f"{total_time:.1f}s — peak memory ≈ "
                f"{get_memory_usage_mb():.0f} MB"
            )
        else:
            status.update(
                label=(
                    f"Batch {bs['next_batch']}/{total_batches} complete"
                ),
                state="complete", expanded=False,
            )
            st.info(
                f"📦 {total_batches - bs['next_batch']} batch(es) remaining "
                f"— click ▶️ Next batch or ⏩ All remaining in the sidebar."
            )
    except Exception as e:
        st.error(f"Batch pipeline error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
    finally:
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()


# ============================================================================
# MICROTRANSFORMER (LatentMoE) — KG‑RAG Reasoning Engine
# ============================================================================

# Relationship index mapping (used by LatentMoE)
RELATIONSHIP_TO_IDX = {rel.name: i for i, rel in enumerate(RelationshipType)}
NUM_EDGE_TYPES = len(RelationshipType)

# 32 specialised latent experts for Li‑ion batteries
BATTERY_EXPERT_LABELS = [
    "Energy Density", "Cycle Life", "Rate Capability", "Thermal Stability",
    "SEI Formation", "Dendrite Growth", "Impedance Growth", "Cathode Materials",
    "Anode Materials", "Electrolyte Design", "Calendering", "Slot‑Die Coating",
    "Doping", "Surface Coating", "Prelithiation", "Formation",
    "Electrode Thickness", "N/P Ratio", "C‑rate", "Temperature",
    "Pressure", "EIS", "dQ/dV", "Galvanostatic Cycling",
    "Coin Cell", "Pouch Cell", "Capacity Fade", "Transition Metal Dissolution",
    "Cation Mixing", "Particle Cracking", "Solid Electrolyte", "Liquid Electrolyte"
]


class LatentMoEKGExtractor(nn.Module):
    def __init__(self, num_nodes, num_edge_types, d_model=96, latent_dim=24,
                 n_experts=32, top_k=4, num_heads=4, num_layers=2):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, d_model)
        self.edge_embedding = nn.Embedding(num_edge_types, d_model)

        self.down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.up_proj = nn.Linear(latent_dim, d_model, bias=False)

        self.router = nn.Linear(latent_dim, n_experts)
        self.experts = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True,
            dim_feedforward=d_model * 2, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, node_seq, edge_seq):
        node_emb = self.node_embedding(node_seq)
        if edge_seq.size(1) > 0:
            edge_emb = self.edge_embedding(edge_seq)
            node_emb[:, 1:, :] = node_emb[:, 1:, :] + edge_emb

        batch_size, seq_len, d_model = node_emb.shape
        latent_repr = self.down_proj(node_emb)

        flat_latent = latent_repr.view(batch_size * seq_len, -1)
        router_logits = self.router(flat_latent)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

        expert_outputs = torch.stack([self.experts[i](flat_latent) for i in range(self.n_experts)], dim=0)
        token_indices = torch.arange(batch_size * seq_len, device=flat_latent.device).unsqueeze(1).expand(-1, self.top_k)
        selected = expert_outputs[topk_indices, token_indices, :]
        weighted = topk_weights.unsqueeze(-1) * selected
        moe_output_flat = weighted.sum(dim=1)

        moe_output = moe_output_flat.view(batch_size, seq_len, -1)
        node_emb = node_emb + self.up_proj(moe_output)
        out = self.transformer(node_emb)
        out = self.output_proj(out)
        return out, routing_weights.view(batch_size, seq_len, -1)


# ----------------------------------------------------------------------------
# Helper functions for Microtransformer visualisations
# ----------------------------------------------------------------------------
def plotly_continuous_scale(cmap_key: str, n: int = 12):
    try:
        import plotly.express as px
        if hasattr(px.colors.sequential, cmap_key):
            palette = getattr(px.colors.sequential, cmap_key)
            return palette[:n] if len(palette) >= n else palette * (n // len(palette) + 1)
    except Exception:
        pass
    try:
        cmap = plt.cm.get_cmap(cmap_key)
        return [matplotlib.colors.to_hex(cmap(i / n)) for i in range(n)]
    except Exception:
        return ['#636efa'] * n


def apply_mt_chart_style(fig, theme, is_axial=True):
    if theme is None:
        theme = THEME_PRESETS["Bright (Default)"]
    fig.update_layout(
        paper_bgcolor=theme.get("plotly_paper", "#ffffff"),
        plot_bgcolor=theme.get("plotly_bg", "#ffffff"),
        font_color=theme.get("font", "#000000"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(font=dict(color=theme.get("font", "#000000"))),
    )
    if is_axial:
        fig.update_xaxes(gridcolor=theme.get("grid_color", "#e2e8f0"),
                         linecolor=theme.get("axis_color", "#64748b"))
        fig.update_yaxes(gridcolor=theme.get("grid_color", "#e2e8f0"),
                         linecolor=theme.get("axis_color", "#64748b"))


def render_chord_diagram(token_labels, routing_np, scale=2.5, theme=None):
    """Render a chord diagram using the 'chord' library if available, otherwise fallback to heatmap."""
    try:
        import chord
        n_tokens = routing_np.shape[0]
        chord_matrix = np.zeros((n_tokens, n_tokens))
        # Use average routing per token to each expert; transpose to get token→expert flow
        # Actually chord diagram shows between experts, but here we want token-to-expert flow.
        # We'll aggregate: each token's distribution across experts.
        # For simplicity, we create a matrix where each row token is source, and columns are experts.
        # But chord expects square matrix of same entities. We'll create a combined list of tokens+experts.
        # Better: use a Sankey diagram instead of chord for token→expert flow.
        # Since chord library expects square, we'll pivot to a flow matrix between tokens and experts.
        # Simpler: just use heatmap.
        raise ImportError("Chord diagram not implemented; falling back to heatmap.")
    except (ImportError, AttributeError):
        # Fallback to heatmap
        fig = px.imshow(
            routing_np,
            labels=dict(x="Expert", y="Token", color="Activation"),
            x=[f"E{i}" for i in range(routing_np.shape[1])],
            y=token_labels,
            title="Token-to-Expert Routing (Heatmap)",
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        apply_mt_chart_style(fig, theme)
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Main Microtransformer KG‑RAG Tab
# ----------------------------------------------------------------------------
QUERY_CONCEPT_MAP = [
    (r"\benergy\s+density\b.*\bcycle\s+life\b", "energy_density", "cycle_life"),
    (r"\bcycle\s+life\b.*\benergy\s+density\b", "cycle_life", "energy_density"),
    (r"\bcalendering\b.*\benergy\s+density\b", "calendering", "energy_density"),
    (r"\bdoping\b.*\bcycle\s+life\b", "doping", "cycle_life"),
    (r"\bsurface\s+coating\b.*\bcapacity\s+fade\b", "surface_coating", "capacity_fade"),
    (r"\bsilicon\b.*\bparticle\s+cracking\b", "silicon", "particle_cracking"),
    (r"\bnmc811\b.*\bthermal\s+stability\b", "nmc811", "thermal_stability"),
    (r"\blfp\b.*\bcycle\s+life\b", "lfp", "cycle_life"),
    (r"\bsolid\s+electrolyte\b.*\bionic\s+conductivity\b", "solid_electrolyte", "ionic_conductivity"),
    (r"\beis\b.*\bimpedance\s+growth\b", "eis", "impedance_growth"),
]

def render_microtransformer_kg_rag_tab(analysis_data: Dict, ontology: DomainOntology) -> None:
    st.subheader("🧠 Microtransformer KG‑RAG (LatentMoE) Reasoning")
    st.markdown("""
    Select a **source** and **target** concept to extract the shortest path in the graph.
    The path is encoded as a token sequence, and a **Mixture‑of‑Experts Transformer** 
    (LatentMoE) computes per‑token expert assignments. The visualisations show which 
    latent reasoning experts are activated along the path, revealing the underlying 
    reasoning patterns.
    """)

    nx_graph = analysis_data["nx_graph"]
    valid_concepts = analysis_data["valid_concepts"]
    concept_abstract_map = analysis_data["concept_abstract_map"]

    if nx_graph.number_of_nodes() < 2:
        st.warning("Graph has fewer than 2 nodes. Cannot run Microtransformer.")
        return

    # Build dropdown options from graph nodes + ontology concepts (union)
    graph_concepts = set(nx_graph.nodes())
    ontology_concepts = set(ontology.concepts.keys())
    all_concepts = sorted(graph_concepts | ontology_concepts)

    # Pre-fill source/target from last query analysis if available
    default_source = None
    default_target = None
    last_analysis = st.session_state.get('last_query_analysis')
    if last_analysis is not None:
        explicit = last_analysis.explicitly_mentioned
        inferred = last_analysis.inferred_concepts
        candidates = explicit + inferred
        # pick first two that exist in all_concepts
        selected = [c for c in candidates if c in all_concepts]
        if len(selected) >= 2:
            default_source = selected[0]
            default_target = selected[1]
        elif len(selected) == 1:
            default_source = selected[0]
            # try to find a second from graph neighbors or ontology
            if selected[0] in nx_graph:
                neighbors = list(nx_graph.neighbors(selected[0]))
                for n in neighbors:
                    if n in all_concepts and n != selected[0]:
                        default_target = n
                        break
            if default_target is None:
                # pick any other concept
                for c in all_concepts:
                    if c != selected[0]:
                        default_target = c
                        break

    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source Concept", all_concepts, index=all_concepts.index(default_source) if default_source in all_concepts else 0)
    with col2:
        target = st.selectbox("Target Concept", all_concepts, index=all_concepts.index(default_target) if default_target in all_concepts else (1 if len(all_concepts)>1 else 0))

    # Quick query field: natural language -> regex mapping to auto-select source/target
    quick_query = st.text_input("Or type a query to auto-select source/target (e.g., 'How does calendering affect energy density?')")
    if quick_query:
        for pattern, s, t in QUERY_CONCEPT_MAP:
            if re.search(pattern, quick_query, re.I):
                if s in all_concepts:
                    source = s
                if t in all_concepts:
                    target = t
                break

    # Optional advanced settings
    with st.expander("⚙️ Model Settings"):
        d_model = st.slider("d_model", 32, 128, 96, step=16)
        latent_dim = st.slider("latent_dim", 8, 64, 24, step=4)
        n_experts = st.slider("Number of experts", 16, 48, 32, step=4)
        top_k = st.slider("Top‑K experts per token", 1, 8, 4, step=1)
        num_heads = st.slider("Transformer heads", 2, 8, 4, step=1)
        num_layers = st.slider("Transformer layers", 1, 4, 2, step=1)
        cmap_name = st.selectbox("Colormap", list(SUPPORTED_COLORMAPS.keys()), index=0)
        theme_name = st.selectbox("Theme", list(THEME_PRESETS.keys()), index=0)
        show_sankey = st.checkbox("Show Sankey diagram", value=True)
        show_chord = st.checkbox("Show Chord diagram (if available)", value=False)

    if source == target:
        st.warning("Source and target must be different.")
        return

    if st.button("🔍 Run Microtransformer Path Analysis", type="primary"):
        if source not in nx_graph or target not in nx_graph:
            st.error("Source or target not in the graph. Please choose concepts that exist in the graph (marked with ✓ in dropdown).")
            return

        # Find shortest path
        try:
            path_nodes = nx.shortest_path(nx_graph, source=source, target=target, weight='weight')
        except nx.NetworkXNoPath:
            st.error(f"No path found between '{source}' and '{target}' in the graph.")
            return

        st.success(f"Shortest path: {' → '.join(path_nodes)}")

        # Build node and edge sequences
        node_seq = [valid_concepts.index(n) for n in path_nodes]
        edge_types = []
        for i in range(len(path_nodes)-1):
            u, v = path_nodes[i], path_nodes[i+1]
            # get edge type if exists
            edge_data = nx_graph.get_edge_data(u, v)
            if edge_data:
                etype = edge_data.get('edge_type', 'semantic')
                try:
                    rel = RelationshipType(etype)
                except ValueError:
                    rel = RelationshipType.SEMANTIC
                edge_types.append(RELATIONSHIP_TO_IDX.get(rel.name, 0))
            else:
                edge_types.append(0)  # fallback

        node_seq_t = torch.tensor([node_seq], dtype=torch.long)
        edge_seq_t = torch.tensor([edge_types], dtype=torch.long)

        num_nodes = len(valid_concepts)
        num_edge_types = NUM_EDGE_TYPES

        # Instantiate model
        with torch.no_grad():
            model = LatentMoEKGExtractor(
                num_nodes=num_nodes,
                num_edge_types=num_edge_types,
                d_model=d_model,
                latent_dim=latent_dim,
                n_experts=n_experts,
                top_k=top_k,
                num_heads=num_heads,
                num_layers=num_layers
            )
            # Run model
            _, routing_weights = model(node_seq_t, edge_seq_t)

        routing_np = routing_weights.squeeze(0).numpy()  # (seq_len, n_experts)
        token_labels = path_nodes

        # Display expert labels
        expert_labels = BATTERY_EXPERT_LABELS[:n_experts]
        if len(expert_labels) < n_experts:
            expert_labels += [f"Expert {i+1}" for i in range(len(expert_labels), n_experts)]

        # Visualisations
        theme = THEME_PRESETS.get(theme_name, THEME_PRESETS["Bright (Default)"])
        cmap_scale = plotly_continuous_scale(cmap_name, n=12)

        # 1. Heatmap
        fig_heat = px.imshow(
            routing_np,
            labels=dict(x="Expert", y="Token", color="Activation"),
            x=expert_labels,
            y=token_labels,
            title="Token‑wise Expert Activation",
            color_continuous_scale=cmap_name,
            aspect="auto"
        )
        # Add custom hovertext: show expert labels
        fig_heat.update_traces(
            hovertemplate="<b>Token:</b> %{y}<br><b>Expert:</b> %{x}<br><b>Activation:</b> %{z:.3f}<extra></extra>"
        )
        apply_mt_chart_style(fig_heat, theme)
        st.plotly_chart(fig_heat, use_container_width=True)

        # 2. Bar chart (average per expert across tokens)
        avg_per_expert = routing_np.mean(axis=0)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=expert_labels,
            y=avg_per_expert,
            marker_color=cmap_scale[:len(expert_labels)],
            text=[f"{v:.3f}" for v in avg_per_expert],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Avg Activation: %{y:.3f}<extra></extra>"
        ))
        fig_bar.update_layout(
            title="Average Expert Activation (across all tokens)",
            xaxis_title="Expert",
            yaxis_title="Activation",
            xaxis_tickangle=-45
        )
        apply_mt_chart_style(fig_bar, theme)
        st.plotly_chart(fig_bar, use_container_width=True)

        # 3. Sankey diagram (token→expert flow)
        if show_sankey:
            # Build Sankey: each token is a source node, each expert is a target node
            n_tokens = len(token_labels)
            n_experts_actual = len(expert_labels)
            # Normalise routing weights to sum to 1 per token (they already do)
            sources = []
            targets = []
            values = []
            for i in range(n_tokens):
                for j in range(n_experts_actual):
                    val = routing_np[i, j]
                    if val > 0.01:  # threshold for readability
                        sources.append(i)
                        targets.append(n_tokens + j)
                        values.append(val)
            # node labels: tokens then experts
            node_labels = token_labels + expert_labels
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=[theme.get("highlight_bg", "#ff6b6b")] * n_tokens + cmap_scale[:n_experts_actual]
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )])
            fig_sankey.update_layout(
                title="Token‑to‑Expert Flow (Sankey)",
                font=dict(size=10, color=theme.get("font", "#000000"))
            )
            apply_mt_chart_style(fig_sankey, theme, is_axial=False)
            st.plotly_chart(fig_sankey, use_container_width=True)

        # 4. Chord diagram (fallback to heatmap if not available)
        if show_chord:
            st.markdown("#### Chord Diagram (Token↔Expert Flow)")
            render_chord_diagram(token_labels, routing_np, theme=theme)

        # 5. Interpretation
        st.subheader("🧐 Scientific Interpretation")
        # Identify top experts per token
        top_experts = []
        for i, token in enumerate(token_labels):
            expert_activations = routing_np[i]
            top_indices = expert_activations.argsort()[-3:][::-1]
            top_labels = [expert_labels[idx] for idx in top_indices]
            top_experts.append(f"**{token}** → {', '.join(top_labels)}")
        st.markdown("**Top‑3 experts per token:**")
        st.markdown("\n".join(top_experts))

        # Overall dominant expert
        overall_dominant = expert_labels[np.argmax(avg_per_expert)]
        st.markdown(f"**Overall dominant expert:** {overall_dominant} (avg activation {np.max(avg_per_expert):.3f})")

        # Provide a system‑like explanation
        st.markdown("""
        The latent experts represent specialised reasoning patterns in battery science. 
        Higher activation indicates that the model relies on that expert's 'knowledge' 
        when processing the corresponding token in the path. This can highlight which 
        scientific concepts (e.g., SEI formation, energy density) are most relevant 
        to the relationship between source and target.
        """)


# ============================================================================
# QDWA — QUERY DISTILLATION & WEIGHTED ALLOCATION MODULE
# ============================================================================
# Six categories for Li-ion battery energy density reasoning
# ============================================================================

class QDWACategory(Enum):
    """Six-rotor categories for energy density focused reasoning."""
    CATHODE_MATERIALS = "cathode_materials"      # NMC811, LFP, NCA, etc.
    ANODE_MATERIALS = "anode_materials"          # Si, Graphite, Li Metal, etc.
    ELECTROLYTE_SYSTEMS = "electrolyte_systems"  # Liquid, Solid, Additives
    MANUFACTURING = "manufacturing"              # Calendering, Coating, Doping
    DEGRADATION = "degradation"                  # SEI, Dendrites, Cracking
    PERFORMANCE = "performance"                  # Energy Density, Capacity, Rate

# Shorthand list used throughout
SIX_CATEGORIES = [cat.value for cat in QDWACategory]

# Human-readable display names
CATEGORY_DISPLAY: Dict[str, str] = {
    "cathode_materials": "⚡ Cathode Materials",
    "anode_materials": "🔋 Anode Materials",
    "electrolyte_systems": "🧪 Electrolyte Systems",
    "manufacturing": "⚙️ Manufacturing",
    "degradation": "⚠️ Degradation",
    "performance": "📊 Performance Metrics",
}

# Category color palette (consistent across all visualizations)
CATEGORY_COLORS: Dict[str, str] = {
    "cathode_materials": "#3b82f6",   # Blue
    "anode_materials":   "#10b981",   # Green
    "electrolyte_systems": "#8b5cf6", # Purple
    "manufacturing":     "#f59e0b",   # Amber
    "degradation":       "#ef4444",   # Red
    "performance":       "#06b6d4",   # Cyan
}

# ============================================================================
# SEED CONCEPTS — define semantic anchors for each category
# These are used in Eq. (1) to compute category anchor vectors μ_k
# ============================================================================

CATEGORY_SEEDS: Dict[str, List[str]] = {
    "cathode_materials": [
        "nmc811 cathode high-nickel layered oxide",
        "nmc622 balanced cathode nickel manganese cobalt",
        "nmc532 stable cathode material",
        "nmc333 standard layered cathode",
        "lfp lifepo4 olivine cathode thermal stability",
        "lco licoo2 cobalt oxide cathode",
        "nca nickel cobalt aluminum oxide high energy",
        "cathode specific capacity voltage platform",
        "high-nickel cathode thermal stability trade-off",
        "cathode particle cracking transition metal dissolution",
    ],
    "anode_materials": [
        "graphite anode standard lithium intercalation",
        "silicon anode high capacity volume expansion",
        "siox silicon oxide anode cycling stability",
        "lithium metal anode dendrite formation",
        "lto lithium titanate zero-strain anode",
        "anode specific capacity mAh/g gravimetric",
        "silicon carbon composite anode design",
        "anode prelithiation first cycle efficiency",
        "anode solid electrolyte interphase formation",
        "high capacity anode energy density improvement",
    ],
    "electrolyte_systems": [
        "liquid electrolyte carbonate solvent li+ transport",
        "solid electrolyte sulfide oxide polymer",
        "fec fluoroethylene carbonate sei additive",
        "vc vinylene carbonate electrolyte additive",
        "electrolyte ionic conductivity li+ mobility",
        "electrolyte oxidation stability voltage window",
        "electrolyte additive sei formation mechanism",
        "solid electrolyte interface lithium metal",
        "electrolyte flame retardant thermal safety",
        "high concentration electrolyte lithium salt",
    ],
    "manufacturing": [
        "calendering electrode compaction density increase",
        "slot die coating electrode slurry deposition",
        "dry electrode processing solvent-free fabrication",
        "doping elemental cation anion modification",
        "surface coating protective layer cathode",
        "prelithiation lithium pre-doping anode",
        "formation cycling initial sei building",
        "electrode thickness optimization energy density",
        "electrode porosity tortuosity ionic transport",
        "manufacturing defect control quality assurance",
    ],
    "degradation": [
        "capacity fade cycling degradation loss",
        "sei formation solid electrolyte interphase growth",
        "dendrite growth lithium plating short circuit",
        "thermal runaway uncontrolled self-heating safety",
        "impedance growth resistance increase polarization",
        "transition metal dissolution cathode leaching",
        "cation mixing ni/li site exchange nmc",
        "particle cracking mechanical degradation volume change",
        "electrolyte decomposition side reaction gas",
        "capacity retention cycle life degradation mechanism",
    ],
    "performance": [
        "energy density wh/kg gravimetric specific energy",
        "volumetric energy density wh/l packing density",
        "specific capacity mAh/g discharge capacity",
        "cycle life cycles to 80% retention longevity",
        "coulombic efficiency charge discharge ratio",
        "rate capability c-rate fast charging power",
        "power density w/kg specific power output",
        "thermal stability decomposition temperature safety",
        "np ratio anode cathode capacity balance",
        "energy density enhancement improvement strategy",
    ],
}

# ============================================================================
# CONCEPT → CATEGORY MAPPING (for fast lookup without embedding)
# ============================================================================

CONCEPT_TO_CATEGORY: Dict[str, str] = {
    # Cathode Materials
    "nmc811": "cathode_materials", "nmc622": "cathode_materials",
    "nmc532": "cathode_materials", "nmc333": "cathode_materials",
    "lfp": "cathode_materials", "lco": "cathode_materials",
    "nca": "cathode_materials",
    # Anode Materials
    "graphite": "anode_materials", "silicon": "anode_materials",
    "siox": "anode_materials", "lithium_metal": "anode_materials",
    "lto": "anode_materials",
    # Electrolyte Systems
    "liquid_electrolyte": "electrolyte_systems",
    "solid_electrolyte": "electrolyte_systems",
    "fec": "electrolyte_systems", "vc": "electrolyte_systems",
    # Manufacturing
    "calendering": "manufacturing", "slot_die_coating": "manufacturing",
    "dry_electrode_processing": "manufacturing", "doping": "manufacturing",
    "surface_coating": "manufacturing", "prelithiation": "manufacturing",
    "formation": "manufacturing",
    # Degradation
    "capacity_fade": "degradation", "sei_formation": "degradation",
    "dendrite_growth": "degradation", "thermal_runaway": "degradation",
    "impedance_growth": "degradation",
    "transition_metal_dissolution": "degradation",
    "cation_mixing": "degradation", "particle_cracking": "degradation",
    # Performance
    "energy_density": "performance", "power_density": "performance",
    "specific_capacity": "performance", "cycle_life": "performance",
    "coulombic_efficiency": "performance", "rate_capability": "performance",
    "thermal_stability": "performance", "ionic_conductivity": "performance",
    "electronic_conductivity": "performance", "electrode_density": "performance",
    "porosity": "performance", "volumetric_energy_density": "performance",
    # Parameters (map to manufacturing for process control)
    "electrode_thickness": "manufacturing", "np_ratio": "manufacturing",
    "c_rate": "performance", "temperature": "degradation",
    "pressure": "manufacturing",
    # Methods (map to performance for measurement)
    "eis": "performance", "dqdv": "performance",
    "galvanostatic_cycling": "performance",
    "coin_cell": "performance", "pouch_cell": "performance",
}

# Default fallback category for unrecognized concepts
DEFAULT_CATEGORY = "performance"


def categorize_concept(canonical_name: str) -> str:
    """Map a concept to one of the six QDWA categories."""
    return CONCEPT_TO_CATEGORY.get(canonical_name.lower(), DEFAULT_CATEGORY)


# ============================================================================
# ENERGY DENSITY ENHANCEMENT KEYWORD LISTS
# ============================================================================

ENERGY_DENSITY_KEYWORDS: Dict[str, List[str]] = {
    "cathode_materials": [
        "nmc811", "nmc 811", "high-nickel", "high nickel", "ni-rich",
        "nmc622", "nmc 622", "nmc532", "nmc 532", "nmc333", "nmc 333",
        "lfp", "lifepo4", "nca", "cathode", "positive electrode",
        "layered oxide", "olivine", "spinel", "nickel", "cobalt", "manganese",
        "voltage platform", "cathode capacity",
    ],
    "anode_materials": [
        "silicon", "si anode", "sio", "siox", "silicon oxide",
        "graphite", "hard carbon", "soft carbon",
        "lithium metal", "li metal", "li-metal", "anode",
        "negative electrode", "lto", "li4ti5o12",
        "high capacity anode", "anode capacity",
    ],
    "electrolyte_systems": [
        "electrolyte", "liquid electrolyte", "solid electrolyte",
        "sse", "sulfide electrolyte", "oxide electrolyte", "polymer electrolyte",
        "fec", "fluoroethylene carbonate", "vc", "vinylene carbonate",
        "additive", "li salt", "lipf6", "liclo4", "lifsi",
        "ionic conductivity", "li+ transport", "transference number",
        "solid-state", "all-solid-state", "assb",
    ],
    "manufacturing": [
        "calender", "compaction", "pressing", "density increase",
        "coating", "slot-die", "slurry", "dry electrode", "solvent-free",
        "doping", "surface coating", "al2o3", "zro2", "aluminum oxide",
        "prelithiation", "pre-lithiation", "lithiation",
        "formation", "formation cycle", "initial cycle",
        "thickness", "electrode thickness", "thick electrode",
        "porosity", "tortuosity", "loading", "mass loading",
        "np ratio", "n/p ratio", "capacity balance",
    ],
    "degradation": [
        "fade", "capacity fade", "capacity loss", "degradation",
        "sei", "solid electrolyte interphase", "sei growth",
        "dendrite", "lithium dendrite", "dendritic",
        "thermal runaway", "safety", "fire", "explosion",
        "impedance", "resistance", "polarization",
        "transition metal dissolution", "tm dissolution", "metal leaching",
        "cation mixing", "ni/li mixing", "site exchange",
        "cracking", "particle cracking", "fracture",
        "side reaction", "gas generation", "electrolyte decomposition",
    ],
    "performance": [
        "energy density", "wh/kg", "wh/l", "specific energy",
        "volumetric energy", "gravimetric energy",
        "capacity", "mah/g", "specific capacity",
        "cycle life", "longevity", "cycling stability",
        "coulombic efficiency", "ce", "first cycle efficiency", "ice",
        "rate capability", "c-rate", "fast charge", "fast charging",
        "power density", "w/kg", "specific power",
        "thermal stability", "decomposition temperature",
        "enhancement", "improvement", "increase", "boost", "optimize",
        "maximize", "trade-off", "balancing",
    ],
}

# ============================================================================
# QDWA ANALYSIS RESULT DATA CLASS
# ============================================================================

@dataclass
class QDWAAnalysis:
    """Container for all QDWA computation results."""
    query: str
    category_weights: Dict[str, float]         # W_k from Eq. (4)
    raw_evidence: Dict[str, float]             # raw_k from Eq. (3)
    concentration: float                       # c from Eq. (5)
    subgraph_depth: int                        # K from Eq. (7)
    term_memberships: Dict[str, Dict[str, float]]  # m_k(t) from Eq. (2)
    extracted_terms: List[str]                 # Terms from query
    primary_category: str                      # Category with highest W_k
    category_anchors: Dict[str, np.ndarray]    # μ_k from Eq. (1)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_focused(self) -> bool:
        """True if query is concentrated on one category (c > 0.6)."""
        return self.concentration > 0.6

    @property
    def is_balanced(self) -> bool:
        """True if query spans multiple categories (c < 0.3)."""
        return self.concentration < 0.3

    def get_ranked_categories(self) -> List[Tuple[str, float]]:
        """Return categories sorted by weight, descending."""
        return sorted(
            self.category_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def get_top_n_categories(self, n: int = 2) -> List[Tuple[str, float]]:
        """Return top N categories by weight."""
        return self.get_ranked_categories()[:n]

    def get_energy_density_relevance(self) -> float:
        """
        Compute energy-density-specific relevance score.
        Combines weights of categories most directly linked to energy density.
        """
        ed_weights = {
            "cathode_materials": 0.30,  # High voltage + capacity
            "anode_materials": 0.25,    # High capacity
            "manufacturing": 0.25,      # Density, thickness
            "performance": 0.15,        # Direct metrics
            "electrolyte_systems": 0.03,
            "degradation": 0.02,
        }
        score = sum(
            self.category_weights.get(cat, 0) * w
            for cat, w in ed_weights.items()
        )
        return min(score, 1.0)


# ============================================================================
# QDWA COMPUTATION ENGINE
# ============================================================================

class QDWAEngine:
    """
    Query Distillation & Weighted Allocation Engine.

    Implements Eqs. (1)-(9) for six-category energy density reasoning.
    """

    # Hyperparameters
    BETA = 8.0       # Softmax sharpness for Eq. (2)
    ALPHA = 0.25     # Laplace prior for Eq. (4)
    RHO = 0.15       # Personalization injection for Eq. (8)
    ETA = 0.50       # Same-category edge boost for Eq. (9)
    KAPPA = 0.50     # Causal edge boost for Eq. (9)

    def __init__(self, ontology: DomainOntology, embedding_model=None):
        self.ontology = ontology
        self.embedding_model = embedding_model
        self._category_anchors: Dict[str, np.ndarray] = {}
        self._anchor_cache_valid = False

    def compute_category_anchors(self) -> Dict[str, np.ndarray]:
        """
        Eq. (1): Compute category anchor vectors.
        μ_k = (1/|S_k|) Σ_{c∈S_k} Enc(c), then normalize.
        """
        if self._anchor_cache_valid and self._category_anchors:
            return self._category_anchors

        if self.embedding_model is None:
            # Fallback: use random but deterministic anchors
            np.random.seed(42)
            dim = 384  # Typical SBERT dimension
            self._category_anchors = {
                cat: self._normalize_vector(np.random.randn(dim))
                for cat in SIX_CATEGORIES
            }
            self._anchor_cache_valid = True
            return self._category_anchors

        anchors = {}
        for cat in SIX_CATEGORIES:
            seeds = CATEGORY_SEEDS.get(cat, [])
            if not seeds:
                anchors[cat] = np.zeros(self.embedding_model.get_sentence_embedding_dimension())
                continue
            embeddings = self.embedding_model.encode(seeds, convert_to_numpy=True)
            mean_embedding = np.mean(embeddings, axis=0)
            anchors[cat] = self._normalize_vector(mean_embedding)

        self._category_anchors = anchors
        self._anchor_cache_valid = True
        return anchors

    @staticmethod
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            return v
        return v / norm

    def soft_category_membership(
        self, term: str
    ) -> Dict[str, float]:
        """
        Eq. (2): Compute soft category membership for a single term.
        z_k(t) = β · μ_k^T · ê_t
        m_k(t) = softmax(z)_k
        """
        anchors = self.compute_category_anchors()

        if self.embedding_model is not None:
            term_emb = self.embedding_model.encode([term], convert_to_numpy=True)[0]
            term_emb = self._normalize_vector(term_emb)
            scores = {}
            for cat, anchor in anchors.items():
                scores[cat] = self.BETA * np.dot(anchor, term_emb)
        else:
            # Fallback: keyword-based scoring
            term_lower = term.lower().replace("_", " ")
            scores = {}
            for cat in SIX_CATEGORIES:
                keywords = ENERGY_DENSITY_KEYWORDS.get(cat, [])
                # Count overlapping n-grams
                score = 0.0
                for kw in keywords:
                    if kw in term_lower:
                        score += 1.0
                    # Partial match
                    kw_words = set(kw.split())
                    term_words = set(term_lower.split())
                    overlap = len(kw_words & term_words)
                    if overlap > 0:
                        score += 0.5 * overlap / max(len(kw_words), 1)
                scores[cat] = self.BETA * score

        # Numerically stable softmax
        max_score = max(scores.values()) if scores else 0
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        if total < 1e-12:
            uniform = 1.0 / len(SIX_CATEGORIES)
            return {cat: uniform for cat in SIX_CATEGORIES}

        return {k: v / total for k, v in exp_scores.items()}

    def compute_raw_evidence(
        self,
        query: str,
        extracted_terms: List[str],
        term_memberships: Dict[str, Dict[str, float]],
        ontology_concepts: List[str] = None,
    ) -> Dict[str, float]:
        """
        Eq. (3): Aggregate raw evidence from three sources.
        raw_k = Σ_{kw∈q} 1[kw∈k] + Σ_{c∈P_def} w_c·1[cat(c)=k] + Σ_{t∈q} m_k(t)
        """
        raw = {cat: 0.0 for cat in SIX_CATEGORIES}
        query_lower = query.lower()

        # Source 1: Direct keyword hits
        for cat, keywords in ENERGY_DENSITY_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    raw[cat] += 1.0

        # Source 2: Ontology concept matches (weighted)
        if ontology_concepts:
            for concept in ontology_concepts:
                cat = categorize_concept(concept)
                # Weight by concept type relevance
                ctype = self.ontology.get_concept_type(concept)
                type_weight = {
                    ConceptType.MATERIAL: 1.5,
                    ConceptType.PROCESS: 1.2,
                    ConceptType.PROPERTY: 1.0,
                    ConceptType.PHENOMENON: 0.8,
                    ConceptType.PARAMETER: 0.7,
                    ConceptType.METHOD: 0.5,
                    ConceptType.GENERAL: 0.3,
                }.get(ctype, 0.5)
                raw[cat] += type_weight

        # Source 3: Soft term membership sums
        for term, memberships in term_memberships.items():
            for cat, m in memberships.items():
                raw[cat] += m

        return raw

    def compute_smoothed_weights(
        self, raw: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Eq. (4): Laplace-smoothed allocation.
        W_k = (α + raw_k) / (6α + Σ_j raw_j)
        Guarantees W_k > 0 for all k.
        """
        total_raw = sum(raw.values())
        denom = len(SIX_CATEGORIES) * self.ALPHA + total_raw
        weights = {}
        for cat in SIX_CATEGORIES:
            weights[cat] = (self.ALPHA + raw.get(cat, 0)) / denom
        return weights

    def compute_concentration(self, W: Dict[str, float]) -> float:
        """
        Eq. (5): Normalized entropy concentration.
        H(W) = -Σ_k W_k ln W_k
        c = 1 - H / ln(6)
        """
        H = 0.0
        for w in W.values():
            if w > 1e-12:
                H -= w * math.log(w)
        H_max = math.log(len(SIX_CATEGORIES))
        c = 1.0 - (H / H_max) if H_max > 1e-12 else 0.0
        return max(0.0, min(1.0, c))

    def compute_subgraph_depth(self, c: float) -> int:
        """
        Eq. (7): Adaptive subgraph depth.
        K = 2 + round(2c) ∈ {2, 3, 4}
        """
        return 2 + round(2 * c)

    def extract_terms(self, query: str) -> List[str]:
        """Extract domain-relevant terms from query text."""
        terms = []

        # 1. Direct ontology concept matches
        query_lower = query.lower()
        for canonical, node in self.ontology.concepts.items():
            if node.is_match(query_lower):
                terms.append(canonical)
            for syn in node.synonyms:
                if syn in query_lower and canonical not in terms:
                    terms.append(canonical)

        # 2. Keyword-based extraction
        for cat, keywords in ENERGY_DENSITY_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    clean_term = kw.replace(" ", "_").replace("-", "_")
                    if clean_term not in terms:
                        terms.append(clean_term)

        # 3. Numerical value extraction (e.g., "350 mAh/g")
        num_patterns = [
            r'(\d+(?:\.\d+)?)\s*mah/g',
            r'(\d+(?:\.\d+)?)\s*wh/kg',
            r'(\d+(?:\.\d+)?)\s*wh/l',
            r'(\d+(?:\.\d+)?)\s*c-rate',
            r'(\d+(?:\.\d+)?)\s*°c',
            r'(\d+(?:\.\d+)?)\s*μm',
        ]
        for pattern in num_patterns:
            matches = re.findall(pattern, query_lower)
            for m in matches:
                term = f"{m}_{pattern.split(r'\\')[1].replace(r'\s*', '_')}"
                if term not in terms:
                    terms.append(term)

        return terms

    def analyze_query(
        self,
        query: str,
        ontology_concepts: List[str] = None,
    ) -> QDWAAnalysis:
        """
        Full QDWA pipeline: Eqs. (1) → (7).
        """
        # Extract terms from query
        extracted_terms = self.extract_terms(query)

        # Compute soft category membership for each term (Eq. 2)
        term_memberships = {}
        for term in extracted_terms:
            term_memberships[term] = self.soft_category_membership(term)

        # If no terms extracted, create uniform membership
        if not term_memberships:
            uniform = {cat: 1/len(SIX_CATEGORIES) for cat in SIX_CATEGORIES}
            term_memberships["_query_fallback"] = uniform

        # Aggregate raw evidence (Eq. 3)
        raw = self.compute_raw_evidence(
            query, extracted_terms, term_memberships, ontology_concepts
        )

        # Smooth weights (Eq. 4)
        W = self.compute_smoothed_weights(raw)

        # Concentration (Eq. 5)
        c = self.compute_concentration(W)

        # Subgraph depth (Eq. 7)
        K = self.compute_subgraph_depth(c)

        # Primary category
        primary = max(W, key=W.get)

        # Category anchors (Eq. 1)
        anchors = self.compute_category_anchors()

        return QDWAAnalysis(
            query=query,
            category_weights=W,
            raw_evidence=raw,
            concentration=c,
            subgraph_depth=K,
            term_memberships=term_memberships,
            extracted_terms=extracted_terms,
            primary_category=primary,
            category_anchors=anchors,
        )

    def compute_personalized_seeds(
        self,
        analysis: QDWAAnalysis,
        graph_nodes: List[str],
    ) -> Dict[str, float]:
        """
        Eq. (8): Category-annealed personalization for PageRank.
        p_n = 1[n ∈ seeds] + ρ · W_{cat(n)}
        """
        W = analysis.category_weights
        seeds = set(analysis.extracted_terms)

        personalization = {}
        for node in graph_nodes:
            cat = categorize_concept(node)
            base = 1.0 if node in seeds else 0.0
            boost = self.RHO * W.get(cat, 1/len(SIX_CATEGORIES))
            personalization[node] = base + boost

        # Normalize
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        return personalization

    def reweight_edges(
        self,
        analysis: QDWAAnalysis,
        graph: nx.Graph,
        causal_edges: Set[Tuple[str, str]] = None,
    ) -> nx.Graph:
        """
        Eq. (9): Edge re-weighting based on QDWA weights.
        w'(u,v) = w(u,v) · (1 + η · W_{cat(u)} · 1[cat(u)=cat(v)])
                  + κ · W_{cat(v)} · 1[edge ∈ causal]
        """
        W = analysis.category_weights
        if causal_edges is None:
            causal_edges = set()

        reweighted = graph.copy()
        for u, v, data in reweighted.edges(data=True):
            orig_weight = data.get("weight", 1.0)
            cat_u = categorize_concept(u)
            cat_v = categorize_concept(v)

            # Same-category boost
            same_cat_boost = 0.0
            if cat_u == cat_v:
                same_cat_boost = self.ETA * W.get(cat_u, 0)

            # Causal edge boost
            causal_boost = 0.0
            if (u, v) in causal_edges or (v, u) in causal_edges:
                causal_boost = self.KAPPA * W.get(cat_v, 0)

            new_weight = orig_weight * (1 + same_cat_boost) + causal_boost
            data["weight"] = new_weight
            data["qdwa_reweighted"] = True
            data["qdwa_original_weight"] = orig_weight

        return reweighted


# ============================================================================
# QDWA VISUALIZATION MODULE
# ============================================================================

def render_qdwa_math_trace(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Display step-by-step mathematical computation trace."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff", "accent": "#3b82f6"}

    st.markdown("### 🧮 QDWA Mathematical Computation Trace")

    with st.expander("📊 Full Step-by-Step Computation", expanded=True):
        alpha = QDWAEngine.ALPHA
        beta = QDWAEngine.BETA
        W = analysis.category_weights
        raw = analysis.raw_evidence
        c = analysis.concentration
        K = analysis.subgraph_depth
        total_raw = sum(raw.values())
        denom = len(SIX_CATEGORIES) * alpha + total_raw

        # Eq (1)
        st.markdown("**Eq. (1) — Category Anchors**")
        st.latex(r"\mu_k = \frac{\bar{e}_k}{\|\bar{e}_k\|}, \quad "
                 r"\bar{e}_k = \frac{1}{|S_k|}\sum_{c \in S_k} \mathrm{Enc}(c)")
        seed_info = " | ".join(
            f"{CATEGORY_DISPLAY[k][:15]}: {len(CATEGORY_SEEDS[k])} seeds"
            for k in SIX_CATEGORIES
        )
        st.caption(f"Seed counts: {seed_info}")

        # Eq (2)
        st.markdown("**Eq. (2) — Soft Category Membership**")
        st.latex(r"z_k(t) = \beta \, \mu_k^\top \hat{e}_t, \quad "
                 r"m_k(t) = \mathrm{softmax}(z)_k, \quad \beta = " + f"{beta}")
        if analysis.term_memberships:
            mf_df = pd.DataFrame(analysis.term_memberships).T
            mf_df.columns = [CATEGORY_DISPLAY.get(k, k)[:12] for k in mf_df.columns]
            st.dataframe(
                mf_df.style.format("{:.4f}").background_gradient(
                    cmap="YlOrRd", axis=1, vmin=0, vmax=1
                ),
                use_container_width=True,
                height=200,
            )

        # Eq (3)
        st.markdown("**Eq. (3) — Raw Evidence Aggregation**")
        st.latex(r"\mathrm{raw}_k = \sum_{\text{kw}} \mathbb{1}[\text{kw} \in k] "
                 r"+ \sum_{c} w_c \, \mathbb{1}[\mathrm{cat}(c)=k] "
                 r"+ \sum_{t \in q} m_k(t)")
        raw_df = pd.DataFrame([
            {
                "Category": CATEGORY_DISPLAY[k],
                "Keyword Hits": raw.get(k, 0),
                "Concept Mass": "—",
                "Soft Term Sum": "—",
                "Total raw_k": f"{raw.get(k, 0):.4f}",
            }
            for k in SIX_CATEGORIES
        ])
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

        # Eq (4)
        st.markdown("**Eq. (4) — Smoothed Allocation (Laplace prior)**")
        st.latex(r"W_k = \frac{\alpha + \mathrm{raw}_k}{6\alpha + \sum_j \mathrm{raw}_j}"
                 r", \quad \alpha = " + f"{alpha}")
        alloc_df = pd.DataFrame([
            {
                "Category": CATEGORY_DISPLAY[k],
                "α (prior)": alpha,
                "raw_k": f"{raw.get(k, 0):.4f}",
                "Numerator": f"{alpha + raw.get(k, 0):.4f}",
                "Denominator": f"{denom:.4f}",
                "**W_k**": f"**{W.get(k, 0):.4f}**",
            }
            for k in SIX_CATEGORIES
        ])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        st.caption(f"✅ Sanity check: Σ W_k = {sum(W.values()):.6f} (must equal 1.0)")

        # Eq (5)
        st.markdown("**Eq. (5) — Concentration (Normalized Entropy)**")
        st.latex(r"H(W) = -\sum_k W_k \ln W_k, \quad "
                 r"c = 1 - \frac{H}{\ln 6}")
        H = -sum(w * math.log(w) for w in W.values() if w > 1e-12)
        H_max = math.log(len(SIX_CATEGORIES))
        st.caption(
            f"H(W) = {H:.4f}  |  ln(6) = {H_max:.4f}  |  "
            f"c = 1 − {H:.4f}/{H_max:.4f} = **{c:.4f}**"
        )

        # Eq (7)
        st.markdown("**Eq. (7) — Adaptive Subgraph Depth**")
        st.latex(r"K = 2 + \mathrm{round}(2c) \in \{2, 3, 4\}")
        st.caption(
            f"c = {c:.4f}  →  2c = {2*c:.4f}  →  "
            f"round(2c) = {round(2*c)}  →  **K = {K}**"
        )


def render_qdwa_sankey(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Sankey diagram: Query → Terms → Categories → Allocation."""
    if theme is None:
        theme = {"font": "#1e293b", "accent": "#3b82f6"}

    st.markdown("### 🌊 QDWA Sankey: Query → Terms → Categories")

    labels, sources, targets, values, colors = [], [], [], [], []

    # Root node
    query_preview = analysis.query[:50] + "..." if len(analysis.query) > 50 else analysis.query
    labels.append(f"Query")
    root_idx = 0
    colors.append(theme.get("accent", "#3b82f6"))

    # Term nodes
    term_indices = {}
    terms = list(analysis.term_memberships.keys())
    for i, term in enumerate(terms):
        display_term = term.replace("_", " ").title()[:20]
        labels.append(display_term)
        term_indices[term] = len(labels) - 1
        colors.append("#94a3b8")
        # Edge: query → term
        sources.append(root_idx)
        targets.append(term_indices[term])
        values.append(1.0 / max(len(terms), 1))

    # Category nodes
    cat_indices = {}
    for cat in SIX_CATEGORIES:
        labels.append(CATEGORY_DISPLAY[cat][:15])
        cat_indices[cat] = len(labels) - 1
        colors.append(CATEGORY_COLORS.get(cat, "#64748b"))

    # Edges: term → category
    for term, memberships in analysis.term_memberships.items():
        if term not in term_indices:
            continue
        for cat, m in memberships.items():
            if m > 0.01:
                sources.append(term_indices[term])
                targets.append(cat_indices[cat])
                values.append(float(m))

    # Sink node
    labels.append("Final W_k")
    sink_idx = len(labels) - 1
    colors.append(theme.get("font", "#000000"))
    for cat in SIX_CATEGORIES:
        sources.append(cat_indices[cat])
        targets.append(sink_idx)
        values.append(float(analysis.category_weights.get(cat, 0)))

    # Link colors
    link_colors = []
    for s in sources:
        hex_c = colors[s]
        if hex_c.startswith('#') and len(hex_c) == 7:
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},0.4)")
        else:
            link_colors.append("rgba(100,100,100,0.4)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15, thickness=20,
            line=dict(color="white", width=0.5),
            label=labels, color=colors,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hovertemplate="<b>%{value:.4f}</b><extra></extra>",
        ),
    ))
    fig.update_layout(
        title="Query Distillation & Weighted Allocation Flow",
        paper_bgcolor=theme.get("bg", "#fff"),
        font_color=theme.get("font", "#000"),
        height=500, margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_qdwa_radar_chart(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Radar/spider chart of category weights."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff"}

    st.markdown("### 🕸️ QDWA Radar: Category Weight Distribution")

    categories = [CATEGORY_DISPLAY[k][:12] for k in SIX_CATEGORIES]
    weights = [analysis.category_weights[k] for k in SIX_CATEGORIES]
    colors = [CATEGORY_COLORS[k] for k in SIX_CATEGORIES]

    # Close the polygon
    categories_closed = categories + [categories[0]]
    weights_closed = weights + [weights[0]]

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=weights_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2),
        name='W_k',
    ))

    # Individual points
    fig.add_trace(go.Scatterpolar(
        r=weights,
        theta=categories,
        mode='markers+text',
        marker=dict(
            color=colors,
            size=12,
            line=dict(width=2, color='white'),
        ),
        text=[f"{w:.3f}" for w in weights],
        textposition='top center',
        textfont=dict(size=10),
        name='Weights',
    ))

    # Uniform reference line
    uniform = 1.0 / len(SIX_CATEGORIES)
    fig.add_trace(go.Scatterpolar(
        r=[uniform] * (len(SIX_CATEGORIES) + 1),
        theta=categories_closed,
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name=f'Uniform ({uniform:.3f})',
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(weights) * 1.2] if max(weights) > 0 else [0, 0.3],
                tickformat='.3f',
            ),
        ),
        showlegend=True,
        title=f"Concentration c = {analysis.concentration:.3f}",
        paper_bgcolor=theme.get("bg", "#fff"),
        font_color=theme.get("font", "#000"),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_qdwa_bar_comparison(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Bar charts: Raw vs Smoothed, Entropy decomposition, K gauge."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff"}

    st.markdown("### 📊 QDWA Bar Charts")

    col1, col2 = st.columns(2)

    with col1:
        df_alloc = pd.DataFrame([
            {
                "Category": CATEGORY_DISPLAY[k][:12],
                "Raw Evidence": analysis.raw_evidence.get(k, 0),
                "Smoothed W_k": analysis.category_weights.get(k, 0),
            }
            for k in SIX_CATEGORIES
        ])
        fig1 = px.bar(
            df_alloc, x="Category", y=["Raw Evidence", "Smoothed W_k"],
            barmode="group",
            title="Raw Evidence vs. Smoothed Weight W_k",
            color_discrete_sequence=["#94a3b8", "#3b82f6"],
        )
        fig1.update_layout(
            paper_bgcolor=theme.get("bg", "#fff"),
            font_color=theme.get("font", "#000"),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        W = analysis.category_weights
        H = -sum(w * math.log(w) for w in W.values() if w > 1e-12)
        H_max = math.log(len(SIX_CATEGORIES))

        fig2 = go.Figure(go.Bar(
            x=["H(W)", "H_max=ln(6)", "Deficit"],
            y=[H, H_max, H_max - H],
            marker_color=["#f59e0b", "#94a3b8", "#10b981"],
            text=[f"{H:.4f}", f"{H_max:.4f}", f"{H_max - H:.4f}"],
            textposition="outside",
        ))
        fig2.update_layout(
            title=f"Entropy → c = {analysis.concentration:.4f}",
            yaxis_title="nats",
            paper_bgcolor=theme.get("bg", "#fff"),
            font_color=theme.get("font", "#000"),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # K-hop gauge
    col3, _ = st.columns([1, 2])
    with col3:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analysis.subgraph_depth,
            title={"text": "Subgraph Depth K"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [2, 4], 'tickwidth': 2},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [2, 2.5], 'color': "#d1fae5"},
                    {'range': [2.5, 3.5], 'color': "#fef3c7"},
                    {'range': [3.5, 4], 'color': "#fee2e2"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': analysis.subgraph_depth,
                },
            },
        ))
        fig3.update_layout(
            paper_bgcolor=theme.get("bg", "#fff"),
            font_color=theme.get("font", "#000"),
            height=280, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)


def render_qdwa_heatmap(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Heatmap: Term × Category membership matrix."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff"}

    st.markdown("### 🔥 QDWA Heatmap: Term × Category Membership")

    if not analysis.term_memberships:
        st.info("No terms extracted from query.")
        return

    df = pd.DataFrame(analysis.term_memberships).T
    df.index = [t.replace("_", " ").title()[:18] for t in df.index]
    df.columns = [CATEGORY_DISPLAY.get(k, k)[:10] for k in df.columns]

    fig = px.imshow(
        df.values,
        x=df.columns, y=df.index,
        color_continuous_scale="Viridis",
        labels=dict(x="Category", y="Term", color="m_k(t)"),
        title="Soft Category Membership m_k(t) per Query Term",
        aspect="auto",
    )
    fig.update_traces(
        hovertemplate="<b>Term:</b> %{y}<br><b>Category:</b> %{x}<br>"
                      "<b>m_k(t):</b> %{z:.4f}<extra></extra>"
    )
    fig.update_layout(
        paper_bgcolor=theme.get("bg", "#fff"),
        font_color=theme.get("font", "#000"),
        height=max(300, len(df) * 40 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_qdwa_chord_matrix(
    graph: nx.Graph,
    theme: Dict[str, str] = None,
) -> None:
    """Category × Category edge count matrix (chord-equivalent)."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff"}

    st.markdown("### 🎯 QDWA Chord: Inter-Category Coupling")

    n = len(SIX_CATEGORIES)
    matrix = np.zeros((n, n))

    for u, v in graph.edges():
        cu = categorize_concept(u)
        cv = categorize_concept(v)
        if cu in SIX_CATEGORIES and cv in SIX_CATEGORIES:
            i = SIX_CATEGORIES.index(cu)
            j = SIX_CATEGORIES.index(cv)
            matrix[i][j] += 1
            if i != j:
                matrix[j][i] += 1

    if matrix.sum() == 0:
        st.info("No inter-category edges in the current graph.")
        return

    cat_labels = [CATEGORY_DISPLAY[k][:12] for k in SIX_CATEGORIES]
    fig = ff.create_annotated_heatmap(
        matrix.tolist(),
        x=cat_labels, y=cat_labels,
        colorscale="Blues",
        showscale=True,
        annotation_text=matrix.astype(int).tolist(),
    )
    fig.update_layout(
        title="Category × Category Edge Count Matrix",
        paper_bgcolor=theme.get("bg", "#fff"),
        font_color=theme.get("font", "#000"),
        height=450,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_qdwa_energy_density_score(
    analysis: QDWAAnalysis,
    theme: Dict[str, str] = None,
) -> None:
    """Dedicated energy density relevance gauge."""
    if theme is None:
        theme = {"font": "#1e293b", "bg": "#ffffff"}

    st.markdown("### ⚡ Energy Density Enhancement Relevance")

    ed_score = analysis.get_energy_density_relevance()

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ed_score * 100,
            delta={'reference': 50, 'increasing': {'color': "#10b981"}},
            title={"text": "ED Relevance %"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2},
                'bar': {'color': "#10b981" if ed_score > 0.5 else "#f59e0b"},
                'steps': [
                    {'range': [0, 30], 'color': "#fee2e2"},
                    {'range': [30, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#d1fae5"},
                ],
                'threshold': {
                    'line': {'color': "#10b981", 'width': 4},
                    'thickness': 0.75,
                    'value': 60,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor=theme.get("bg", "#fff"),
            font_color=theme.get("font", "#000"),
            height=250, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Breakdown by category
        ed_weights = {
            "cathode_materials": 0.30,
            "anode_materials": 0.25,
            "manufacturing": 0.25,
            "performance": 0.15,
            "electrolyte_systems": 0.03,
            "degradation": 0.02,
        }
        contrib = []
        for cat, factor in ed_weights.items():
            w = analysis.category_weights.get(cat, 0)
            contrib.append({
                "Category": CATEGORY_DISPLAY[cat][:15],
                "W_k": w,
                "ED Factor": factor,
                "Contribution": w * factor,
            })
        contrib_df = pd.DataFrame(contrib)
        st.dataframe(
            contrib_df.style.format("{:.4f}").background_gradient(
                subset=["Contribution"], cmap="Greens"
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_qdwa_full_dashboard(
    analysis: QDWAAnalysis,
    graph: nx.Graph = None,
    theme: Dict[str, str] = None,
) -> None:
    """Master render: all QDWA visualizations in sequence."""
    if analysis is None:
        st.warning("No QDWA analysis available. Run a query first.")
        return

    # Summary metrics at top
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Primary Category", CATEGORY_DISPLAY.get(
            analysis.primary_category, analysis.primary_category
        )[:20])
    with col2:
        st.metric("Concentration c", f"{analysis.concentration:.4f}")
    with col3:
        st.metric("Subgraph Depth K", analysis.subgraph_depth)
    with col4:
        ed_score = analysis.get_energy_density_relevance()
        st.metric("ED Relevance", f"{ed_score*100:.1f}%",
                  delta=f"{(ed_score-0.5)*100:+.1f}%")

    st.markdown("---")

    # Ranked categories table
    st.markdown("#### 📋 Category Weight Rankings")
    ranked = analysis.get_ranked_categories()
    rank_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Category": CATEGORY_DISPLAY.get(cat, cat),
            "W_k": f"{w:.4f}",
            "Raw Evidence": f"{analysis.raw_evidence.get(cat, 0):.2f}",
            "Color": "🟢" if w > 0.2 else ("🟡" if w > 0.15 else "⚪"),
        }
        for i, (cat, w) in enumerate(ranked)
    ])
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # All visualizations
    render_qdwa_math_trace(analysis, theme)
    st.markdown("---")
    render_qdwa_sankey(analysis, theme)
    st.markdown("---")
    render_qdwa_radar_chart(analysis, theme)
    st.markdown("---")
    render_qdwa_bar_comparison(analysis, theme)
    st.markdown("---")
    render_qdwa_heatmap(analysis, theme)
    st.markdown("---")
    render_qdwa_energy_density_score(analysis, theme)
    st.markdown("---")

    if graph is not None:
        render_qdwa_chord_matrix(graph, theme)


# ============================================================================
# QDWA GLOBAL INSTANCE (initialize once)
# ============================================================================
#
@st.cache_resource
def get_qdwa_engine(_ontology: DomainOntology) -> QDWAEngine:
    """Get or create the QDWA engine (cached as resource)."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    except Exception:
        st.warning("SentenceTransformer not available, using keyword fallback.")
        model = None
    return QDWAEngine(_ontology, embedding_model=model)


def initialize_qdwa_in_session():
    """Initialize QDWA engine and store in session state."""
    if "qdwa_engine" not in st.session_state:
        ontology = st.session_state.get("ontology", DomainOntology())
        st.session_state.qdwa_engine = get_qdwa_engine(ontology)

    if "last_qdwa_analysis" not in st.session_state:
        st.session_state.last_qdwa_analysis = None

    if "last_query_text" not in st.session_state:
        st.session_state.last_query_text = ""


def render_qdwa_sidebar_preview(analysis: QDWAAnalysis) -> None:
    """Compact QDWA preview for sidebar."""
    st.sidebar.markdown("#### ⚖️ QDWA Weights")
    cols = st.sidebar.columns(3)
    ranked = analysis.get_ranked_categories()
    for i, (cat, w) in enumerate(ranked[:6]):
        col = cols[i % 3]
        short_name = CATEGORY_DISPLAY.get(cat, cat)[:8]
        col.metric(short_name, f"{w:.2f}")

    st.sidebar.caption(
        f"c={analysis.concentration:.2f} → K={analysis.subgraph_depth}"
    )

    if st.sidebar.button(
        "🔍 Open Full QDWA Dashboard",
        key="open_qdwa_dashboard_btn",
        use_container_width=True,
    ):
        st.session_state["show_qdwa_dashboard"] = True


def run_qdwa_analysis(query: str, ontology_concepts: List[str] = None) -> QDWAAnalysis:
    """Run QDWA and store results in session state."""
    engine = st.session_state.get("qdwa_engine")
    if engine is None:
        initialize_qdwa_in_session()
        engine = st.session_state.qdwa_engine

    analysis = engine.analyze_query(query, ontology_concepts)

    st.session_state.last_qdwa_analysis = analysis
    st.session_state.last_query_text = query

    return analysis


def render_qdwa_tab() -> None:
    """Full QDWA dashboard tab content."""
    st.header("⚖️ Query Distillation & Weighted Allocation")
    st.caption(
        "QDWA analyzes your query to determine which knowledge domains "
        "(Cathode, Anode, Electrolyte, Manufacturing, Degradation, Performance) "
        "are most relevant for energy density enhancement."
    )

    analysis = st.session_state.get("last_qdwa_analysis")
    query_text = st.session_state.get("last_query_text", "")

    # Manual query input
    with st.expander("📝 Enter Query for QDWA Analysis", expanded=analysis is None):
        manual_query = st.text_area(
            "Query about energy density enhancement:",
            value=query_text,
            height=100,
            key="qdwa_manual_query",
        )
        if st.button("Run QDWA Analysis", key="run_qdwa_manual", type="primary"):
            if manual_query.strip():
                with st.spinner("Computing QDWA weights..."):
                    analysis = run_qdwa_analysis(manual_query.strip())
                st.success("QDWA analysis complete!")
                st.rerun()

    if analysis is None:
        st.info(
            "👈 Enter a query above, or ask a question in the LLM panel "
            "to populate QDWA weights."
        )
        return

    # Theme
    theme_name = st.session_state.get("theme", "Bright (Default)")
    theme = {
        "font": "#1e293b",
        "bg": "#ffffff",
        "accent": "#3b82f6",
    }

    # Get graph if available
    graph = None
    if "analysis_data" in st.session_state:
        graph = st.session_state.analysis_data.get("nx_graph")

    # Render full dashboard
    render_qdwa_full_dashboard(analysis, graph=graph, theme=theme)


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ Configuration v7.0 (QDWA)")

        # ============================================================================
        # RECORD SAMPLING — prevents OOM on 4000+ records
        # ============================================================================
        st.markdown("---")
        st.markdown("### 📊 Record Sampling")
        st.caption("Full database may exceed 1 GB. Select a subset:")

        SAMPLING_OPTIONS = {
            "All records (100%)":    1.0,
            "Half (50%)":           0.50,
            "Quarter (25%)":        0.25,
            "One-tenth (10%)":      0.10,
            "One-twentieth (5%)":   0.05,
            "100 records":          100,
            "250 records":          250,
            "500 records":          500,
        }

        sampling_choice = st.selectbox(
            "Dataset fraction", options=list(SAMPLING_OPTIONS.keys()), index=2,
            key="sampling_select",
        )
        sampling_seed = st.sidebar.number_input(
            "Random seed", min_value=0, max_value=999999, value=42, step=1,
            key="sampling_seed",
        )

        st.subheader("🎨 Theme")
        st.session_state['theme'] = st.selectbox(
            "Color theme:",
            options=list(THEME_PRESETS.keys()),
            index=0,
        )

        st.subheader("🔍 Query-Focused Graph Mode")
        query_focused_enabled = st.checkbox("Build graph only for current query concepts", key="query_focused_build")
        if query_focused_enabled:
            whitelist = st.session_state.get('last_query_whitelist', set())
            if whitelist:
                st.success(f"Will extract {len(whitelist)} focused concepts")
                if st.session_state.get('batch_mode', False):
                    st.info(
                        "📦 **Batch mode compatible** — frequency threshold will be "
                        "auto-lowered to 1–2 so all whitelisted concepts survive."
                    )
                with st.expander("Preview whitelisted concepts"):
                    st.write(sorted(whitelist))
            else:
                st.info("Ask a question in the 🤖 LLM-Guided Q&A tab to generate a whitelist.")
        theme = THEME_PRESETS[st.session_state['theme']]
        st.subheader("🔬 Lithium‑Ion Battery Focus Areas")
        st.markdown("- **Materials:** NMC811, LFP, Graphite, Silicon, Solid Electrolytes, etc.")
        st.markdown("- **Manufacturing:** Calendering, Slot‑Die Coating, Doping, Surface Coating, Prelithiation, Formation")
        st.markdown("- **Properties:** Energy Density, Specific Capacity, Cycle Life, Coulombic Efficiency, Rate Capability, Thermal Stability")
        st.markdown("- **Phenomena:** Capacity Fade, SEI Formation, Dendrite Growth, Thermal Runaway, Impedance Growth, TM Dissolution, Cation Mixing, Particle Cracking")
        st.markdown("- **Parameters:** Electrode Thickness, N/P Ratio, C‑rate, Temperature, Pressure")
        st.markdown("- **Methods:** EIS, dQ/dV, Galvanostatic Cycling, Coin Cell, Pouch Cell")
        st.subheader("🧠 NLP Reasoning Options")
        st.session_state['use_ontology'] = st.checkbox(
            "Use ontology-based resolution", value=True,
            help="Maps synonyms like 'NMC811' to canonical concepts",
        )
        st.session_state['use_embedding_resolution'] = st.checkbox(
            "Use embedding-based semantic equivalence", value=True,
            help="Detects semantic similarity >0.85 even for unseen variants",
        )
        st.session_state['use_relationship_extraction'] = st.checkbox(
            "Extract cause-effect relationships", value=True,
            help="Identifies causal links between manufacturing parameters and properties",
        )
        st.session_state['use_inference'] = st.checkbox(
            "Enable reasoning-based edge inference", value=True,
            help="Infers manufacturing→property chains even when not co-occurring",
        )
        st.session_state['context_window'] = st.slider(
            "Context window (chars)", 20, 200, 50,
            help="Window size for context-based disambiguation",
        )
        st.subheader("📊 Visualization")
        st.session_state['viz_backend'] = st.selectbox(
            "Engine:",
            ["PyVis (Interactive)", "Plotly 2D", "Plotly 3D", "Text Summary"],
            index=0,
        )
        st.session_state['show_edge_weights'] = st.toggle(
            "Show edge weights", value=False,
            help="Display numerical weight labels on graph edges.",
        )
        st.session_state['edge_label_mode'] = st.selectbox(
            "Edge label mode:", ["hover", "threshold", "all"], index=0,
            help="hover=tooltip only, threshold=top 20% edges, all=all edges",
        )
        st.session_state['cmap_name'] = st.selectbox(
            "Colormap:",
            options=list(SUPPORTED_COLORMAPS.keys()),
            index=0,
        )
        st.subheader("⚡ Physics & Layout")
        st.session_state['physics_preset'] = st.selectbox(
            "Physics preset:",
            options=list(PHYSICS_PRESETS.keys()),
            index=0,
        )
        preset = PHYSICS_PRESETS[st.session_state['physics_preset']]
        st.session_state['physics_enabled'] = st.checkbox(
            "Enable physics", value=(preset["gravity"] != 0),
        )
        with st.expander("Advanced Physics Overrides"):
            st.session_state['adv_damping'] = st.slider(
                "Damping", 0.05, 0.95, preset["damping"], step=0.05,
            )
            st.session_state['adv_gravity'] = st.slider(
                "Repulsion", -8000, -500, preset["gravity"], step=100,
            )
            st.session_state['adv_spring_length'] = st.slider(
                "Spring length", 40, 300, preset["spring_length"], step=10,
            )
            st.session_state['adv_spring_strength'] = st.slider(
                "Spring strength", 0.01, 0.20,
                preset["spring_strength"], step=0.01,
            )
            st.session_state['adv_central_gravity'] = st.slider(
                "Central gravity", 0.0, 0.5,
                preset["central_gravity"], step=0.05,
            )
            st.session_state['adv_stabilization'] = st.slider(
                "Stabilization iter", 0, 5000,
                preset["stabilization"], step=250,
            )
        base_preset = PHYSICS_PRESETS[
            st.session_state['physics_preset']
        ].copy()
        if st.session_state.get('adv_damping') is not None:
            base_preset["damping"] = st.session_state['adv_damping']
            base_preset["gravity"] = st.session_state['adv_gravity']
            base_preset["spring_length"] = st.session_state['adv_spring_length']
            base_preset["spring_strength"] = st.session_state['adv_spring_strength']
            base_preset["central_gravity"] = st.session_state['adv_central_gravity']
            base_preset["stabilization"] = st.session_state['adv_stabilization']
        st.session_state['effective_physics'] = base_preset
        st.subheader("📏 Display Limits")
        col_all1, col_slider1 = st.columns([0.3, 0.7])
        with col_all1:
            all_graph = st.checkbox("All", value=True, key="all_graph_chk")
        with col_slider1:
            st.session_state['top_n_graph'] = st.slider(
                "Max nodes", 10, 500, 200, step=10,
                disabled=all_graph, key="top_n_graph_slider",
            )
        if all_graph:
            st.session_state['top_n_graph'] = 0
        col_all2, col_slider2 = st.columns([0.3, 0.7])
        with col_all2:
            all_sun = st.checkbox("All", value=True, key="all_sun_chk")
        with col_slider2:
            st.session_state['top_n_sunburst'] = st.slider(
                "Max children/category", 10, 100, 40, step=10,
                disabled=all_sun, key="top_n_sunburst_slider",
            )
        if all_sun:
            st.session_state['top_n_sunburst'] = 0
        col_all3, col_slider3 = st.columns([0.3, 0.7])
        with col_all3:
            all_radar = st.checkbox("All", value=True, key="all_radar_chk")
        with col_slider3:
            st.session_state['top_n_radar'] = st.slider(
                "Top K for radar", 5, 30, 15,
                disabled=all_radar, key="top_n_radar_slider",
            )
        if all_radar:
            st.session_state['top_n_radar'] = 0
        st.subheader("🔧 Graph Parameters")
        st.session_state['min_freq'] = st.slider(
            "Min concept frequency", 1, 20, 1,
        )
        st.session_state['min_words'] = st.slider(
            "Min words per concept", 2, 5, 2,
        )
        st.session_state['sim_threshold'] = st.slider(
            "Semantic threshold", 0.6, 0.95, 0.85, step=0.05,
        )
        st.session_state['cooc_weight'] = st.slider(
            "Co-occurrence weight", 0.5, 1.0, 0.7, step=0.1,
        )
        st.session_state['sem_weight'] = st.slider(
            "Semantic weight", 0.0, 0.5, 0.2, step=0.1,
        )
        st.session_state['inf_weight'] = st.slider(
            "Inference weight", 0.0, 0.3, 0.1, step=0.05,
        )
        
        # Batch Processing Controls
        render_batch_processing_controls()

        st.subheader("📈 Statistics")
        st.session_state['bootstrap_samples'] = st.slider(
            "Bootstrap samples", 100, 2000, 500, step=100,
        )
        st.session_state['alpha_level'] = st.selectbox(
            "Significance alpha", [0.01, 0.05, 0.10], index=1,
        )

        st.markdown("---")
        st.subheader("🎨 Visualization Customization")
        st.session_state['enable_node_highlight'] = st.checkbox(
            "🔍 Enable Node Selection Highlight & Descriptions",
            value=False,
            help=(
                "When enabled, clicking a node highlights connected nodes "
                "with gold borders and overlays edge weights/relationship descriptions."
            ),
        )
        with st.expander("Node & Label Settings"):
            st.markdown("##### 🏷️ Node Label Display  ·  engine v4")
            label_mode_choice = st.selectbox(
                "Label mode (pick ONE — three orthogonal options):",
                options=list(LABEL_MODE_OPTIONS.keys()),
                index=0,
                key="pyvis_label_mode",
                help=(
                    "1) Full Name = full concept name inside each node.\n"
                    "2) Annotations = N1, N2, … inside each node + legend table below the graph.\n"
                    "3) Custom Blank = nodes are blank inside; you type the text you want."
                ),
            )
            st.session_state['label_mode'] = LABEL_MODE_OPTIONS[label_mode_choice]

            if st.session_state['label_mode'] == NodeLabelMode.FULL_NAME:
                st.caption("✏️ Full concept name will appear inside each node.")
                st.session_state['node_label_size'] = st.slider(
                    "Font size (px)", 8, 50, 25, step=1, key="full_label_font_size",
                )
                st.session_state['node_label_position'] = st.selectbox(
                    "Label position inside node",
                    ["center", "top", "bottom"], index=0,
                )

            elif st.session_state['label_mode'] == NodeLabelMode.ANNOTATION:
                st.caption("🔢 Each node becomes N1, N2, … A legend maps them to full names.")
                st.session_state['node_label_size'] = st.slider(
                    "Annotation font size (px)", 8, 40, 18, step=1, key="annot_font_size",
                )
                _annot_prefix = st.text_input(
                    "Prefix (optional)", value="N", key="annot_prefix",
                    help="Use 'N' for N1, N2 … or 'C' for C1, C2 … or any prefix you like.",
                )
                st.session_state['node_label_position'] = "center"

            elif st.session_state['label_mode'] == NodeLabelMode.CUSTOM_BLANK:
                st.caption("⬜ Nodes are blank inside. Type your own text below.")
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state['external_font_size'] = st.slider(
                        "Font size (px)", 6, 60, 14, step=1, key="blank_font_size",
                    )
                with c2:
                    st.session_state['external_font_color'] = st.color_picker(
                        "Font colour", "#1e293b", key="blank_font_color",
                    )
                st.session_state['external_label_align'] = st.radio(
                    "Text alignment", ["center", "top", "bottom", "left", "right"],
                    index=0, horizontal=True, key="blank_label_align",
                )
                st.session_state['external_label_text'] = st.text_input(
                    "Common text for ALL nodes (optional)", "",
                    key="blank_common_text",
                    help=(
                        "If you type something here (e.g. '•'), EVERY node shows that "
                        "same string. Leave blank for truly empty nodes, OR use the "
                        "per-node overrides below."
                    ),
                )
                _custom = st.text_area(
                    "Per-node overrides  (node_key = Your Text, one per line)",
                    key="custom_label_text", height=110,
                    placeholder=(
                        "nmc811 = NMC811\n"
                        "lfp = LFP\n"
                        "silicon = Si\n"
                        "energy_density = E\n"
                        "capacity_fade = Fade"
                    ),
                    help="These override the common text on a per-node basis.",
                )
                _map = {}
                for _ln in _custom.splitlines():
                    if "=" in _ln:
                        _k, _v = _ln.split("=", 1)
                        if _k.strip() and _v.strip():
                            _map[_k.strip()] = _v.strip()
                st.session_state['custom_label_map'] = _map
                st.session_state['node_label_size'] = st.session_state.get('external_font_size', 14)
                st.session_state['node_label_position'] = st.session_state.get('external_label_align', 'left')

            st.session_state['node_font_face'] = st.selectbox(
                "Font family",
                [
                    "Inter, Segoe UI, Roboto, sans-serif",
                    "Arial, Helvetica, sans-serif",
                    "Georgia, serif",
                    "Courier New, monospace",
                    "Times New Roman, serif",
                ],
                index=0,
            )
            st.slider(
                "Node legend font size", 8, 50, 25, step=1,
                help="Font size for the legend table below the graph.",
                key="node_legend_font_size",
            )
        st.session_state['use_abbreviated_labels'] = (
            st.session_state.get('label_mode') == NodeLabelMode.ANNOTATION
        )
        st.session_state['max_label_length'] = 0
        st.session_state['show_definitions'] = st.checkbox(
            "📖 Show concept definitions in tooltips",
            value=True,
            help="When enabled, hovering over a node displays its ontology definition in the tooltip.",
        )
        with st.expander("Edge Label Settings"):
            st.session_state['edge_label_size'] = st.slider(
                "Edge label font size", 6, 18, 10, step=1,
                help="Font size for edge weight labels",
            )
            st.session_state['edge_label_color'] = st.color_picker(
                "Edge label color", value="#000000",
                help="Color for edge weight labels (default matches theme)",
            )
            st.session_state['edge_label_position'] = st.selectbox(
                "Edge label position",
                ["middle", "top", "bottom", "from", "to"],
                index=0,
                help="Where to place edge labels along the edge",
            )
        with st.expander("Edge Color Customization"):
            st.selectbox(
                "Edge color mode",
                ["theme", "uniform_grey", "custom"],
                index=0,
                help="theme: based on relationship type (lightened), uniform_grey: single grey, custom: your pick",
                key="edge_color_mode",
            )
            if st.session_state['edge_color_mode'] == "custom":
                st.color_picker(
                    "Custom edge color", value="#AAAAAA",
                    key="custom_edge_color",
                )
            else:
                st.session_state['custom_edge_color'] = "#AAAAAA"
            st.slider(
                "Edge lightness (0=original, 1=white)", 0.0, 1.0, 0.6, step=0.05,
                help="Higher values make edges lighter, improving node visibility.",
                key="edge_lightness",
            )
        edge_color_value = st.session_state.get('edge_label_color')
        if not edge_color_value or edge_color_value == '':
            edge_color_value = '#000000'
        st.session_state['edge_label_color'] = edge_color_value

        st.markdown("---")
        st.subheader("🖥️ Hardware")
        cpu_toggle = st.checkbox("Force CPU mode (disable CUDA)", value=is_force_cpu(), key="force_cpu")
        if cpu_toggle:
            st.info("CPU mode active — all models will run on CPU.")
        st.markdown("---")
        st.subheader("✏️ Graph Editing")
        with st.expander("Remove Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_remove = st.multiselect(
                    "Select nodes to remove:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="remove_nodes_select",
                )
                st.session_state['nodes_to_remove'] = nodes_to_remove
            else:
                st.info("Build graph first to edit nodes.")
                st.session_state['nodes_to_remove'] = []
        with st.expander("Merge Nodes"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                nodes_to_merge = st.multiselect(
                    "Select nodes to merge:",
                    options=st.session_state['analysis_data']['valid_concepts'],
                    key="merge_nodes_select",
                )
                merge_name = st.text_input(
                    "New merged concept name:", key="merge_name_input",
                )
                st.session_state['nodes_to_merge'] = nodes_to_merge
                st.session_state['merge_name'] = merge_name
            else:
                st.info("Build graph first to merge nodes.")
                st.session_state['nodes_to_merge'] = []
                st.session_state['merge_name'] = ""
        with st.expander("Add Edge"):
            if (
                st.session_state.get('analysis_data')
                and st.session_state['analysis_data'].get('valid_concepts')
            ):
                all_concepts = st.session_state['analysis_data']['valid_concepts']
                edge_u = st.selectbox(
                    "Source concept:", options=all_concepts, key="edge_u_select",
                )
                edge_v = st.selectbox(
                    "Target concept:", options=all_concepts, key="edge_v_select",
                )
                edge_weight = st.number_input(
                    "Edge weight:", min_value=0.1, max_value=10.0,
                    value=1.0, step=0.1, key="edge_weight_input",
                )
                st.session_state['new_edge'] = (
                    (edge_u, edge_v) if edge_u != edge_v else None
                )
                st.session_state['new_edge_weight'] = edge_weight
            else:
                st.info("Build graph first to add edges.")
                st.session_state['new_edge'] = None
                st.session_state['new_edge_weight'] = 1.0
        with st.expander("Filter by Degree/Frequency"):
            st.session_state['filter_min_degree'] = st.slider(
                "Min degree", 0, 20, 0, key="filter_degree_slider",
            )
            st.session_state['filter_min_freq'] = st.slider(
                "Min frequency", 0, 50, 0, key="filter_freq_slider",
            )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            if st.button("Apply Graph Edits", key="apply_edits_btn"):
                st.session_state['apply_edits'] = True
        if (
            st.session_state.get('analysis_data')
            and st.session_state.get('edit_history')
        ):
            col_undo, col_redo = st.columns(2)
            with col_undo:
                if (
                    st.button("↩️ Undo", key="undo_btn")
                    and st.session_state['edit_history'].can_undo()
                ):
                    snapshot = st.session_state['edit_history'].undo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Undo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
            with col_redo:
                if (
                    st.button("↪️ Redo", key="redo_btn")
                    and st.session_state['edit_history'].can_redo()
                ):
                    snapshot = st.session_state['edit_history'].redo()
                    if snapshot:
                        st.session_state['analysis_data']['nx_graph'] = snapshot['nx_graph']
                        st.session_state['analysis_data']['valid_concepts'] = snapshot['valid_concepts']
                        st.session_state['analysis_data']['concept_to_id'] = snapshot['concept_to_id']
                        st.session_state['analysis_data']['id_to_concept'] = snapshot['id_to_concept']
                        st.session_state['analysis_data']['concept_abstract_map'] = snapshot['concept_abstract_map']
                        st.success("Redo applied!")
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()

        st.markdown("---")
        st.subheader("☀️ Sunburst Chart Customization")
        st.session_state['sunburst_cmap'] = st.selectbox(
            "Colormap:",
            options=[
                "viridis", "plasma", "inferno", "magma", "cividis",
                "turbo", "rainbow", "hsv", "coolwarm", "RdBu", "Spectral",
                "tab10", "tab20", "Pastel1", "Set1", "Set2", "Set3",
                "YlOrRd", "PuBuGn", "GnBu", "YlGnBu",
            ],
            index=0,
            help="Choose color scheme for sunburst categories",
            key="sunburst_cmap_select",
        )
        st.session_state['sunburst_font_family'] = st.selectbox(
            "Sunburst font family",
            [
                "Arial, sans-serif",
                "Inter, Segoe UI, Roboto, sans-serif",
                "Georgia, serif",
                "Courier New, monospace",
                "Times New Roman, serif",
            ],
            index=0,
            help="Font family for sunburst chart labels",
            key="sunburst_font_family_select",
        )
        col_labels, col_values = st.columns(2)
        with col_labels:
            st.session_state['sunburst_show_labels'] = st.checkbox(
                "Show symbols", value=True,
                help="Display symbol combinations inside chart segments",
                key="sunburst_show_labels_chk",
            )
        with col_values:
            st.session_state['sunburst_show_values'] = st.checkbox(
                "Show values", value=False,
                help="Display numerical values inside chart segments",
                key="sunburst_show_values_chk",
            )
        st.session_state['sunburst_hover_info'] = st.selectbox(
            "Hover information:",
            options=["all", "minimal", "none"],
            index=0,
            help="Amount of information shown on hover tooltip",
            key="sunburst_hover_select",
        )
        st.session_state['sunburst_branchvalues'] = st.selectbox(
            "Branch values mode:", ["total", "remainder"], index=0,
            help="How to calculate branch sizes: total=sum of children, remainder=parent minus children",
            key="sunburst_branch_mode",
        )
        col_w, col_h = st.columns(2)
        with col_w:
            st.session_state['sunburst_width'] = st.slider(
                "Chart width (px)", 600, 1400, 900, step=50,
                key="sunburst_width_slider",
            )
        with col_h:
            st.session_state['sunburst_height'] = st.slider(
                "Chart height (px)", 500, 1200, 700, step=50,
                key="sunburst_height_slider",
            )
        st.session_state['sunburst_label_size'] = st.slider(
            "Symbol font size", 8, 30, 20, step=1,
            help="Size of symbols inside sunburst slices",
            key="sunburst_label_size_slider",
        )
        st.slider(
            "Sunburst legend font size", 8, 50, 24, step=1,
            help="Font size for the symbol-to-label legend below the sunburst chart.",
            key="sunburst_legend_font_size",
        )
        st.session_state['sunburst_show_legend'] = st.checkbox(
            "Show symbol legend", value=True,
            help="Display symbol-to-label mapping table below chart",
            key="sunburst_show_legend_chk",
        )
        if (
            st.session_state.get('analysis_data')
            and st.session_state['analysis_data'].get('valid_concepts')
        ):
            all_cats = list(set(
                abstract_concepts_to_categories(
                    st.session_state['analysis_data']['valid_concepts']
                ).values()
            ))
            st.session_state['sunburst_categories'] = st.multiselect(
                "Filter categories:", options=all_cats,
                default=all_cats, key="sunburst_cat_filter",
            )
        else:
            st.info("Build graph first to filter categories.")
            st.session_state['sunburst_categories'] = []

        st.markdown("---")
        with st.expander("⚡ Performance Monitor"):
            if st.button("Show Timing Report"):
                report = PerformanceMonitor.get_report()
                if report:
                    st.code(report, language="text")
                else:
                    st.info("No timing data yet. Run analysis first.")
            if st.button("Reset Timings"):
                PerformanceMonitor.reset()
                st.success("Timing data reset!")

        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            gc.collect()
            st.success("Cache cleared!")
        dev = get_device()
        st.caption(f"Device: {dev.upper()}")

        ontology = st.session_state.ontology
        expander = st.session_state.qa_expander
        full_graph = st.session_state.analysis_data.get("nx_graph") if st.session_state.get('analysis_data') else nx.Graph()
        render_llm_query_panel(ontology, expander, full_graph)
        render_mutation_controls(expander)
        render_query_history()


# ============================================================================
# ★★★ LLM-GUIDED QUERY ANALYSIS & GRAPHRAG INTEGRATOR (v6.2) ★★★
# ============================================================================
import json
import re
import copy
import tempfile
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st

# ============================================================================
# 0. LOCAL LLM MODEL REGISTRY (< 1B parameters for Streamlit Cloud)
# ============================================================================
# (will be overridden dynamically in render_llm_query_panel)
LOCAL_LLM_REGISTRY: Dict[str, Optional[str]] = {
    "Fallback (Rule-based, no LLM)": None,
    "[Ollama] qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
    "[Ollama] qwen2.5:1.5b (Balanced)": "ollama:qwen2.5:1.5b",
    "[Ollama] qwen2.5:7b (Recommended for RAG)": "ollama:qwen2.5:7b",
    "[Ollama] qwen2.5:14b (Max Reasoning)": "ollama:qwen2.5:14b",
    "[Ollama] llama3.1:8b (Meta Standard)": "ollama:llama3.1:8b",
    "[Ollama] mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
    "[Ollama] gemma2:9b (Scientific Nuance)": "ollama:gemma2:9b",
    "[Ollama] falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
}

# ============================================================================
# 1. QUERY ANALYSIS DATA STRUCTURES (Lithium‑Ion Battery)
# ============================================================================
class LIBProblem(Enum):
    ENERGY_DENSITY_ENHANCEMENT = "energy_density_enhancement"
    CYCLE_LIFE_EXTENSION = "cycle_life_extension"
    FAST_CHARGING = "fast_charging"
    SAFETY_THERMAL_RUNAWAY = "safety_thermal_runaway"
    MANUFACTURING_REPRODUCIBILITY = "manufacturing_reproducibility"
    GENERAL = "general"
    MULTI_PROBLEM = "multi_problem"

@dataclass
class LIBProblemDefinition:
    problem_id: LIBProblem
    title: str
    scientific_description: str
    root_cause: str
    key_concepts: List[str]
    key_relationships: List[Tuple[str, str, str]]
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
            concepts.update([src, tgt])
        return concepts

# Pre-defined LIB Problem Definitions
LIB_PROBLEM_DEFINITIONS: Dict[LIBProblem, LIBProblemDefinition] = {
    LIBProblem.ENERGY_DENSITY_ENHANCEMENT: LIBProblemDefinition(
        problem_id=LIBProblem.ENERGY_DENSITY_ENHANCEMENT, title="Maximizing Energy Density",
        scientific_description="Increasing cell‑level energy density requires high‑capacity materials and thick electrodes, but trade‑offs exist with power and cycle life.",
        root_cause="Limitations of specific capacity, electrode density, and cell design parameters.",
        key_concepts=["energy_density", "specific_capacity", "electrode_thickness", "electrode_density"],
        key_relationships=[("electrode_thickness", "INFLUENCES", "energy_density"),
                           ("specific_capacity", "INFLUENCES", "energy_density")],
        solution_directions=["Use high‑capacity NMC811 or Si anodes", "Increase electrode thickness and density via calendering", "Optimize N/P ratio"],
        relevant_materials=["nmc811", "silicon", "graphite"],
        relevant_phenomena=[],
        relevant_properties=["energy_density", "specific_capacity", "electrode_density"],
        example_queries=["How can we increase the energy density of NMC811/graphite cells?", "What is the optimal electrode thickness for high energy density?"],
        visualization_focus=["energy_density_comparison", "thickness_optimization"]
    ),
    LIBProblem.CYCLE_LIFE_EXTENSION: LIBProblemDefinition(
        problem_id=LIBProblem.CYCLE_LIFE_EXTENSION, title="Extending Cycle Life",
        scientific_description="Capacity fade limits lifetime, caused by SEI growth, particle cracking, and TM dissolution.",
        root_cause="Degradation mechanisms such as SEI growth, particle cracking, and transition metal dissolution.",
        key_concepts=["cycle_life", "capacity_fade", "sei_formation", "particle_cracking", "transition_metal_dissolution"],
        key_relationships=[("sei_formation", "CAUSES", "capacity_fade"),
                           ("particle_cracking", "CAUSES", "capacity_fade"),
                           ("surface_coating", "PREVENTS", "transition_metal_dissolution")],
        solution_directions=["Use surface coating on cathode", "Add FEC/VC to electrolyte", "Optimize formation protocol"],
        relevant_materials=["nmc811", "lfp", "graphite", "fec", "vc"],
        relevant_phenomena=["capacity_fade", "sei_formation", "particle_cracking", "transition_metal_dissolution"],
        relevant_properties=["cycle_life", "coulombic_efficiency"],
        example_queries=["How can we improve the cycle life of NMC811 batteries?", "What role does FEC play in extending cycle life?"],
        visualization_focus=["cycle_life_curve", "degradation_mechanisms"]
    ),
    LIBProblem.FAST_CHARGING: LIBProblemDefinition(
        problem_id=LIBProblem.FAST_CHARGING, title="Enabling Fast Charging",
        scientific_description="Fast charging requires high rate capability, but is limited by lithium plating and impedance growth.",
        root_cause="Kinetic limitations and lithium plating at high currents.",
        key_concepts=["rate_capability", "c_rate", "impedance_growth", "dendrite_growth"],
        key_relationships=[("c_rate", "INFLUENCES", "rate_capability"),
                           ("impedance_growth", "INFLUENCES", "rate_capability")],
        solution_directions=["Use high‑conductivity electrolytes", "Reduce electrode thickness for shorter diffusion paths", "Modify electrode porosity"],
        relevant_materials=["liquid_electrolyte", "solid_electrolyte", "graphite"],
        relevant_phenomena=["impedance_growth", "dendrite_growth"],
        relevant_properties=["rate_capability", "ionic_conductivity", "porosity"],
        example_queries=["How can we improve fast charging capability?", "What is the impact of electrode porosity on rate performance?"],
        visualization_focus=["rate_performance_plot", "impedance_spectra"]
    ),
    LIBProblem.SAFETY_THERMAL_RUNAWAY: LIBProblemDefinition(
        problem_id=LIBProblem.SAFETY_THERMAL_RUNAWAY, title="Preventing Thermal Runaway",
        scientific_description="Thermal runaway is a critical safety hazard, often triggered by internal shorts or oxygen release.",
        root_cause="Cathode instability (e.g., NMC811 oxygen release) and dendrite‑induced shorts.",
        key_concepts=["thermal_runaway", "thermal_stability", "dendrite_growth", "solid_electrolyte"],
        key_relationships=[("dendrite_growth", "CAUSES", "thermal_runaway"),
                           ("thermal_stability", "INFLUENCES", "thermal_runaway")],
        solution_directions=["Use LFP or solid electrolytes", "Apply protective coatings", "Manage cell temperature and pressure"],
        relevant_materials=["lfp", "solid_electrolyte", "nmc811"],
        relevant_phenomena=["thermal_runaway", "dendrite_growth"],
        relevant_properties=["thermal_stability", "safety"],
        example_queries=["How can thermal runaway be prevented in NMC811 cells?", "What is the role of solid electrolytes in battery safety?"],
        visualization_focus=["safety_comparison", "thermal_analysis"]
    ),
    LIBProblem.MANUFACTURING_REPRODUCIBILITY: LIBProblemDefinition(
        problem_id=LIBProblem.MANUFACTURING_REPRODUCIBILITY, title="Improving Manufacturing Reproducibility",
        scientific_description="Batch‑to‑batch variations in electrode properties affect cell performance.",
        root_cause="Sensitivity to coating thickness, calendering pressure, and drying conditions.",
        key_concepts=["calendering", "slot_die_coating", "electrode_thickness", "electrode_density"],
        key_relationships=[("calendering", "INFLUENCES", "electrode_density"),
                           ("slot_die_coating", "INFLUENCES", "electrode_thickness")],
        solution_directions=["Standardize coating and calendering processes", "Use in‑line thickness monitoring", "Apply design of experiments (DoE)"],
        relevant_materials=[],
        relevant_phenomena=[],
        relevant_properties=["electrode_thickness", "electrode_density"],
        example_queries=["What parameters most affect electrode thickness uniformity?", "How can we reduce batch‑to‑batch variability?"],
        visualization_focus=["process_control_chart", "thickness_distribution"]
    ),
    LIBProblem.GENERAL: LIBProblemDefinition(
        problem_id=LIBProblem.GENERAL, title="General Li‑ion Inquiry",
        scientific_description="General inquiry about lithium‑ion batteries.",
        root_cause="N/A", key_concepts=["lithium_ion_battery"], key_relationships=[],
        solution_directions=[], relevant_materials=[], relevant_phenomena=[], relevant_properties=[],
        example_queries=["What are lithium‑ion batteries?"], visualization_focus=["general_overview"]
    ),
    LIBProblem.MULTI_PROBLEM: LIBProblemDefinition(
        problem_id=LIBProblem.MULTI_PROBLEM, title="Multi‑Problem Li‑ion Inquiry",
        scientific_description="Inquiry spanning multiple core problems.", root_cause="N/A",
        key_concepts=[], key_relationships=[], solution_directions=[],
        relevant_materials=[], relevant_phenomena=[], relevant_properties=[],
        example_queries=[], visualization_focus=["multi_problem_comparison"]
    )
}

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

# ============================================================================
# 2. LLM QUERY ANALYZERS (Abstract + Implementations)
# ============================================================================
class LLMQueryAnalyzer(ABC):
    @abstractmethod
    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult: pass
    @abstractmethod
    def is_available(self) -> bool: pass

class FallbackAnalyzer(LLMQueryAnalyzer):
    PROBLEM_KEYWORDS = {
        LIBProblem.ENERGY_DENSITY_ENHANCEMENT: {"energy", "density", "specific", "capacity", "thick", "calender"},
        LIBProblem.CYCLE_LIFE_EXTENSION: {"cycle", "life", "fade", "degradation", "sei", "stability", "long"},
        LIBProblem.FAST_CHARGING: {"fast", "charge", "rate", "c‑rate", "impedance", "conductivity"},
        LIBProblem.SAFETY_THERMAL_RUNAWAY: {"safety", "thermal", "runaway", "dendrite", "short", "fire"},
        LIBProblem.MANUFACTURING_REPRODUCIBILITY: {"reproduc", "batch", "variation", "coating", "calender", "thickness"},
    }
    def is_available(self) -> bool: return True

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        q = query.lower().strip()
        problem_scores = {p: sum(1 for kw in kws if kw in q) for p, kws in self.PROBLEM_KEYWORDS.items()}
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
                concept_name=concept, concept_type=ontology.get_concept_type(concept).value,
                composite_score=(1.0 if is_explicit else 0.6) * 0.5 + (1.0 if concept in problem_concept_set else 0.4) * 0.5,
                direct_score=1.0 if is_explicit else 0.6, problem_affinity_score=1.0 if concept in problem_concept_set else 0.4,
                causal_path_score=0.5, is_explicitly_mentioned=is_explicit, is_inferred=not is_explicit,
                inference_reason="problem_affinity" if not is_explicit else "explicit_mention"
            )

        query_type = "general"
        if any(w in q for w in ["compare", "vs", "versus", "difference"]): query_type = "comparison"
        elif any(w in q for w in ["why", "cause", "reason", "lead to"]): query_type = "causal"
        elif any(w in q for w in ["how", "improve", "enhance", "optimize", "strategy"]): query_type = "solution"

        highlight_paths = [[src, tgt] for src, rel, tgt in pdef.key_relationships if src in ontology.concepts and tgt in ontology.concepts]
        total = max(sum(problem_scores.values()), 1)
        
        return QueryAnalysisResult(
            original_query=query, normalized_query=q, primary_problem=primary, secondary_problems=secondary,
            problem_confidences={p.value: s / total for p, s in problem_scores.items()},
            explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=all_relevant,
            concept_priorities=priorities, query_type=query_type, emphasis_direction="cause" if query_type == "causal" else "neutral",
            subgraph_depth=2, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
            suggested_layout="force" if query_type != "comparison" else "bisected", highlight_paths=highlight_paths,
            visualization_focus=pdef.visualization_focus, reasoning_chain=[f"Query normalized: '{q}'", f"Primary problem: {primary.value}"],
            confidence=min(sum(problem_scores.values()) / 3.0, 1.0)
        )

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

    def is_available(self) -> bool: return bool(self.api_key) and self._get_client() is not None

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        client = self._get_client()
        if client is None: return FallbackAnalyzer().analyze_query(query, ontology)

        concept_list = list(ontology.concepts.keys())[:50]
        system_prompt = """You are an expert in lithium‑ion battery materials science. Analyze the user's query and return ONLY valid JSON with:
        1. "primary_problem": One of: energy_density_enhancement, cycle_life_extension, fast_charging, safety_thermal_runaway, manufacturing_reproducibility, general, multi_problem
        2. "explicitly_mentioned": List of canonical concept names from the query (use snake_case)
        3. "inferred_concepts": List of additional relevant concepts the query implies
        4. "query_type": One of: causal, comparison, solution, definition, general
        5. "highlight_paths": List of [source, target] concept pairs to highlight
        6. "reasoning_chain": List of strings explaining analysis steps
        7. "new_concepts": List of objects with "name" (snake_case), "type" (material/property/phenomenon/process/method/parameter), "definition", "synonyms" (list)
        8. "new_relationships": List of [source, relationship_type, target, confidence] for NEW relationships between EXISTING concepts."""
        
        try:
            response = client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Analyze: '{query}'. Available concepts: {', '.join(concept_list)}"}],
                temperature=0.1, max_tokens=1500, response_format={"type": "json_object"}
            )
            parsed = json.loads(response.choices[0].message.content)
            self._pending_new_concepts = parsed.get("new_concepts", [])
            self._pending_new_relationships = parsed.get("new_relationships", [])
            
            problem_map = {p.value: p for p in LIBProblem}
            primary = problem_map.get(parsed.get("primary_problem", "general"), LIBProblem.GENERAL)
            explicitly_mentioned = [c for c in parsed.get("explicitly_mentioned", []) if c in ontology.concepts]
            inferred = [c for c in parsed.get("inferred_concepts", []) if c in ontology.concepts and c not in explicitly_mentioned]
            
            priorities = {c: ConceptPriority(c, ontology.get_concept_type(c).value, 0.9 if c in explicitly_mentioned else 0.6, 1.0 if c in explicitly_mentioned else 0.5, 0.8, 0.5, c in explicitly_mentioned, c not in explicitly_mentioned, "llm_inferred") for c in list(dict.fromkeys(explicitly_mentioned + inferred))}
            
            return QueryAnalysisResult(
                original_query=query, normalized_query=query.lower().strip(), primary_problem=primary, secondary_problems=[],
                problem_confidences={}, explicitly_mentioned=explicitly_mentioned, inferred_concepts=inferred, all_relevant_concepts=list(dict.fromkeys(explicitly_mentioned + inferred)),
                concept_priorities=priorities, query_type=parsed.get("query_type", "general"), emphasis_direction="cause",
                subgraph_depth=2, priority_threshold=0.3, focus_nodes=explicitly_mentioned[:5], bridge_nodes=inferred[:3],
                suggested_layout="bisected" if parsed.get("query_type") == "comparison" else "force",
                highlight_paths=[[p[0], p[1]] for p in parsed.get("highlight_paths", []) if len(p) >= 2],
                visualization_focus=LIB_PROBLEM_DEFINITIONS[primary].visualization_focus, reasoning_chain=parsed.get("reasoning_chain", ["LLM analysis completed"]), confidence=0.85
            )
        except Exception as e:
            st.warning(f"OpenAI analysis failed ({e}), falling back to rule-based.")
            return FallbackAnalyzer().analyze_query(query, ontology)

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
            import torch

            st.info(f"⏳ Loading local model: `{self.model_name}`… (first run may take 1–2 min)")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            load_kwargs: Dict[str, Any] = {}
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
                maybe_empty_cache()

    def is_available(self) -> bool:
        self._load_model()
        return self._loaded

    def analyze_query(self, query: str, ontology: Any) -> QueryAnalysisResult:
        if not self.is_available():
            return FallbackAnalyzer().analyze_query(query, ontology)

        prompt = (
            f"[INST] You are an expert in lithium‑ion batteries. Analyze: '{query}'. "
            "Return ONLY valid JSON with: primary_problem, explicitly_mentioned "
            "(snake_case list), inferred_concepts (list), query_type, highlight_paths "
            "(list of [src, tgt]), reasoning_chain (list). [/INST]"
        )

        try:
            if self._is_ollama:
                ollama_model_name = self.model_name.split(":", 1)[1]
                payload = {
                    "model": ollama_model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
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
                maybe_empty_cache()

        return FallbackAnalyzer().analyze_query(query, ontology)

    def unload_model(self) -> None:
        """Explicitly free the LLM model and pipeline from memory."""
        if self._is_ollama:
            try:
                ollama_model_name = self.model_name.split(":", 1)[1]
                payload = {
                    "model": ollama_model_name,
                    "prompt": "",
                    "keep_alive": 0
                }
                requests.post(
                    f"{self._ollama_url}/api/generate",
                    json=payload,
                    timeout=5
                )
            except Exception:
                pass
            self._loaded = False
            return

        if self._pipeline is not None:
            if hasattr(self._pipeline, 'tokenizer'):
                del self._pipeline.tokenizer
            if hasattr(self._pipeline, 'model'):
                del self._pipeline.model
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            maybe_empty_cache()

class LLMQueryAnalyzerFactory:
    def __init__(self):
        self._openai_cache: Optional[OpenAIQueryAnalyzer] = None
        self._local_cache: Dict[str, LocalLLMQueryAnalyzer] = {}
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

# ============================================================================
# 3. DYNAMIC ONTOLOGY EXPANDER
# ============================================================================
class DynamicOntologyExpander:
    REL_STR_TO_ENUM = {r.value: r for r in RelationshipType}
    for _k, _v in list(REL_STR_TO_ENUM.items()): REL_STR_TO_ENUM[_k.upper()] = _v
    TYPE_STR_TO_ENUM = {t.value: t for t in ConceptType}

    def __init__(self, ontology: Any):
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
        return {"base_concepts": self._base_concept_count, "base_relationships": self._base_rel_count,
                "concepts_added": len(self.session_concepts_added), "relationships_added": len(self.session_relationships_added),
                "bridge_concepts": len(self.query_bridge_concepts), "total_mutations": len(self.mutation_log)}

    def apply_query_analysis(self, analysis: QueryAnalysisResult, analyzer: LLMQueryAnalyzer = None) -> Dict[str, Any]:
        changes = {"concepts_added": [], "relationships_added": [], "bridges_created": []}
        for concept_name, priority in analysis.concept_priorities.items():
            if concept_name in self.ontology.concepts:
                self.priority_overrides[concept_name] = priority.composite_score

        new_concepts_raw = getattr(analyzer, '_pending_new_concepts', []) if hasattr(analyzer, '_pending_new_concepts') else []
        new_rels_raw = getattr(analyzer, '_pending_new_relationships', []) if hasattr(analyzer, '_pending_new_relationships') else []

        for concept_data in new_concepts_raw:
            result = self._add_concept_from_llm(concept_data, analysis.original_query)
            if result: changes["concepts_added"].append(result)
        for rel_data in new_rels_raw:
            result = self._add_relationship_from_llm(rel_data, analysis.original_query)
            if result: changes["relationships_added"].append(result)

        for concept in analysis.inferred_concepts:
            if concept not in self.ontology.concepts:
                bridge_result = self._create_bridge_concept(concept, analysis.original_query, analysis.primary_problem)
                if bridge_result: changes["bridges_created"].append(bridge_result)
        
        self.ontology._build_synonym_index()
        return changes

    def _add_concept_from_llm(self, concept_data: Dict, source_query: str) -> Optional[Dict]:
        name = concept_data.get("name", "").strip().lower().replace(" ", "_")
        if not name or name in self.ontology.concepts or name in self.session_concepts_added: return None
        concept_type = self.TYPE_STR_TO_ENUM.get(concept_data.get("type", "general"), ConceptType.GENERAL)
        synonyms = set(s.lower().strip() for s in concept_data.get("synonyms", []) if isinstance(s, str))
        definition = concept_data.get("definition", f"LLM-inferred concept from query: {source_query}")
        
        self.ontology._add_concept(name, concept_type, synonyms=synonyms, definition=definition)
        self.ontology.synonym_to_canonical[name.lower()] = name
        for syn in synonyms: self.ontology.synonym_to_canonical[syn] = name
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
        if len(rel_data) < 3: return None
        source, rel_type_str, target = str(rel_data[0]).strip().lower().replace(" ", "_"), str(rel_data[1]).upper(), str(rel_data[2]).strip().lower().replace(" ", "_")
        confidence = float(rel_data[3]) if len(rel_data) > 3 else 0.7
        if source not in self.ontology.concepts or target not in self.ontology.concepts: return None
        
        rel_enum = self.REL_STR_TO_ENUM.get(rel_type_str, RelationshipType.INFLUENCES)
        self.ontology.relationships.append(Relationship(source, target, rel_enum, confidence))
        self.session_relationships_added.append((source, target, rel_enum, confidence))
        self.mutation_log.append({"type": "add_relationship", "source": source, "target": target, "rel_type": rel_enum.value, "source_query": source_query})
        return {"source": source, "target": target, "rel_type": rel_enum.value, "confidence": confidence}

    def _create_bridge_concept(self, missing_concept: str, source_query: str, problem: LIBProblem) -> Optional[Dict]:
        bridge_name = f"query_bridge_{missing_concept.replace(' ', '_').lower()}"
        if bridge_name in self.ontology.concepts: return None
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

    def get_priority_boosted_scores(self, base_priorities: Dict[str, ConceptPriority]) -> Dict[str, ConceptPriority]:
        boosted = {}
        for name, priority in base_priorities.items():
            boost = self.priority_overrides.get(name, 0.0)
            if boost > 0:
                bp = copy.deepcopy(priority)
                bp.composite_score = min(bp.composite_score + boost * 0.2, 1.0)
                bp.causal_path_score = boost * 0.2
                boosted[name] = bp
            else:
                boosted[name] = priority
        return boosted

    def undo_last_mutation(self) -> Optional[Dict]:
        if not self.mutation_log: return None
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
            if name in self.ontology.concepts: del self.ontology.concepts[name]
        self.ontology.relationships = self.ontology.relationships[:self._base_rel_count]
        self.session_concepts_added.clear()
        self.session_relationships_added.clear()
        self.query_bridge_concepts.clear()
        self.priority_overrides.clear()
        self.mutation_log.clear()
        self.ontology._build_synonym_index()
        return {"concepts_removed": len(self.session_concepts_added), "relationships_removed": len(self.ontology.relationships) - self._base_rel_count}

# ============================================================================
# 4. PRIORITY-GUIDED SUBGRAPH EXTRACTOR & VISUALIZER
# ============================================================================
class PriorityGuidedSubgraphExtractor:
    def __init__(self, full_graph: nx.Graph, ontology: Any, expander: DynamicOntologyExpander):
        self.full_graph = full_graph
        self.ontology = ontology
        self.expander = expander

    def extract(self, analysis: QueryAnalysisResult, query_embedding: np.ndarray = None) -> nx.Graph:
        raw_seed_nodes = set(analysis.focus_nodes + analysis.get_concepts_above_threshold())
        seed_nodes = {n for n in raw_seed_nodes if n in self.full_graph}
        if not seed_nodes:
            seed_nodes = {n for n, d in self.full_graph.nodes(data=True)
                          if d.get("priority_score", 0) >= 0.3}

        personalization = {n: 1.0 if n in seed_nodes else 0.0 for n in self.full_graph.nodes()}
        try:
            ppr_scores = nx.pagerank(self.full_graph, personalization=personalization, alpha=0.85)
        except Exception:
            ppr_scores = {n: 1.0/len(self.full_graph) for n in self.full_graph.nodes()}

        qc_pmi = {}

        for node in self.full_graph.nodes():
            ppr = ppr_scores.get(node, 0.0)
            srs = self._compute_semantic_resonance(node, query_embedding) if query_embedding is not None else 0.5
            combined = 0.6 * ppr + 0.4 * srs
            self.full_graph.nodes[node]["priority_score"] = combined
            self.full_graph.nodes[node]["ppr_score"] = ppr
            self.full_graph.nodes[node]["semantic_resonance"] = srs

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
        selected_nodes = {n for n, d in self.full_graph.nodes(data=True)
                          if d.get("priority_score", 0) >= threshold}
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
    def __init__(self, ontology: Any):
        self.ontology = ontology
        self.type_colors = {"material": "#FF6B6B", "property": "#4ECDC4", "phenomenon": "#FFE66D", "method": "#95E1D3", "parameter": "#F38181", "process": "#AA96DA", "model": "#FCBAD3", "general": "#A8D8EA"}

    def render_pyvis(self, subgraph: nx.Graph, analysis: QueryAnalysisResult, height: str = "700px",
                     physics_enabled: bool = True,
                     gravity: float = -800.0,
                     central_gravity: float = 0.1,
                     spring_length: float = 120,
                     spring_strength: float = 0.02,
                     damping: float = 0.95) -> str:
        from pyvis.network import Network
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
            color = self.type_colors.get(concept_type, "#A8D8EA")
            if is_explicit: border_width, border_color, shape = 4, "#FF0000", "dot"
            elif is_llm_added: border_width, border_color, shape = 3, "#00FF00", "diamond"
            else: border_width, border_color, shape = 1, "#666666", "dot"
            title = "<b>" + node + "</b><br>Type: " + concept_type + "<br>Priority: " + str(round(priority, 2))
            if is_llm_added: title += "<br>⚠️ LLM-inferred concept"
            defn = attrs.get("definition", "")
            if defn: title += "<br><i>" + defn[:150] + "...</i>"
            net.add_node(node, label=node.replace("_", " ").title(), size=size, color=color, border_width=border_width, border_color=border_color, shape=shape, title=title, font={"size": 10 + priority * 6})
        for u, v, attrs in subgraph.edges(data=True):
            color = attrs.get("color", "#888888")
            width = attrs.get("width", 1.0)
            highlighted = any(len(p) >= 2 and ((p[0] == u and p[1] == v) or (p[1] == u and p[0] == v)) for p in analysis.highlight_paths)
            if highlighted: color, width = "#FF0000", max(width, 4.0)
            net.add_edge(u, v, color=color, width=width, dashes=attrs.get("style") == "dashed" or attrs.get("inferred", False), title=u + " → " + v + "<br>Type: " + attrs.get('edge_type','unknown'), arrows="to")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            net.save_graph(f.name)
            return Path(f.name).read_text(encoding='utf-8')

class GraphRAGAnswerGenerator:
    def __init__(self, analyzer: LLMQueryAnalyzer):
        self.analyzer = analyzer

    def generate_ground_response(self, query: str, analysis: QueryAnalysisResult, subgraph: nx.Graph, concept_abstract_map: Dict[str, List[int]], all_texts: Union[List[str], Dict[int, str]], max_docs_per_concept: int = 2) -> str:
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
                    evidence_snippets.append("- **" + node + "**: " + clean_text + "...")
        nl = chr(10)
        prompt = "You are an expert in lithium‑ion battery materials science. Answer the user's query based *strictly* on the provided graph context and evidence snippets." + nl
        prompt += "User Query: " + repr(query) + nl
        prompt += "Identified Core Problem: " + analysis.primary_problem.value.replace("_", " ").title() + nl
        prompt += "Key Graph Concepts: " + ", ".join([n for n, _ in top_nodes]) + nl
        prompt += "Evidence Snippets from Literature:" + nl
        if evidence_snippets:
            prompt += nl.join(evidence_snippets) + nl
        else:
            prompt += "No direct text snippets found. Rely on your general knowledge of Li‑ion batteries but note the lack of specific retrieved context." + nl
        prompt += "Instructions:" + nl
        prompt += "1. Provide a direct, scientifically accurate answer (2-3 paragraphs)." + nl
        prompt += "2. Explicitly mention how the key concepts interact (e.g., causal chains like 'electrode thickness influences energy density')." + nl
        prompt += "3. If the retrieved evidence is insufficient, state what specific data is missing."
        if isinstance(self.analyzer, OpenAIQueryAnalyzer) and self.analyzer.is_available():
            return self._call_llm_for_answer(prompt, self.analyzer, query, analysis, top_nodes, evidence_snippets)
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _call_llm_for_answer(self, prompt: str, analyzer: LLMQueryAnalyzer, query: str, analysis: QueryAnalysisResult, top_nodes, evidence_snippets) -> str:
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
                fallback_text = self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)
                return "⚠️ LLM API Error: " + str(e) + chr(10) + chr(10) + fallback_text
        return self._generate_fallback_answer(query, analysis, top_nodes, evidence_snippets)

    def _generate_fallback_answer(self, query: str, analysis: Optional[QueryAnalysisResult], top_nodes, snippets: List[str]) -> str:
        nl = chr(10)
        fallback_text = "### Analysis of: '" + query + "'" + nl + nl
        if analysis is not None:
            primary = getattr(analysis, 'primary_problem', None)
            fallback_text += "**Core Problem Identified:** " + (primary.value.replace('_', ' ').title() if primary else 'Unknown') + nl + nl
        else:
            fallback_text += "**Core Problem Identified:** (analysis unavailable)" + nl + nl
        fallback_text += "**Key Concepts in Focus:**" + nl
        fallback_text += nl.join(["- **" + node + "** (" + attrs.get("concept_type", "general") + "): Priority Score " + str(round(attrs.get("priority_score", 0), 2)) for node, attrs in top_nodes])
        if snippets:
            fallback_text += nl + "**Retrieved Evidence Context:**" + nl + nl.join(snippets[:3]) + nl
        else:
            fallback_text += nl + "*Note: No direct text snippets were linked to these concepts in the current dataset.*" + nl
        fallback_text += nl + "**System Reasoning Chain:**" + nl
        if analysis is not None:
            reasoning_chain = getattr(analysis, 'reasoning_chain', [])
            fallback_text += nl.join(["- " + step for step in reasoning_chain])
        else:
            fallback_text += "- No reasoning chain available (analysis was None)." + nl
        return fallback_text

class QuerySessionManager:
    SESSION_KEY = "lib_query_session"
    @classmethod
    def init_session(cls) -> Dict[str, Any]:
        if cls.SESSION_KEY not in st.session_state:
            st.session_state[cls.SESSION_KEY] = {"query_history": [], "analysis_history": [], "mutation_history": [], "analyzer_mode": "auto", "total_concepts_added": 0, "total_relationships_added": 0}
        return st.session_state[cls.SESSION_KEY]

    @classmethod
    def record_query(cls, query: str, analysis: QueryAnalysisResult, mutations: Dict[str, Any]) -> None:
        session = cls.init_session()
        session["query_history"].append(query)
        session["analysis_history"].append({"query": query, "primary_problem": analysis.primary_problem.value, "query_type": analysis.query_type, "concepts_found": len(analysis.all_relevant_concepts), "explicit": len(analysis.explicitly_mentioned), "inferred": len(analysis.inferred_concepts), "confidence": analysis.confidence, "timestamp": datetime.now().isoformat()})
        session["mutation_history"].append({"query": query, "concepts_added": len(mutations.get("concepts_added", [])), "relationships_added": len(mutations.get("relationships_added", [])), "bridges_created": len(mutations.get("bridges_created", [])), "timestamp": datetime.now().isoformat()})
        session["total_concepts_added"] += len(mutations.get("concepts_added", []))
        session["total_relationships_added"] += len(mutations.get("relationships_added", []))

    @classmethod
    def get_session(cls) -> Dict[str, Any]: return cls.init_session()
    @classmethod
    def clear_session(cls) -> None:
        if cls.SESSION_KEY in st.session_state: del st.session_state[cls.SESSION_KEY]

# ============================================================================
# 7. STREAMLIT UI INTEGRATORS
# ============================================================================
def render_llm_query_panel(ontology: Any, expander: DynamicOntologyExpander, full_graph: nx.Graph) -> Optional[QueryAnalysisResult]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 LLM-Guided Query")
    st.sidebar.caption("Ask a question to dynamically expand the ontology and focus the graph")

    session = QuerySessionManager.get_session()
    mode = st.sidebar.selectbox("Analysis Engine", ["auto", "fallback", "openai", "local"], index=["auto", "fallback", "openai", "local"].index(session.get("analyzer_mode", "auto")), key="llm_mode_select")
    session["analyzer_mode"] = mode

    api_key = None
    if mode in ("auto", "openai"):
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.environ.get("OPENAI_API_KEY", ""), key="openai_key_input")

    # --- DYNAMIC LOCAL MODEL SELECTION (UPDATED) ---
    local_model = None
    if mode in ("auto", "local"):
        st.sidebar.markdown("#### 🖥️ Local LLM Model")

        # Detect Ollama once per call
        _has_ollama = False
        try:
            _has_ollama = requests.get(
                "http://localhost:11434/api/tags", timeout=2
            ).status_code == 200
        except Exception:
            pass

        if _has_ollama:
            st.sidebar.caption("✅ Ollama detected on localhost:11434")
            LOCAL_LLM_REGISTRY = {
                "Fallback (Rule-based, no LLM)": None,
                "qwen2.5:0.5b (Fastest, CPU OK)": "ollama:qwen2.5:0.5b",
                "qwen2.5:1.5b (Balanced)":        "ollama:qwen2.5:1.5b",
                "qwen2.5:7b (Recommended)":       "ollama:qwen2.5:7b",
                "qwen2.5:14b (Max Reasoning)":    "ollama:qwen2.5:14b",
                "llama3.1:8b (Meta Standard)":    "ollama:llama3.1:8b",
                "mistral:7b (High JSON Reliability)": "ollama:mistral:7b",
                "gemma2:9b (Scientific Nuance)":   "ollama:gemma2:9b",
                "falcon3:10b (Instruction Following)": "ollama:falcon3:10b",
            }
        else:
            st.sidebar.caption("⚠️ Streamlit Cloud ≈1 GB RAM. Pick a small model or use Fallback.")
            LOCAL_LLM_REGISTRY = {
                "Fallback (Rule-based, no LLM)": None,
                "DistilGPT-2 (82M, fastest)": "distilgpt2",
                "GPT-Neo-125M (125M)": "EleutherAI/gpt-neo-125M",
                "Pythia-410M (410M, balanced)": "EleutherAI/pythia-410m",
                "BLOOM-560M (560M, multilingual)": "bigscience/bloom-560m",
                "Qwen2-0.5B-Instruct (500M, best JSON)": "Qwen/Qwen2-0.5B-Instruct",
                "Qwen2.5-0.5B-Instruct (500M, newest)": "Qwen/Qwen2.5-0.5B-Instruct",
            }

        model_display_names = list(LOCAL_LLM_REGISTRY.keys())
        selected_display = st.sidebar.selectbox(
            "Select model:", options=model_display_names, index=0,
            key="local_model_select",
        )
        local_model = LOCAL_LLM_REGISTRY[selected_display]
        st.session_state['selected_local_model'] = local_model

    # --- end of dynamic local model block ---

    example_queries = [q for pdef in LIB_PROBLEM_DEFINITIONS.values() for q in pdef.example_queries[:1]]
    selected_example = st.sidebar.selectbox("Or select an example:", [""] + example_queries, key="example_query_select")
    query = st.sidebar.text_area("Your Li‑ion question:", value=selected_example, height=100, key="llm_query_input", placeholder="e.g., How can we improve the cycle life of NMC811 batteries?")
    
    submitted = st.sidebar.button("🚀 Analyze & Expand Ontology", type="primary", key="llm_submit")
    if not submitted or not query.strip(): return None

    factory = LLMQueryAnalyzerFactory()
    analyzer = factory.get_analyzer(mode=mode, api_key=api_key, local_model=local_model)

    if isinstance(analyzer, OpenAIQueryAnalyzer): st.sidebar.info("🤖 Using **OpenAI GPT-4o-mini**")
    elif isinstance(analyzer, LocalLLMQueryAnalyzer): st.sidebar.info("🖥️ Using **Local LLM**")
    else: st.sidebar.info("📋 Using **Rule-based fallback**")

    with st.spinner("🔍 Analyzing query via Ollama..."):
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

    # Additionally, run QDWA analysis on the query
    run_qdwa_analysis(query, ontology_concepts=analysis.explicitly_mentioned + analysis.inferred_concepts)

    QuerySessionManager.record_query(query, analysis, mutations)

    st.sidebar.success(f"✅ Analysis complete (confidence: {analysis.confidence:.0%})")
    st.sidebar.caption(f"Primary problem: **{analysis.primary_problem.value}**")
    st.sidebar.caption(f"Explicit concepts: {len(analysis.explicitly_mentioned)} | Inferred: {len(analysis.inferred_concepts)}")
    if mutations["concepts_added"]:
        st.sidebar.warning(f"🆕 {len(mutations['concepts_added'])} new concept(s) added")
        for c in mutations["concepts_added"]: st.sidebar.markdown(f"  - `{c['name']}` ({c['type']})")
    if mutations["bridges_created"]:
        st.sidebar.info(f"🌉 {len(mutations['bridges_created'])} bridge concept(s) created")
        for b in mutations["bridges_created"]: st.sidebar.markdown(f"  - `{b['bridge']}` ← `{b['for']}`")
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
                if mut["type"] == "add_concept": st.sidebar.markdown(f"{i}. ➕ `{mut['concept']}`")
                elif mut["type"] == "add_relationship": st.sidebar.markdown(f"{i}. 🔗 `{mut['source']}` → `{mut['target']}`")
                elif mut["type"] == "create_bridge": st.sidebar.markdown(f"{i}. 🌉 `{mut['bridge_name']}`")
        col_undo, col_reset = st.sidebar.columns(2)
        if col_undo.button("↩️ Undo Last", key="undo_mutation"):
            undone = expander.undo_last_mutation()
            if undone: st.sidebar.toast(f"Undone: {undone['type']}"); st.rerun()
        if col_reset.button("🔄 Reset All", key="reset_mutations"):
            result = expander.reset_to_base()
            st.sidebar.toast(f"Reset: {result['concepts_removed']} concepts, {result['relationships_removed']} relations removed")
            st.rerun()

def render_query_history() -> None:
    session = QuerySessionManager.get_session()
    if not session["query_history"]: return
    st.sidebar.markdown("---")
    with st.sidebar.expander("📜 Query History", expanded=False):
        for i, entry in enumerate(reversed(session["analysis_history"][-10:]), 1):
            st.sidebar.markdown(f"**{i}.** {entry['query'][:60]}...")
            st.sidebar.caption(f"  Problem: {entry['primary_problem']} | Type: {entry['query_type']} | Concepts: {entry['concepts_found']}")

def render_analysis_details(analysis: QueryAnalysisResult) -> None:
    st.markdown("## 📊 Query Analysis Results")
    with st.expander("🧠 Reasoning Chain", expanded=True):
        for step in analysis.reasoning_chain: st.markdown(f"→ {step}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Primary Problem", analysis.primary_problem.value.replace("_", " "))
    col2.metric("Query Type", analysis.query_type)
    col3.metric("Confidence", f"{analysis.confidence:.0%}")
    
    st.markdown("### Concept Priority Rankings")
    top = analysis.get_top_concepts(15)
    if top:
        df = pd.DataFrame([cp.to_dict() for cp in top])
        def highlight_row(row):
            if row.get("explicit", False): return ["background-color: #d4edda"] * len(row)
            elif row.get("inferred", False): return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), use_container_width=True)

def render_llm_qa_tab(analysis_data: Dict, ontology: Any):
    st.subheader("🤖 LLM-Guided Graph Q&A")
    st.markdown("Ask a specific scientific question about lithium‑ion batteries. The system will dynamically expand the ontology, extract a relevant subgraph, and generate a grounded answer using retrieved literature snippets.")
    
    if "qa_factory" not in st.session_state: st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if "qa_expander" not in st.session_state: st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if "qa_generator" not in st.session_state: st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    factory = st.session_state.qa_factory
    expander = st.session_state.qa_expander
    generator = st.session_state.qa_generator

    col1, col2 = st.columns([3, 1])
    with col1: query = st.text_input("Enter your research question:", placeholder="e.g., How does electrode thickness affect energy density?")
    with col2: mode = st.selectbox("Engine", ["auto", "openai", "local", "fallback"], index=0)
        
    if st.button("🔍 Analyze & Answer", type="primary"):
        if not query.strip(): st.warning("Please enter a query."); return
            
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

            # Run QDWA on the query
            qdwa_analysis = run_qdwa_analysis(query, ontology_concepts=analysis.explicitly_mentioned + analysis.inferred_concepts)
            
            # Show compact QDWA preview
            st.markdown("---")
            st.markdown("#### ⚖️ QDWA Category Weights")
            cols = st.columns(6)
            for col, (cat, w) in zip(cols, qdwa_analysis.get_ranked_categories()):
                col.metric(CATEGORY_DISPLAY[cat][:10], f"{w:.3f}")
            
            if st.button("View Full QDWA Dashboard"):
                st.session_state["show_qdwa_dashboard"] = True
                st.rerun()

            if st.session_state.get('query_focused_build'):
                st.success(f"✅ Query analysis complete. Whitelist contains {len(whitelist)} concepts.")
                if st.button("🔧 Rebuild Graph for This Query", type="primary", key="rebuild_for_query_btn"):
                    st.session_state['force_rebuild'] = True
                    st.rerun()

        with st.spinner("🕸️ Extracting priority-guided subgraph..."):
            full_graph = analysis_data["nx_graph"]
            extractor = PriorityGuidedSubgraphExtractor(full_graph, ontology, expander)
            embed_model = analysis_data.get("embed_model")
            if embed_model is not None:
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
                query=query, analysis=analysis, subgraph=subgraph,
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
                key="subgraph_phys_preset",
                help="'Stable' uses high damping to stop oscillation. 'Off' freezes the layout."
            )
            presets = {
                "Stable (No Jiggle)": {"gravity": -800, "central_gravity": 0.1, "spring_length": 120, "spring_strength": 0.02, "damping": 0.95},
                "Fluid": {"gravity": -500, "central_gravity": 0.2, "spring_length": 150, "spring_strength": 0.04, "damping": 0.8},
                "Tight": {"gravity": -2000, "central_gravity": 0.3, "spring_length": 80, "spring_strength": 0.08, "damping": 0.6},
                "Off": {"gravity": 0, "central_gravity": 0, "spring_length": 100, "spring_strength": 0, "damping": 0.99},
            }
            p = presets[phys_preset]
            col1, col2 = st.columns(2)
            with col1:
                grav = st.slider("Gravity (Repulsion)", -5000, 0, p["gravity"], step=100, key="sub_grav")
                spring_len = st.slider("Spring Length", 50, 300, p["spring_length"], step=10, key="sub_slen")
                damp = st.slider("Damping (Anti-jiggle)", 0.1, 0.99, p["damping"], step=0.01, key="sub_damp")
            with col2:
                cent_grav = st.slider("Central Gravity", 0.0, 1.0, p["central_gravity"], step=0.05, key="sub_cgrav")
                spring_str = st.slider("Spring Strength", 0.0, 0.5, p["spring_strength"], step=0.01, key="sub_sstr")
                phys_on = st.checkbox("Enable Physics", value=(phys_preset != "Off"), key="sub_phys_on")
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
            for step in analysis.reasoning_chain: st.markdown("- " + step)
            if mutations.get("concepts_added") or mutations.get("bridges_created"):
                st.markdown("**Dynamic Ontology Updates:**")
                for c in mutations.get("concepts_added", []): st.markdown("➕ Added Concept: `" + c['name'] + "` (" + c['type'] + ")")
                for b in mutations.get("bridges_created", []): st.markdown("🌉 Created Bridge: `" + b['bridge'] + "` for `" + b['for'] + "`")


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    st.title(
        "🔋 Lithium‑Ion Battery Concept Graph v7.0 (QDWA)"
    )
    st.caption(
        "Multi-level reasoning concept graph for Li‑ion battery materials | "
        "Focus: Energy Density, Degradation, Manufacturing, and Performance | "
        "Memory-Safe | Batch Processing (≤1 GB) | Interactive Visualization | "
        "Ontology-aware resolution | LLM-Guided Q&A | QDWA Weighted Allocation"
    )

    if 'ontology' not in st.session_state:
        st.session_state.ontology = DomainOntology()
    ontology = st.session_state.ontology

    # Initialize QDWA engine
    initialize_qdwa_in_session()

    if 'qa_factory' not in st.session_state:
        st.session_state.qa_factory = LLMQueryAnalyzerFactory()
    if 'qa_expander' not in st.session_state:
        st.session_state.qa_expander = DynamicOntologyExpander(ontology)
    if 'qa_generator' not in st.session_state:
        st.session_state.qa_generator = GraphRAGAnswerGenerator(st.session_state.qa_factory.get_analyzer("auto"))

    render_sidebar()

    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "input_hash" not in st.session_state:
        st.session_state.input_hash = None
    if "apply_edits" not in st.session_state:
        st.session_state.apply_edits = False
    if "edit_history" not in st.session_state:
        st.session_state.edit_history = GraphEditHistory()
    if "burst_df" not in st.session_state:
        st.session_state.burst_df = None
    if "drift_df" not in st.session_state:
        st.session_state.drift_df = None
    if "genealogy_df" not in st.session_state:
        st.session_state.genealogy_df = None
    if "bridge_df" not in st.session_state:
        st.session_state.bridge_df = None
    if "motifs" not in st.session_state:
        st.session_state.motifs = {}

    st.header("📁 Data Loading")
    st.info(f"Place JSON/BibTeX/CSV files in: `{JSON_METADATA_DIR}`")
    with st.spinner("Scanning json_metadatabase..."):
        file_records = load_all_json_files(JSON_METADATA_DIR)
        df = build_master_dataframe(file_records)

    if not file_records:
        st.warning("No .json/.bib/.csv files found in the directory.")
        st.info(
            "Please place your metadata files in the `json_metadatabase/` folder."
        )
        return
    successful_files = [f for f in file_records if f[1]]
    if not successful_files:
        st.error(
            "Files found but none could be parsed. Check error messages above."
        )
        return
    st.success(
        f"Loaded {len(successful_files)} file(s) | {len(df)} record(s)"
    )

    # --- APPLY RECORD SAMPLING ---
    total_records = len(df)
    st.sidebar.caption(f"Total in database: **{total_records:,}**")

    # Get sampling settings from sidebar (defined in render_sidebar)
    sampling_choice = st.session_state.get("sampling_select", "Quarter (25%)")
    sampling_seed = st.session_state.get("sampling_seed", 42)
    SAMPLING_OPTIONS = {
        "All records (100%)":    1.0,
        "Half (50%)":           0.50,
        "Quarter (25%)":        0.25,
        "One-tenth (10%)":      0.10,
        "One-twentieth (5%)":   0.05,
        "100 records":          100,
        "250 records":          250,
        "500 records":          500,
    }
    sel = SAMPLING_OPTIONS[sampling_choice]
    if isinstance(sel, float):
        n = max(1, int(total_records * sel))
    else:
        n = min(int(sel), total_records)

    if n < total_records:
        df = df.sample(n=n, random_state=sampling_seed).reset_index(drop=True)
        st.sidebar.success(f"Using **{n:,}** / **{total_records:,}** records")
    else:
        st.sidebar.info(f"Using all **{total_records:,}** records")
    # --- END SAMPLING ---

    file_names = [f[0] for f in successful_files]
    selected_files = st.multiselect(
        "Filter by source file", file_names, default=file_names,
    )
    if selected_files:
        df_filtered = df[df["_source_file"].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    st.write(f"Working with **{len(df_filtered)}** records")
    with st.expander("Preview Data Structure"):
        st.dataframe(df_filtered.head(5), use_container_width=True)
        st.markdown("**Available columns:**")
        st.write(list(df_filtered.columns))

    text_cols = [
        c for c in df_filtered.columns
        if any(
            k in c.lower()
            for k in ['abstract', 'title', 'summary', 'text', 'content', 'description']
        )
    ]
    if not text_cols:
        text_cols = [
            c for c in df_filtered.columns if df_filtered[c].dtype == 'object'
        ]
    selected_text_cols = st.multiselect(
        "Select text columns for concept extraction:",
        options=text_cols,
        default=text_cols[:2] if len(text_cols) >= 2 else text_cols,
    )
    if not selected_text_cols:
        st.error("Please select at least one text column.")
        return

    build_clicked = st.button(
        "🚀 Build Concept Graph with Reasoning",
        type="primary", use_container_width=True,
    )
    batch_trigger = st.session_state.pop("batch_trigger", None)
    batch_mode_on = st.session_state.get("batch_mode", False)
    force_rebuild = st.session_state.pop("force_rebuild", False)

    should_build = build_clicked or force_rebuild

    if batch_mode_on and (should_build or batch_trigger):
        if force_rebuild and st.session_state.get('query_focused_build'):
            _wl = st.session_state.get('last_query_whitelist')
            if _wl:
                st.info(
                    f"🎯 Query-focused batch mode: building graph for "
                    f"{len(_wl)} whitelisted concepts only."
                )
            else:
                st.warning(
                    "Query-focused build enabled but no whitelist found. "
                    "Running standard batch analysis."
                )
        run_batch_analysis(
            df_filtered=df_filtered,
            selected_text_cols=selected_text_cols,
            ontology=ontology,
            run_mode=(batch_trigger or "all"),
        )
    elif should_build:
        progress_bar = st.progress(0.0)
        status = st.status(
            "Initializing advanced NLP analysis...", expanded=True,
        )
        overall_start = time.perf_counter()
        try:
            with status:
                st.write("Preparing text corpus...")
                all_texts: List[str] = []
                for idx, row in df_filtered.iterrows():
                    text = " ".join([
                        str(row[col]) for col in selected_text_cols
                        if col in row and pd.notna(row[col])
                    ])
                    all_texts.append(text)
                num_abstracts = len(all_texts)
                st.write(f"Prepared {num_abstracts} documents")
                progress_bar.progress(0.05)

                st.write("Loading embedding model...")
                embed_model = load_embedding_model()
                st.success("Embedding model loaded")
                progress_bar.progress(0.10)

                config = get_adaptive_config(num_abstracts)
                config["MIN_CONCEPT_FREQ"] = st.session_state.get('min_freq', 5)
                config["MIN_CONCEPT_LENGTH_WORDS"] = st.session_state.get('min_words', 2)
                config["SIMILARITY_THRESHOLD"] = st.session_state.get('sim_threshold', 0.85)
                config["COOCCURRENCE_WEIGHT"] = st.session_state.get('cooc_weight', 0.7)
                config["SEMANTIC_WEIGHT"] = st.session_state.get('sem_weight', 0.2)
                config["INFERENCE_WEIGHT"] = st.session_state.get('inf_weight', 0.1)

                whitelist = build_query_whitelist(st.session_state)
                if whitelist is not None:
                    if len(whitelist) <= 15:
                        config["MIN_CONCEPT_FREQ"] = 1
                        st.info("Frequency threshold lowered to 1 for focused query.")
                    else:
                        config["MIN_CONCEPT_FREQ"] = 2
                        st.info(f"Query-focused build: {len(whitelist)} concepts whitelisted. MIN_CONCEPT_FREQ set to {config['MIN_CONCEPT_FREQ']}.")

                st.write(f"Adaptive config: {config}")
                progress_bar.progress(0.15)

                use_ontology = st.session_state.get('use_ontology', True)
                use_embedding = st.session_state.get('use_embedding_resolution', True)
                use_inference = st.session_state.get('use_inference', True)

                if use_ontology:
                    st.write("Initializing ontology-based concept resolver...")
                    resolver = AdvancedConceptResolver(ontology, embed_model)
                    extractor = EnhancedConceptExtractor(ontology, resolver)
                    st.session_state.resolver = resolver
                    st.session_state.extractor = extractor
                    st.success("Ontology and resolver initialized")
                else:
                    st.write("Using legacy extraction (no ontology)...")
                    resolver = None
                    extractor = None
                progress_bar.progress(0.20)

                st.write("Extracting concepts from abstracts (Parallel)...")
                all_concepts: List[Optional[List[str]]] = [None] * len(df_filtered)
                all_metrics: List[Optional[Dict]] = [None] * len(df_filtered)

                def _process_single_row(idx, row, allowed_concepts=None):
                    text = " ".join([
                        str(row[col]) for col in selected_text_cols
                        if col in row and pd.notna(row[col])
                    ])
                    concepts = extractor.extract_from_text(text, idx, allowed_concepts=allowed_concepts)
                    metrics: Dict[str, Any] = {}
                    # Li-ion metric extraction
                    capacity_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:mah/g)', text, re.I
                    )
                    if capacity_matches:
                        metrics['specific_capacity_mAh_g'] = [float(m) for m in capacity_matches]
                    energy_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:wh/kg)', text, re.I
                    )
                    if energy_matches:
                        metrics['energy_density_Wh_kg'] = [float(m) for m in energy_matches]
                    cycle_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:cycles)', text, re.I
                    )
                    if cycle_matches:
                        metrics['cycle_life_cycles'] = [float(m) for m in cycle_matches]
                    impedance_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:ohm|Ω)', text, re.I
                    )
                    if impedance_matches:
                        metrics['impedance_Ohm'] = [float(m) for m in impedance_matches]
                    thickness_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:μm|um)', text, re.I
                    )
                    if thickness_matches:
                        metrics['electrode_thickness_um'] = [float(m) for m in thickness_matches]
                    temp_matches = re.findall(
                        r'(\d+(?:\.\d+)?)\s*(?:°c|celsius|k)', text, re.I
                    )
                    if temp_matches:
                        metrics['temperature_C'] = [float(m) for m in temp_matches]
                    return idx, concepts, metrics

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(_process_single_row, idx, row, whitelist): idx
                        for idx, row in df_filtered.iterrows()
                    }
                    completed = 0
                    total = len(futures)
                    for future in as_completed(futures):
                        idx, concepts, metrics = future.result()
                        all_concepts[idx] = concepts
                        all_metrics[idx] = metrics
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            progress_bar.progress(
                                0.20 + (completed / total) * 0.15
                            )
                            status.write(
                                f"Extracted {completed}/{total} documents..."
                            )

                all_concepts = [
                    c if c is not None else [] for c in all_concepts
                ]
                all_metrics = [
                    m if m is not None else {} for m in all_metrics
                ]

                if use_ontology and extractor is not None:
                    concept_freq = extractor.get_concept_frequencies()
                    valid_concepts = [
                        c for c, f in concept_freq.items()
                        if f >= config.get("MIN_CONCEPT_FREQ", 2)
                    ]
                    concept_abstract_map: Dict[str, List[int]] = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts):
                            concept_abstract_map[c].append(doc_idx)
                else:
                    concept_freq: Dict[str, int] = defaultdict(int)
                    for concepts in all_concepts:
                        for c in concepts:
                            concept_freq[c] += 1
                    valid_concepts = [
                        c for c, f in concept_freq.items()
                        if f >= config.get("MIN_CONCEPT_FREQ", 2)
                    ]
                    concept_abstract_map = defaultdict(list)
                    for doc_idx, concepts in enumerate(all_concepts):
                        for c in set(concepts):
                            concept_abstract_map[c].append(doc_idx)

                st.write(f"✅ Extraction complete. Found {len(valid_concepts)} valid concepts.")
                progress_bar.progress(0.35)

                valid_concepts = sorted(
                    valid_concepts,
                    key=lambda c: concept_abstract_map.get(c, []).__len__(),
                    reverse=True,
                )
                top_n = config.get("TOP_N_CONCEPTS", 1000)
                if len(valid_concepts) > top_n:
                    valid_concepts = valid_concepts[:top_n]
                concept_to_id = {
                    c: i for i, c in enumerate(valid_concepts)
                }
                id_to_concept = {
                    i: c for i, c in enumerate(valid_concepts)
                }
                st.write(f"**{len(valid_concepts)}** valid concepts retained")
                progress_bar.progress(0.45)

                if len(valid_concepts) < 5:
                    st.error(
                        "Too few concepts extracted. "
                        "Try lowering frequency thresholds."
                    )
                    return

                st.write("Building concept graph...")
                if use_ontology and use_inference:
                    graph_builder = ReasoningEnhancedGraphBuilder(
                        ontology, extractor
                    )
                    nx_graph = graph_builder.build_graph(
                        all_concepts, valid_concepts,
                        concept_to_id, embed_model, config,
                    )
                else:
                    nx_graph = build_hybrid_graph(
                        all_concepts, valid_concepts,
                        concept_to_id, embed_model, config, ontology,
                    )
                pos_pairs, neg_pairs = sample_edges_for_training(
                    nx_graph, valid_concepts, concept_to_id, config,
                )
                st.write(
                    f"Graph: {len(valid_concepts)} nodes, "
                    f"{nx_graph.number_of_edges()} edges"
                )
                progress_bar.progress(0.55)

                st.write("Generating node embeddings...")
                try:
                    with torch.no_grad():
                        embeddings = embed_model.encode(
                            valid_concepts, show_progress_bar=False,
                            batch_size=64, convert_to_numpy=True,
                        )
                    node_features = torch.tensor(
                        embeddings, dtype=torch.float32,
                    )
                except Exception:
                    node_features = torch.randn(len(valid_concepts), 384)
                st.write(f"Node features: {node_features.shape}")
                progress_bar.progress(0.65)

                st.write("Training GraphSAGE...")

                def training_progress(epoch, loss):
                    progress = 0.65 + (epoch / 50) * 0.15
                    progress_bar.progress(min(1.0, progress))
                    if epoch % 10 == 0:
                        status.write(
                            f"Epoch {epoch}/50 | Loss: {loss:.4f}"
                        )

                gnn_model, final_emb, adj_indices, adj_values = train_gnn(
                    node_features, nx_graph, concept_to_id,
                    pos_pairs, neg_pairs, training_progress,
                )
                st.success("GNN training complete")
                progress_bar.progress(0.80)

                st.write("Scoring research directions...")
                concept_properties: Dict[str, float] = {}
                for concept in valid_concepts:
                    doc_indices = concept_abstract_map.get(concept, [])
                    values: List[float] = []
                    for idx in doc_indices:
                        if idx < len(all_metrics):
                            metric_dict = all_metrics[idx]
                            if metric_dict is not None:
                                for metric_values in metric_dict.values():
                                    values.extend(metric_values)
                    concept_properties[concept] = (
                        float(np.median(values)) if values else 0.0
                    )
                X_feat: List[List[float]] = []
                y_target: List[float] = []
                for u, v in nx_graph.edges():
                    pu = concept_properties.get(u, 0)
                    pv = concept_properties.get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(
                        max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
                    )
                ridge = None
                if len(X_feat) > 5:
                    ridge = Ridge(alpha=1.0).fit(
                        np.array(X_feat), np.array(y_target)
                    )
                top_scores = compute_research_direction_scores(
                    gnn_model, node_features, final_emb, nx_graph,
                    valid_concepts, concept_properties, ridge, embed_model,
                )
                st.write(f"Scored {len(top_scores)} novel pairs")
                progress_bar.progress(0.90)

                st.write("Computing distillation metrics...")
                distill_df = compute_concept_distillation(
                    valid_concepts, concept_abstract_map, all_texts,
                )

                st.write("Running advanced analytics...")
                burst_df = detect_keyword_bursts(
                    df_filtered, valid_concepts,
                    concept_abstract_map, selected_text_cols,
                )
                drift_df = detect_semantic_drift(
                    df_filtered, valid_concepts,
                    concept_abstract_map, selected_text_cols,
                )
                genealogy_df = build_concept_genealogy(
                    nx_graph, valid_concepts, concept_abstract_map,
                )
                bridge_df = detect_cross_domain_bridges(
                    nx_graph, valid_concepts, concept_abstract_map,
                )
                motifs = analyze_network_motifs(nx_graph)

                st.session_state.burst_df = burst_df
                st.session_state.drift_df = drift_df
                st.session_state.genealogy_df = genealogy_df
                st.session_state.bridge_df = bridge_df
                st.session_state.motifs = motifs

                total_time = time.perf_counter() - overall_start
                st.success(f"Analysis complete in {total_time:.1f}s!")
                progress_bar.progress(1.00)
                status.update(
                    label=f"Analysis complete! ({total_time:.1f}s)",
                    state="complete", expanded=False,
                )

                analysis_data = {
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
                    "df_filtered": df_filtered,
                    "selected_text_cols": selected_text_cols,
                }
                if use_ontology:
                    analysis_data.update({
                        "ontology": ontology,
                        "resolver": resolver,
                        "extractor": extractor,
                        "graph_builder": graph_builder if use_inference else None,
                        "reasoning_paths": graph_builder.reasoning_paths if use_inference else [],
                    })
                st.session_state.analysis_data = analysis_data

                st.session_state.edit_history = GraphEditHistory()
                st.session_state.edit_history.save_snapshot(
                    nx_graph, valid_concepts, concept_to_id,
                    id_to_concept, concept_abstract_map,
                )
        except Exception as e:
            st.error(f"Pipeline Error: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            return
        finally:
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_cache()

    if (
        st.session_state.get('apply_edits')
        and st.session_state.analysis_data is not None
    ):
        data = st.session_state.analysis_data
        st.session_state.edit_history.save_snapshot(
            data["nx_graph"], data["valid_concepts"],
            data["concept_to_id"], data["id_to_concept"],
            data["concept_abstract_map"],
        )
        (
            nx_graph, valid_concepts, concept_to_id,
            id_to_concept, concept_abstract_map, edited,
        ) = apply_graph_edits(
            data["nx_graph"], data["valid_concepts"],
            data["concept_to_id"], data["id_to_concept"],
            data["concept_abstract_map"],
            nodes_to_remove=st.session_state.get('nodes_to_remove', []),
            nodes_to_merge=st.session_state.get('nodes_to_merge', []),
            merge_name=st.session_state.get('merge_name', None),
            new_edge=st.session_state.get('new_edge', None),
            new_edge_weight=st.session_state.get('new_edge_weight', 1.0),
            min_degree=st.session_state.get('filter_min_degree', 0),
            min_freq=st.session_state.get('filter_min_freq', 0),
        )
        if edited:
            st.session_state.analysis_data["nx_graph"] = nx_graph
            st.session_state.analysis_data["valid_concepts"] = valid_concepts
            st.session_state.analysis_data["concept_to_id"] = concept_to_id
            st.session_state.analysis_data["id_to_concept"] = id_to_concept
            st.session_state.analysis_data["concept_abstract_map"] = concept_abstract_map
            st.success("Graph edits applied successfully!")
            st.session_state['apply_edits'] = False
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        valid_concepts = data["valid_concepts"]
        concept_abstract_map = data["concept_abstract_map"]
        nx_graph = data["nx_graph"]
        top_scores = data["top_scores"]
        distill_df = data["distill_df"]
        df_filtered = data.get("df_filtered", pd.DataFrame())
        selected_text_cols = data.get("selected_text_cols", [])
        cmap = st.session_state.get('cmap_name', 'viridis')
        top_n_graph = st.session_state.get('top_n_graph', 200)

        has_reasoning = "ontology" in data
        tab_names = [
            "📊 Visualization",
            "⚖️ QDWA Weights",
            "🧪 Distillation",
            "🎯 Research Directions",
            "✅ Validation",
            "📥 Export",
            "📈 Extra Viz",
            "🔬 Advanced Analytics",
        ]
        if has_reasoning:
            tab_names.append("🧠 Reasoning Dashboard")
        tab_names.append("🧠 Microtransformer #2")
        tab_names.append("🤖 LLM-Guided Q&A")
        tabs = st.tabs(tab_names)
        tab_idx = 0

        with tabs[tab_idx]:
            st.subheader("Interactive Concept Graph")
            if nx_graph.number_of_nodes() == 0:
                st.warning("No nodes to display.")
            elif nx_graph.number_of_edges() == 0:
                st.warning("No edges - building semantic fallback")
                nx_graph = nx.complete_graph(len(valid_concepts))
                nx_graph = nx.relabel_nodes(
                    nx_graph, {i: valid_concepts[i] for i in range(len(valid_concepts))}
                )
            viz_choice = st.session_state.get('viz_backend', 'PyVis (Interactive)')
            physics = st.session_state.get('physics_enabled', True)
            physics_preset = st.session_state.get(
                'effective_physics', PHYSICS_PRESETS["Stable (Default)"]
            )
            theme = THEME_PRESETS.get(
                st.session_state.get('theme', 'Bright (Default)'),
                THEME_PRESETS["Bright (Default)"],
            )
            top_n = st.session_state.get('top_n_graph', 0)
            show_weights = st.session_state.get('show_edge_weights', False)
            edge_label_mode = st.session_state.get('edge_label_mode', 'hover')

            if viz_choice == "PyVis (Interactive)":
                render_pyvis_graph(
                    nx_graph, concept_abstract_map,
                    physics_enabled=physics,
                    cmap_name=cmap,
                    top_n_nodes=top_n,
                    theme=theme,
                    physics_preset=physics_preset,
                    show_edge_weights=show_weights,
                    edge_label_mode=edge_label_mode,
                    node_label_size=st.session_state.get('node_label_size') or 12,
                    node_label_position=st.session_state.get('node_label_position') or 'center',
                    node_font_face=st.session_state.get('node_font_face') or 'Inter, Segoe UI, Roboto, sans-serif',
                    edge_label_size=st.session_state.get('edge_label_size') or 10,
                    edge_label_color=st.session_state.get('edge_label_color') or None,
                    edge_label_position=st.session_state.get('edge_label_position') or 'middle',
                    use_abbreviated_labels=st.session_state.get('use_abbreviated_labels', False),
                    max_label_length=st.session_state.get('max_label_length', 15),
                    enable_node_highlight=st.session_state.get('enable_node_highlight', False),
                    show_definitions=st.session_state.get('show_definitions', True),
                    edge_lightness=st.session_state.get('edge_lightness', 0.6),
                    edge_color_mode=st.session_state.get('edge_color_mode', 'theme'),
                    custom_edge_color=st.session_state.get('custom_edge_color', '#AAAAAA'),
                    tooltip_font_size=st.session_state.get('tooltip_font_size', 13),
                    node_legend_font_size=st.session_state.get('node_legend_font_size', 13),
                    label_mode=st.session_state.get('label_mode', NodeLabelMode.FULL_NAME),
                    external_label_text=st.session_state.get('external_label_text', ''),
                    external_font_size=st.session_state.get('external_font_size', 14),
                    external_font_color=st.session_state.get('external_font_color', '#333333'),
                    external_label_align=st.session_state.get('external_label_align', 'left'),
                )
            elif viz_choice == "Plotly 2D":
                render_graph_plotly_2d(
                    nx_graph, concept_abstract_map,
                    cmap_name=cmap,
                    top_n_nodes=top_n,
                    theme=theme,
                    show_edge_weights=show_weights,
                    node_label_size=st.session_state.get('node_label_size') or 10,
                )
            elif viz_choice == "Plotly 3D":
                render_graph_plotly_3d(
                    nx_graph, concept_abstract_map,
                    cmap_name=cmap, top_n_nodes=top_n,
                    theme=theme, show_edge_weights=show_weights,
                )
            else:
                render_graph_fallback(
                    nx_graph, concept_abstract_map,
                    theme=theme, show_edge_weights=show_weights,
                )
            with st.expander("Graph Metrics"):
                metrics = compute_graph_metrics(nx_graph)
                display_metric_dashboard(metrics, theme=theme)
            with st.expander("Domain Hierarchy (Sunburst)"):
                cat_filter = st.session_state.get('sunburst_categories', [])
                if cat_filter:
                    filtered_concepts = [
                        c for c in valid_concepts
                        if abstract_concepts_to_categories([c]).get(c, 'general') in cat_filter
                    ]
                    filtered_map = {
                        c: concept_abstract_map[c]
                        for c in filtered_concepts if c in concept_abstract_map
                    }
                else:
                    filtered_concepts = valid_concepts
                    filtered_map = concept_abstract_map
                
                labels, parents, values = build_category_hierarchy(
                    filtered_concepts, filtered_map,
                    top_n_per_category=st.session_state.get('top_n_sunburst', 0),
                )
                
                render_sunburst_chart(
                    labels, parents, values,
                    cmap_name=st.session_state.get('sunburst_cmap', cmap),
                    theme=theme,
                    branchvalues=st.session_state.get('sunburst_branchvalues', 'total'),
                    label_size=st.session_state.get('sunburst_label_size') or 20,
                    width=st.session_state.get('sunburst_width') or 900,
                    height=st.session_state.get('sunburst_height') or 700,
                    show_labels=st.session_state.get('sunburst_show_labels', True),
                    show_values=st.session_state.get('sunburst_show_values', False),
                    hover_info=st.session_state.get('sunburst_hover_info', 'all'),
                    font_family=st.session_state.get('sunburst_font_family', 'Inter, Segoe UI, Roboto, sans-serif'),
                    legend_font_size=st.session_state.get('sunburst_legend_font_size', 12),
                )
            with st.expander("Concept Radar"):
                radar_k = st.session_state.get('top_n_radar', 15)
                if radar_k == 0:
                    radar_k = min(15, len(distill_df))
                render_radar_chart(
                    distill_df, top_k=radar_k, cmap_name=cmap, theme=theme,
                )

        # QDWA Tab
        tab_idx += 1
        with tabs[tab_idx]:
            render_qdwa_tab()

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Concept Distillation Efficiency")
            top_n = st.slider(
                "Show Top N", 10, min(200, len(distill_df)), 50,
                key="distill_top_n",
            )
            display_df = distill_df.head(top_n)
            st.dataframe(display_df, use_container_width=True)
            st.markdown("**Efficiency vs Frequency:**")
            chart_df = display_df.set_index('concept')[['distillation_efficiency']]
            st.bar_chart(chart_df)
            st.markdown("**Multi-Metric Comparison:**")
            metric_cols = [
                c for c in [
                    'frequency', 'tfidf_weight',
                    'semantic_density', 'coherence_score',
                ]
                if c in display_df.columns
            ]
            if metric_cols:
                compare_df = display_df[['concept'] + metric_cols].set_index('concept')
                st.line_chart(compare_df)

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Top Research Direction Recommendations")
            if top_scores.empty:
                st.info(
                    "No novel pairs scored. "
                    "The graph may be too dense or too sparse."
                )
            else:
                st.write(f"Top {len(top_scores)} novel concept pairs:")
                st.dataframe(
                    top_scores[[
                        'concept_u', 'concept_v', 'composite_score',
                        'gnn_affinity', 'semantic_novelty',
                        'expected_property_gain', 'feasibility_score',
                    ]].head(20),
                    use_container_width=True,
                )
                csv_scores = top_scores.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Scores (CSV)", data=csv_scores,
                    file_name="lib_research_directions.csv", mime="text/csv",
                )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Mathematical Validation")
            val_metrics = validate_graph_metrics(nx_graph, valid_concepts)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Modularity", f"{val_metrics.get('modularity', 0):.3f}"
            )
            col2.metric(
                "Silhouette",
                f"{val_metrics.get('silhouette_score', 0):.3f}",
            )
            col3.metric(
                "Communities", val_metrics.get('n_communities', 0)
            )
            col4.metric(
                "Significant Edges",
                val_metrics.get('edge_significant_count', 0),
            )
            if not top_scores.empty:
                n_boot = st.session_state.get('bootstrap_samples', 500)
                alpha = st.session_state.get('alpha_level', 0.05)
                mean_score, ci_low, ci_high = compute_bootstrap_ci(
                    top_scores['composite_score'].values,
                    n_bootstrap=n_boot, alpha=alpha,
                )
                st.success(
                    f"Composite Score: `{mean_score:.3f}` | "
                    f"{int((1 - alpha) * 100)}% CI: "
                    f"`[{ci_low:.3f}, {ci_high:.3f}]`"
                )
                X_feat: List[List[float]] = []
                y_target: List[float] = []
                for u, v in nx_graph.edges():
                    pu = data["concept_properties"].get(u, 0)
                    pv = data["concept_properties"].get(v, 0)
                    w = nx_graph[u][v].get('weight', 1)
                    X_feat.append([pu, pv, w])
                    y_target.append(
                        max(pu, pv) * 1.08 if max(pu, pv) > 0 else 0
                    )
                if data["ridge"] is not None and len(X_feat) > 5:
                    y_pred = data["ridge"].predict(np.array(X_feat))
                    st.markdown("### Ridge Regression (Property Prediction)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R2", f"{r2_score(y_target, y_pred):.3f}")
                    c2.metric(
                        "MAE", f"{mean_absolute_error(y_target, y_pred):.2f}"
                    )
                    c3.metric(
                        "RMSE",
                        f"{np.sqrt(mean_squared_error(y_target, y_pred)):.2f}",
                    )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Export & Post-Processing")
            export_format = st.selectbox("Format:", [
                "GraphML", "JSON (Full Metadata)", "JSON (Compact)",
                "CSV (Edges + Metadata)", "CSV (Nodes + Metadata)",
                "PNG", "SVG", "GEXF",
            ])
            include_metadata = st.checkbox(
                "Include metadata in export", value=True,
            )
            if st.button("Generate Export"):
                result = export_graph(
                    nx_graph, concept_abstract_map,
                    export_format, include_metadata,
                )
                if result[0]:
                    data_bytes, mime, filename = result
                    st.download_button(
                        "💾 Save File", data=data_bytes,
                        file_name=filename, mime=mime,
                    )
            st.markdown("---")
            st.subheader("Publication-Ready Figure")
            pub_dpi = st.slider("DPI", 150, 600, 300, step=50)
            pub_figsize = st.selectbox(
                "Figure size:",
                [(10, 8), (12, 10), (14, 12), (16, 14)],
                index=2,
            )
            if st.button("Generate Publication Figure"):
                pub_bytes = export_publication_figure(
                    nx_graph, valid_concepts, concept_abstract_map,
                    cmap_name=cmap, dpi=pub_dpi, figsize=pub_figsize,
                )
                if pub_bytes:
                    st.download_button(
                        "📥 Download Publication PNG",
                        data=pub_bytes,
                        file_name="lib_graph_publication.png",
                        mime="image/png",
                    )
            st.markdown("---")
            st.subheader("Automated Analysis Report")
            if st.button("Generate Markdown Report"):
                burst_df = st.session_state.get('burst_df', pd.DataFrame())
                drift_df = st.session_state.get('drift_df', pd.DataFrame())
                genealogy_df = st.session_state.get('genealogy_df', pd.DataFrame())
                bridge_df = st.session_state.get('bridge_df', pd.DataFrame())
                motifs = st.session_state.get('motifs', {})
                report = generate_analysis_report(
                    nx_graph, valid_concepts, concept_abstract_map,
                    top_scores, distill_df, burst_df, drift_df,
                    genealogy_df, bridge_df, motifs, val_metrics, df_filtered,
                )
                st.download_button(
                    "📄 Download Report (Markdown)",
                    data=report.encode('utf-8'),
                    file_name="lib_analysis_report.md",
                    mime="text/markdown",
                )
                with st.expander("Preview Report"):
                    st.markdown(report)
            concept_list_df = pd.DataFrame({
                'concept': valid_concepts,
                'frequency': [
                    len(concept_abstract_map.get(c, [])) for c in valid_concepts
                ],
                'degree': [nx_graph.degree(c) for c in valid_concepts],
                'category': [
                    abstract_concepts_to_categories([c]).get(c, 'general')
                    for c in valid_concepts
                ],
                'concept_type': [
                    nx_graph.nodes[c].get('concept_type', 'general')
                    for c in valid_concepts
                ],
                'definition': [
                    nx_graph.nodes[c].get('definition', '')
                    for c in valid_concepts
                ],
            })
            csv_concepts = concept_list_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📋 Download Concept List (CSV)",
                data=csv_concepts,
                file_name="lib_concepts_enhanced.csv", mime="text/csv",
            )
            with st.expander("📖 Concept Definitions & Meanings"):
                defs_df = concept_list_df[
                    concept_list_df['definition'] != ''
                ][['concept', 'definition', 'category']]
                if not defs_df.empty:
                    st.dataframe(defs_df, use_container_width=True)
                else:
                    st.info(
                        "No definitions available. "
                        "Enable ontology-based resolution to see concept definitions."
                    )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Extra Visualizations")
            theme = THEME_PRESETS.get(
                st.session_state.get('theme', 'Bright (Default)'),
                THEME_PRESETS["Bright (Default)"],
            )
            with st.expander("Concept Timeline", expanded=True):
                render_concept_timeline(
                    df_filtered, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Co-occurrence Heatmap"):
                heatmap_n = st.slider(
                    "Top N concepts for heatmap", 5, 50, 25,
                    key="heatmap_n_slider",
                )
                render_cooccurrence_heatmap(
                    nx_graph, valid_concepts, concept_abstract_map,
                    top_n=heatmap_n, theme=theme,
                )
            with st.expander("t-SNE Projection"):
                embed_model = data.get("embed_model")
                if embed_model:
                    render_tsne_projection(
                        valid_concepts, concept_abstract_map,
                        embed_model, theme=theme,
                    )
                else:
                    st.info("Embedding model not available. Rebuild the graph.")
            with st.expander("Community Detection"):
                render_community_detection(
                    nx_graph, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Concept Growth Rate"):
                render_concept_growth(
                    df_filtered, valid_concepts,
                    concept_abstract_map, theme=theme,
                )
            with st.expander("Bubble Chart (Importance)"):
                render_bubble_chart(
                    nx_graph, valid_concepts,
                    concept_abstract_map, distill_df, theme=theme,
                )

        tab_idx += 1
        with tabs[tab_idx]:
            st.subheader("Advanced Analytics")
            with st.expander("Keyword Burst Detection", expanded=True):
                burst_df = st.session_state.get('burst_df')
                if burst_df is not None and not burst_df.empty:
                    st.dataframe(burst_df.head(20), use_container_width=True)
                    fig = px.bar(
                        burst_df.head(15), x='concept', y='burst_score',
                        color='burst_year',
                        title=(
                            "Keyword Bursts "
                            "(Sudden Spikes in Publication Frequency)"
                        ),
                        labels={
                            'burst_score': 'Burst Score',
                            'concept': 'Concept',
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No burst data available. "
                        "Build graph with temporal data."
                    )
            with st.expander("Semantic Drift Detection"):
                drift_df = st.session_state.get('drift_df')
                if drift_df is not None and not drift_df.empty:
                    st.dataframe(drift_df.head(20), use_container_width=True)
                    fig = px.bar(
                        drift_df.head(15), x='concept', y='semantic_drift',
                        title=(
                            "Semantic Drift "
                            "(Contextual Meaning Shift Over Time)"
                        ),
                        labels={
                            'semantic_drift': 'Drift Score',
                            'concept': 'Concept',
                        },
                        color='semantic_drift',
                        color_continuous_scale='RdYlBu_r',
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "No drift data available. "
                        "Build graph with temporal data spanning multiple years."
                    )
            with st.expander("Concept Genealogy"):
                genealogy_df = st.session_state.get('genealogy_df')
                if genealogy_df is not None and not genealogy_df.empty:
                    st.dataframe(
                        genealogy_df.head(20), use_container_width=True,
                    )
                    gen_counts = genealogy_df['generation'].value_counts()
                    fig = px.pie(
                        values=gen_counts.values, names=gen_counts.index,
                        title="Concept Generations Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No genealogy data available.")
            with st.expander("Cross-Domain Bridge Detection"):
                bridge_df = st.session_state.get('bridge_df')
                if bridge_df is not None and not bridge_df.empty:
                    st.dataframe(
                        bridge_df.head(20), use_container_width=True,
                    )
                    fig = px.scatter(
                        bridge_df.head(30),
                        x='betweenness', y='connected_categories',
                        size='bridge_score', color='own_category',
                        hover_data=['concept', 'categories'],
                        title="Cross-Domain Bridge Concepts",
                        labels={
                            'betweenness': 'Betweenness Centrality',
                            'connected_categories': 'Categories Connected',
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No bridge data available.")
            with st.expander("Network Motif Analysis"):
                motifs = st.session_state.get('motifs', {})
                if motifs:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        "Triangles", motifs.get('total_triangles', 0)
                    )
                    col2.metric("Cliques", motifs.get('total_cliques', 0))
                    col3.metric(
                        "Max Clique Size", motifs.get('max_clique_size', 0)
                    )
                    col4.metric(
                        "Star Motifs", motifs.get('star_motifs', 0)
                    )
                    if motifs.get('top_stars'):
                        st.markdown(
                            "**Top Star Motifs (Central Hubs):**"
                        )
                        star_df = pd.DataFrame(
                            motifs['top_stars'],
                            columns=['Concept', 'Degree', 'Clustering'],
                        )
                        st.dataframe(
                            star_df, use_container_width=True,
                        )
                else:
                    st.info("No motif data available.")
            with st.expander("Centrality Comparison & Degree Distribution"):
                centrality_df = compute_centrality_comparison(
                    nx_graph, valid_concepts,
                )
                if not centrality_df.empty:
                    st.dataframe(
                        centrality_df.head(20), use_container_width=True,
                    )
                    corr_cols = [
                        'degree', 'betweenness', 'closeness',
                        'eigenvector', 'pagerank',
                    ]
                    available = [
                        c for c in corr_cols if c in centrality_df.columns
                    ]
                    if len(available) >= 2:
                        corr_matrix = centrality_df[available].corr()
                        fig = px.imshow(
                            corr_matrix, text_auto=True, aspect="auto",
                            title="Centrality Correlation Matrix",
                            color_continuous_scale='RdBu_r',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    fig = plot_degree_distribution(nx_graph, theme=theme)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No centrality data available.")

        if has_reasoning:
            tab_idx += 1
            with tabs[tab_idx]:
                ontology_data = data.get("ontology")
                extractor_data = data.get("extractor")
                if ontology_data and extractor_data:
                    render_reasoning_dashboard(
                        nx_graph, valid_concepts, ontology_data, extractor_data,
                    )
                else:
                    st.info(
                        "Reasoning data not available. "
                        "Rebuild graph with ontology enabled."
                    )

        # Microtransformer tab
        tab_idx += 1
        with tabs[tab_idx]:
            if st.session_state.analysis_data is not None and "ontology" in st.session_state.analysis_data:
                render_microtransformer_kg_rag_tab(st.session_state.analysis_data, st.session_state.analysis_data["ontology"])
            else:
                st.info("Please build the concept graph with ontology enabled first.")

        # LLM Q&A tab
        tab_idx += 1
        with tabs[tab_idx]:
            if st.session_state.analysis_data is not None and "ontology" in st.session_state.analysis_data:
                render_llm_qa_tab(st.session_state.analysis_data, st.session_state.analysis_data["ontology"])
            else:
                st.info("Please build the concept graph with ontology enabled first.")

if __name__ == "__main__":
    main()
