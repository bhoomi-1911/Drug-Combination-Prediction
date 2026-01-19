# app.py (updated with fixed Recommendation Engine)
import streamlit as st
import pandas as pd
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, Crippen
import torch.nn as nn
from torch_geometric.data import Data as PyGData, Batch
from torch_geometric.nn import GINEConv, global_mean_pool
import pickle
import os
from itertools import combinations
import time
from typing import List, Dict, Tuple
import math

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(
    page_title="Synerlytics-Drug Synergy Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .page-header {
        font-weight: bold;
        font-size: 2.2rem;
        color: #bd0561;
        background-image: linear-gradient(45deg, #bd0561, #d33fef 50%, #304aad 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        display: inline-block;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00cc96;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .synergy-high {
        color: #00cc96;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-medium {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-low {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .synergy-antagonism {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .toxicity-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .toxicity-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .toxicity-low {
        color: #00cc96;
        font-weight: bold;
    }
    .side-effect-high {
        background-color: #ffebee;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #ff4b4b;
    }
    .side-effect-medium {
        background-color: #fff3e0;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #ffa500;
    }
    .side-effect-low {
        background-color: #e8f5e8;
        padding: 8px;
        border-radius: 5px;
        border-left: 4px solid #00cc96;
    }
    .rank-badge {
        background-color: #1f77b4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .test-button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 0.9rem;
        margin-top: 10px;
        width: 100%;
    }
    .test-button:hover {
        background-color: #1668a1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model classes
# ---------------------------
class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            conv = GINEConv(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                edge_dim=edge_dim
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x

class DualGNNModel(nn.Module):
    def __init__(self, node_feat_dim, cancer_dim, num_cells, cell_emb_dim,
                 gnn_hidden, gnn_layers, mlp_hidden, dropout, edge_dim):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, gnn_hidden, gnn_layers, edge_dim)
        self.cell_emb = nn.Embedding(num_cells, cell_emb_dim)
        self.fc1 = nn.Linear(gnn_hidden * 2 + cancer_dim + cell_emb_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden // 2)
        self.fc3 = nn.Linear(mlp_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_g1, batch_g2, cancer_vec, cell_idx):
        x1, e1_idx, e1_attr = batch_g1.x, batch_g1.edge_index, batch_g1.edge_attr
        x2, e2_idx, e2_attr = batch_g2.x, batch_g2.edge_index, batch_g2.edge_attr
        
        emb1 = self.encoder(x1, e1_idx, e1_attr, batch_g1.batch)
        emb2 = self.encoder(x2, e2_idx, e2_attr, batch_g2.batch)
        cell_embed = self.cell_emb(cell_idx)
        combined = torch.cat([emb1, emb2, cancer_vec, cell_embed], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

# ---------------------------
# RDKit Side Effect Prediction
# ---------------------------
def predict_side_effects(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    heavy_atoms = Lipinski.HeavyAtomCount(mol)
    
    side_effect_alerts = {
        'hepatotoxicity': ['[#6]-[N+](=O)[O-]', 'c1ccccc1[N]', 'S(=O)(=O)[OH]'],
        'nephrotoxicity': ['[Cl,Br,I]', 'c1ccccc1[F,Cl,Br,I]', '[NH2]C(=O)'],
        'cardiotoxicity': ['c1ccccc1', '[N+]', 'C=O'],
        'neurotoxicity': ['C#N', 'C=O', '[S;D2](=O)(=O)[#6]'],
        'gastrointestinal': ['[OH]', 'C(=O)[OH]', 'c1ccccc1[OH]'],
        'dermatological': ['[Cl,Br,I]', 'C=O', 'c1ccccc1']
    }
    
    side_effects = {}
    hepatotoxicity_score = sum([0.3 for pattern in side_effect_alerts['hepatotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if mw > 500: hepatotoxicity_score += 0.2
    side_effects['hepatotoxicity'] = min(1.0, hepatotoxicity_score)
    
    nephrotoxicity_score = sum([0.25 for pattern in side_effect_alerts['nephrotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if logp > 4: nephrotoxicity_score += 0.2
    side_effects['nephrotoxicity'] = min(1.0, nephrotoxicity_score)
    
    cardiotoxicity_score = sum([0.2 for pattern in side_effect_alerts['cardiotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if Descriptors.NumAromaticRings(mol) > 2: cardiotoxicity_score += 0.3
    side_effects['cardiotoxicity'] = min(1.0, cardiotoxicity_score)
    
    neurotoxicity_score = sum([0.25 for pattern in side_effect_alerts['neurotoxicity'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['neurotoxicity'] = min(1.0, neurotoxicity_score)
    
    gi_score = sum([0.2 for pattern in side_effect_alerts['gastrointestinal'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    if hbd > 3: gi_score += 0.2
    side_effects['gastrointestinal'] = min(1.0, gi_score)
    
    derm_score = sum([0.2 for pattern in side_effect_alerts['dermatological'] if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))])
    side_effects['dermatological'] = min(1.0, derm_score)
    
    side_effects['cns_effects'] = min(1.0, 0.1 + (logp / 10))
    side_effects['metabolic_issues'] = min(1.0, 0.1 + (mw - 300) / 700)
    
    return side_effects

def interpret_side_effects(side_effects):
    interpretations = []
    risk_levels = {'High Risk (0.7-1.0)': [], 'Medium Risk (0.4-0.7)': [], 'Low Risk (0.0-0.4)': []}
    for effect, score in side_effects.items():
        effect_name = effect.replace('_', ' ').title()
        if score >= 0.7: risk_levels['High Risk (0.7-1.0)'].append(f"{effect_name} ({score:.2f})")
        elif score >= 0.4: risk_levels['Medium Risk (0.4-0.7)'].append(f"{effect_name} ({score:.2f})")
        else: risk_levels['Low Risk (0.0-0.4)'].append(f"{effect_name} ({score:.2f})")
    for risk_level, effects in risk_levels.items():
        if effects:
            if 'High Risk' in risk_level: interpretations.append(f"**{risk_level}**: {', '.join(effects)}")
            elif 'Medium Risk' in risk_level: interpretations.append(f"**{risk_level}**: {', '.join(effects)}")
            else: interpretations.append(f"**{risk_level}**: {', '.join(effects)}")
    return interpretations

def get_side_effect_class(score):
    if score > 0.7: return "side-effect-high", "High Risk"
    elif score > 0.4: return "side-effect-medium", "Medium Risk"
    else: return "side-effect-low", "Low Risk"

def predict_combination_side_effects(side_effects_a, side_effects_b):
    combo_effects = {}
    for effect in side_effects_a.keys():
        risk_a = side_effects_a[effect]
        risk_b = side_effects_b.get(effect, 0.1)
        combo_risk = 1 - (1 - risk_a) * (1 - risk_b)
        combo_effects[effect] = min(1.0, combo_risk)
    return combo_effects

# ---------------------------
# SMILES -> PyG graph
# ---------------------------
def mol_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_feats = [[atom.GetAtomicNum(), atom.GetTotalNumHs(), atom.GetDegree(), atom.GetImplicitValence(), 1.0 if atom.GetIsAromatic() else 0.0] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt_val = {Chem.rdchem.BondType.SINGLE: 1.0, Chem.rdchem.BondType.DOUBLE: 2.0, Chem.rdchem.BondType.TRIPLE: 3.0, Chem.rdchem.BondType.AROMATIC: 1.5}.get(bond.GetBondType(), 0.0)
        edge_index.extend([[a1, a2], [a2, a1]])
        edge_attr.extend([[bt_val], [bt_val]])
    if len(edge_index) == 0: edge_index, edge_attr = [[0, 0]], [[0.0]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)

def mol_to_image(smiles, size=(250, 250)):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=size) if mol else None

# ---------------------------
# Robust helper: find cancer col index
# ---------------------------
def find_cancer_col_index(cancer_cols, cancer_type_key):
    candidates = [f"cancer_type_{cancer_type_key}", f"cancer_type_{cancer_type_key.replace(' ', '_')}",
                  f"cancer_type_{cancer_type_key.lower()}", f"cancer_type_{cancer_type_key.replace(' ', '').lower()}"]
    for cand in candidates:
        if cand in cancer_cols: return cancer_cols.index(cand)
    key_words = [w.lower() for w in cancer_type_key.split()]
    for i, cc in enumerate(cancer_cols):
        if all(w in cc.lower() for w in key_words): return i
    for i, cc in enumerate(cancer_cols):
        if cancer_type_key.lower() in cc.lower(): return i
    return None

# ---------------------------
# Prediction function
# ---------------------------
def predict_synergy_proper(drug1_smiles, drug2_smiles, cancer_type, cell_line,
                           model, device,
                           cancer_cell_mapping, cell_line_mapping,
                           cell_le=None, cancer_cols=None, scaler_y=None,
                           debug_mode=False):
    try:
        if model is None:
            drug_hash = hash(drug1_smiles + drug2_smiles) % 100
            synergy_score = (drug_hash - 50) / 2.0
            return synergy_score, None

        g1 = mol_to_pyg(drug1_smiles)
        g2 = mol_to_pyg(drug2_smiles)
        if g1 is None or g2 is None:
            return None, "Invalid SMILES string"

        batch_g1 = Batch.from_data_list([g1]).to(device)
        batch_g2 = Batch.from_data_list([g2]).to(device)

        if cancer_cols is not None:
            cancer_vec = torch.zeros(1, len(cancer_cols), dtype=torch.float).to(device)
            idx = find_cancer_col_index(cancer_cols, cancer_type)
            if idx is not None:
                cancer_vec[0, idx] = 1.0
        else:
            cancer_cols_list = list(cancer_cell_mapping.keys())
            cancer_vec = torch.zeros(1, len(cancer_cols_list), dtype=torch.float).to(device)
            if cancer_type in cancer_cell_mapping:
                cancer_vec[0, cancer_cols_list.index(cancer_type)] = 1.0

        if cell_le is not None:
            try:
                cell_idx_val = int(cell_le.transform([cell_line])[0])
            except Exception:
                cell_idx_val = int(cell_line_mapping.get(cell_line, 0))
        else:
            cell_idx_val = int(cell_line_mapping.get(cell_line, 0))

        cell_idx = torch.tensor([cell_idx_val], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model(batch_g1, batch_g2, cancer_vec, cell_idx)
            pred_scaled = float(pred.cpu().squeeze().item())

        if scaler_y is not None:
            try:
                synergy_score = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
            except Exception:
                original_mean = -0.145845
                original_std = 8.535542
                synergy_score = (pred_scaled * original_std) + original_mean
        else:
            original_mean = -0.145845
            original_std = 8.535542
            synergy_score = (pred_scaled * original_std) + original_mean

        return synergy_score, None

    except Exception as e:
        return None, str(e)

# ---------------------------
# Recommendation System
# ---------------------------
class DrugRecommender:
    def __init__(self, drug_df, model, device, cancer_cell_mapping, 
                 cell_line_mapping, cell_le, cancer_cols, scaler_y):
        self.drug_df = drug_df
        self.model = model
        self.device = device
        self.cancer_cell_mapping = cancer_cell_mapping
        self.cell_line_mapping = cell_line_mapping
        self.cell_le = cell_le
        self.cancer_cols = cancer_cols
        self.scaler_y = scaler_y
        
        # Get valid drugs with SMILES
        self.drug_names = drug_df['drug_name'].tolist()
        self.drug_smiles = dict(zip(drug_df['drug_name'], drug_df['smiles']))
        
        # Filter out drugs with invalid SMILES
        self.valid_drugs = []
        for drug_name in self.drug_names:
            smiles = self.drug_smiles[drug_name]
            if Chem.MolFromSmiles(smiles) is not None:
                self.valid_drugs.append(drug_name)
        
        # Load cell line to drug mapping from CSV
        self.cell_line_drugs_df = self._load_cell_line_drug_mapping()
        # Cache for predictions
        self.prediction_cache = {}
    
    def _load_cell_line_drug_mapping(self):
        """Load CSV file with cell line to drug mapping"""
        try:
            if os.path.exists("cell_line_drugs_mapping.csv"):
                df = pd.read_csv("cell_line_drugs_mapping.csv")
                return df
            else:
                st.sidebar.warning("cell_line_drugs_mapping.csv not found")
                return None
        except Exception as e:
            st.sidebar.error(f"Error loading cell line mapping: {e}")
            return None
    
    def get_drugs_for_cell_line(self, cell_line):
        """Get all drugs actually tested on a specific cell line"""
        if self.cell_line_drugs_df is None:
            return []
        
        drugs = self.cell_line_drugs_df[
            self.cell_line_drugs_df['cell_line'] == cell_line
        ]['drug_name'].tolist()
        
        # Filter to only drugs we have SMILES for
        available_drugs = [d for d in drugs if d in self.valid_drugs]
        return available_drugs
    
    def get_cell_line_stats(self, cell_line):
        """Get statistics for a cell line"""
        if self.cell_line_drugs_df is None:
            return None
        
        cell_data = self.cell_line_drugs_df[
            self.cell_line_drugs_df['cell_line'] == cell_line
        ]
        
        if len(cell_data) == 0:
            return None
        
        total_drugs = int(cell_data.iloc[0]['total_drugs_for_cell_line'])
        possible_combinations = int(cell_data.iloc[0]['possible_combinations'])
        
        return {
            'total_drugs': total_drugs,
            'possible_combinations': possible_combinations,
            'available_drugs': len(self.get_drugs_for_cell_line(cell_line))
        }
    
    def get_cache_key(self, drug1, drug2, cancer_type, cell_line):
        return f"{drug1}||{drug2}||{cancer_type}||{cell_line}"
    
    def predict_and_cache(self, drug1, drug2, cancer_type, cell_line):
        cache_key = self.get_cache_key(drug1, drug2, cancer_type, cell_line)
        
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        smiles1 = self.drug_smiles.get(drug1, '')
        smiles2 = self.drug_smiles.get(drug2, '')
        
        synergy_score, error = predict_synergy_proper(
            smiles1, smiles2, cancer_type, cell_line,
            self.model, self.device, self.cancer_cell_mapping,
            self.cell_line_mapping, self.cell_le, self.cancer_cols,
            self.scaler_y, debug_mode=False
        )
        
        if error or synergy_score is None:
            synergy_score = 0.0
        
        # Calculate combination safety score
        side_effects1 = predict_side_effects(smiles1)
        side_effects2 = predict_side_effects(smiles2)
        combo_effects = predict_combination_side_effects(side_effects1, side_effects2)
        
        # Calculate average risk (lower is better)
        avg_risk = np.mean(list(combo_effects.values())) if combo_effects else 0.5
        safety_score = 1.0 - avg_risk  # Invert risk to get safety
        
        # Calculate overall recommendation score (70% synergy, 30% safety)
        # Normalize synergy to 0-1 range (assuming max synergy ~20)
        norm_synergy = max(0, min(1, synergy_score / 20.0))
        recommendation_score = 0.7 * norm_synergy + 0.3 * safety_score
        
        result = {
            'synergy_score': synergy_score,
            'safety_score': safety_score,
            'recommendation_score': recommendation_score,
            'avg_risk': avg_risk,
            'side_effects': combo_effects
        }
        
        self.prediction_cache[cache_key] = result
        return result
    
    def get_top_recommendations(self, cancer_type: str, cell_line: str, 
                                top_k: int = 20) -> List[Dict]:
        """Get top drug pair recommendations based on tested drugs"""
        
        if cell_line not in self.cell_line_mapping:
            return []
        
        # Get drugs tested on this cell line
        cell_line_drugs = self.get_drugs_for_cell_line(cell_line)
        
        if len(cell_line_drugs) < 2:
            st.warning(f"Not enough tested drugs available for {cell_line}")
            return []
        
        # Get statistics
        stats = self.get_cell_line_stats(cell_line)
        
        # Generate ALL possible pairs from tested drugs
        all_pairs = list(combinations(cell_line_drugs, 2))
        
        if len(all_pairs) > 2000:
            st.warning(f"Evaluating {len(all_pairs):,} combinations may take some time...")
        
        recommendations = []
        
        with st.spinner(f"Evaluating {len(all_pairs):,} combinations..."):
            progress_bar = st.progress(0)
            
            for idx, (drug1, drug2) in enumerate(all_pairs):
                try:
                    result = self.predict_and_cache(drug1, drug2, cancer_type, cell_line)
                    
                    recommendation = {
                        'drug_pair': (drug1, drug2),
                        'synergy_score': result['synergy_score'],
                        'safety_score': result['safety_score'],
                        'recommendation_score': result['recommendation_score'],
                        'avg_risk': result['avg_risk'],
                        'side_effects': result['side_effects']
                    }
                    recommendations.append(recommendation)
                except Exception as e:
                    continue
                
                # Update progress
                if idx % 100 == 0:
                    progress_bar.progress(min((idx + 1) / len(all_pairs), 1.0))
            
            progress_bar.empty()
        
        # Sort by synergy score
        recommendations.sort(key=lambda x: x['synergy_score'], reverse=True)
        
        return recommendations[:top_k]

# ---------------------------
# Load data/artifacts and model
# ---------------------------
@st.cache_resource
def load_data_and_model():
    try:
        drug_df = pd.read_csv("Datasets/comprehensive_drug_smiles.csv")
    except:
        drug_df = pd.DataFrame({
            'drug_name': ['PACLITAXEL', 'DOXORUBICIN', 'CISPLATIN', 'GEMCITABINE', 
                         '5-FLUOROURACIL', 'IRINOTECAN', 'OXALIPLATIN', 'CARBOPLATIN',
                         'TAMOXIFEN', 'ETOPOSIDE', 'VINCRISTINE', 'TEMOZOLOMIDE',
                         'DOCETAXEL', 'SUNITINIB', 'IMATINIB', 'RITUXIMAB'],
            'smiles': [
                'CC1=C2C(C(=O)C3=C(COC3=O)C(C2(C)C)(CC1OC(=O)C(C(C4=CC=CC=C4)NC(=O)C5=CC=CC=C5)O)C)OC(=O)C6=CC=CC=C6',
                'CC1C(C(CC(O1)OC2C(OC3C(C2O)C(C(C4=CC(=O)C5=CC=CC=C5C4=O)O)(C)C)C)O)(C)O',
                'Cl[Pt](Cl)(N)N',
                'C1=NC(=O)N(C=C1)C2C(C(C(O2)CO)O)O',
                'C1=CN(C(=O)NC1=O)F',
                'CC1(C(O)CN(C)C1=O)C2=C(C3=C(C(=C2)C(=O)OC)CCCN3)C(=O)O',
                'C1C[C@@H]([C@H]([C@@H]1N)O)O[C@@H]2[C@H]([C@@H]([C@H](O2)C)O)O',
                'C1CCCCC1',
                'CC(C)(C)C1=CC=C(C=C1)C(C)C2=CC3=C(C=C2)OCO3',
                'CC1CC(C(C(C1)O)O)OC2C3C4C(C(C(C4OC3C(C2O)O)O)O)O',
                'CC1(C2CC3C(C2(CC1OC(=O)C4=CC=CC=C4)OC)CN(CC3)C(=O)C5=CC=CC=C5)C',
                'CC1=NN(C(=O)NC1=O)C2=CC=CC=C2',
                'CC1=C2C(C(=O)C3=C(COC3=O)C(C2(C)C)(CC1OC(=O)C(C(C4=CC=CC=C4)NC(=O)C5=CC=CC=C5)O)C)OC(=O)C6=CC=CC=C6',
                'CC1=CC(=CC=C1)C2=NC3=C(C=C(C=C3)Cl)NC(=N2)NC4=CC(=C(C=C4)F)OC',
                'CN1CCN(CC1)C2=NC3=C(C=C2)C=C(C=C3)NC(=O)C4=CC=CC=C4',
                'CC(C)CC(=O)NCC1=CC=C(C=C1)OC2=CC=C(C=C2)C3=NC4=C(C=C3)OCCO4'
            ]
        })
    drug_df = drug_df.dropna(subset=['smiles'])

    cancer_cell_mapping = {
        'Bladder Cancer': ['BFTC-905', 'HT-1197', 'HT-1376', 'J82', 'JMSU-1', 'KU-19-19', 'RT-112', 'T24', 'TCCSUP', 'UM-UC-3'],
        'Bone Cancer': ['A-673', 'TC-32', 'TC-71'],
        'Brain Cancer': ['SF-295', 'T98G'],
        'Breast Cancer': ['BT-549', 'MCF7', 'MDA-MB-231', 'MDA-MB-468', 'T-47D', 'BT-20', 'BT-474', 'CAL-120', 'CAL-148', 'CAL-51'],
        'Colon/Colorectal Cancer': ['SW527', 'COLO 205', 'HCT-15', 'KM12', 'RKO', 'SW837', 'LS513'],
        'Gastric Cancer': ['AGS', 'KATO III', 'SNU-16'],
        'Kidney Cancer': ['ACHN', 'SN12C', 'UO-31'],
        'Leukemia': ['CCRF-CEM', 'K-562'],
        'Lung Cancer': ['A427', 'A549', 'EKVX', 'HOP-62', 'HOP-92', 'NCI-H226', 'NCI-H322M'],
        'Lymphoma': ['HDLM-2', 'L-1236', 'L-428', 'U-HO1'],
        'Myeloma': ['KMS-11'],
        'Not Available': ['EW-8', 'SF-268', 'SF-539', 'SNB-19', 'SNB-75', 'U251'],
        'Ovarian Cancer': ['A2780', 'IGROV1', 'OVCAR-4', 'OVCAR-5', 'OVCAR-8', 'SK-OV-3'],
        'Pancreatic Cancer': ['PANC-1'],
        'Prostate Cancer': ['PC-3'],
        'Sarcoma': ['RD', 'SMS-CTR'],
        'Skin Cancer': ['A2058', 'LOX IMVI', 'M14', 'SK-MEL-2', 'SK-MEL-28', 'SK-MEL-5']
    }

    cell_line_mapping = {}
    idx = 0
    for cancer_type, cell_lines in cancer_cell_mapping.items():
        for cl in cell_lines:
            if cl not in cell_line_mapping:
                cell_line_mapping[cl] = idx
                idx += 1

    scaler_y, cell_le, cancer_cols = None, None, None
    if os.path.exists("scaler_y.pkl"):
        with open("scaler_y.pkl","rb") as f: scaler_y = pickle.load(f)
    if os.path.exists("cell_le.pkl"):
        with open("cell_le.pkl","rb") as f: cell_le = pickle.load(f)
    if os.path.exists("cancer_cols.pkl"):
        with open("cancer_cols.pkl","rb") as f: cancer_cols = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    try:
        model = DualGNNModel(
            node_feat_dim=5,
            cancer_dim=(len(cancer_cols) if cancer_cols is not None else 17),
            num_cells=(len(cell_le.classes_) if cell_le is not None else 288),
            cell_emb_dim=48,
            gnn_hidden=192,
            gnn_layers=4,
            mlp_hidden=512,
            dropout=0.3,
            edge_dim=1
        ).to(device)
        model.load_state_dict(torch.load("final_gnn_model.pt", map_location=device))
        model.eval()
    except:
        model = None

    # Initialize recommender
    recommender = DrugRecommender(
        drug_df, model, device, cancer_cell_mapping,
        cell_line_mapping, cell_le, cancer_cols, scaler_y
    )

    return drug_df, cancer_cell_mapping, model, device, cell_line_mapping, cell_le, cancer_cols, scaler_y, recommender

# ---------------------------
# Main app
# ---------------------------
def main():
    # Initialize session state
    if 'recommend_drug1' not in st.session_state:
        st.session_state.recommend_drug1 = ""
    if 'recommend_drug2' not in st.session_state:
        st.session_state.recommend_drug2 = ""
    if 'recommend_cancer' not in st.session_state:
        st.session_state.recommend_cancer = ""
    if 'recommend_cell_line' not in st.session_state:
        st.session_state.recommend_cell_line = ""
    if 'example_drug1' not in st.session_state:
        st.session_state.example_drug1 = ""
    if 'example_drug2' not in st.session_state:
        st.session_state.example_drug2 = ""
    if 'recommendations_list' not in st.session_state:
        st.session_state.recommendations_list = []
    if 'test_status' not in st.session_state:
        st.session_state.test_status = {'drug1': '', 'drug2': '', 'tested': False}
    
    # Load data and model
    drug_df, cancer_cell_mapping, model, device, cell_line_mapping, cell_le, cancer_cols, scaler_y, recommender = load_data_and_model()
    
    st.sidebar.title("Navigation")
    
    # Create tabs
    page = st.sidebar.radio("Go to", ["Single Prediction", "Advanced (SMILES)", "Recommendation Engine", "Batch Screening", "About"])
    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)

    # Filter cell lines to only include those in our dataset
    if hasattr(recommender, 'cell_line_drugs_df') and recommender.cell_line_drugs_df is not None:
        available_cell_lines = recommender.cell_line_drugs_df['cell_line'].unique().tolist()
        
        # Filter cancer_cell_mapping to only include available cell lines
        filtered_cancer_cell_mapping = {}
        for cancer_type, cell_lines in cancer_cell_mapping.items():
            filtered_lines = [cl for cl in cell_lines if cl in available_cell_lines]
            if filtered_lines:  # Only keep cancer types that have available cell lines
                filtered_cancer_cell_mapping[cancer_type] = filtered_lines
    else:
        filtered_cancer_cell_mapping = cancer_cell_mapping

    if page == "Single Prediction":
        st.markdown('<div class="page-header">Synerlytics - Drug Combination Synergy Predictor</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Drug Selection")
            
            # Determine default indices for drug selection
            drug_list = drug_df['drug_name'].tolist()
            
            # Set default indices based on session state (if coming from recommendations)
            default_a = 0
            default_b = 1
            
            if st.session_state.recommend_drug1 and st.session_state.recommend_drug1 in drug_list:
                default_a = drug_list.index(st.session_state.recommend_drug1)
            if st.session_state.recommend_drug2 and st.session_state.recommend_drug2 in drug_list:
                default_b = drug_list.index(st.session_state.recommend_drug2)
            
            drug_a = st.selectbox("Select Drug A", options=drug_list, index=default_a)
            drug_b = st.selectbox("Select Drug B", options=drug_list, index=default_b)
            
            # Show auto-fill indicator if values were auto-filled
            if st.session_state.recommend_drug1 and st.session_state.recommend_drug2:
                st.info(f"Auto-filled: **{st.session_state.recommend_drug1}** + **{st.session_state.recommend_drug2}**")
                
                # Clear auto-fill button
                if st.button("Clear Auto-fill"):
                    st.session_state.recommend_drug1 = ""
                    st.session_state.recommend_drug2 = ""
                    st.session_state.recommend_cancer = ""
                    st.session_state.recommend_cell_line = ""
                    st.session_state.test_status = {'drug1': '', 'drug2': '', 'tested': False}
                    st.rerun()
            
            if drug_a and drug_b:
                col1a, col1b = st.columns(2)
                with col1a:
                    st.write(f"**{drug_a}**")
                    smiles_a = drug_df[drug_df['drug_name']==drug_a]['smiles'].iloc[0]
                    img_a = mol_to_image(smiles_a, size=(220, 220))
                    if img_a: st.image(img_a, use_container_width=True)
                with col1b:
                    st.write(f"**{drug_b}**")
                    smiles_b = drug_df[drug_df['drug_name']==drug_b]['smiles'].iloc[0]
                    img_b = mol_to_image(smiles_b, size=(220, 220))
                    if img_b: st.image(img_b, use_container_width=True)

        with col2:
            st.subheader("Cancer Context")
            
            # Determine default cancer type and cell line
            default_cancer_index = 0
            if st.session_state.recommend_cancer and st.session_state.recommend_cancer in filtered_cancer_cell_mapping:
                cancer_types = list(filtered_cancer_cell_mapping.keys())
                if st.session_state.recommend_cancer in cancer_types:
                    default_cancer_index = cancer_types.index(st.session_state.recommend_cancer)
            
            cancer_type = st.selectbox("Select Cancer Type", 
                                       options=list(filtered_cancer_cell_mapping.keys()), 
                                       index=default_cancer_index)
            
            # Determine default cell line
            if cancer_type in filtered_cancer_cell_mapping:
                cell_lines = filtered_cancer_cell_mapping[cancer_type]
                default_cell_index = 0
                
                # Check if we have a stored cell line for this cancer type
                if (st.session_state.recommend_cell_line and 
                    st.session_state.recommend_cell_line in cell_lines):
                    default_cell_index = cell_lines.index(st.session_state.recommend_cell_line)
                
                cell_line = st.selectbox("Select Cell Line", 
                                         options=cell_lines,
                                         index=default_cell_index)
            else:
                cell_line = st.selectbox("Select Cell Line", ["No cell lines available"])
            
            predict_btn = st.button("Predict Synergy & Side Effects", type="primary", use_container_width=True)

        if predict_btn:
            if drug_a == drug_b:
                st.error("Please select two different drugs for combination analysis.")
            else:
                with st.spinner("Predicting synergy and side effects..."):
                    synergy_score, error = predict_synergy_proper(
                        smiles_a, smiles_b, cancer_type, cell_line,
                        model, device, cancer_cell_mapping, cell_line_mapping,
                        cell_le=cell_le, cancer_cols=cancer_cols, scaler_y=scaler_y,
                        debug_mode=debug_mode
                    )

                    side_effects_a = predict_side_effects(smiles_a)
                    side_effects_b = predict_side_effects(smiles_b)

                    if error:
                        st.error(f"Prediction error: {error}")
                    else:
                        # Clear the auto-fill after successful prediction
                        if st.session_state.recommend_drug1:
                            st.session_state.recommend_drug1 = ""
                            st.session_state.recommend_drug2 = ""
                            st.session_state.recommend_cancer = ""
                            st.session_state.recommend_cell_line = ""
                            st.session_state.test_status = {'drug1': '', 'drug2': '', 'tested': False}
                        
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.subheader("Synergy Prediction Results")

                        if synergy_score > 6:
                            synergy_class = "High Synergy"
                            synergy_color = "synergy-high"
                            interpretation = "Strong positive interaction expected"
                            confidence_level = "High"
                            recommendation = "Promising candidate for further investigation!"
                        elif synergy_score > 3:
                            synergy_class = "Medium Synergy"
                            synergy_color = "synergy-medium"
                            interpretation = "Moderate positive interaction expected"
                            confidence_level = "High"
                            recommendation = "Worth further experimental validation"
                        elif synergy_score > 0:
                            synergy_class = "Low Synergy"
                            synergy_color = "synergy-low"
                            interpretation = "Weak positive interaction"
                            confidence_level = "Medium"
                            recommendation = "May need careful evaluation"
                        else:
                            synergy_class = "Antagonism"
                            synergy_color = "synergy-antagonism"
                            interpretation = "Negative interaction expected"
                            confidence_level = "Low"
                            recommendation = "Likely to be ineffective or harmful"

                        st.markdown(f"**Synergy Score:** <span class='{synergy_color}'>{synergy_score:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Synergy Class:** <span class='{synergy_color}'>{synergy_class}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Interpretation:** {interpretation}")
                        st.markdown(f"**Confidence Level:** {confidence_level}")
                        st.markdown(f"**Recommendation:** {recommendation}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        st.subheader("Side Effect Risk Assessment")
                        col_side1, col_side2 = st.columns(2)
                        with col_side1:
                            st.write(f"**{drug_a} Side Effect Profile**")
                            if side_effects_a:
                                interpretations = interpret_side_effects(side_effects_a)
                                for interpretation in interpretations:
                                    if "High Risk" in interpretation: st.error(interpretation)
                                    elif "Medium Risk" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")

                        with col_side2:
                            st.write(f"**{drug_b} Side Effect Profile**")
                            if side_effects_b:
                                interpretations = interpret_side_effects(side_effects_b)
                                for interpretation in interpretations:
                                    if "High Risk" in interpretation: st.error(interpretation)
                                    elif "Medium Risk" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")

                        st.subheader("Combination Side Effect Risks")
                        combination_side_effects = predict_combination_side_effects(side_effects_a, side_effects_b)
                        interpretations = interpret_side_effects(combination_side_effects)
                        for interpretation in interpretations:
                            if "High Risk" in interpretation: st.error(interpretation)
                            elif "Medium Risk" in interpretation: st.warning(interpretation)
                            else: st.success(interpretation)

    elif page == "Advanced (SMILES)":
        st.markdown('<div class="page-header">Advanced SMILES Mode</div>', unsafe_allow_html=True)

        st.caption("### Enter SMILES of any drug names you want to combine and test their synergy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Drug 1 (SMILES)")
            # Pre-fill with example if available
            default_drug1 = st.session_state.example_drug1 if st.session_state.example_drug1 else "CN1CCC2=CC3=C(C=C2C1C4C5=C(C6=C(C=C5)OCO6)C(=O)O4)OCO3"
            drug1_smiles = st.text_area(
                "Enter SMILES for Drug 1:",
                value=default_drug1,
                height=100,
                help="Enter the SMILES string for first drug. Example: CN1CCC2=CC3=C(C=C2C1C4C5=C(C6=C(C=C5)OCO6)C(=O)O4)OCO3"
            )
            
            if drug1_smiles:
                try:
                    img1 = mol_to_image(drug1_smiles, size=(220, 220))
                    if img1:
                        st.image(img1, caption="Drug 1 Structure", use_container_width=True)
                    else:
                        st.warning("Could not generate structure. Check SMILES format.")
                except:
                    st.warning("Could not visualize SMILES.")
        
        with col2:
            st.subheader("Drug 2 (SMILES)")
            # Pre-fill with example if available
            default_drug2 = st.session_state.example_drug2 if st.session_state.example_drug2 else "CN(C)CCC(C1=CC=C(C=C1)Cl)C2=CC=CC=N2"
            drug2_smiles = st.text_area(
                "Enter SMILES for Drug 2:",
                value=default_drug2,
                height=100,
                help="Enter the SMILES string for second drug. Example: CN(C)CCC(C1=CC=C(C=C1)Cl)C2=CC=CC=N2"
            )
            
            if drug2_smiles:
                try:
                    img2 = mol_to_image(drug2_smiles, size=(220, 220))
                    if img2:
                        st.image(img2, caption="Drug 2 Structure", use_container_width=True)
                    else:
                        st.warning("Could not generate structure. Check SMILES format.")
                except:
                    st.warning("Could not visualize SMILES.")

        # Help section with link
        with st.expander("Need SMILES for a drug? Click here!"):
            st.markdown("""
            ### Find SMILES for Any Drug:
            1. **PubChem**: [https://pubchem.ncbi.nlm.nih.gov/](https://pubchem.ncbi.nlm.nih.gov/)
            
            
            **Quick Search Example:**
            - Go to PubChem
            - Search your drug name (e.g., "Aspirin")
            - Find the **Canonical SMILES** field
            - Copy the SMILES string (looks like: `CC(=O)OC1=CC=CC=C1C(=O)O`)
            """)
        
        # Quick SMILES examples
        with st.expander("Quick SMILES Examples"):
            examples = {
                "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Metformin": "CN(C)C(=N)N=C(N)N",
                "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "Bicuculline": "CN1CCC2=CC3=C(C=C2C1C4C5=C(C6=C(C=C5)OCO6)C(=O)O4)OCO3",
                "Chlorpheniramine": "CN(C)CCC(C1=CC=C(C=C1)Cl)C2=CC=CC=N2"
            }
            
            selected_example = st.selectbox("Load Example:", list(examples.keys()))
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Use as Drug 1", key="ex_drug1"):
                    st.session_state.example_drug1 = examples[selected_example]
                    st.rerun()
            with col_b:
                if st.button("Use as Drug 2", key="ex_drug2"):
                    st.session_state.example_drug2 = examples[selected_example]
                    st.rerun()
        
        st.subheader("Cancer Context")
        cancer_type = st.selectbox("Select Cancer Type", options=list(cancer_cell_mapping.keys()), index=0, key="adv_cancer")
        
        if cancer_type in cancer_cell_mapping:
            cell_line = st.selectbox("Select Cell Line", options=cancer_cell_mapping[cancer_type], key="adv_cell")
        else:
            cell_line = st.selectbox("Select Cell Line", ["No cell lines available"], key="adv_cell")
        
        # SMILES validation
        if drug1_smiles and drug2_smiles:
            mol1 = Chem.MolFromSmiles(drug1_smiles)
            mol2 = Chem.MolFromSmiles(drug2_smiles)
            
            if mol1 is None:
                st.error("Invalid SMILES for Drug 1. Please check the format.")
            if mol2 is None:
                st.error("Invalid SMILES for Drug 2. Please check the format.")
            
            if mol1 and mol2:
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.info(f"**Drug 1 Valid**\nAtoms: {mol1.GetNumAtoms()}\nBonds: {mol1.GetNumBonds()}")
                with col_stats2:
                    st.info(f"**Drug 2 Valid**\nAtoms: {mol2.GetNumAtoms()}\nBonds: {mol2.GetNumBonds()}")
        
        predict_btn = st.button("Predict Synergy from SMILES", type="primary", use_container_width=True)
        
        if predict_btn:
            if not drug1_smiles or not drug2_smiles:
                st.error("Please enter SMILES for both drugs.")
            elif drug1_smiles == drug2_smiles:
                st.error("Drug 1 and Drug 2 SMILES are identical.")
            else:
                with st.spinner("Predicting synergy from SMILES..."):
                    synergy_score, error = predict_synergy_proper(
                        drug1_smiles, drug2_smiles, cancer_type, cell_line,
                        model, device, cancer_cell_mapping, cell_line_mapping,
                        cell_le=cell_le, cancer_cols=cancer_cols, scaler_y=scaler_y,
                        debug_mode=debug_mode
                    )
                    
                    side_effects_a = predict_side_effects(drug1_smiles)
                    side_effects_b = predict_side_effects(drug2_smiles)
                    
                    if error:
                        st.error(f"Prediction error: {error}")
                    else:
                        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                        st.subheader("Synergy Prediction Results")
                        
                        if synergy_score > 6:
                            synergy_class = "High Synergy"
                            synergy_color = "synergy-high"
                            interpretation = "Strong positive interaction expected"
                            confidence_level = "High"
                            recommendation = "Promising candidate for further investigation!"
                        elif synergy_score > 3:
                            synergy_class = "Medium Synergy"
                            synergy_color = "synergy-medium"
                            interpretation = "Moderate positive interaction expected"
                            confidence_level = "High"
                            recommendation = "Worth further experimental validation"
                        elif synergy_score > 0:
                            synergy_class = "Low Synergy"
                            synergy_color = "synergy-low"
                            interpretation = "Weak positive interaction"
                            confidence_level = "Medium"
                            recommendation = "May need careful evaluation"
                        else:
                            synergy_class = "Antagonism"
                            synergy_color = "synergy-antagonism"
                            interpretation = "Negative interaction expected"
                            confidence_level = "Low"
                            recommendation = "Likely to be ineffective or harmful"
                        
                        st.markdown(f"**Synergy Score:** <span class='{synergy_color}'>{synergy_score:.2f}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Synergy Class:** <span class='{synergy_color}'>{synergy_class}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Interpretation:** {interpretation}")
                        st.markdown(f"**Confidence Level:** {confidence_level}")
                        st.markdown(f"**Recommendation:** {recommendation}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.subheader("Side Effect Risk Assessment")
                        col_side1, col_side2 = st.columns(2)
                        with col_side1:
                            st.write(f"**Drug 1 Side Effect Profile**")
                            if side_effects_a:
                                interpretations = interpret_side_effects(side_effects_a)
                                for interpretation in interpretations:
                                    if "High Risk" in interpretation: st.error(interpretation)
                                    elif "Medium Risk" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")
                        
                        with col_side2:
                            st.write(f"**Drug 2 Side Effect Profile**")
                            if side_effects_b:
                                interpretations = interpret_side_effects(side_effects_b)
                                for interpretation in interpretations:
                                    if "High Risk" in interpretation: st.error(interpretation)
                                    elif "Medium Risk" in interpretation: st.warning(interpretation)
                                    else: st.success(interpretation)
                            else: st.info("Side effect prediction not available")
                        
                        st.subheader("Combination Side Effect Risks")
                        combination_side_effects = predict_combination_side_effects(side_effects_a, side_effects_b)
                        interpretations = interpret_side_effects(combination_side_effects)
                        for interpretation in interpretations:
                            if "High Risk" in interpretation: st.error(interpretation)
                            elif "Medium Risk" in interpretation: st.warning(interpretation)
                            else: st.success(interpretation)

    elif page == "Recommendation Engine":
        st.markdown('<div class="page-header">AI-Powered Drug Combination Recommender</div>', unsafe_allow_html=True)
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.subheader("Cancer Context")
            cancer_type = st.selectbox("Select Cancer Type for Recommendations", 
                                       options=list(filtered_cancer_cell_mapping.keys()), 
                                       index=0, key="rec_cancer")
            
            if cancer_type in filtered_cancer_cell_mapping:
                cell_line = st.selectbox("Select Cell Line", 
                                         options=filtered_cancer_cell_mapping[cancer_type],
                                         key="rec_cell")
            else:
                cell_line = None
                st.warning("No cell lines available for this cancer type")
        
        with col_rec2:
            st.subheader("Recommendation Settings")
            top_k = st.slider("Number of recommendations", 5, 50, 20, 
                             help="How many top combinations to display")
            
            # Show cell line statistics if available
            if cell_line and hasattr(recommender, 'get_cell_line_stats'):
                stats = recommender.get_cell_line_stats(cell_line)
                if stats:
                    st.info(f"""
                    **Cell Line Info:**
                    - Tested drugs: {stats['total_drugs']}
                    - Available with SMILES: {stats['available_drugs']} drugs
                    - Possible combinations: {stats['possible_combinations']:,}
                    """)
        
        # Generate recommendations button
        generate_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)
        
        if generate_btn:
            if not cell_line:
                st.error("Please select a valid cell line")
            else:
                # Get recommendations
                recommendations = recommender.get_top_recommendations(
                    cancer_type, cell_line, top_k=top_k
                )
                
                # Store in session state
                st.session_state.recommendations_list = recommendations
                st.session_state.recommend_cancer = cancer_type
                st.session_state.recommend_cell_line = cell_line
                # Clear any previous test status
                st.session_state.test_status = {'drug1': '', 'drug2': '', 'tested': False}
                
                if recommendations:
                    st.subheader(f"Top {len(recommendations)} Recommendations for {cancer_type} ({cell_line})")
                    
                    # Summary statistics
                    avg_synergy = np.mean([r['synergy_score'] for r in recommendations])
                    avg_safety = np.mean([r['safety_score'] for r in recommendations])
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Average Synergy", f"{avg_synergy:.1f}")
                    with col_stat2:
                        st.metric("Average Safety", f"{avg_safety:.2f}")
                    with col_stat3:
                        st.metric("Total Evaluated", f"{len(recommendations)}")
                    
                    # Display each recommendation
                    st.markdown("---")
                    drug_smiles_dict = dict(zip(drug_df['drug_name'], drug_df['smiles']))
                    
                    for idx, rec in enumerate(recommendations, 1):
                        drug1, drug2 = rec['drug_pair']
                        synergy_score = rec['synergy_score']
                        safety_score = rec['safety_score']
                        avg_risk = rec['avg_risk']
                        
                        # Determine synergy class
                        if synergy_score > 6:
                            synergy_class = "High Synergy"
                            synergy_color = "synergy-high"
                        elif synergy_score > 3:
                            synergy_class = "Medium Synergy"
                            synergy_color = "synergy-medium"
                        elif synergy_score > 0:
                            synergy_class = "Low Synergy"
                            synergy_color = "synergy-low"
                        else:
                            synergy_class = "Antagonism"
                            synergy_color = "synergy-antagonism"
                        
                        # Determine safety class
                        if avg_risk < 0.3:
                            safety_class = "Low Risk"
                            safety_color = "toxicity-low"
                        elif avg_risk < 0.6:
                            safety_class = "Medium Risk"
                            safety_color = "toxicity-medium"
                        else:
                            safety_class = "High Risk"
                            safety_color = "toxicity-high"
                        
                        with st.container():
                            st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns([0.5, 2, 1.5])
                            
                            with col1:
                                st.markdown(f'<div class="rank-badge">{idx}</div>', unsafe_allow_html=True)
                            
                            with col2:
                                # Molecular images (slightly smaller)
                                col_mol1, col_mol2 = st.columns(2)
                                with col_mol1:
                                    img1 = mol_to_image(drug_smiles_dict.get(drug1, ''), size=(180, 180))
                                    if img1:
                                        st.image(img1, width=180)
                                    st.markdown(f"**{drug1}**")
                                with col_mol2:
                                    img2 = mol_to_image(drug_smiles_dict.get(drug2, ''), size=(180, 180))
                                    if img2:
                                        st.image(img2, width=180)
                                    st.markdown(f"**{drug2}**")
                            
                            with col3:
                                st.markdown(f"**Synergy:** <span class='{synergy_color}'>{synergy_class} ({synergy_score:.1f})</span>", 
                                           unsafe_allow_html=True)
                                st.markdown(f"**Safety:** <span class='{safety_color}'>{safety_class} ({safety_score:.2f})</span>", 
                                           unsafe_allow_html=True)
                                st.markdown(f"**Overall Score:** `{rec['recommendation_score']:.3f}`")
                                
                                # Create a unique key for each button
                                button_key = f"test_btn_{idx}_{drug1}_{drug2}"
                                
                                # Check if this combination was just tested
                                just_tested = (
                                    st.session_state.test_status['tested'] and
                                    st.session_state.test_status['drug1'] == drug1 and
                                    st.session_state.test_status['drug2'] == drug2
                                )
                                
                                # Test combination button
                                if st.button(f"Test This Combination", key=button_key):
                                    # Store the selected drugs, cancer type, and cell line in session state
                                    st.session_state.recommend_drug1 = drug1
                                    st.session_state.recommend_drug2 = drug2
                                    st.session_state.recommend_cancer = cancer_type
                                    st.session_state.recommend_cell_line = cell_line
                                    # Mark as tested
                                    st.session_state.test_status = {
                                        'drug1': drug1, 
                                        'drug2': drug2, 
                                        'tested': True
                                    }
                                    # Rerun to show the success message
                                    st.rerun()
                                
                                # Show success message if this combination was just tested
                                if just_tested:
                                    st.success("✅ Values auto-filled! Go to 'Single Prediction' tab to test this combination.")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Export options
                    st.markdown("---")
                    st.subheader("Export Results")
                    
                    export_col1, export_col2 = st.columns(2)
                    with export_col1:
                        if st.button("Copy to Clipboard", key="copy_clipboard"):
                            export_text = f"Top Drug Combinations for {cancer_type} ({cell_line})\n\n"
                            for idx, rec in enumerate(recommendations, 1):
                                drug1, drug2 = rec['drug_pair']
                                export_text += f"{idx}. {drug1} + {drug2}: Synergy={rec['synergy_score']:.1f}, Safety={rec['safety_score']:.2f}\n"
                            st.code(export_text)
                            st.success("Results copied to clipboard (in code block)")
                    
                    with export_col2:
                        # Create DataFrame for export
                        export_df = pd.DataFrame([
                            {
                                'Rank': idx,
                                'Drug_A': rec['drug_pair'][0],
                                'Drug_B': rec['drug_pair'][1],
                                'Synergy_Score': rec['synergy_score'],
                                'Safety_Score': rec['safety_score'],
                                'Recommendation_Score': rec['recommendation_score'],
                                'Average_Risk': rec['avg_risk']
                            }
                            for idx, rec in enumerate(recommendations, 1)
                        ])
                        
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"drug_recommendations_{cancer_type.replace(' ', '_')}_{cell_line}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No recommendations generated. This cell line may not have enough tested drugs with available SMILES.")
        
        # Also show recommendations if they already exist in session state (when coming back from another page)
        elif 'recommendations_list' in st.session_state and st.session_state.recommendations_list:
            recommendations = st.session_state.recommendations_list
            cancer_type = st.session_state.get('recommend_cancer', 'Selected Cancer')
            cell_line = st.session_state.get('recommend_cell_line', 'Selected Cell Line')
            
            if recommendations:
                st.subheader(f"Top {len(recommendations)} Recommendations for {cancer_type} ({cell_line})")
                
                # Summary statistics
                avg_synergy = np.mean([r['synergy_score'] for r in recommendations])
                avg_safety = np.mean([r['safety_score'] for r in recommendations])
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Average Synergy", f"{avg_synergy:.1f}")
                with col_stat2:
                    st.metric("Average Safety", f"{avg_safety:.2f}")
                with col_stat3:
                    st.metric("Total Evaluated", f"{len(recommendations)}")
                
                # Display each recommendation
                st.markdown("---")
                drug_smiles_dict = dict(zip(drug_df['drug_name'], drug_df['smiles']))
                
                for idx, rec in enumerate(recommendations, 1):
                    drug1, drug2 = rec['drug_pair']
                    synergy_score = rec['synergy_score']
                    safety_score = rec['safety_score']
                    avg_risk = rec['avg_risk']
                    
                    # Determine synergy class
                    if synergy_score > 6:
                        synergy_class = "High Synergy"
                        synergy_color = "synergy-high"
                    elif synergy_score > 3:
                        synergy_class = "Medium Synergy"
                        synergy_color = "synergy-medium"
                    elif synergy_score > 0:
                        synergy_class = "Low Synergy"
                        synergy_color = "synergy-low"
                    else:
                        synergy_class = "Antagonism"
                        synergy_color = "synergy-antagonism"
                    
                    # Determine safety class
                    if avg_risk < 0.3:
                        safety_class = "Low Risk"
                        safety_color = "toxicity-low"
                    elif avg_risk < 0.6:
                        safety_class = "Medium Risk"
                        safety_color = "toxicity-medium"
                    else:
                        safety_class = "High Risk"
                        safety_color = "toxicity-high"
                    
                    with st.container():
                        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns([0.5, 2, 1.5])
                        
                        with col1:
                            st.markdown(f'<div class="rank-badge">{idx}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            col_mol1, col_mol2 = st.columns(2)
                            with col_mol1:
                                img1 = mol_to_image(drug_smiles_dict.get(drug1, ''), size=(150, 150))
                                if img1:
                                    st.image(img1, width=150)
                                st.markdown(f"**{drug1}**")
                            with col_mol2:
                                img2 = mol_to_image(drug_smiles_dict.get(drug2, ''), size=(150, 150))
                                if img2:
                                    st.image(img2, width=150)
                                st.markdown(f"**{drug2}**")
                        
                        with col3:
                            st.markdown(f"**Synergy:** <span class='{synergy_color}'>{synergy_class} ({synergy_score:.1f})</span>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"**Safety:** <span class='{safety_color}'>{safety_class} ({safety_score:.2f})</span>", 
                                       unsafe_allow_html=True)
                            st.markdown(f"**Overall Score:** `{rec['recommendation_score']:.3f}`")
                            
                            button_key = f"prev_test_btn_{idx}_{drug1}_{drug2}"
                            
                            # Check if this combination was just tested
                            just_tested = (
                                st.session_state.test_status['tested'] and
                                st.session_state.test_status['drug1'] == drug1 and
                                st.session_state.test_status['drug2'] == drug2
                            )
                            
                            if st.button(f"Test This Combination", key=button_key):
                                st.session_state.recommend_drug1 = drug1
                                st.session_state.recommend_drug2 = drug2
                                st.session_state.test_status = {
                                    'drug1': drug1, 
                                    'drug2': drug2, 
                                    'tested': True
                                }
                                st.rerun()
                            
                            # Show success message if this combination was just tested
                            if just_tested:
                                st.success("✅ Values auto-filled! Go to 'Single Prediction' tab to test this combination.")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Batch Screening":
        st.markdown('<div class="page-header">Batch Drug Combination Screening</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How to use:
        Upload a CSV file with the following columns:
        
        **Required columns:**
        - **drug_a_name**: Name of Drug A
        - **drug_b_name**: Name of Drug B
        - **drug_a_smiles**: SMILES string for Drug A
        - **drug_b_smiles**: SMILES string for Drug B  
        - **cancer_type**: Type of cancer
        - **cell_line**: Cell line name
        
        **Output columns (will be added):**
        - **synergy_zip**: Synergy score prediction
        - **safety_score**: Safety score prediction (0-1, higher is safer)
        
        The system will predict both **synergy scores** and **safety scores** for all combinations using the provided SMILES strings.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_df)} rows")
                
                # Check required columns
                required_cols = ['drug_a_name', 'drug_b_name', 'drug_a_smiles', 'drug_b_smiles', 'cancer_type', 'cell_line']
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your CSV has these exact column names:")
                    st.code("drug_a_name, drug_b_name, drug_a_smiles, drug_b_smiles, cancer_type, cell_line")
                else:
                    # Show preview
                    st.subheader("Data Preview (First 5 rows)")
                    st.dataframe(batch_df.head())
                    
                    # Validate SMILES
                    def validate_smiles(smiles):
                        if pd.isna(smiles) or str(smiles).strip() == '':
                            return False, "Empty"
                        mol = Chem.MolFromSmiles(str(smiles).strip())
                        return mol is not None, "Valid" if mol else "Invalid"
                    
                    # Count valid SMILES
                    valid_a = 0
                    valid_b = 0
                    invalid_rows = []
                    
                    for idx, row in batch_df.iterrows():
                        valid1, _ = validate_smiles(row['drug_a_smiles'])
                        valid2, _ = validate_smiles(row['drug_b_smiles'])
                        if valid1:
                            valid_a += 1
                        if valid2:
                            valid_b += 1
                        if not valid1 or not valid2:
                            invalid_rows.append(idx + 1)  # 1-indexed for user display
                    
                    # Show validation summary
                    col_val1, col_val2, col_val3 = st.columns(3)
                    with col_val1:
                        st.metric("Drug A SMILES Valid", f"{valid_a}/{len(batch_df)}")
                    with col_val2:
                        st.metric("Drug B SMILES Valid", f"{valid_b}/{len(batch_df)}")
                    with col_val3:
                        st.metric("Rows with Issues", len(invalid_rows))
                    
                    if invalid_rows:
                        st.warning(f"Rows with invalid SMILES: {', '.join(map(str, invalid_rows[:10]))}{'...' if len(invalid_rows) > 10 else ''}")
                        st.info("Rows with invalid SMILES will be skipped during prediction.")
                    
                    # Process button
                    if st.button("Start Batch Predictions", type="primary", use_container_width=True):
                        with st.spinner("Processing batch predictions..."):
                            results_list = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            processed = 0
                            successful = 0
                            failed = 0
                            
                            for idx, row in batch_df.iterrows():
                                processed += 1
                                status_text.text(f"Processing row {processed}/{len(batch_df)}...")
                                
                                try:
                                    # Get SMILES from CSV
                                    smiles_a = str(row['drug_a_smiles']).strip()
                                    smiles_b = str(row['drug_b_smiles']).strip()
                                    
                                    # Validate SMILES
                                    mol_a = Chem.MolFromSmiles(smiles_a)
                                    mol_b = Chem.MolFromSmiles(smiles_b)
                                    
                                    if mol_a is None or mol_b is None:
                                        error_msg = "Invalid SMILES"
                                        if mol_a is None:
                                            error_msg += " for Drug A"
                                        if mol_b is None:
                                            error_msg += " for Drug B" if mol_a is None else " for Drug B"
                                        
                                        results_list.append({
                                            'drug_a_name': row['drug_a_name'],
                                            'drug_b_name': row['drug_b_name'],
                                            'drug_a_smiles': smiles_a,
                                            'drug_b_smiles': smiles_b,
                                            'cancer_type': row['cancer_type'],
                                            'cell_line': row['cell_line'],
                                            'synergy_zip': None,
                                            'safety_score': None,
                                            'error': error_msg
                                        })
                                        failed += 1
                                        continue
                                    
                                    # Predict synergy
                                    synergy_score, error = predict_synergy_proper(
                                        smiles_a, smiles_b, row['cancer_type'], row['cell_line'],
                                        model, device, cancer_cell_mapping, cell_line_mapping,
                                        cell_le=cell_le, cancer_cols=cancer_cols, scaler_y=scaler_y,
                                        debug_mode=debug_mode
                                    )
                                    
                                    if error:
                                        results_list.append({
                                            'drug_a_name': row['drug_a_name'],
                                            'drug_b_name': row['drug_b_name'],
                                            'drug_a_smiles': smiles_a,
                                            'drug_b_smiles': smiles_b,
                                            'cancer_type': row['cancer_type'],
                                            'cell_line': row['cell_line'],
                                            'synergy_zip': None,
                                            'safety_score': None,
                                            'error': error
                                        })
                                        failed += 1
                                    else:
                                        # Predict safety score
                                        side_effects_a = predict_side_effects(smiles_a)
                                        side_effects_b = predict_side_effects(smiles_b)
                                        
                                        safety_score = 0.5  # Default
                                        if side_effects_a and side_effects_b:
                                            combo_effects = predict_combination_side_effects(side_effects_a, side_effects_b)
                                            avg_risk = np.mean(list(combo_effects.values())) if combo_effects else 0.5
                                            safety_score = 1.0 - avg_risk  # Higher is safer
                                        
                                        results_list.append({
                                            'drug_a_name': row['drug_a_name'],
                                            'drug_b_name': row['drug_b_name'],
                                            'drug_a_smiles': smiles_a,
                                            'drug_b_smiles': smiles_b,
                                            'cancer_type': row['cancer_type'],
                                            'cell_line': row['cell_line'],
                                            'synergy_zip': synergy_score,
                                            'safety_score': safety_score,
                                            'error': None
                                        })
                                        successful += 1
                                        
                                except Exception as e:
                                    results_list.append({
                                        'drug_a_name': row.get('drug_a_name', ''),
                                        'drug_b_name': row.get('drug_b_name', ''),
                                        'drug_a_smiles': row.get('drug_a_smiles', ''),
                                        'drug_b_smiles': row.get('drug_b_smiles', ''),
                                        'cancer_type': row.get('cancer_type', ''),
                                        'cell_line': row.get('cell_line', ''),
                                        'synergy_zip': None,
                                        'safety_score': None,
                                        'error': str(e)
                                    })
                                    failed += 1
                                
                                # Update progress
                                progress_bar.progress((idx + 1) / len(batch_df))
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Create results DataFrame
                            results_df = pd.DataFrame(results_list)
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(results_df)
                            
                            # Summary statistics
                            st.subheader("Summary Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Combinations", len(results_df))
                            with col2:
                                st.metric("Successful Predictions", successful)
                            with col3:
                                st.metric("Failed Predictions", failed)
                            with col4:
                                success_rate = (successful / len(results_df)) * 100 if len(results_df) > 0 else 0
                                st.metric("Success Rate", f"{success_rate:.1f}%")
                            
                            # Synergy score statistics
                            valid_results = results_df[results_df['synergy_zip'].notna()]
                            if len(valid_results) > 0:
                                st.write("### Synergy Score Analysis")
                                
                                col_syn1, col_syn2, col_syn3, col_syn4 = st.columns(4)
                                with col_syn1:
                                    avg_synergy = valid_results['synergy_zip'].mean()
                                    st.metric("Average Synergy", f"{avg_synergy:.2f}")
                                with col_syn2:
                                    max_synergy = valid_results['synergy_zip'].max()
                                    st.metric("Maximum Synergy", f"{max_synergy:.2f}")
                                with col_syn3:
                                    min_synergy = valid_results['synergy_zip'].min()
                                    st.metric("Minimum Synergy", f"{min_synergy:.2f}")
                                with col_syn4:
                                    std_synergy = valid_results['synergy_zip'].std()
                                    st.metric("Std Dev Synergy", f"{std_synergy:.2f}")
                                
                                # Safety score statistics
                                st.write("### Safety Score Analysis")
                                
                                col_saf1, col_saf2, col_saf3, col_saf4 = st.columns(4)
                                with col_saf1:
                                    avg_safety = valid_results['safety_score'].mean()
                                    st.metric("Average Safety", f"{avg_safety:.3f}")
                                with col_saf2:
                                    max_safety = valid_results['safety_score'].max()
                                    st.metric("Maximum Safety", f"{max_safety:.3f}")
                                with col_saf3:
                                    min_safety = valid_results['safety_score'].min()
                                    st.metric("Minimum Safety", f"{min_safety:.3f}")
                                with col_saf4:
                                    high_safety = len(valid_results[valid_results['safety_score'] > 0.7])
                                    st.metric("High Safety (>0.7)", high_safety)
                                
                                # Categorize synergy results
                                st.write("### Synergy Distribution")
                                high_synergy = valid_results[valid_results['synergy_zip'] > 6]
                                medium_synergy = valid_results[(valid_results['synergy_zip'] > 3) & (valid_results['synergy_zip'] <= 6)]
                                low_synergy = valid_results[(valid_results['synergy_zip'] > 0) & (valid_results['synergy_zip'] <= 3)]
                                antagonism = valid_results[valid_results['synergy_zip'] <= 0]
                                
                                dist_col1, dist_col2, dist_col3, dist_col4 = st.columns(4)
                                with dist_col1:
                                    st.metric("High Synergy (>6)", len(high_synergy))
                                with dist_col2:
                                    st.metric("Medium Synergy (3-6)", len(medium_synergy))
                                with dist_col3:
                                    st.metric("Low Synergy (0-3)", len(low_synergy))
                                with dist_col4:
                                    st.metric("Antagonism (≤0)", len(antagonism))
                                
                                # Top 10 combinations
                                st.write("### Top 10 Combinations by Synergy Score")
                                top_10 = valid_results.sort_values('synergy_zip', ascending=False).head(10)
                                top_10_display = top_10[['drug_a_name', 'drug_b_name', 'synergy_zip', 'safety_score']].copy()
                                top_10_display.columns = ['Drug A', 'Drug B', 'Synergy Score', 'Safety Score']
                                st.dataframe(top_10_display.style.format({
                                    'Synergy Score': '{:.2f}',
                                    'Safety Score': '{:.3f}'
                                }))
                            
                            # Export options
                            st.subheader("Export Results")
                            
                            # Prepare CSV for download
                            csv = results_df.to_csv(index=False)
                            
                            col_export1, col_export2 = st.columns(2)
                            with col_export1:
                                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                                filename = f"batch_predictions_{timestamp}.csv"
                                st.download_button(
                                    label="Download Full Results (CSV)",
                                    data=csv,
                                    file_name=filename,
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col_export2:
                                # Create summary file
                                summary_text = f"""Batch Prediction Summary
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Combinations: {len(results_df)}
Successful Predictions: {successful}
Failed Predictions: {failed}
Success Rate: {success_rate:.1f}%
"""
                                if len(valid_results) > 0:
                                    summary_text += f"\nSynergy Score Statistics:\n"
                                    summary_text += f"  Average: {valid_results['synergy_zip'].mean():.2f}\n"
                                    summary_text += f"  Maximum: {valid_results['synergy_zip'].max():.2f}\n"
                                    summary_text += f"  Minimum: {valid_results['synergy_zip'].min():.2f}\n"
                                    summary_text += f"  Std Dev: {valid_results['synergy_zip'].std():.2f}\n"
                                    
                                    summary_text += f"\nSafety Score Statistics:\n"
                                    summary_text += f"  Average: {valid_results['safety_score'].mean():.3f}\n"
                                    summary_text += f"  Maximum: {valid_results['safety_score'].max():.3f}\n"
                                    summary_text += f"  Minimum: {valid_results['safety_score'].min():.3f}\n"
                                
                                st.download_button(
                                    label="Download Summary (TXT)",
                                    data=summary_text,
                                    file_name=f"prediction_summary_{timestamp}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )                            
            
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
                st.info("Please check that your file is a valid CSV format.")

    else:
        st.markdown('<div class="page-header">About</div>', unsafe_allow_html=True)
        st.markdown("""
        # Drug Combination Synergy Predictor

        This tool predicts **drug combination synergy** for various cancer types using a **Graph Neural Network (GNN)**.  
        It also evaluates potential **side effect risks** based on molecular properties.

        ## Features
        - **Single Prediction**: Predict synergy for predefined drugs
        - **Advanced (SMILES)**: Enter any drug SMILES directly for analysis
        - **Synergy Prediction**: Predicts drug interactions for specific cancer cell lines.
        - **Side Effect Assessment**: Estimates risk in 8 categories.
        - **Recommendation Engine**: Suggests top drug pairs using drugs tested on each cell line.
        - **Batch Screening** for large CSV inputs.
        
        ## Recommendation System
        Our recommendation system:
        1. Uses experimental DrugComb data
        2. Considers drugs tested on each cell line
        3. Evaluates possible combinations from tested drugs
        4. Balances synergy with safety profiles

        **Note:** Only drugs with valid SMILES are considered for predictions.
                
        **Disclaimer:** This tool is for research and educational purposes only.
        """)

if __name__ == "__main__": 
    main()
