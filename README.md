Synerlytics: Drug Combination Synergy Predictor
A Graph Neural Network-based system for predicting synergistic effects of drug combinations in cancer therapy. The model is trained on the DrugComb dataset and achieves state-of-the-art performance with R² = 0.6137 and MAE = 3.84 (10.7% of data range).

Live Demo
🔗 Try the live application: https://drug-combination-prediction-zptjtqn7ym9usoalhkf2gn.streamlit.app/

(Note: The app may take a few seconds to wake up if inactive)

Overview
Synerlytics predicts synergy scores (ZIP scale) for drug pairs across different cancer types and cell lines. Each drug is represented as a molecular graph, processed through a dual GNN architecture, combined with cancer context and cell line embeddings to output a continuous synergy score.

Key Features
Single Prediction Mode: Select drugs from a curated database, visualize molecular structures, and get synergy scores with clinical interpretations

Advanced SMILES Mode: Input custom SMILES strings for novel drug combinations not in the database

Recommendation Engine: Suggests top drug pairs for specific cancer types and cell lines based on experimentally validated combinations

Batch Screening: Upload CSV files with multiple drug combinations for high-throughput screening

Side Effect Assessment: Predicts toxicity risks across 8 categories for individual drugs and combinations

Model Architecture
Graph Neural Network Components
Graph Encoder: 4-layer GINEConv (Graph Isomorphism Network with Edge features)

Node Features: 5-dimensional (atomic number, hydrogen count, degree, valence, aromaticity)

Edge Features: 1-dimensional (bond type encoding: single=1.0, double=2.0, triple=3.0, aromatic=1.5)

Drug Embeddings: 192-dimensional vectors per drug via global mean pooling

Context Integration: Cancer type one-hot encoding (17-dim) + Cell line embeddings (48-dim, vocabulary size 288)

Prediction Head: MLP with architecture 449 → 512 → 256 → 1, dropout 0.3

Training Configuration
Loss Function: Mean Squared Error (MSE)

Optimizer: AdamW (learning rate 1e-3, weight decay 1e-5)

Scheduler: CosineAnnealingWarmRestarts (T_0=10)

Batch Size: 128

Gradient Clipping: max_norm=1.0

Mixed Precision: Automatic Mixed Precision (AMP) for GPU acceleration

Early Stopping: Patience of 6 epochs

Data Split: 80% training, 20% testing

Performance Metrics
text
Dataset Characteristics:
- Total samples after symmetry augmentation: ~500,000
- Data Range: 36.05 synergy points (ZIP scale)

Model Performance (Test Set):
- MAE: 3.84 (10.7% of data range)
- RMSE: 5.31 (14.7% of data range)
- R²: 0.6137

Side Effect Prediction
The system includes a rule-based toxicity assessment module that evaluates:

Hepatotoxicity

Nephrotoxicity

Cardiotoxicity

Neurotoxicity

Gastrointestinal effects

Dermatological effects

CNS effects

Metabolic issues

Each toxicity category is scored 0-1 based on SMARTS pattern matching and molecular descriptors (molecular weight, LogP, TPSA, hydrogen bond donors/acceptors, rotatable bonds). Combination risk is calculated as: Risk_combined = 1 - (1 - Risk_A) × (1 - Risk_B)

Data Sources
Training Data: DrugComb dataset (drug pairs, cancer types, cell lines, synergy ZIP scores)

Drug Database: Comprehensive drug SMILES catalog with 50+ anticancer drugs

Cell Line Mappings: Experimentally validated drug-cell line associations from literature

