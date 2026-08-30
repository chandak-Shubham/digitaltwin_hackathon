# Model Architectures for Vehicle Assembly Anomaly Detection

## Input Data Format
- **Vehicle Sequences**: Each vehicle is a sequence of 10 stations
- **Features per Station**: 13 numeric (sensor readings) + 5 categorical (metadata)
- **Shape**: `(num_vehicles, 10_stations, 18_features)`
- **Output**: Binary label (anomalous=1, normal=0)

---

## 1. LightGBM (Gradient Boosting Trees)

### Architecture Flow
```
Vehicle Sequences (10 stations × 18 features)
           ↓
    Flatten to Row-Level
    (concatenate all 10 stations)
           ↓
    Single Feature Vector (180 features)
           ↓
    ┌─────────────────────────────┐
    │  Gradient Boosting Ensemble │
    │  (Sequential Tree Addition)  │
    │                              │
    │  Tree 1 → Residuals         │
    │  Tree 2 → Residuals         │
    │  ...                         │
    │  Tree N → Final Prediction  │
    └─────────────────────────────┘
           ↓
    Classification Score (0–1)
```

### Key Characteristics
- **Tabular Data**: Works directly on flattened row features
- **Tree-based**: Ensemble of decision trees with gradient boosting
- **No Sequential Dependency**: Treats all 10 stations equally; order is implicit in feature names
- **Output**: Probability per vehicle (via averaging predictions across all rows)

---

## 2. LSTM (Long Short-Term Memory)

### Architecture Flow
```
Vehicle Sequence (10 stations × 18 features)
           ↓
    ┌──────────────────────────────────┐
    │    LSTM Layer Stack              │
    │  (1–2 layers, 32–64 hidden)      │
    │                                   │
    │  Input: [station_1, station_2... │
    │          ...station_10]           │
    │                                   │
    │  Hidden State Propagation:       │
    │  h₀ → LSTM → h₁ → LSTM → h₂ ... │
    │                       ... → h₁₀   │
    │                                   │
    │  Output: Sequence of hidden      │
    │  states, one per station         │
    └──────────────────────────────────┘
           ↓
    Final Hidden State (h₁₀)
    (captures sequential context)
           ↓
    ┌──────────────────────────────┐
    │  Dense Head (Linear Layer)    │
    │  hidden_dim → 1 output        │
    └──────────────────────────────┘
           ↓
    Raw Logit
           ↓
    Sigmoid Activation
           ↓
    Classification Score (0–1)
```

### Key Characteristics
- **Sequence-Aware**: Processes stations in order (S1 → S2 → ... → S10)
- **Memory**: LSTM cells maintain hidden states across time steps
- **Recurrent**: Each station's output depends on all previous stations
- **Output**: Single prediction based on final station (end of line context)

---

## 3. Transformer (Self-Attention)

### Architecture Flow
```
Vehicle Sequence (10 stations × 18 features)
           ↓
    ┌──────────────────────────────────┐
    │  Linear Projection Layer          │
    │  18 features → hidden_dim         │
    └──────────────────────────────────┘
           ↓
    ┌──────────────────────────────────────────────┐
    │    Transformer Encoder Stack                  │
    │  (1–2 layers, 2 attention heads, hidden×2)   │
    │                                               │
    │  For each Layer:                             │
    │    Multi-Head Self-Attention                 │
    │    • Queries from all stations                │
    │    • Keys/Values from all stations            │
    │    • Computes attention weights               │
    │    • Re-weights station features              │
    │                                               │
    │    Feed-Forward Network                      │
    │    • Dense transformation                     │
    │    • Non-linearity                            │
    │                                               │
    │  Output: Transformed sequence                │
    │  (same shape: 10 stations × hidden_dim)      │
    └──────────────────────────────────────────────┘
           ↓
    Final Layer Outputs (10 × hidden_dim)
           ↓
    Select Last Token (station_10)
           ↓
    ┌──────────────────────────────┐
    │  Dense Head (Linear Layer)    │
    │  hidden_dim → 1 output        │
    └──────────────────────────────┘
           ↓
    Raw Logit
           ↓
    Sigmoid Activation
           ↓
    Classification Score (0–1)
```

### Key Characteristics
- **Parallel Processing**: All stations attend to all other stations at once
- **Attention Mechanism**: Learns which stations are important for prediction
- **Position-Aware**: Implicitly preserves station order via positional encoding
- **Flexible Dependencies**: Can learn arbitrary dependencies between stations (not just left-to-right)
- **Output**: Single prediction based on learned attention over all stations

---

## Comparison Summary

| Aspect | LightGBM | LSTM | Transformer |
|--------|----------|------|-------------|
| **Input Shape** | Flattened (180,) | Sequence (10, 18) | Sequence (10, 18) |
| **Processing** | Parallel trees | Sequential | Parallel attention |
| **Station Dependencies** | Implicit (feature names) | Left-to-right only | All-to-all (learned) |
| **Computational Cost** | Low | Medium | Medium–High |
| **Interpretability** | Feature importance | Gate values | Attention weights |
| **Long-Range Context** | Built-in | Gradient dependent | Direct |

---

## Training & Evaluation Pipeline

For each dataset and model:

1. **Split**: Vehicles → 80% train, 20% test (stratified by anomaly label)
2. **Hyperparameter Tuning**: 
   - 3-fold stratified cross-validation on training vehicles only
   - Grid search over parameter combinations
   - Select best by mean CV AUC
3. **Final Training**: Refit best model on all training vehicles
4. **Test Evaluation**: Single held-out test set evaluation
5. **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

---

## Data Flow Diagram (High-Level)

```
┌─────────────────────────────────┐
│   Five Datasets                 │
│  (base, sparse, drift, ...)     │
└────────────┬────────────────────┘
             │
             ↓
    ┌────────────────────────┐
    │ Vehicle-Level Split    │
    │ (80% train, 20% test)  │
    └────────┬───────────────┘
             │
             ├─────────────────┬─────────────────┬──────────────────┐
             ↓                 ↓                 ↓                  ↓
        ┌────────────┐   ┌────────────┐   ┌────────────┐     
        │ LightGBM   │   │   LSTM     │   │Transformer │   
        │            │   │            │   │            │   
        │ Tuning:    │   │ Tuning:    │   │ Tuning:    │   
        │ CV 3-fold  │   │ CV 3-fold  │   │ CV 3-fold  │   
        │            │   │            │   │            │   
        │ Best Params│   │Best Params │   │Best Params │   
        └────┬───────┘   └────┬───────┘   └────┬───────┘
             │                 │                │
             └─────────┬───────┴────────┬───────┘
                       ↓
            ┌──────────────────────────┐
            │  Refit Best Configs      │
            │  on All Training Data    │
            └────────────┬─────────────┘
                         ↓
            ┌──────────────────────────┐
            │  Evaluate on Test Set    │
            │  (Report AUC, F1, etc.)  │
            └────────────┬─────────────┘
                         ↓
            ┌──────────────────────────┐
            │  Aggregate Results       │
            │  Across 5 Datasets       │
            │  Rank Models             │
            └──────────────────────────┘
```

---

## Notes for Visualization

- **Station Sequence**: Visualize 10 stations left-to-right (S1 → S2 → ... → S10)
- **Feature Bundle**: Show 18 features as a colored "feature vector" at each station
- **LightGBM**: Animate flattening into a single vector, then trees being applied sequentially
- **LSTM**: Show hidden state flowing left-to-right through cells, with gates opening/closing
- **Transformer**: Show attention heads as connections between all pairs of stations, highlighting which station pairs matter most
- **Output**: All models converge to a single binary decision (anomaly/normal)
