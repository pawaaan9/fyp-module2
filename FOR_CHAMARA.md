# Module 1 → Module 2 handover (3 things)

Module 2 is already built and tested against a placeholder scorer.
To swap in the real model, only these three are needed:

### 1. Save a checkpoint
```python
import torch
torch.save({
    "state_dict":      model.state_dict(),
    "arch":            {"node_dim": 6, "hidden_dim": 32, "heads": 4},
    "feature_version": "1.0",
    "val_auc":         0.63,          # whatever it actually is
}, "models/module1_gat.pt")
```

### 2. Fill in `pitchpulse/graph.py`
Two functions, both currently `NotImplementedError`:
- `build_graph(scenario) -> Data`
- `load_model(path, device) -> nn.Module`

`build_graph` must be **pure**: everything it needs comes from `scenario`,
nothing from notebook globals. Module 2 calls it ~121 times per optimisation.

### 3. Train through the same function
The training notebook must build its graphs via
`contract.from_record(record)` → `graph.build_graph(scenario)`.

This is the important one. Module 1 currently has four different graph
builders (notebook cells 28, 44, 54, 58). If training uses one and the
dashboard uses another, the coach sees a score the model was never
trained to produce — and it fails silently, which is the worst kind.

---

Once these land, Module 2 needs no code changes. `get_scorer()` detects the
checkpoint automatically and prints which scorer is active.
