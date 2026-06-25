# Task sequence (dependency graph)

Issues are [#1–#18](https://github.com/lcubrilo/PRML-Projekat/issues). Arrows = "must finish before". GitHub renders the diagram below.

```mermaid
graph LR
  A2["A2 proposal ✅"]:::done
  B1["B1 load + 6-class label"] --> B2["B2 features (diffs/abs, encode)"]
  B2 --> B3["B3 symmetrize corners"]
  B3 --> B4["B4 chrono split + scale"]
  B2 --> C1["C1 EDA: distributions"]
  B2 --> C2["C2 EDA: red-corner confound"]
  B2 --> C3["C3 EDA: leakage check"]
  B4 --> D1["D1 baselines LDA/QDA/kNN"]
  B4 --> D2["D2 SAMME from scratch"]
  B1 --> D3["D3 naive baselines"]
  D1 --> D4["D4 odds benchmark"]
  D1 --> E1["E1 metrics"]
  D2 --> E1
  D1 --> E2["E2 hyperparam sweeps"]
  D2 --> E2
  D1 --> E3["E3 DR ablation"]
  E1 --> G1["G1 report"]
  D4 --> G1
  E2 --> G1
  E3 --> G1
  G1 --> G2["G2 slides (last)"]
  classDef done fill:#cfc,stroke:#393;
```

Critical path: **B1 → B2 → B3 → B4 → D1/D2 → E1/E2 → G1 → G2**.
EDA (C1–C3), D3 (naive), and D4 (odds) hang off earlier nodes and can run in parallel.
