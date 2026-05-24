# Group Recommendations

A two-stage pipeline for recommending financial assets to groups of investors using only implicit feedback (purchase transactions — no ratings or explicit preferences).

Based on *"Secure Artificial Intelligence of Things for Implicit Group Recommendations"* (arXiv:2104.11699v1).

---

## Components

| File | Description |
|---|---|
| `src/data_loader.py` | FAR-Trans loader — maps investors → sectors, computes π(i,d), generates negative samples |
| `src/models/cbn.py` | BaseCBN — learns I(i,d) and S(i,d) via mini-batch SGD with Gaussian priors |
| `src/models/cbn_new.py` | ImprovedCBN — sector-popularity prior on S and I–S decorrelation loss |
| `src/models/game.py` | Non-cooperative game — Nash equilibrium |
| `src/utils/metrics.py` | Six evaluation metrics: EucDist, ManDist, CheDist, CorDist, MAEDist, MSEDist |
| `src/utils/sector_embeddings.py` | Sector name embeddings via `sentence-transformers`; sector-affinity group formation |
| `main.py` | End-to-end pipeline |
| `compare.py` | Comparison of BaseCBN vs ImprovedCBN on identical groups |

---

## ImprovedCBN extension

The base CBN learns I and S from the same BCE gradient, causing them to correlate — both encode the same interaction pattern rather than distinct factors.

**Fix:** two additions to the loss:

**Sector-popularity prior** 

> μ_S(d) = μ₂ + pop_scale · popularity(d)

**Decorrelation loss** — 

> L_decorr = mean_d [ corr(I[:,d], S[:,d])² ]

Full loss:

> L = BCE + λ_reg · (prior_I + prior_S) + λ_decorr · L_decorr

| File | Description |
|---|---|
| `src/models/cbn_new.py` | `ImprovedCBN`, `compute_sector_popularity`, `train_improved_cbn` |
| `src/utils/sector_embeddings.py` | `compute_sector_embeddings`, `form_groups_by_sector_affinity` |
| `compare.py` | Runs both models|

---

## Data

Download FAR-Trans from [doi.org/10.5525/gla.researchdata.1658](https://doi.org/10.5525/gla.researchdata.1658) and place the extracted files under `data/far_trans/` so that the following are present:

```
data/far_trans/transactions.csv
data/far_trans/asset_information.csv
```

18,813 investors · 13 market sectors · 183,079 buy transactions

---

## Quickstart

```bash
pip install -r requirements.txt

python main.py          # full pipeline with ImprovedCBN
python compare.py       # BaseCBN vs ImprovedCBN side-by-side
```

---