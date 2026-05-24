# FAR-Trans Data

1. Go to [doi.org/10.5525/gla.researchdata.1658](https://doi.org/10.5525/gla.researchdata.1658)
2. Download the dataset archive
3. Extract and place the files here so the directory contains at minimum:

```
data/far_trans/
├── transactions.csv       ← buy/sell records per investor (required)
├── asset_information.csv  ← ISIN → sector mapping (required)
├── close_prices.csv       ← daily close prices per asset (optional)
├── customer_information.csv
├── limit_prices.csv
├── markets.csv
└── questionnaires.csv
```

Only `transactions.csv` and `asset_information.csv` are required to run the pipeline.

## Citation

> Javier Sanz-Cruzado, Nikolaos Droukas, Richard McCreadie.
> *FAR-Trans: An Investment Dataset for Financial Asset Recommendation.*
> IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys). Jeju, South Korea, August 2024.