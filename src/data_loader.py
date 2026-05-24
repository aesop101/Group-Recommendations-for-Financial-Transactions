import os
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir

    def load_far_trans(self, data_dir):
        """
        Loads the FAR-Trans financial asset recommendation dataset.

        Download from: https://doi.org/10.5525/gla.researchdata.1658
        user  → investor (customerID)
        item  → financial asset (ISIN)
        topic → market sector (sector column in asset_information.csv)
        B=1   → Buy transaction (positive implicit feedback)
        B=0   → randomly sampled unobserved (user, sector) pair
        """
        tx_path = os.path.join(data_dir, 'transactions.csv')
        asset_path = os.path.join(data_dir, 'asset_information.csv')

        for p in (tx_path, asset_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found.\n"
                    "Download FAR-Trans from https://doi.org/10.5525/gla.researchdata.1658 "
                    f"and place transactions.csv and asset_information.csv in {data_dir}/"
                )

        tx_df = pd.read_csv(tx_path)
        asset_df = pd.read_csv(asset_path)

        tx_df.columns = tx_df.columns.str.lower().str.strip()
        asset_df.columns = asset_df.columns.str.lower().str.strip()

        cust_col = self._find_col(tx_df,    ['customerid', 'customer_id', 'client_id', 'user_id'])
        isin_tx = self._find_col(tx_df,    ['isin', 'asset_id', 'assetid', 'item_id', 'security_id'])
        isin_asset = self._find_col(asset_df, ['isin', 'asset_id', 'assetid', 'item_id', 'security_id'])
        type_col = self._find_col(tx_df,    ['transactiontype', 'transaction_type', 'type', 'order_type', 'operation'])
        sector_col = self._find_col(asset_df, ['sector', 'gics_sector', 'category', 'asset_category'])

        buys = tx_df[tx_df[type_col].str.lower().isin(['buy', 'b', 'acquisition', 'purchase'])].copy()

        asset_meta = asset_df[[isin_asset, sector_col]].drop_duplicates(subset=isin_asset)
        merged = buys.merge(asset_meta, left_on=isin_tx, right_on=isin_asset, how='inner')
        merged = merged.dropna(subset=[sector_col])

        if merged.empty:
            raise RuntimeError(
                "No transactions remain after joining with asset sectors. "
                "Check that the sector column is populated and column names match."
            )

        uid_map = {uid: i for i, uid in enumerate(sorted(merged[cust_col].unique()))}
        sector_names = sorted(merged[sector_col].unique())
        sector_map = {s: i for i, s in enumerate(sector_names)}

        merged['u'] = merged[cust_col].map(uid_map)
        merged['d'] = merged[sector_col].map(sector_map)

        num_users = len(uid_map)
        num_sectors = len(sector_names)

        user_total = merged.groupby('u').size().reset_index(name='total')
        user_sector_cnts = merged.groupby(['u', 'd']).size().reset_index(name='count_ud')
        merged = merged.merge(user_total, on='u').merge(user_sector_cnts, on=['u', 'd'])
        merged['pi'] = merged['count_ud'] / merged['total']

        interactions = list(zip(
            merged['u'].astype(int).tolist(),
            merged['d'].astype(int).tolist(),
            merged['pi'].astype(float).tolist(),
            [1.0] * len(merged),
        ))

        rng = np.random.default_rng(42)
        neg_u = rng.integers(0, num_users,   len(interactions))
        neg_d = rng.integers(0, num_sectors, len(interactions))
        total_lookup = user_total.set_index('u')['total'].to_dict()
        sec_lookup = user_sector_cnts.set_index(['u', 'd'])['count_ud'].to_dict()

        for u_val, d_val in zip(neg_u.tolist(), neg_d.tolist()):
            total = total_lookup.get(u_val, 0)
            count_ud = sec_lookup.get((u_val, d_val), 0)
            pi_val = count_ud / total if total > 0 else 0.0
            interactions.append((u_val, d_val, pi_val, 0.0))

        return interactions, num_users, num_sectors, sector_names

    def _find_col(self, df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(
            f"None of {candidates} found in DataFrame columns: {list(df.columns)}\n"
            "Check that the CSV uses one of the expected column names."
        )
