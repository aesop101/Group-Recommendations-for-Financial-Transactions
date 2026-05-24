import sys
import os
import torch
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import DataLoader
from models.cbn import CollaborativeBayesianNetwork, train_cbn
from models.cbn_new import ImprovedCBN, compute_sector_popularity, train_improved_cbn
from models.game import GroupRecommendationGame
from utils.metrics import Metrics
from utils.sector_embeddings import (compute_sector_embeddings,
                                     sector_similarity_matrix,
                                     form_groups_by_sector_affinity)

def run_pipeline(dataset='far_trans', use_improved_cbn=True):
    print(f"--- Starting SAIoT-GR Pipeline [{dataset}] "
          f"[{'ImprovedCBN' if use_improved_cbn else 'BaseCBN'}] ---")

    loader = DataLoader()
    sector_names = None

    if dataset == 'far_trans':
        data_path = os.path.join(os.getcwd(), 'data', 'far_trans')
        print(f"Loading FAR-Trans dataset from {data_path} ...")
        interactions, num_users, num_topics, sector_names = loader.load_far_trans(data_path)
        print(f"  {num_users:,} investors | {num_topics} sectors: {', '.join(sector_names)}")
        print(f"  {len(interactions):,} interactions (positive + negative)")

    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    print("Stage 1: Collaborative Inference via CBN...")
    if use_improved_cbn:
        print("Computing sector popularity prior for ImprovedCBN...")
        sector_popularity = compute_sector_popularity(interactions, num_users, num_topics)
        cbn_model = ImprovedCBN(num_users, num_topics, sector_popularity)
        trained_model = train_improved_cbn(cbn_model, interactions, num_epochs=50)
    else:
        cbn_model = CollaborativeBayesianNetwork(num_users, num_topics)
        trained_model = train_cbn(cbn_model, interactions, num_epochs=50)

    I_learned = trained_model.I.detach().numpy()
    S_learned = trained_model.S.detach().numpy()

    def normalize(matrix):
        m_min, m_max = matrix.min(), matrix.max()
        if m_max == m_min:
            return np.zeros_like(matrix)
        return (matrix - m_min) / (m_max - m_min)

    I_norm = normalize(I_learned)
    S_norm = normalize(S_learned)

    print("Stage 2: Group Recommendation via Non-cooperative Game...")
    game = GroupRecommendationGame(eta1=0.6, eta2=0.4)

    group_size = 5
    n_groups = 10
    rng = np.random.default_rng(42)

    print("Embedding sectors to compute inter-sector closeness...")
    sector_embs = compute_sector_embeddings(sector_names)
    sim_matrix = sector_similarity_matrix(sector_embs)
    print("  Sector similarity matrix (13×13) computed.")
    if sector_names:
        closest = {sector_names[i]: sector_names[np.argsort(sim_matrix[i])[-2]]
                   for i in range(len(sector_names))}
        print("  Closest sector pairs:")
        for s, neighbour in closest.items():
            print(f"    {s:30s} ↔ {neighbour}")

    print("Forming groups by sector affinity (diverse sector clusters per group)...")
    groups, sector_cluster_labels, _ = form_groups_by_sector_affinity(
        interactions, num_users, num_topics, sector_embs,
        group_size=group_size, n_groups=n_groups, rng=rng)
    if sector_names:
        cluster_summary = {}
        for s_idx, c in enumerate(sector_cluster_labels):
            cluster_summary.setdefault(int(c), []).append(sector_names[s_idx])
        print("  Sector clusters:")
        for c, sectors in sorted(cluster_summary.items()):
            print(f"    Cluster {c}: {', '.join(sectors)}")

    all_metrics = {k: [] for k in Metrics.calculate_all(
        np.zeros(num_topics), np.zeros(num_topics)).keys()}

    for g_idx, group in enumerate(groups):
        optimal_strategies = game.recommend_items(group, num_topics, I_norm, S_norm)

        if sector_names:
            print(f"\nGroup {g_idx + 1} — investors {group}:")
            for uid, topic in zip(group, optimal_strategies):
                print(f"  Investor {uid:>5}: {sector_names[topic]} (sector {topic})")
        else:
            print(f"Group {g_idx + 1} strategies: {optimal_strategies}")

        target_interest = np.mean(I_norm[group], axis=0)
        predicted_interest = np.zeros(num_topics)
        for topic in optimal_strategies:
            predicted_interest[topic] += 1
        predicted_interest /= len(optimal_strategies)

        metrics = Metrics.calculate_all(predicted_interest, target_interest)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    print("\n--- Results (averaged over {} groups) ---".format(n_groups))
    for metric, values in all_metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")

if __name__ == "__main__":
    run_pipeline(dataset='far_trans', use_improved_cbn=True)
