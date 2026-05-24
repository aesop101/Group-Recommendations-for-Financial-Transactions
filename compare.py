import sys
import os
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

GROUP_SIZE = 5
N_GROUPS = 10
SEED = 42
NUM_EPOCHS = 50


def normalize(matrix):
    m_min, m_max = matrix.min(), matrix.max()
    if m_max == m_min:
        return np.zeros_like(matrix)
    return (matrix - m_min) / (m_max - m_min)


def compute_pi_avg(interactions, num_users, num_topics):
    pi_sum = np.zeros((num_users, num_topics), dtype=np.float32)
    pi_cnt = np.zeros((num_users, num_topics), dtype=np.float32)
    for u, d, pi, b in interactions:
        if b == 1.0:
            pi_sum[u, d] += pi
            pi_cnt[u, d] += 1.0
    return np.where(pi_cnt > 0, pi_sum / np.maximum(pi_cnt, 1.0), 0.0)


def evaluate_model(trained_model, groups, num_topics, game, pi_avg):
    I_norm = normalize(trained_model.I.detach().numpy())
    S_norm = normalize(trained_model.S.detach().numpy())

    all_metrics = {k: [] for k in Metrics.calculate_all(
        np.zeros(num_topics), np.zeros(num_topics)).keys()}

    for group in groups:
        strategies = game.recommend_items(group, num_topics, I_norm, S_norm)

        target = np.mean(pi_avg[group], axis=0)
        predicted = np.zeros(num_topics)
        for t in strategies:
            predicted[t] += 1
        predicted /= len(strategies)

        m = Metrics.calculate_all(predicted, target)
        for k, v in m.items():
            all_metrics[k].append(v)

    return {k: np.mean(v) for k, v in all_metrics.items()}


def print_comparison(base_results, improved_results):
    col_w = 14
    header = (f"{'Metric':<10}"
              f"{'BaseCBN':>{col_w}}"
              f"{'ImprovedCBN':>{col_w}}"
              f"{'Δ (imp-base)':>{col_w}}"
              f"  Winner")
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    wins = {"BaseCBN": 0, "ImprovedCBN": 0, "Tie": 0}
    for metric in base_results:
        base_val = base_results[metric]
        imp_val  = improved_results[metric]
        delta    = imp_val - base_val         

        if abs(delta) < 1e-6:
            winner = "Tie"
            wins["Tie"] += 1
        elif delta < 0:
            winner = "ImprovedCBN"
            wins["ImprovedCBN"] += 1
        else:
            winner = "BaseCBN"
            wins["BaseCBN"] += 1

        print(f"{metric:<10}"
              f"{base_val:>{col_w}.4f}"
              f"{imp_val:>{col_w}.4f}"
              f"{delta:>{col_w}.4f}"
              f"  {winner}")

    print(sep)
    print(f"Wins — BaseCBN: {wins['BaseCBN']}  "
          f"ImprovedCBN: {wins['ImprovedCBN']}  "
          f"Ties: {wins['Tie']}")
    print(sep)


def main():
    print("=" * 60)
    print("Loading FAR-Trans dataset...")
    data_path = os.path.join(os.getcwd(), 'data', 'far_trans')
    loader = DataLoader()
    interactions, num_users, num_topics, sector_names = loader.load_far_trans(data_path)
    print(f"  {num_users:,} investors | {num_topics} sectors | "
          f"{len(interactions):,} interactions")

    print("\nEmbedding sectors for group formation...")
    sector_embs = compute_sector_embeddings(sector_names)
    sim_matrix  = sector_similarity_matrix(sector_embs)

    print("  Sector similarity (closest neighbour per sector):")
    for i, name in enumerate(sector_names):
        sims = sim_matrix[i].copy()
        sims[i] = -1
        neighbour = sector_names[int(np.argmax(sims))]
        print(f"    {name:<30} ↔ {neighbour}")

    rng = np.random.default_rng(SEED)
    groups, sector_cluster_labels, _ = form_groups_by_sector_affinity(
        interactions, num_users, num_topics, sector_embs,
        group_size=GROUP_SIZE, n_groups=N_GROUPS, rng=rng)

    print(f"\n  {N_GROUPS} groups of {GROUP_SIZE} investors formed "
          f"(sector-affinity clustering).")
    cluster_summary = {}
    for s_idx, c in enumerate(sector_cluster_labels):
        cluster_summary.setdefault(int(c), []).append(sector_names[s_idx])
    for c, sectors in sorted(cluster_summary.items()):
        print(f"    Cluster {c}: {', '.join(sectors)}")

    game = GroupRecommendationGame(eta1=0.6, eta2=0.4)
    pi_avg = compute_pi_avg(interactions, num_users, num_topics)

    print("\n" + "=" * 60)
    print("Training BaseCBN...")
    base_model = CollaborativeBayesianNetwork(num_users, num_topics)
    base_model = train_cbn(base_model, interactions, num_epochs=NUM_EPOCHS)
    print("Evaluating BaseCBN...")
    base_results = evaluate_model(base_model, groups, num_topics, game, pi_avg)

    print("\n" + "=" * 60)
    print("Computing sector popularity for ImprovedCBN prior...")
    sector_popularity = compute_sector_popularity(interactions, num_users, num_topics)
    print("Training ImprovedCBN...")
    improved_model = ImprovedCBN(num_users, num_topics, sector_popularity)
    improved_model = train_improved_cbn(improved_model, interactions,
                                        num_epochs=NUM_EPOCHS)
    print("Evaluating ImprovedCBN...")
    improved_results = evaluate_model(improved_model, groups, num_topics, game, pi_avg)

    print("\n" + "=" * 60)
    print("RESULTS")
    print_comparison(base_results, improved_results)


if __name__ == "__main__":
    main()
