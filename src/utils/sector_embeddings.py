import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def compute_sector_embeddings(sector_names, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sector_names, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def sector_similarity_matrix(sector_embeddings):
    norms = np.linalg.norm(sector_embeddings, axis=1, keepdims=True)
    normed = sector_embeddings / np.maximum(norms, 1e-8)
    return normed @ normed.T


def form_groups_by_sector_affinity(interactions, num_users, num_topics,
                                   sector_embeddings, group_size=5,
                                   n_groups=10, n_sector_clusters=4, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    pi_sum = np.zeros((num_users, num_topics), dtype=np.float32)
    pi_cnt = np.zeros((num_users, num_topics), dtype=np.float32)
    for u, d, pi, b in interactions:
        if b == 1.0:
            pi_sum[u, d] += pi
            pi_cnt[u, d] += 1.0
    pi_avg = np.where(pi_cnt > 0, pi_sum / np.maximum(pi_cnt, 1.0), 0.0)

    km = KMeans(n_clusters=n_sector_clusters, random_state=42, n_init=10)
    sector_cluster_labels = km.fit_predict(sector_embeddings)

    dominant_sector = np.argmax(pi_avg, axis=1)              # (num_users,)
    user_cluster = sector_cluster_labels[dominant_sector]    # (num_users,)

    cluster_to_users = {}
    for u in range(num_users):
        c = int(user_cluster[u])
        cluster_to_users.setdefault(c, []).append(u)

    clusters = sorted(cluster_to_users.keys())
    n_clusters = len(clusters)

    groups = []
    for _ in range(n_groups):
        chosen_clusters = [clusters[i % n_clusters]
                           for i in rng.permutation(max(n_clusters, group_size))[:group_size]]

        group = []
        used = set()
        for c in chosen_clusters:
            candidates = [u for u in cluster_to_users[c] if u not in used]
            if not candidates:
                candidates = cluster_to_users[c]
            u = int(rng.choice(candidates))
            group.append(u)
            used.add(u)

        groups.append(group)

    return groups, sector_cluster_labels, pi_avg
