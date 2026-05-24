import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CollaborativeBayesianNetwork(nn.Module):
    def __init__(self, num_users, num_topics, mu1=0.5, sigma1=0.1, mu2=0.5, sigma2=0.1, lambda_reg=0.01):
        super(CollaborativeBayesianNetwork, self).__init__()
        self.num_users = num_users
        self.num_topics = num_topics
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.lambda_reg = lambda_reg

        self.I = nn.Parameter(torch.normal(mu1, sigma1, size=(num_users, num_topics)))
        self.S = nn.Parameter(torch.normal(mu2, sigma2, size=(num_users, num_topics)))

    def forward(self, user_indices, topic_indices, pi):
        I_batch = self.I[user_indices, topic_indices]
        S_batch = self.S[user_indices, topic_indices]
        logits = pi * I_batch + S_batch
        return torch.sigmoid(logits)

    def compute_loss(self, prob, B_actual):
        bce_loss = nn.functional.binary_cross_entropy(prob, B_actual, reduction='mean')

        n = B_actual.shape[0]
        prior_I = torch.sum((self.I - self.mu1) ** 2) / (2 * self.sigma1 ** 2) / n
        prior_S = torch.sum((self.S - self.mu2) ** 2) / (2 * self.sigma2 ** 2) / n

        return bce_loss + self.lambda_reg * (prior_I + prior_S)


def train_cbn(model, interactions, num_epochs=50, lr=0.01, batch_size=2048, convergence_threshold=0.001):
    optimizer = optim.SGD(model.parameters(), lr=lr)

    n = len(interactions)
    u_arr  = torch.tensor([x[0] for x in interactions], dtype=torch.long)
    d_arr  = torch.tensor([x[1] for x in interactions], dtype=torch.long)
    pi_arr = torch.tensor([x[2] for x in interactions], dtype=torch.float32)
    B_arr  = torch.tensor([x[3] for x in interactions], dtype=torch.float32)

    prev_loss = float('inf')

    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            optimizer.zero_grad()
            prob = model(u_arr[idx], d_arr[idx], pi_arr[idx])
            loss = model.compute_loss(prob, B_arr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if abs(prev_loss - avg_loss) < convergence_threshold:
            print(f"Converged at epoch {epoch+1} (|Δloss| = {abs(prev_loss - avg_loss):.6f} < {convergence_threshold})")
            break

        prev_loss = avg_loss

    return model
