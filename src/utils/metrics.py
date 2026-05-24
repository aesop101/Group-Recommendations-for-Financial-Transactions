import numpy as np

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def chebyshev_distance(v1, v2):
    return np.max(np.abs(v1 - v2))

def correlation_distance(v1, v2):
    v1_mean = np.mean(v1)
    v2_mean = np.mean(v2)
    num = np.sum((v1 - v1_mean) * (v2 - v2_mean))
    den = np.sqrt(np.sum((v1 - v1_mean)**2) * np.sum((v2 - v2_mean)**2))
    if den == 0:
        return 0
    return 1 - (num / den)

def mae_distance(v1, v2):
    return np.mean(np.abs(v1 - v2))

def mse_distance(v1, v2):
    return np.mean((v1 - v2)**2)

class Metrics:
    @staticmethod
    def calculate_all(predicted_interest, target_interest):
        v1 = np.array(predicted_interest)
        v2 = np.array(target_interest)
        
        return {
            "EucDist": euclidean_distance(v1, v2),
            "ManDist": manhattan_distance(v1, v2),
            "CheDist": chebyshev_distance(v1, v2),
            "CorDist": correlation_distance(v1, v2),
            "MAEDist": mae_distance(v1, v2),
            "MSEDist": mse_distance(v1, v2)
        }
