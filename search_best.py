import torch
from tqdm import tqdm
import numpy as np
import pickle
import torch.nn.functional as F
from skopt import Optimizer

def load_results(model_files):
    results = []
    for file in model_files:
        with open(file, 'rb') as f:
            results.append(list(pickle.load(f).items()))
    return results

def objective(alpha, results, labels):
    num_samples = len(results[0])
    num_classes = 155
    ensemble_scores = torch.zeros((num_samples, num_classes), dtype=torch.float32, device=labels.device)

    for i in range(num_samples):
        weighted_sum = torch.zeros(num_classes, dtype=torch.float32, device=labels.device)

        for j in range(len(results)):
            _, r_j = results[j][i]
            r_j_tensor = torch.FloatTensor(r_j).to(labels.device)
            weighted_sum += r_j_tensor * alpha[j]

        ensemble_scores[i] = weighted_sum

    pred = F.softmax(ensemble_scores, dim=1)
    preds = pred.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total * 100

    return -accuracy


def tqdm_gp_minimize(func, space, n_calls, *args, **kwargs):

    opt = Optimizer(space)
    for i in tqdm(range(n_calls)):
        x = opt.ask()
        y = func(x, *args, **kwargs)
        opt.tell(x, y)


    best_x = opt.Xi[np.argmin(opt.yi)]
    best_fun = min(opt.yi)

    return best_x, best_fun

model_files = [
    'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (15).pkl',
    'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (16).pkl',
    # 'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (17).pkl',
    'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (18).pkl',
    'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (19).pkl',
    'data/CRYD/CRYD-main/CTR-GCN-main/epoch1_test_score (20).pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/j/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ja/epoch1_test_score.pkl',
    # 'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/jm/epoch1_test_score.pkl',
    # 'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/b/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/ba/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/bm/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2a/epoch1_test_score.pkl',
    # 'data/CRYD/CRYD-main/HDBN/ICMEW2024-Track10-main/Model_inference/Mix_Former/work_dir_test/k2M/epoch1_test_score.pkl',
    'data/CRYD/CRYD-main/skate/epoch1_test_score (21).pkl',
    'data/CRYD/CRYD-main/skate/epoch1_test_score (22).pkl',
    # 'data/CRYD/CRYD-main/skate/jm.pkl',
    'data/CRYD/CRYD-main/skate/epoch1_test_score (23).pkl',
    'data/CRYD/CRYD-main/skate/epoch1_test_score (24).pkl'
    # 'data/CRYD/CRYD-main/skate/bm.pkl',
    # 'data/CRYD/CRYD-main/DeGCN/DeGCN_pytorch-main/work_dir_test/j/20241108122412/epoch1_test_score.pkl',
    # 'data/CRYD/CRYD-main/DeGCN/DeGCN_pytorch-main/work_dir_test/jm/ 20241108 122429/epoch1_test_score.pkl',
    # 'data/CRYD/CRYD-main/DeGCN/DeGCN_pytorch-main/work_dir_test/b/ 20241108 122441/epoch1_test_score.pkl'
    # 'data/CRYD/CRYD-main/DeGCN/DeGCN_pytorch-main/work_dir_test/bm/ 20241108 122444/epoch1_test_score.pkl'
]

results = load_results(model_files)

labels = np.load('data/val_label.npy')
for i in range(len(labels)):
    if labels[i] == -1:
        labels[i] = 0
labels = torch.LongTensor(labels)


space = [(0.0, 20.0) for _ in range(len(model_files))]


best_alpha, best_accuracy = tqdm_gp_minimize(
    lambda alpha: objective(torch.FloatTensor(alpha), results, labels),
    space,
    n_calls=300
)


print(f'Best Alpha: {best_alpha}')
print(f'Best accuracy: {-best_accuracy}')