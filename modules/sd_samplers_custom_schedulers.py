import torch


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [
        inner_model.t_to_sigma(ts)
        for ts in torch.linspace(start, end, n)[:-1]
    ]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)
