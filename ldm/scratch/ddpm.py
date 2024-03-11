import torch
from tqdm import tqdm

class DDPM:
    def __init__(self, beta_min, beta_max, n_steps, device):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_steps = n_steps
        self.device = device
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps, device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)
        self.sigma = self.beta.sqrt()

    def schedule(self, x):
        device = x.device
        t = torch.randint(0, self.n_steps, (len(x),), dtype=torch.long, device=device)
        noise = torch.randn_like(x).to(device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        mean = alpha_bar_t.sqrt()*x
        variance = (1 - alpha_bar_t).sqrt()*noise
        xt = mean + variance
        return xt, t, noise

    @torch.inference_mode()
    def sample(self, model, sz, c):
        device = next(model.parameters()).device
        batch_size = sz[0]
        xt = torch.randn(sz).to(device)
        c = c.expand(batch_size).to(device)
        preds = []
        for i in tqdm(reversed(range(self.n_steps))):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            noise_pred = model(xt, t, c)
            noise = (torch.randn_like(xt) if i > 0 else torch.zeros_like(xt)).to(device)
            alpha_t1 = self.alpha[i - 1] if i > 0 else torch.tensor(1)
            beta_bar_t = 1 - self.alpha[i]
            beta_bar_t1 = 1 - alpha_t1
            x_0_hat = ((x_t - beta_bar_t.sqrt() * noise_pred) / self.alpha[t].sqrt()).clamp(-1, 1)
            x0_coeff = alpha_t1.sqrt() * (1 - self.alpha[i]) / beta_bar_t
            xt_coeff = self.alpha[t].sqrt() * beta_bar_t1 / beta_bar_t
            x_t = x_0_hat * x0_coeff + x_t * xt_coeff + self.sigma[i] * noise
            preds.append(x_t.cpu())
        return preds