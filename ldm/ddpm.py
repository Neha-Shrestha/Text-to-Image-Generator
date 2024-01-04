import torch
from utils import init_attr

class DDPM:
    def __init__(self, beta_min, beta_max, n_steps, device):
        init_attr(self, locals())
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0).to(self.device)
        self.sigma = self.beta.sqrt()

    def schedule(self, x):
        t = torch.randint(0, self.n_steps, (len(x),), dtype=torch.long, device=self.device)
        noise = torch.randn_like(x).to(self.device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        mean = alpha_bar_t.sqrt()*x
        variance = (1 - alpha_bar_t).sqrt()*noise
        x = mean + variance
        return x, t, noise

    @torch.no_grad()
    def sample(self, model, sz, c):
        device = self.device
        xt = torch.randn(sz, device=device)
        c = c.to(device)
        preds = []
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
            if t > 0:
                noise = torch.randn(xt.shape).to(device)   
                alpha_t_1 = self.alpha_bar[t-1]
            else:
                noise = torch.zeros(xt.shape).to(device)
                alpha_t_1 = torch.tensor(1)
            beta_t = 1 - self.alpha_bar[t]
            beta_t_1 = 1 - alpha_t_1
            noise_pred = model(xt, t_batch, c)
            x0 = ((xt - beta_t.sqrt() * noise_pred)/self.alpha_bar[t].sqrt()).clamp(-1,1)
            x0_coeff = (alpha_t_1.sqrt()*(1-self.alpha[t])) / beta_t
            xt_coeff = (self.alpha[t].sqrt() * beta_t_1) / beta_t
            xt = x0*x0_coeff + xt*xt_coeff + self.sigma[t]*noise
            preds.append(xt)
        return preds