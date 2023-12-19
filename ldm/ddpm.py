class DDPM:
    def __init__(self, beta_min, beta_max, n_steps):
        init_attr(self, locals=locals())
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(dim=0)
        self.sigma = self.beta.sqrt()
    
    def forward(self, x0):
        device = x0.device
        n = len(x0)
        t = torch.randint(0, self.n_steps, (n,), device=device, dtype=torch.long)
        noise = torch.randn(x0.shape, device=device)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1).to(device)
        mean = alpha_bar_t.sqrt().to(device) * x0.to(device) 
        variance = (1-alpha_bar_t).sqrt().to(device) * noise.to(device) 
        xt = mean + variance
        return (xt, t.to(device)), noise