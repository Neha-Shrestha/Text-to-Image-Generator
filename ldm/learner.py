import torch
from tqdm.notebook import trange
from IPython.display import display
from torcheval.metrics import MulticlassAccuracy, Mean
from utils import plot_curves

class Learner:
    def __init__(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_metric = Mean(device=self.device)
        self.accuracy_metric = MulticlassAccuracy(device=self.device)
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
    
    def _update_results(self):
        self.loss_metric.update(self.loss)
        self.accuracy_metric.update(self.pred, self.y)
        
    def _compute_results(self):
        lm = self.loss_metric.compute().item()
        am = self.accuracy_metric.compute().item()
        mode = "train" if self.training else "test"
        self.results[f"{mode}_loss"].append(lm)
        self.results[f"{mode}_acc"].append(am)
        self.loss_metric.reset()
        self.accuracy_metric.reset()
        print(f"{self.epoch}\t{mode}\t{lm:.4f}\t{am:.4f}")

    def _run_batch(self):
        self.pred = self.model(self.X)
        self.loss = self.loss_fn(self.pred, self.y)
        self._update_results()
        if self.training:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

    def _run_epoch(self, train=True):
        self.training = train
        self.dl = self.train_dataloader if self.training else self.test_dataloader
        for self.X, self.y in self.dl:
            self.X, self.y = self.X.to(self.device), self.y.to(self.device)
            self._run_batch()
        self._compute_results()
            
    def fit(self, epochs, train=True, test=True):
        self.epochs = epochs
        self.model.to(self.device)
        progress_bar = trange(self.epochs, desc="Progress")
        display(progress_bar)
        print("Epoch\tMode\tLoss\tAccuracy")
        for self.epoch in progress_bar:
            if train:
                self.model.train()
                self._run_epoch(True)
            if test: 
                self.model.eval()
                with torch.inference_mode(): 
                    self._run_epoch(False)
            self.scheduler.step()
        plot_curves(self.results)