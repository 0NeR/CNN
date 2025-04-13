from time import gmtime, strftime
import torch
import numpy as np
from tqdm import tqdm

import neptune

from ex1_prepare_data import *
from ex6_CLDataset import load_datasets
from ex6_PreModel import PreModel, plot_features
from ex6_SimCLR_Loss import SimCLR_Loss



class BaseTrainProcess:
    def __init__(self, hyp, experiment=None):
        start_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
        
        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.experiment = experiment  # Neptune experiment object

        self.hyp = hyp

        self.lr_scheduler: Optional[torch.optim.lr_scheduler] = None
        self.model: Optional[torch.nn.modules] = None
        self.optimizer: Optional[torch.optim] = None
        self.criterion: Optional[torch.nn.modules] = None

        self.train_loader: Optional[Dataloader] = None
        self.valid_loader: Optional[Dataloader] = None

        # Log hyperparameters to Neptune
        if self.experiment is not None:
            self._log_hyperparameters()
        
        self.init_params()

    def _log_hyperparameters(self):
        """Log all hyperparameters to Neptune"""
        self.experiment["parameters"] = {
            "batch_size": self.hyp['batch_size'],
            "epochs": self.hyp['epochs'],
            "lr": self.hyp['lr'],
            "weight_decay": self.hyp['weight_decay'],
            "temperature": self.hyp['temperature'],
            "n_workers": self.hyp['n_workers']
        }
        self.experiment["parameters/optimizer"] = "AdamW"
        self.experiment["parameters/scheduler"] = "CosineAnnealingWarmRestarts"

    def init_params(self):
        self._init_data()
        self._init_model()

    def _init_data(self):
        train_dataset, valid_dataset = load_datasets(X_train, y_train, X_val, y_val, crop_coef=1.4)
        print('Train size:', len(train_dataset), 'Valid size:', len(valid_dataset))

        if self.experiment is not None:
            self.experiment["data/train_size"] = len(train_dataset)
            self.experiment["data/valid_size"] = len(valid_dataset)

        self.train_loader = DataLoader(train_dataset,
                                     batch_size=self.hyp['batch_size'],
                                     shuffle=True,
                                     num_workers=self.hyp['n_workers'],
                                     pin_memory=True,
                                     drop_last=True)

        self.valid_loader = DataLoader(valid_dataset,
                                     batch_size=self.hyp['batch_size'],
                                     shuffle=True,
                                     num_workers=self.hyp['n_workers'],
                                     pin_memory=True,
                                     drop_last=True)

    def _init_model(self):
        self.model = PreModel()
        self.model.to(self.device)

        if self.experiment is not None:
            self.experiment["model/architecture"] = str(self.model)

        model_params = [params for params in self.model.parameters() if params.requires_grad]
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp['lr'], weight_decay=self.hyp['weight_decay'])

        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            500,
            eta_min=0.05,
            last_epoch=-1,
        )

        self.criterion = SimCLR_Loss(batch_size=self.hyp['batch_size'],
                                   temperature=self.hyp['temperature']).to(self.device)

    def save_checkpoint(self, loss_valid, path):
        if loss_valid[0] <= self.best_loss:
            self.best_loss = loss_valid[0]
            self.save_model(path)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.mainscheduler.state_dict()
        }, path)

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()
        
        cum_loss = 0.0
        cum_acc = 0.0
        proc_loss = 0.0
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (xi, xj, _, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)
        
            with torch.set_grad_enabled(True):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
        
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()
        
            cur_loss = loss.detach().cpu().item()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)
        
            sim = torch.cosine_similarity(zi, zj)
            acc = (sim > 0.5).float().mean().item()
            cum_acc += acc
        
            if self.experiment is not None:
                self.experiment["train/batch_loss"].log(cur_loss)
                self.experiment["train/batch_acc"].log(acc)
        
            s = f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {acc:4.3f}'
            pbar.set_description(s)
        
        epoch_loss = cum_loss / len(self.train_loader)
        epoch_acc = cum_acc / len(self.train_loader)
        
        if self.experiment is not None:
            self.experiment["train/epoch_loss"].log(epoch_loss)
            self.experiment["train/epoch_acc"].log(epoch_acc)
        
        return [epoch_loss]


    def valid_step(self):
        self.model.eval()
        
        cum_loss = 0.0
        cum_acc = 0.0
        proc_loss = 0.0
        
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (xi, xj, _, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)
        
            with torch.set_grad_enabled(False):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
        
            cur_loss = loss.detach().cpu().item()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)
        
            # Similarity accuracy
            sim = torch.cosine_similarity(zi, zj)
            acc = (sim > 0.5).float().mean().item()
            cum_acc += acc
        
            if self.experiment is not None:
                self.experiment["valid/batch_loss"].log(cur_loss)
                self.experiment["valid/batch_acc"].log(acc)
        
            s = f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, Loss: {proc_loss:4.3f}, Acc: {acc:4.3f}'
            pbar.set_description(s)
        
        epoch_loss = cum_loss / len(self.valid_loader)
        epoch_acc = cum_acc / len(self.valid_loader)
        
        if self.experiment is not None:
            self.experiment["valid/epoch_loss"].log(epoch_loss)
            self.experiment["valid/epoch_acc"].log(epoch_acc)
        
        return [epoch_loss]

    def run(self):
        best_w_path = 'best.pt'
        last_w_path = 'last.pt'

        train_losses = []
        valid_losses = []

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            loss_train = self.train_step()
            train_losses.append(loss_train)

            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            if self.experiment is not None:
                self.experiment["train/learning_rate"].log(lr)

            loss_valid = self.valid_step()
            valid_losses.append(loss_valid)

            self.save_checkpoint(loss_valid, best_w_path)

            if (epoch + 1) % 10 == 0 or epoch == self.hyp['epochs'] - 1:
                fig = plot_features(self.model, self.valid_loader, device=self.device)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                if self.experiment is not None:
                    self.experiment["visualizations/tsne"].log(
                        neptune.types.File.as_image(image_from_plot)
                    )

        self.save_model(last_w_path)
        torch.cuda.empty_cache()

        # Log final metrics
        if self.experiment is not None:
            self.experiment["final/train_loss"] = train_losses[-1][0]
            self.experiment["final/valid_loss"] = valid_losses[-1][0]
            self.experiment["artifacts/best_model"].upload(best_w_path)
            self.experiment["artifacts/last_model"].upload(last_w_path)

        return train_losses, valid_losses

if __name__ == "__main__":

    hyps = {'batch_size': 128, 'lr': 0.2, 'epochs': 100}
    run = neptune.init_run(project="eduard-andreev/Images-homework-3")
    trainer = BaseTrainProcess(hyps, experiment=run)
    trainer.run()
