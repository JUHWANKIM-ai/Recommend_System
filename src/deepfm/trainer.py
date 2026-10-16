import sys
import time
import torch
import pandas as pd
from src.utils.plot import *
from src.utils.metrics import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    def __init__(self, config, dataloaders):
        self.config = config
        self.dataloaders = dataloaders
        self.model = self.config['model']

        self.lr = self.config['train'].lr
        self.epochs = self.config['train'].epochs
        self.device = self.config['train'].device
        self.patience = self.config['train'].patience
        self.batch_size = self.config['train'].batch_size

        self.weight_path = f'weights/DeepFM_movie.pth'
        self.saving_path = f'results/DeepFM_movie.csv'

    def plot_bar(self, mode, i, len_data):
        progress = i / len_data
        bar_length = 30
        block = int(round(bar_length * progress))
        progress_bar = f'{mode}: [{"-" * block}{"." * (bar_length - block)}] {progress * 100:.2f}%'
        sys.stdout.write('\r' + progress_bar)

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def eval_fn(self, model, dataset_dl, initial):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl)
        model.to(self.device)

        model.eval()
        with torch.no_grad():
            i = 0
            for data, gt in dataset_dl:
                i += 1
                if not initial:
                    self.plot_bar('Val', i, len_data)

                data = data.to(self.device)
                gt = gt.to(self.device)

                output, loss = model(data, gt)

                total_loss += loss
                accuracy = R2(output, gt)
                total_accuracy += accuracy

            total_loss = total_loss / len_data
            total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_fn(self, model, dataset_dl, opt):
        total_loss = 0.0
        total_accuracy = 0.0
        len_data = len(dataset_dl)
        model.to(self.device)

        model.train()
        i = 0
        for data, gt in dataset_dl:
            i += 1
            self.plot_bar('Train', i, len_data)

            data = data.to(self.device)
            gt = gt.to(self.device)

            opt.zero_grad()
            output, loss = model(data, gt)
            loss.backward()
            opt.step()

            total_loss += loss
            accuracy = R2(output, gt)
            total_accuracy += accuracy

        total_loss = total_loss / len_data
        total_accuracy = total_accuracy / len_data

        return total_loss, total_accuracy

    def train_and_eval(self, model, params):
        backbone_weight_path = params['backbone_weight_path']
        opt = params["optimizer"]
        lr_scheduler = params["lr_scheduler"]

        loss_history = pd.DataFrame(columns=['train', 'val'])
        accuracy_history = pd.DataFrame(columns=['train', 'val'])

        if backbone_weight_path:
            val_loss1, val_accuracy = self.eval_fn(model, self.dataloaders['val'], True)
            model.load_state_dict(torch.load(backbone_weight_path))
            val_loss2, val_accuracy = self.eval_fn(model, self.dataloaders['val'], True)
            if val_loss1 > val_loss2:
                print('backbone')
                best_loss = val_loss2
                best_model_wts = torch.load(backbone_weight_path)
            else:
                print('retrain')
                best_loss = val_loss1
                best_model_wts = torch.load(self.weight_path)
                model.load_state_dict(best_model_wts)

        else:
            val_loss1, val_accuracy = self.eval_fn(model, self.dataloaders['val'], True)
            best_loss = val_loss1

        start_time = time.time()

        counter = 0
        for epoch in range(self.epochs):
            current_lr = self.get_lr(opt)
            print(f'\nEpoch {epoch + 1}/{self.epochs}, current lr={current_lr}')

            train_loss, train_accuracy = self.train_fn(model, self.dataloaders['train'], opt)
            loss_history.loc[epoch, 'train'] = train_loss
            accuracy_history.loc[epoch, 'train'] = train_accuracy

            val_loss, val_accuracy = self.eval_fn(model, self.dataloaders['val'], False)
            loss_history.loc[epoch, 'val'] = val_loss
            accuracy_history.loc[epoch, 'val'] = val_accuracy

            lr_scheduler.step(val_loss)

            if val_loss < best_loss:
                counter = 0
                best_loss = val_loss
                torch.save(model.state_dict(), self.weight_path)
                best_model_wts = model.state_dict().copy()
                print('\nSaved model weight!')
            else:
                counter += 1
                if counter >= self.patience:
                    model.load_state_dict(best_model_wts)
                    print("\nEarly stopped.")
                    break

            print(f'\ntrain loss: {train_loss:.5f}, val loss: {val_loss:.5f}')
            print(f'Metric: {val_accuracy:.5f}, time: {(time.time() - start_time) / 60:.2f}')

        return model, loss_history, accuracy_history

    def train(self):
        print('\nTraining model...')

        opt = Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=self.patience)

        parameters = {
            'backbone_weight_path': None,
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        self.model, self.loss_hist, self.metric_hist = self.train_and_eval(self.model, parameters)

        print('\nModel: train complete.')

    def evaluate(self):
        print('\nSaving loss and metric...')

        loss_hist_numpy = self.loss_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        metric_hist_numpy = self.metric_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

        # plot loss progress
        plot_hist(self.epochs, loss_hist_numpy, 'Loss')

        # plot accuracy progress
        plot_hist(self.epochs, metric_hist_numpy, 'R2')

        print('Loss and metrics: save complete.')

    def test(self):
        print('\nSaving test set results...')
        self.model.load_state_dict(torch.load(self.weight_path))
        self.model.to(self.device)
        self.model.eval()
        batch_results = []
        with ((torch.no_grad())):
            for X_train, gt in self.dataloaders['test']:
                X_train = X_train.to(self.device)
                output = ensure_tensor_array(self.model(X_train))
                # y_pred = np.argmax(output, axis=1).astype(int)
                gt = ensure_tensor_array(gt.squeeze()).astype(float)
                user_id = X_train[:, 0].cpu().numpy().astype(int)
                item_id = X_train[:, 1].cpu().numpy().astype(int)
                batch_results.extend(zip(user_id, item_id, output, gt))

        self.pred = pd.DataFrame(batch_results, columns=['user_id', 'movie_id', 'y_pred', 'gt'])
        self.pred.to_csv(self.saving_path)
        print(f"Results in {self.saving_path}: save complete.")
