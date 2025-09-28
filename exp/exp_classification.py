from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import wandb
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        wandb.init(
            project="hummob",  # Change project name as needed
            config={                              # Log hyperparameters
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.batch_size,
                "epochs": self.args.train_epochs,
                "model": self.args.model,
            },
            mode="disabled"
        )
        train_data, train_loader = self._get_data(flag='train')
        self.args.num_classes = train_data.get_num_class()
        self.num_classes = train_data.get_num_class()
        
        print(f"Dataset: {self.args.data}, City: {self.args.city}, Num Users: {self.args.num_users}, Num Classes: {self.args.num_classes}")

        
        
    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        self.args.num_classes = train_data.get_num_class()
        self.args.num_users = train_data.get_num_users()
        
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
            model = model.to(self.device)
        return model

    def _get_data(self, flag, **kwargs):
        data_set, data_loader = data_provider(self.args, flag, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.AdamW([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        if self.args.data == 'yj':
            criterion = nn.CrossEntropyLoss(ignore_index=40000, label_smoothing=self.args.label_smoothing)
        elif self.args.data == 'us':
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion

    
    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = 0
        total_samples = 0
        correct_counts = torch.zeros(10, device=self.device)  # Track top-1 through top-10 in one tensor
        mrr_sum = 0
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Move data to device all at once
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
                batch_y = batch_y.long().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)

                # Handle test vs validation sequence lengths
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:]
                
                # Reshape for computation
                batch_y = batch_y.view(-1)
                outputs = outputs.view(-1, outputs.size(-1))

                # Filter valid samples (handle padding)
                padding_idx = 40000 if self.args.data == 'yj' else 0
                valid_mask = batch_y != padding_idx
                valid_outputs = outputs[valid_mask]
                valid_targets = batch_y[valid_mask]
                
                # Calculate loss efficiently
                loss = criterion(valid_outputs, valid_targets)
                total_loss += loss.item() * valid_targets.size(0)
                
                # Get top-k predictions efficiently (k=10)
                scores, topk_indices = valid_outputs.topk(10, dim=1, sorted=True)
                correct = topk_indices.eq(valid_targets.unsqueeze(1))
                
                # Update top-k accuracy counts (cumulative sum along k dimension)
                correct_at_k = correct.cumsum(dim=1)
                correct_counts.add_(correct_at_k.sum(dim=0))
                
                # Calculate MRR efficiently
                first_correct_pos = correct.float().mul(
                    torch.arange(1, 11, device=self.device).float()
                )
                first_correct_pos[first_correct_pos == 0] = float('inf')
                mrr_sum += (1.0 / first_correct_pos.min(dim=1).values).sum().item()
                
                total_samples += valid_targets.size(0)

                if (i + 1) % 500 == 0:  # Reduced frequency of progress updates
                    print(f"\tValidation batch: {i + 1}/{len(vali_loader)}")
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
                
        # Calculate final metrics efficiently
        accuracies = {
            f'acc@{k}': (correct_counts[k-1] / total_samples).item()
            for k in (1, 3, 5, 10)
        }
        accuracies['MRR'] = mrr_sum / total_samples
        
        avg_loss = total_loss / total_samples
        accuracy = accuracies['acc@1']

        return avg_loss, accuracy, accuracies



    def train(self, setting):
        """
        Train the model to predict the next POI ID.
        Args:
            setting (str): Name for saving checkpoints and logs.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        print(f'Best model will be saved to {path}') 

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        best_val_accuracy = 0.0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale=8)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            total_loss = torch.tensor(0.0, device=self.device)
            total_correct = torch.tensor(0, device=self.device)
            total_samples = torch.tensor(0, device=self.device)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_y_f = batch_y_f.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y = batch_y.view(-1)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)

                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print("NaN detected in outputs before loss computation!")
                            continue
                     
                        outputs = outputs.view(-1, outputs.size(-1)) 

                        loss = criterion(outputs, batch_y)
                        
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                    
                    try:
                        outputs = outputs.view(-1, outputs.size(-1))
                    except RuntimeError as e:
                        outputs = outputs.reshape(-1, outputs.size(-1)).float()
                    
                    loss = criterion(outputs, batch_y)
                    
                
                

                if (i + 1) % 500 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        

                if self.args.use_amp:
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print("NaN detected in loss before loss computation!")
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

                total_loss += loss.detach()
                _, predicted = torch.max(outputs, dim=1)
                if not self.args.label_missing:
                    total_correct += (predicted == batch_y).sum()
                    total_samples += batch_y.size(0)
                else:
                    valid_mask = batch_y != (self.num_classes - 1)
                    total_correct += ((predicted == batch_y) & valid_mask).sum().item()
                    total_samples += valid_mask.sum().item()

            
            # if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
             
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)    

            train_loss = (total_loss / total_samples).item()
            train_accuracy = (total_correct / total_samples).item()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f'Training time: {time.time() - epoch_time:.2f}s')
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': model_optim.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict(),
            # }, os.path.join(path, 'current_checkpoint.pth'))

            wandb.log({
                        "iteration_loss": train_loss,
                        "iteration_accuracy": train_accuracy,
                        "iteration": i + 1,
                        "epoch": epoch + 1
                    })
            vali_loss, vali_accuracy, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            torch.cuda.empty_cache()
            test_loss, test_accuracy, test_metrics = self.vali(test_data, test_loader, criterion, is_test=True)
            torch.cuda.empty_cache()

            if vali_accuracy > best_val_accuracy:
                best_val_accuracy = vali_accuracy
                if not self.args.use_multi_gpu or (self.args.use_multi_gpu and dist.get_rank() == 0):
                    best_model_path = os.path.join(path, 'best_checkpoint.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

            wandb.log({
                "vali_loss": vali_loss,
                "vali_accuracy": vali_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "iteration": i + 1,
                "epoch": epoch + 1
            })
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.4f} Vali Loss: {:.4f} Test Loss: {:.4f} | | Train Acc: {:.4f} Vali Acc: {:.4f} Test Acc: {:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss, train_accuracy, vali_accuracy, test_accuracy))
                print(f"Vali: Acc@1: {vali_metrics['acc@1']:.4f}, Acc@3: {vali_metrics['acc@3']:.4f}, Acc@5: {vali_metrics['acc@5']:.4f}, Acc@10: {vali_metrics['acc@10']:.4f}, MRR: {vali_metrics['MRR']:.4f}")
                print(f"Test: Acc@1: {test_metrics['acc@1']:.4f}, Acc@3: {test_metrics['acc@3']:.4f}, Acc@5: {test_metrics['acc@5']:.4f}, Acc@10: {test_metrics['acc@10']:.4f}, MRR: {test_metrics['MRR']:.4f}")
                print("Epoch: {} cost time: {:.0f}s".format(epoch + 1, time.time() - epoch_time))  
            if self.args.enable_early_stopping:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("Early stopping")
                    break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
            
            torch.cuda.empty_cache()

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            # If using multiple GPUs, save the model's state_dict
            if torch.distributed.get_rank() == 0:  # Save only on the main process
                torch.save(self.model.state_dict(), best_model_path)
            dist.barrier()
        else:
            # Save the model's state_dict directly
            torch.save(self.model.state_dict(), best_model_path)

        print("Training completed.")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        correct_predictions = 0
        total_samples = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y_f = batch_y_f.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_y = batch_y.view(-1)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                        outputs = outputs.view(-1, outputs.size(-1)).float()
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                    outputs = outputs.view(-1, outputs.size(-1)).float()

                # Get predicted labels
                _, predicted = torch.max(outputs, dim=1)
                if not self.args.label_missing:
                    correct_predictions += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)
                else:
                    valid_mask = batch_y != (self.num_classes - 1)
                    correct_predictions += ((predicted == batch_y) & valid_mask).sum().item()
                    total_samples += valid_mask.sum().item() # TODO: adjust to different dataset

                preds.append(predicted.detach().cpu())
                trues.append(batch_y.detach().cpu())

                # Logging
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            # Aggregate predictions and ground truths
            preds = torch.cat(preds, dim=0).numpy()
            trues = torch.cat(trues, dim=0).numpy()

            # Calculate accuracy
            accuracy = correct_predictions / total_samples
            print(f'Test Accuracy: {accuracy:.4f}')

            # Calculate precision, recall, F1-score
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds, target_names=None)
            #print("Classification Report:\n", report)
            wandb.log({
                "test_accuracy": accuracy,
                "classification_report": report
            })

            # Save the classification report to a file
            with open("result_classification_report.txt", 'a') as f:
                f.write(setting + "\n")
                f.write(f'Test Accuracy: {accuracy:.4f}\n')
                f.write(report)
                f.write('\n')

        return accuracy
