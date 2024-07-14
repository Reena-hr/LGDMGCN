import numpy as np
import torch
from .basetrainer import BaseTrainer
import math
import time


class Trainer(BaseTrainer):
    """
    trainer class
    """
    def __init__(self, device, model_type, model, loss, metrics, optimizer, scaler, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, val_len_epoch=None, supports=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.device = device
        self.model_type = model_type
        self.scaler = scaler
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len_epoch
        self.val_len_epoch = val_len_epoch
        self.cl_decay_steps = config["trainer"]["cl_decay_steps"]

        self.max_grad_norm = config["trainer"]["max_grad_norm"]
        self.valid_data_loader = valid_data_loader
        # self.do_validation = self.valid_data_loader is not None
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(config["trainer"]["log_steps"])
        self.supports = supports
        # self.batch_size = int(config["arch"]["args"]["batch_size"])

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch, dy_supports, tinterval, order2time_map):
        
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        start_time = time.time()
        loss_list = []
        for batch_idx, (data, target) in enumerate(self.data_loader.get_iterator()):
            data = torch.FloatTensor(data)            
            target = torch.FloatTensor(target)            
            label = target[..., :self.model.output_dim]
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.len_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.cl_decay_steps)

            if self.model_type == "seq2seq":
                output = self.model(self.supports, data, target, teacher_forcing_ratio, batch_idx, self.model.batch_size, dy_supports, tinterval, order2time_map)
                output = torch.stack(output, dim=1)
                
            elif self.model_type == "lstm":
                output = self.model(data)  # (batch_size, horizon, num_nodes)
                output = output.unsqueeze_(dim=-1)
            else:
                raise ValueError("Invalid model type.")

            loss = self.loss(output.cpu(), label)  # loss is self-defined, need cpu input
            loss_list.append(loss)
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(self.scaler.inverse_transform(output.detach().cpu().numpy()),
                                                self.scaler.inverse_transform(label.numpy()))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        training_time = time.time() - start_time
        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch, dy_supports, tinterval, order2time_map)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        log.update({'Training Time': "{:.4f}s".format(training_time)})
        return log, training_time, loss_list

    def _valid_epoch(self, epoch, dy_supports, tinterval, order2time_map):
       
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader.get_iterator()):
                data = torch.FloatTensor(data)
                # data = data.permute(0, 2, 1, 3).contiguous()
                target = torch.FloatTensor(target)
                # target = target.permute(0, 2, 1, 3).contiguous()
                label = target[..., :self.model.output_dim]  # (..., 1)  supposed to be numpy array
                data, target = data.to(self.device), target.to(self.device)

                if self.model_type == "seq2seq":                   
                    output = self.model(self.supports, data, target, 0, batch_idx, self.model.batch_size, dy_supports, tinterval, order2time_map)
                    output = torch.stack(output, dim=1)                   
                elif self.model_type == "lstm":
                    output = self.model(data)  
                    output = output.unsqueeze_(dim=-1)
                else:
                    raise ValueError("Invalid model type.")

                loss = self.loss(output.cpu(), label)

                self.writer.set_step((epoch - 1) * self.val_len_epoch + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(self.scaler.inverse_transform(output.detach().cpu().numpy()),
                                                        self.scaler.inverse_transform(label.numpy()))               

       
    @staticmethod
    def _compute_sampling_threshold(global_step, k):       
        return k / (k + math.exp(global_step / k))
