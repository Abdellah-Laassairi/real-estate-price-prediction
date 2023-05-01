import lightning.pytorch as pl
import torch
from efficientnet_pytorch import EfficientNet
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy
from torchmetrics import R2Score
from torchmetrics.classification import MulticlassAccuracy

logger = TensorBoardLogger('tb_logs', name='my_model')


def my_loss(output, target):
    mse = torch.nn.MSELoss()
    loss = torch.sqrt(mse(output, target))
    # loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss


class RealModel(pl.LightningModule):
    def __init__(self,
                 num_target_classes=int(400),
                 arch='efficientnet-b0',
                 lr=0.1,
                 weight_decay=0.001,
                 batch_size=8,
                 max_epochs=100,
                 *args,
                 **kwargs):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.val_output_list = []
        self.train_output_list = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.net = EfficientNet.from_pretrained(
            arch, advprop=True, num_classes=self.num_target_classes)

        self.linear1 = torch.nn.Linear(self.num_target_classes, 1, bias=True)
        self.linear1 = torch.nn.Linear(self.num_target_classes, 1, bias=True)

        self.relu = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(
            num_features=self.num_target_classes)
        self.loss_fn = my_loss
        self.acc = R2Score(num_outputs=self.batch_size,
                           multioutput='raw_values')
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.dropout(x1)
        x3 = self.linear1(x2)
        x4 = self.dropout(x3)
        # x5 = self.relu(x4)
        # x6 = self.linear(x5)
        # x7 = self.dropout(x6)
        output = self.relu(x4).squeeze(1)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            max_lr=self.lr,
            epochs=self.max_epochs,
            optimizer=optimizer,
            steps_per_epoch=self.num_training_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
            base_momentum=0.90,
            max_momentum=0.95,
        )
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def step(self, batch):
        x, y = batch
        y = y.float()
        y_hat = self.forward(x)
        # print("y", y.size())
        # print("y_hat", y_hat.size())
        return y, y_hat

    def training_step(self, batch):
        # hardware agnostic training
        y, y_hat = self.step(batch)

        loss = self.loss_fn(y_hat, y)

        self.log('Train_loss', loss, prog_bar=True, sync_dist=True)
        self.train_output_list.append({
            'y': y,
            'y_hat': y_hat,
        })
        return loss

    def on_train_epoch_end(self):
        preds = []
        targets = []

        for output in self.train_output_list:
            if output['y_hat'].size()[0] == self.batch_size:
                preds.append(output['y_hat'])
                targets.append(output['y'])

        targets = torch.stack(targets)
        preds = torch.stack(preds)
        acc = self.acc(preds, targets)
        self.log('R2_Train', acc.mean(), sync_dist=True)
        print('R2_Train:', acc.mean())
        self.train_output_list.clear()
        return {
            'R2': acc,
        }

    def validation_step(self, batch, batch_nb):
        y, y_hat = self.step(batch)
        loss = self.loss_fn(y_hat, y)
        results = {'val_loss': loss, 'y': y.detach(), 'y_hat': y_hat.detach()}
        self.log('Val_loss', loss, prog_bar=True, sync_dist=True)

        self.val_output_list.append(results)
        return results

    def on_validation_epoch_end(self):
        preds = []
        targets = []

        for output in self.val_output_list:
            if output['y_hat'].size()[0] == self.batch_size:
                preds.append(output['y_hat'])
                targets.append(output['y'])
        targets = torch.stack(targets)
        preds = torch.stack(preds)
        acc = self.acc(preds, targets)
        self.log('R2_val', acc.mean(), sync_dist=True)
        print('R2_VAL:', acc.mean())
        self.val_output_list.clear()
        return {
            'R2': acc,
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.train_dataloader())
        batches = (min(batches, limit_batches) if isinstance(
            limit_batches, int) else int(limit_batches * batches))

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
