from config import config
from data.MPII.dataset import MPII_Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import gc
import time
import utils.model


def train(model, start_lr, lr_schedule, epochs, bs, criterion, optimizer, train_ds, valid_ds, checkpoint_path=None):
    model.train()
    train_dataloader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=bs, shuffle=True)

    for epoch in range(epochs):
        # if at milestone epoch, adapt learning rate
        if epoch in lr_schedule.keys():
            for g in optimizer.param_groups:
                g['lr'] = lr_schedule[epoch]

        epoch_train_losses = []
        epoch_valid_losses = []

        outputs_train = []
        targets_train = []

        best_valid_loss = float('inf')

        for i, data in enumerate(train_dataloader):
            inputs, heatmaps = data # input has shape (batch_size, channels, height, width) = (bs, 3, 256, 256)
            inputs = inputs.cuda(device=0)

            # model requires input shape (batch_size, channels, height, width)
            optimizer.zero_grad()
            preds = model(inputs) # forward pass. returns shape (bs, 8, 16, 64, 64) = (bs, hg_modules, 16 kp, height, width)

            preds = preds.cpu()
            loss = criterion(combined_hm_preds=preds, heatmaps=heatmaps) # loss shape = (16, 8)
            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

            outputs_train.append(preds)
            targets_train.append(heatmaps)

        outputs_train = torch.cat(outputs_train)
        targets_train = torch.cat(targets_train)

        model.eval()
        with torch.no_grad():
            outputs_valid = []
            targets_valid = []
            for j, data in enumerate(valid_dataloader):
                inputs, heatmaps = data
                inputs = inputs.cuda(device=0)
                preds_valid = model(inputs) # returns shape (16, 8, 16, 64, 64) = (bs, hg_modules, 16 kp, height, width)
                preds_valid = preds_valid.cpu()
                valid_loss = criterion(combined_hm_preds=preds_valid, heatmaps=heatmaps) # loss shape = (16, 8)
                valid_loss = torch.mean(valid_loss)
                epoch_valid_losses.append(valid_loss.item())
                outputs_valid.append(preds_valid)
                targets_valid.append(heatmaps)

            outputs_valid = torch.cat(outputs_valid)
            targets_valid = torch.cat(targets_valid)

        overall_valid_loss = sum(epoch_valid_losses)/len(epoch_valid_losses)
        print(f'EPOCH {epoch} -- TRAINING_LOSS: {round(sum(epoch_train_losses)/len(epoch_train_losses), 4)} -- VALIDATION_LOSS: {round(sum(epoch_valid_losses)/len(epoch_valid_losses), 4)}')

        # save checkpoint
        if overall_valid_loss < best_valid_loss:
            best_valid_loss = overall_valid_loss
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, checkpoint_path)



if __name__ == '__main__':
    # set up hyperparameters
    bs = config['train']['batchsize']
    # lr = config['train']['learning_rate']
    input_res = config['train']['input_res']
    epochs = config['train']['epochs']
    checkpoint_path = config['inference']['checkpoint_path']

    # adaptive learning rate (linear)
    start_lr = config['train']['start_learning_rate']  # 2.5e-4
    end_lr = config['train']['end_learning_rate'] # 1e-5
    decay_epochs = config['train']['decay_epochs'] # [75, 100, 150]

    # linear learning rate
    lr_step_size = (end_lr - start_lr)/ len(decay_epochs)
    lr_schedule = {}
    current = start_lr
    for i in range(len(decay_epochs)):
        current = current + lr_step_size
        lr_schedule[decay_epochs[i]] = current

    valid_ds = MPII_Dataset(config, mode='valid')
    train_ds = MPII_Dataset(config, mode='train')

    torch.cuda.empty_cache()
    gc.collect()

    net = utils.model.load_model(config)

    start_cuda = time.time()
    net = net.cuda(device=0)
    end_cuda = time.time()
    print('model loading to cuda time: ', (end_cuda - start_cuda)/60, 'minutes')

    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    criterion = net.calc_loss

    net = DataParallel(net, device_ids=[0, 1, 2, 3])

    print('start training')
    train(net, start_lr, lr_schedule, epochs, bs, criterion, optimizer, train_ds, valid_ds, checkpoint_path)
