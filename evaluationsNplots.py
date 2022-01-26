# General imports
import numpy as np
import matplotlib.pyplot as plt
import os, time, math
# Project imports
from wisdmDataLoader import *
from dataPreprocessing import *
from transformerModel import *
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as f


def train(train_xyz, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len):
    model.train()
    total_loss = 0.
    train_loss = 0.
    start_time = time.time()

    for batch, fs in enumerate(range(0, np.shape(train_xyz)[0] - 1, batch_size)):
        data, targets = get_batch(train_xyz, fs, batch_size, [0, 1, 2])
        if np.random.random() > 0.5:
             data, targets = data.flip(dims=[1, 2]), targets.flip(dims=[1, 2])
        if np.random.random() > 0.5:
            data, targets = -data, -targets

        optimizer.zero_grad()
        output = model(data)
        output = f.normalize(output, dim=1)*np.sqrt(sample_len)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.cpu().item()
        train_loss += loss.cpu().item() * data.size(0)
        log_interval = int(np.shape(train_xyz)[0] / batch_size / 5)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2e} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, np.shape(train_xyz)[0] // batch_size, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    train_loss /= np.shape(train_xyz)[0]
    return train_loss


def plot_examples(data_source, eval_model, criterion, epoch_num, sample_len, pe_type):
    eval_model.eval()
    total_loss = 0.

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    with torch.no_grad():
        for k, j in enumerate(np.random.randint(0, np.shape(data_source)[0] - 1, size=4)):
            data, target = get_batch(data_source, j, 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            curr_loss = criterion(output, target).item()
            test_result = output[0].view(-1).cpu()
            truth = target[0].view(-1).cpu()

            fig.add_subplot(2, 2, k + 1)
            plt.plot(test_result, color="red", label="pred")
            plt.plot(truth, color="blue", label="gt")
            # plt.plot(test_result - truth, color="green", label="diff")
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.title('sample ' + str(j) + ', MSE ' + '{:2.4f}'.format(curr_loss))
            if k < 2:
                plt.tick_params(labelbottom=False)
        plt.legend()
        fig.savefig('plots_' + pe_type + '/epoch%d_evaluation_example.png' % epoch_num)
        plt.close()

    return total_loss / np.shape(data_source)[0]


    eval_model.eval()
    loss = np.empty(np.shape(data_source)[0])

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    with torch.no_grad():
        for s in range(np.shape(data_source)[0]):
            data, targets = get_batch(data_source, s, 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            loss[s] = criterion(output, targets).cpu().item()
            best_samples = np.argsort(loss)

        for k in range(4):
            data, target = get_batch(data_source, best_samples[k], 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            curr_loss = criterion(output, target).item()
            test_result = output[0].view(-1).cpu()
            truth = target[0].view(-1).cpu()

            fig.add_subplot(2, 2, k + 1)
            plt.plot(test_result, color="red", label="pred")
            plt.plot(truth, color="blue", label="gt")
            # plt.plot(test_result - truth, color="green", label="diff")
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.title('sample ' + str(best_samples[k]) + ', MSE ' + '{:2.4f}'.format(curr_loss))
            if k < 2:
                plt.tick_params(labelbottom=False)
        plt.legend()
        fig.savefig('plots_' + pe_type + '/epoch%d_best_examples.png' % epoch_num)
        plt.close()


def plot_learning_curves(train_loss_vec, val_loss_vec, vline, pe_type):
    xscale = np.arange(1, np.shape(train_loss_vec)[0]+1)

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.semilogx(xscale, train_loss_vec, 'b-x', label='Training')
    plt.semilogx(xscale, val_loss_vec, 'r-x', label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss plot')
    plt.legend()
    plt.grid(True, which='both')
    plt.axvline(x=vline, color='k')
    fig.savefig('plots_' + pe_type + '/learning_curve.png')
    plt.close()


def evaluate(data_source, eval_model, criterion, batch_size, sample_len):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for e in range(0, np.shape(data_source)[0] - 1, batch_size):
            data, targets = get_batch(data_source, e, batch_size, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            total_loss += data.size(0) * criterion(output, targets).cpu().item()
    return total_loss / np.shape(data_source)[0]
