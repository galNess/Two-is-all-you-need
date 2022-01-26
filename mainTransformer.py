# General imports
import numpy as np
import matplotlib.pyplot as plt
import os, time, math
# Project imports
from wisdmDataLoader import *
from dataPreprocessing import *
from transformerModel import *
from evaluationsNplots import *
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as f

# Torch initialization
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyper-parameters
pe_type = 'sine'            # Choose between 'sine', 'sawtooth' or 'sign'
pe_weaving = False

batch_size = 128
sample_len = 80

lr = 5e-7
lr_step_size = 30
lr_gamma = 0.8
epochs = 1000
evaluate_after_epoch = 10
val_split = .1

# Load and preprocess data
all_data = load_oscillatory_data()
train_data0, val_n_test_data = order_data(all_data, sample_len, val_split)
val_data = torch.FloatTensor(val_n_test_data[batch_size:, :, :]).to(device)
test_data = torch.FloatTensor(val_n_test_data[:batch_size, :, :]).to(device)

# Define the model
model = TransformerModel(pe_type=pe_type, pe_weaving=pe_weaving, samp_len=sample_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# Make folders
if not os.path.exists('models_' + pe_type):
    os.makedirs('models_' + pe_type)
if not os.path.exists('plots_' + pe_type):
    os.makedirs('plots_' + pe_type)

# Training loop
train_losses = []
val_losses = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data = torch.FloatTensor(reshuffle_train_data(train_data0, sample_len)).to(device)    # Random cropping
    cur_train_loss = train(train_data, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len)
    train_losses.append(cur_train_loss)
    torch.save(model.state_dict(), 'models_' + pe_type + '/epoch_' + str(epoch).zfill(3) + '.pt')

    if epoch % evaluate_after_epoch == 0:
        plot_examples(val_data, model, criterion, epoch, sample_len, pe_type)

    cur_train_loss = evaluate(train_data, model, criterion, batch_size, sample_len)
    cur_val_loss = evaluate(val_data, model, criterion, batch_size, sample_len)
    val_losses.append(cur_val_loss)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), cur_val_loss, math.exp(cur_val_loss)))
    print('-' * 89)

    scheduler.step()

# Aftermaths - choose best epoch
max_idx = 0
for i in range(epochs):
    if val_losses[i] < val_losses[max_idx]:
        max_idx = i
checkpoint = torch.load('models_' + pe_type + '/epoch_' + str(max_idx).zfill(3) + '.pt')
model.load_state_dict(checkpoint)
test_loss = evaluate(test_data, model, criterion, batch_size, sample_len)
print('Test loss is {:2.4f}'.format(test_loss))

# Store results
plot_learning_curves(train_losses, val_losses, max_idx, pe_type)
with open('models_' + pe_type + '/losses.npy', 'wb') as f:
    np.save(f, train_losses)
    np.save(f,  val_losses)
    np.save(f, max_idx)
