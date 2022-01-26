# Two (gyro's) is all you need

> This repository contains our project assignment made for the Technion's EE course 046211 "Deep Learning" in  2022.


## Introduction

<img src="https://github.com/?.png" width="700">




## Files in the repository

|File name         | Purpose |
|----------------------|------|
|`mainTransformer.py`| main script for training and evaluating the model, includes the parameters and PE type selection |
|`transformerModel.py`| includes pytorch model and PE layers definitions |
|`evaluationsNplots.py`| here are the trinity - train, evaluate, and plots (examples and learning curves)|
|`dataPreprocessing.py`| loades 'wisdm_oscillatory.npy' numpy datafile and preprocesses it for the model |
|`wisdmDataLoader.py`| makes 'wisdm_oscillatory.npy' numpy datafile from the WISDM datasets |
|`wisdm_oscillatory.npy`| includes the oscillatory data samples from the original WISDM dataset |


### mainTransformer

|Parameter         | Purpose |
|----------------------|------|
|`pe_type`| selection of Positional Encoding layer - chose 'sine', 'sawtooth', or 'sign' |
|`pe_weaving`| enables weaved PE layers of the two inputs, otherwise use independent |
|`batch_size`| batch size |
|`sample_len`| datapoints per sample (sampling rate is 20Hz) so the default 80 is 4 seconds |
|`lr`| learning rate |
|`lr_step_size`| learning rate step size, for the scheduler|
|`lr_gamma`| learning rate multiplication factor, for the scheduler|
|`epochs`| number of epochs, training time is ~4 seconds per epoch on Titan V | 
|`evaluate_after_epoch`| after this number of epochs plots are generated |
|`val_split`| validation split, includes test data as well |

### transformerModel

|Function/class        | Purpose |
|----------------------|------|
|`PositionalEncodingLayer'| defines independent PE layer for a single input|
|`WeavedPositionalEncodingLayer`| defines weaved PE layer for a single input|
|`_trinagular_mask`| make triangular mask for attention causality |
|`TransformerModel`| defines the main torch model with all layers |


### evaluationsNplots
|Function/class        | Purpose |
|----------------------|------|
|`train`|  runs one loop over all the data it gets, prints losses each 5 batches and returns training loss |
|`plot_examples`|  plots 4 random examples of model outputs vs. ground truth|
|`plot_learning_curves`|  plots learning curves post training|
|`evaluate`|  evaluates the model and returns loss |


### dataPreprocessing
|Function/class        | Purpose |
|----------------------|------|
|`order_data`| splits the data to training and validation and prepares it according to the desired sample length |
|`reshuffle_train_data`| reshuffles the data for each epoch |
|`get_batch`| reshapes  the input data for each batch |
|`mean_std_norm`| normalizes the data |


### wisdmDataLoader

to load the dataset from scratch, make sure to download it and have it in the `wisdm-dataset\raw` subfolders
|Function/class        | Purpose |
|----------------------|------|
|`import_all_data`| imports all WISDM dataset from the raw files  |
|`import_oscillatory_data`| like `import_all_data`, but only for the usable samples |
|`load_all_data`| load all data from `wisdm_data.npy` if it exists, else uses `import_all_data` to make it |
|`load_oscillatory_data`| load oscillatory data from `wisdm_oscillatory.npy` if it exists, else uses `import_oscillatory_data` to make it |


## Datasets

## Model
> DESAT architecture
## Results

## References

## Contact

Gal Ness - [gn@technion.ac.il](mailto:gn@technion.ac.il)
Elad Zohar - [elad.zohar@campus.technion.ac.il](mailto:elad.zohar@campus.technion.ac.il)

## Project report
[Two is all you need](/aux/Two_is_all_you_need.pdf)
 
## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.6.8`|
|`torch`|  `1.10.1`|
|`torchtext`|  `0.11.1`|
|`scikit-learn`|  `0.24.2`|
|`scipy`|  `1.5.4`|
|`numpy`|  `1.19.5`|
|`pandas`|  `1.1.5`|
|`matplotlib`|  `3.3.4`|




