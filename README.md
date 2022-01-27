# Two (gyro's) is all you need

> This repository contains our project assignment made for the Technion's EE course 046211 "Deep Learning" in  2022.


## Introduction

From the information in the temporal evolution of two axes, the third
can be predicted by exploiting inter-axis correlations. For this, we
employ a multi-head auto-attention transformer adapted to process two
input signals. We further show that, using novel positional encoding
layers, cross-temporal correlations can boost the prediction accuracy.

The physical motion of a device, whether a phone or a smartwatch, is
usually not restricted to a single axis, and it tends to feature
periodic patterns. For example, a running carrying a smartwatch moves
his arm in a particular motion, but unless pathologically intending, his
movement is not co-linear with the watch’s coordinate system. Therefore,
the recorded signal would involve accelerations in each of the three

To attain the inter-axis prediction, we use a multi-headed transformer
model. Self-attention transformers were designed to handle sequential
input data. However, unlike recurrent neural nets, transformers do not
process the data in order. Instead, the attention mechanism provides
context for any position in the input sequence. A cardinal component of
language attention networks is positional encoding, which in our case
translates to temporal encoding. We propose nontrivial temporal encoders
and comparatively study their performance.




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


## Data & preprocessing

### Dataset

We use the WISDM (Wireless Sensor Data Mining) dataset for human
activity recognition (HAR), which was recorded and labeled in the WISDM
lab at Fordham University with the original intention of training
activity classification models. It contains 18 different labeled
activities (labeled A-K) of 51 users, which vary from walking and
running to eating pasta and folding clothes, recorded using
accelerometers and gyroscopes of both a phone and a smartwatch.

The instances in the dataset are about 3 minutes long, with a sampling
rate of 20Hz, which gives about 3600 points of data for each axis
(*x*, *y*, *z*). An example from the dataset reads:

```
User, activity, timestamp,       x,           y,           z;
1600, A,        252207918580802, -0.85321045, 0.29722595,  0.8901825;
1600, A,        252207968934806, -0.8751373,  0.015472412, 0.16223145;
1600, A,        252208019288809, -0.72016907, 0.38848877,  -0.28401184;
1600, A,        252208069642813, -0.57164,    1.2274017,   -0.2416687;
```

We chose to focus our attention on the gyroscope reading of specific
watch and phone activities which feature some temporal periodicity, as
elaborated bellow. The input data therefore includes 300 useful samples
overall.

### Identifying oscillatory data

At first, our model (detailed below) utilized the entirety of the WISDM
dataset, and performed rather poorly. Specifically, we noticed the
model’s output signal would tend to “die out” after a few epochs. This
prompted us to first try solving the problem using a more traditional
recurrent neural network (RNN). The idea is that if the RNN would also
perform poorly, then there is an inherent problem in our data. Using an
of-the-shelf RNN, we got similar results to those of the transformer.
Our next step was to inspect the dataset a little more carefully and
find that a lot of the activities recorded on it showed little to no
signal most of the time, and zero periodicity. We thus decided to choose
the subset of activities that did exhibit temporally periodic signals,
namely, sports activities such as walking, jogging, dribbling etc., this
can be justified by the fact that wearable motion sensors are primarily
used for recording such activities.

### Sampling and augmentations

<figure>
<img src="/aux/sample_example.png" style="width:55.0%"
alt="Example of three axes data with temporal correlations. Different colors represent different axes, all normalized to zero mean and unity standard deviation, and plotted vs. time ticks of duration 1/20Hz=50ms." />
</figure>
> Example of three axes data with temporal correlations. Different colors represent different axes, all normalized to zero mean and unity standard deviation, and plotted vs. time ticks of duration 1/20Hz=50ms.

The data is divided into 12k sequences, each 80 points (4 seconds) long.
Above is an example of such a sequence (sample). First, we chop half a
sample at the beginning and end of each long recording to avoid
irregularities. Then, we reshape the data to have mini-sequences of
around 10 fold of the desired lengths. The first sample of each
mini-sequence is allocated for validation/testing. The rest was chopped
with 71 points of buffer on the edges, which allows us to randomly
sample the data from a different starting point each epoch, effectively
increasing the number of new instances the model “sees” each epoch by a
factor of 71. The specific amount of mini-sequences and buffer length
automatically vary according to the validation split parameter.

We also use random temporal flipping and sign inversion augmentations to
provide even more diversity in the sequences the model encounters. All
of which are made to all three axes simultaneously to preserve the
inter-axis correlations.

## Dual encoder self-attention transformer (DESAT)

### Architecture

The DESAT model is a multi-headed attention transformer (reference
below) that utilizes two encoders and one decoder, corresponding to the
two axes input to one axis output.

Essentially, we modify the usual sequence-to-sequence transformer by
applying two additional linear feed-forward layers: one combining the
positional encoded inputs that are fed forward to the decoder without
going through the attention mechanism, and another combining the two
encoders’ attention outputs into a single layer, such that the
dimensions of the inputs to the decoder layer are the same as in a
vanilla sequence-to-sequence model, but encode data derived from two
inputs. Both combining layers mentioned above are fully connected
layers, to allow the model to learn the weighted correlation between the
three axes. In addition, the model is designed with integrated
positional encoding options, which enable the exploration of a few novel
ideas, as detailed below.

<figure>
<img src="/aux/model.png" style="width:90.0%"
alt="DESAT architecture. A dual encoder design is linked to a single decoder, all containing a self attention mechanism." />
</figure>
>DESAT architecture. A dual encoder design is linked to a single decoder, all containing a self attention mechanism.

### Positional encoding beyond the Sine

A key ingredient of attention transformers is a positional encoding
layer. This is a quasi-spanning set of periodic functions added to the
input data to break the translational symmetry of fully connected
(attention) layers. The original implementation used a Sine-Cosine pair
of decreasing frequencies to extract the input sequence’s positional
(temporal) information.

In the following, we explore different positional encoding concepts
which diverge from the mainstream Sine-Cosine set. First is the sawtooth
and the sign function employed on the original Sine-Cosine encoding such
that the final encoding is a binary version of the original one. A more
advanced positional encoding manipulation we explore is weaving.
Specifically, we weave the encoding in a more intricately – instead of
using the Sine function for even cells and the Cosine for odd cells, we
weave another set of Sine + Cosine which are now inlaid every four
cells. This method, which is visualized in the figure below should allow
the model to leverage differences in the inter-axis correlations as it
embeds an intrinsic phase delay on the positional encoding layer. This
again is done for all three (sine, sawtooth, sign) positional encoding
types we examine.

<figure>
<img src="/aux/PEfigure" style="width:85.0%"
alt="Positional encoding weaving." />
</figure>
> Positional encoding weaving. Example of the first depth layers where the positional encoding frequency kept to its highest. (a) independent positional encoding, both channels (x and y) are used with the same encoding, in this example – sign of sine and cosine. (b) weaved positional encoding, here at each frequency the channels feature one in-phase repetition and one out-of-phase repetition.

## Results

After the model finishes the 1k epochs training, it loops through the
validation losses of each epoch and chooses the best one, it then checks
the model on the test set and outputs the resulting loss. The best
epochs and test losses results are detailed in the following table:

|        Model         | Best epoch | Test loss |
|:--------------------:|:----------:|:---------:|
|   Independent sine   |    839     |  1.7217   |
| Independent sawtooth |    946     |  1.7215   |
|   Independent sign   |    839     |  1.7275   |
|     Weaved sine      |    840     |  1.7178   |
|   Weaved sawtooth    |    946     |  1.7177   |
|     Weaved sign      |    840     |  1.7268   |

>DESAT different positional encoding performances

All models discussed above achieved similar results with marginal
advantage in favor of the weaved sine model. Moreover, most models
achieved optimal performance around epoch 840, which we attribute to the
scheduler parameters.

Here is a couple of examples for each of the weaved sine positional
encoding prediction DESAT model, comparing the predicted signal (red)
and the ground truth (blue):

<img src="/aux/EE_sine_weaved.png" alt="examples" />

## References
-   **The dataset**: Gary M. Weiss et al. Smartphone and
    smartwatch-based biometrics using activities of daily living. *IEEE
    Access*, **7** 133190 (2019). [doi:10.1109/access.2019.2940729][http://dx.doi.org/10.1109/access.2019.2940729].

-   **Self attention transformers**: Ashish Vaswani et al. Attention is
    all you need. [arXiv:1706.03762][https://arxiv.org/abs/1706.03762] (2017).

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




