# NASNet-Large Features with MRMR, LDA, and KNN

MATLAB pipeline using fine-tuned NASNet-Large global-average-pooling features, MRMR selection, and validation-tuned LDA/KNN classifiers for TartanAir **Nature**, **Rural**, and **Urban** scenes.

| Classifier | Test accuracy |
|---|---:|
| LDA | 63.26% |
| KNN | **64.09%** |

## Run

Configure the provided dataset under `data/` or through `TARTANAIR_DATASET_ROOT`, then run:

```matlab
train_nasnet
classify_nasnet_features
```

Requires Deep Learning Toolbox, Statistics and Machine Learning Toolbox, Computer Vision Toolbox, and the NASNet-Large support package.

Author: [Zeyad Elsaber](https://github.com/zeyadelsaber), University of Rome Tor Vergata.
