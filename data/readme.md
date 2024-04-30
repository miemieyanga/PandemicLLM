# PandemicLLM Dataset

## Dataset 

PandemicLLM is fine-tuned using multiple disparate categories of data including spatial (static), epidemiological time series (dynamic), genomic surveillance (dynamic), and public health policy data (dynamic). Our data covers all 50 states in the United States, ensuring a comprehensive nationwide scope for our study. All the spatial data are available at the state resolution, while all the time-varying data are available at the weekly resolution. Please check our paper for more details about the data.

### Load the Dataset


```
import pickle
path = '/data/processed_v5_4.pkl'
data = pickle.load(open(path, 'rb'))
```

### Data Columns

`data`: raw data
`sta_dy_aug_data`: `data` with static and dynamic augmentation. 

`base_splits`, `sta_aug_splits`, `dy_aug_splits`, `sta_dy_aug_splits`: indexes for `sta_dy_aug_data`, representing for raw data, data with static augmentation, data with dynamic augmentation, and data with static and dynamic augmentation respectively. 

`label_info`, `mse_val_map`: information used for training


## Raw Data and Source Code for Processing Data

The raw data and source code are located at `\data\src`. Before runing the code, please unzip the `raw_data.zip` with:

```
cd data/src
unzip raw_data.zip
```