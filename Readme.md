<div align="center">

# PandemicLLM: Adapting Large Language Models to Forecast Pandemics in Real-time: A COVID-19 Case Study

Hongru Du\*, Jianan Zhao\*, Yang Zhao\*, Shaochong Xu, Xihong Lin, Lauren M. Gardner\#, Yiran Chen\#, and Hao (Frank) Yang\#. 


\* *Equal Contribution*, \# *Corresponding Authors*

</div>

![Cow1](https://github.com/AndyJZhao/CovidLLM/blob/main/PandemicLLM.jpg?raw=true)

# Steps to reproduce

## Python Environment
We conduct experiments under CUDA 12.1 and Ubuntu 22.04 on Nvidia A100 GPU. 

Create conda environment

```shell
conda create -n PLLM python=3.10
source activate PLLM
```

Install the related packages. 
```shell
pip install -r requirements.txt
conda install mpi4py
```
## Set up Environment Variables

We recommend to use [wandb](https://wandb.ai/) to monitor and manage the training process. Before runing the training script, make sure to set up the wandb environment properly. **You can also skip wandb by setting the `use_wandb=False`**. 

The `HF_ACCESS_TOKEN` is also necessary for accessing [LLaMA](https://huggingface.co/) models. 

```shell
mkdir configs/user
echo "
# @package _global_
env:
  vars:
    WANDB_API_KEY: YOUR_WANDB_API_KEY
    HF_ACCESS_TOKEN: YOUR_HF_ACCESS_TOKEN
" >> configs/user/env.yaml
```

## Supervised  Fine-tuning (SFT)
CovidLLM supports instruction fine-tuning a LLM on graph. An RNN is used to map the continuous sequence to text space (as tokens). We recommend to use BF16 for stable training. To reproduce our results, you should run the following scripts under the same environment we listed above:
### PandemicLLM-7B 1-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-7b lr=2e-05 save_model=False seed=2024 splits_type=base_splits target=t1 total_steps=1501 use_cont_fields=True use_deepspeed=True use_wandb=True
```
### PandemicLLM-7B 3-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-7b lr=2e-05 save_model=False seed=2024 splits_type=sta_aug_splits target=t3 total_steps=1501 use_cont_fields=True use_deepspeed=True use_wandb=True
```

### PandemicLLM-13B 1-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-13b lr=1e-05 save_model=False seed=2023 splits_type=sta_aug_splits target=t1 total_steps=1501 use_cont_fields=True use_deepspeed=True use_wandb=True
```

### PandemicLLM-13B 3-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-13b lr=2e-05 save_model=False seed=2023 splits_type=base_splits target=t3 total_steps=1501 use_cont_fields=True use_deepspeed=True use_wandb=True
```

### PandemicLLM-70B 1-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-70b lr=1e-05 save_model=False seed=2023 splits_type=sta_dy_aug_splits target=t1 total_steps=1501 use_cont_fields=True use_deepspeed=True use_int4=True use_wandb=True
```

### PandemicLLM-70B 3-week
```shell
cd src/scripts
python run_covid_llm_sft.py data_file=processed_v5_4.pkl data_file=processed_v5_4.pkl eq_batch_size=4 in_weeks=3 llm_base_model=llama2-70b lr=2e-05 save_model=False seed=2023 splits_type=sta_dy_aug_splits target=t3 total_steps=1501 use_cont_fields=True use_deepspeed=True use_int4=True use_wandb=True
```


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


### Credits

This work is supported by NSF Award ID 2229996 and computational resources at NSF AI Institute for Edge Computing Leveraging Next Generation Networks (Athena), Duke University. 
