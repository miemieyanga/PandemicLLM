import os
import sys

root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
os.chdir(root_path)
sys.path.append(root_path + 'src')

from utils.basics import init_env_variables, print_important_cfg, time_logger
from tqdm import tqdm
from math import ceil

init_env_variables()

from utils.pkg.distributed import initialize_deepspeed, initialize_distributed
from utils.project.exp import init_experiment
import logging
import hydra
import numpy as np
import pandas as pd

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from covid_llm.agent import DeepSpeedAgent, Agent
from covid_llm.instruction_dataset import InstructionDataset, load_sft_dataset
from covid_llm.model import CovidLLM
from utils.data.covid_data import CovidData
import torch as th

from covid_llm.metrics import calc_prediction_class_distribution


@time_logger()
@hydra.main(config_path=f'{root_path}/configs', config_name='main', version_base=None)
def train_covid_llm(cfg):
    cfg, logger = init_experiment(cfg)
    data = CovidData(cfg=cfg)

    if not th.cuda.is_available():
        cfg.llm.base_model = 'tinygpt'
        cfg.use_bf16 = False
    else:
        cfg.use_bf16 = th.cuda.is_bf16_supported() and cfg.use_bf16

    initialize_distributed(cfg, logger)
    initialize_deepspeed(cfg)
    
    if cfg.get('use_flash_attn', False):  # and ( == 'Ampere':  # CPU Debug only
        logger.critical('Using FlashAttn2 for training')
        from covid_llm.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    else:
        logger.critical('FlashAttn2 disabled for training')
    logger.critical(f'eq_batch_size={cfg.eq_batch_size}, bsz_per_gpu={cfg.bsz_per_gpu}, '
                    f'grad_acc_steps={cfg.grad_acc_steps}')
    
    model = CovidLLM(cfg, data, logger)
    
    if th.cuda.is_available():
        th.cuda.set_device(cfg.local_rank)
        device = th.device("cuda", cfg.local_rank)
    else:
        device =  th.device('cpu')
        
    model.init_rank(cfg.local_rank, device)
    
    if cfg.use_deepspeed:
        logger.critical('Using DeepSpeed agent for training')
        agent = DeepSpeedAgent(model, cfg, data, logger)
    else:
        model = model.to(device)
        logger.critical(f'Using normal agent for training.')
        agent = Agent(model, cfg, data, logger)
        
    print_important_cfg(cfg, logger.debug)
    # Initialize DataLoaders
    batch_size = cfg.world_size * cfg.ds['train_micro_batch_size_per_gpu']
    full_dataset = InstructionDataset(data, cfg, cfg.mode, use_variant_prompt = cfg.training_variant_prompt)

    # ! Full data for link prediction
    train_ids = data.split_ids['train'][:cfg.data.max_train_samples]
    train_data, train_iter, sampler = load_sft_dataset(
        cfg,
        full_dataset=full_dataset, split_ids=train_ids,
        batch_size=batch_size,
        split='train', world_size=cfg.world_size, rank=cfg.local_rank
    )
    
    val_data, val_iter, sampler = load_sft_dataset(
        cfg,
        full_dataset=full_dataset, split_ids=data.split_ids['val'][:cfg.data.max_eval_samples],
        batch_size=cfg.inf_batch_size,
        split='val', world_size=cfg.world_size, rank=cfg.local_rank
    )
    
    test_with_variant_data, test_with_variant_iter, sampler = load_sft_dataset(
        cfg,
        full_dataset=full_dataset, split_ids=data.split_ids['test'][:cfg.data.max_test_samples],
        batch_size=cfg.inf_batch_size,
        split='test', world_size=cfg.world_size, rank=cfg.local_rank
    )
    
    if cfg.test_without_variant:
        test_without_variant_dataset = InstructionDataset(data, cfg, cfg.mode, use_variant_prompt=False)
        _, test_without_variant_iter, _ = load_sft_dataset(
                cfg,
                full_dataset=test_without_variant_dataset, split_ids=data.split_ids['test'][:cfg.data.max_test_samples],
                batch_size=cfg.inf_batch_size,
                split='test', world_size=cfg.world_size, rank=cfg.local_rank
            )


    epochs = min(cfg.get('max_epochs', 1000), ceil(ceil(cfg.total_steps / (len(train_data) / cfg.eq_batch_size))))
    logger.warning(f'Begin training {cfg.total_steps} steps ({epochs} epochs).')
    current_step = 0
    is_eval = cfg.local_rank == 0
    pbar_refresh_freq = max(agent.total_batch_steps // 100, 10)
    pbar = tqdm(total=agent.total_batch_steps, desc="Training", dynamic_ncols=True, disable=cfg.local_rank > 0)
    df_to_save = agent.data.df.copy()
    for epoch_i in range(epochs):
        logger.critical(f'Started epoch {epoch_i}.')
        for batch in train_iter:
            results = agent.train_model_batch(batch, current_step=current_step)
            if is_eval and current_step % cfg.eval_freq == 0 and current_step >= cfg.min_eval_step:
                eval_results = agent.evaluate({'val': val_iter, 'test_with_variant': test_with_variant_iter}, logger)
                results.update(eval_results)
                
                if cfg.test_without_variant:
                    eval_results = agent.evaluate({'test_without_variant': test_without_variant_iter}, logger, df_suffix='without_variant')
                    results.update(eval_results)
                
                # record the best one's results and do zero shot
                if logger.is_the_best_metric(cfg.best_eval_metrics, results, cfg.max_best_eval_metrics):
                    df_to_save = agent.data.df.copy()
                    if cfg.save_model:
                        agent.save_model(cfg.save_path, current_step, is_final=True)
            
            logger.wandb_metric_log({**results, **{'train/epoch': epoch_i}})
            agent.torch_distributed_barrier()

            if current_step % cfg.save_freq == 0 and epoch_i > 0 and th.cuda.is_available():
                agent.save_model(cfg.save_path, current_step)
            if current_step % pbar_refresh_freq == 0:
                pbar.update(pbar_refresh_freq)

            current_step += 1  # Every gradient update or every batch forward
            if current_step >= agent.total_batch_steps:
                break
    pbar.close()
    
    test_df = df_to_save.loc[data.split_ids['test']]
    test_df.to_csv(cfg.save_file)
    logger.critical(f"Saved results to {cfg.save_file}")
    logger.save_file_to_wandb(cfg.save_file, base_path=cfg.out_dir)
    
    # save confusion Matrix
    pred = test_df.pred[~pd.isna(test_df.pred)]
    logger.save_confusion_matrix_to_wandb(df_to_save[cfg.target].loc[pred.keys()], pred, data.mse_val_map, data.label_info.label_token)
    # update final valid and test acc.
    final_results = logger.lookup_metric_checkpoint_by_best_eval(cfg.best_eval_metrics, out_metrics=None, max_val=cfg.max_best_eval_metrics)
    # save histograms
    class_distribution = calc_prediction_class_distribution(df_to_save['confidence'][df_to_save['confidence'].notnull()])
    logger.save_histograms_to_wandb(class_distribution)
    # compare the results with variant and that without variant
    logger.compare_results(test_df, cfg.target)
    
    logger.wandb_summary_update(final_results, finish_wandb=True)


if __name__ == "__main__":
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        train_covid_llm()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profiling.prof')
