import logging
from collections import defaultdict

import hydra
import wandb
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
import numpy as np
from sklearn.metrics import confusion_matrix

install()
logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(
        rich_tracebacks=False, tracebacks_suppress=[hydra],
        console=Console(width=165),
        enable_link_path=False
    )],
)
# Default logger
logger = rich_logger = logging.getLogger("rich")
# from rich.traceback import install
# install(show_locals=True, width=150, suppress=[hydra])
logger.info("Rich Logger initialized.")

PercentageFloatMetrics = ['acc']


def get_best_by_val_perf(res_list, prefix, metric):
    results = max(res_list, key=lambda x: x[f'val_{metric}'])
    return {f'{prefix}_{k}': v for k, v in results.items()}


def judge_by_partial_match(k, match_dict, case_sensitive=False):
    k = k if case_sensitive else k.lower()
    return len([m for m in match_dict if m in k]) > 0


def metric_processing(log_dict):
    # Round floats and process percentage
    for k, v in log_dict.items():
        if isinstance(v, float):
            is_percentage = judge_by_partial_match(k, PercentageFloatMetrics)
            if is_percentage:
                log_dict[k] *= 100
            log_dict[k] = round(log_dict[k], 4)
    return log_dict


def get_split(metric):
    split = 'train'
    if 'val' in metric:
        split = 'val'
    elif 'test' in metric:
        split = 'test'
    return split


class WandbExpLogger:
    '''Wandb Logger with experimental metric saving logics'''

    def __init__(self, cfg):
        self.wandb = cfg.wandb
        self.wandb_on = cfg.wandb.id is not None
        self.local_rank = cfg.local_rank
        self.logger = rich_logger  # Rich logger
        self.logger.setLevel(getattr(logging, cfg.logging.level.upper()))
        self.info = self.logger.info
        self.critical = self.logger.critical
        self.warning = self.logger.warning
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.error = self.logger.error
        self.log_metric_to_stdout = (not self.wandb_on and cfg.local_rank <= 0) or \
                                    cfg.logging.log_wandb_metric_to_stdout
        # ! Experiment Metrics
        self.results = defaultdict(list)

    # ! Log functions
    def log(self, *args, level='', **kwargs):
        if self.local_rank <= 0:
            self.logger.log(getattr(logging, level.upper()), *args, **kwargs)

    def log_fig(self, fig_name, fig_file):
        if self.wandb_on and self.local_rank <= 0:
            wandb.log({fig_name: wandb.Image(fig_file)})
        else:
            self.error('Figure not logged to Wandb since Wandb is off.', 'ERROR')

    def wandb_metric_log(self, metric_dict, level='info'):
        # Preprocess metric
        metric_dict = metric_processing(metric_dict)
        for metric, value in metric_dict.items():
            self.results[metric].append(value)

        if self.wandb_on and self.local_rank <= 0:
            wandb.log(metric_dict)
        if self.log_metric_to_stdout:
            self.log(metric_dict, level=level)

    def lookup_metric_checkpoint_by_best_eval(self, eval_metric, out_metrics=None, max_val=True):
        if len(self.results[eval_metric]) == 0:
            return {}
        best_val_ind = self.results[eval_metric].index(max(self.results[eval_metric])) if max_val\
            else self.results[eval_metric].index(min(self.results[eval_metric]))
        out_metrics = out_metrics or self.results.keys()
        return {m: self.results[m][best_val_ind] for m in out_metrics}
    
    def is_the_best_metric(self, eval_metric, cur_data, max_val):
        if len(self.results[eval_metric]) == 0:
            return True
        cur_value = cur_data[eval_metric]
        is_the_best = (cur_value>=max(self.results[eval_metric])) if max_val\
            else (cur_value<=min(self.results[eval_metric]))
        return is_the_best

    # ! Experiment metrics functions
    def wandb_summary_update(self, result, finish_wandb=True):
        # ! Finish wandb
        if self.wandb_on and self.local_rank <= 0:
            wandb.summary.update(result)
            if finish_wandb:
                wandb.finish()

    def save_file_to_wandb(self, file, base_path, policy='now', **kwargs):
        if self.wandb_on and self.local_rank <= 0:
            wandb.save(file, base_path=base_path, policy=policy, **kwargs)
            
    def save_histograms_to_wandb(self, class_distribution):
        if self.wandb_on and self.local_rank <= 0:
            for k, v in class_distribution.items():
                table = wandb.Table(data=v, columns=[k])
                wandb.log({f'class_distribution/{k}': wandb.plot.histogram(table, k,
 	                title=f"{k} Class Distribution")})
                
    def compare_results(self, test_res, target, metrics='mse'):
        # build function map for metircs
        cats = ['SUBSTANTIAL DECREASING', 'MODERATE DECREASING', 'STABLE', 'MODERATE INCREASING', 'SUBSTANTIAL INCREASING']
        def get_mse(test_dct, target):
            pred = max(list(test_dct.items()), key=lambda x:x[1])[0]
            mse = (cats.index(pred) - cats.index(target))**2
            return mse
        metrics_fn = {
            # 'brier score': get_brier_score,
            'mse': get_mse,
            # 'wmse': get_wmse,
            # 'acc': get_acc
        }
        
        # calculate metrics
        ind_res = {}
        for i, week in enumerate(test_res['Week_start'].unique()):
            cur_week_data = test_res[test_res['Week_start'] == week][[target, 'confidence', 'confidence_without_variant']]
            ori_metrics = 0
            new_metrics = 0
            cnt = 0
            for item in cur_week_data.iterrows():
                _, (cur_target, cur_conf, cur_conf_zero_shot) = item
                ori_metrics += metrics_fn[metrics](cur_conf, cur_target)
                new_metrics += metrics_fn[metrics](cur_conf_zero_shot, cur_target)
                cnt += 1
            ori_metrics = round(ori_metrics/cnt,3)
            new_metrics = round(new_metrics/cnt,3)
            ind_res[week] = {f'{metrics}_with_variant': ori_metrics, 
                             f'{metrics}_without_variant': new_metrics, 
                             'week': i}
        
        # log to wandb
        for week, acc_dct in ind_res.items():
            wandb.log(acc_dct)
                
    def save_confusion_matrix_to_wandb(self, targets, predictions, mse_val_map, label_token):
        if self.wandb_on and self.local_rank <= 0:
            
            # 'SUBSTANTIAL DECREASING': '0SUBSTANTIAL DECREASING', add num in front of label names
            label_names_dct = {k:str(v)+k for k,v in mse_val_map.items()}
            
            # get ordered label names
            label_names = []
            for k, v in mse_val_map.items():
                if k in label_token.to_list():
                    label_names.append([label_names_dct[k], v])
            label_names.sort(key=lambda x:x[1])
            label_names = list(np.array(label_names)[:, 0])
            
            # confusion matrix -- type 1
            # 'SUBSTANTIAL DECREASING' --> 0
            predictions_digits = predictions.apply(mse_val_map.get)
            targets_digits = targets.apply(mse_val_map.get)
            wandb.log({'Confusion Matrix/Confusion Matrix1': wandb.plot.confusion_matrix(
                y_true = targets_digits.to_list(), preds = predictions_digits.to_list(), class_names = label_names)})
            
            # confusion matrix -- type 2
            # 'SUBSTANTIAL DECREASING' --> '0SUBSTANTIAL DECREASING'
            new_preds = predictions.apply(label_names_dct.get)
            new_targets = targets.apply(label_names_dct.get)
            cm = confusion_matrix(new_targets.to_list(), new_preds.to_list(), labels=label_names)
            cm_sum = np.sum(cm, axis=0) + 10e-5
            cm = cm/cm_sum
            cm = np.round(cm, decimals=2)
            wandb.log({'Confusion Matrix/Confusion Matrix2': wandb.plots.HeatMap(label_names, label_names, cm, show_text=True)})
        

def wandb_finish(result=None):
    if wandb.run is not None:
        wandb.summary.update(result or {})
        wandb.finish()
