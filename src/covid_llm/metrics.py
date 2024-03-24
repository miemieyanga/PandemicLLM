import numpy as np
from sklearn.metrics import mean_squared_error


def calc_acc(label, pred):
    return np.mean(label == pred) if len(label) == len(pred) else -1


def calc_mse_from_cls_labels(label, pred, val_map):
    val_map_func = np.vectorize(dict(val_map).get)
    y_true, y_pred = val_map_func(label), val_map_func(pred)
    return mean_squared_error(y_true, y_pred)

# weighted mse
def calc_weighted_mse_from_cls_labels(label, confidence, val_map):
    assert len(confidence) > 0
    val_map_func = np.vectorize(dict(val_map).get)
    labels = val_map_func(label)

    res = []
    for i, conf in enumerate(confidence):
        cur_label = labels[i]
        cur_res = 0
        for k,v in conf.items():
            
            # if no matched label, wmse is set to 16
            if np.isnan(v):
                cur_res = 16
                break
            
            k = dict(val_map).get(k)
            diff = k-cur_label
            cur_res += v*(diff**2)
            
        res.append(cur_res)
    res = np.array(res)
    return res.sum()/len(res)


# brier score
def calc_brier_score(label, confidence, val_map):
    assert len(confidence) > 0
    val_map_func = np.vectorize(dict(val_map).get)
    labels = val_map_func(label)

    res = []
    for i, conf in enumerate(confidence):
        cur_label = labels[i]
        cur_res = 0
        for k,v in conf.items():
            # if no matched label, bs is set to 5
            if np.isnan(v):
                cur_res = 5
                break
            k = dict(val_map).get(k)
            cur_res += (v-(k==cur_label))**2
        res.append(cur_res)
    res = np.array(res)
    return res.sum()/len(res)


def calc_prediction_distribution(pred, class_names):
    unique, counts = np.unique(pred, return_counts=True)
    res = {val: cnt / len(pred)
           for val, cnt in zip(unique, counts)}
    # Set non-predicted labels to 0 frequency
    for cls in class_names:
        res[cls] = res.get(cls, 0)
    return res


def calc_prediction_class_distribution(confidence):
    distribution = {}
    for rows in confidence:
        for k,v in rows.items():
            if k not in distribution.keys():
                distribution[k] = [[v]]
            else:
                distribution[k].append([v])
    return distribution
