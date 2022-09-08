
from typing import Union
import numpy as np
import torch
from monai.metrics import DiceMetric, MeanIoU, ROCAUCMetric
from utils.cl_dice_loss import clDiceLoss
from monai.losses import DiceLoss


class Task:
    VESSEL_SEGMENTATION = "ves-seg"
    AREA_SEGMENTATION = "area-seg"
    IMAGE_QUALITY_CLASSIFICATION = "img-qual-clf"
    RETINOPATHY_CLASSIFICATION = "ret-clf"

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator

class QuadraticWeightedKappa:
    """
    Implementation following https://github.com/zhuanjiao2222/DRAC2022/blob/main/evaluation/metric_classification.py
    """
    def __init__(self) -> None:
        self.preds=[]
        self.labels=[]

    def __call__(self, y_pred: list[torch.Tensor], y: list[torch.Tensor]) -> None:
        for y_pred_i, y_i in zip(y_pred, y):
            pred_label = np.argmax(y_pred_i.detach().cpu().numpy())
            true_label = np.argmax(y_i.numpy())
            self.preds.append(pred_label)
            self.labels.append(true_label)

    def aggregate(self) -> torch.Tensor:
        if len(self.preds) > 0:
            return torch.tensor(quadratic_weighted_kappa(self.labels,self.preds))
        else:
            return torch.tensor(0)

    def reset(self):
        self.preds = []
        self.labels = []

class SigmoidDiceBCELoss():
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return (self.dice(y_pred, y) + self.bce(y_pred, y))/2
       

class MetricsManager():
    def __init__(self, task: Task):
        if task == Task.VESSEL_SEGMENTATION or task == Task.AREA_SEGMENTATION:
            self.metrics = {
                "DSC": DiceMetric(include_background=True, reduction="mean"),
                "IoU": MeanIoU(include_background=True, reduction="mean")
            }
            self.comp = "DSC"
        elif task == Task.IMAGE_QUALITY_CLASSIFICATION or task == Task.RETINOPATHY_CLASSIFICATION:
            self.metrics =  {
                "QwK": QuadraticWeightedKappa(),
                "ROC_AUC": ROCAUCMetric(average="macro")
            }
            self.comp = "QwK"

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        for v in self.metrics.values():
            v(y_pred=y_pred, y=y)

    def aggregate_and_reset(self, prefix: str = ''):
        d = dict()
        for k,v in self.metrics.items():
            d[f'{prefix}_{k}'] = v.aggregate().item()
            v.reset()
        return d

    def get_comp_metric(self, prefix: str):
        return f'{prefix}_{self.comp}'

class WeightedCosineLoss():
    def __init__(self, weights=[1,1,1]) -> None:
        self.weights = weights

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred_norm = torch.nn.functional.normalize(y_pred, dim=-1)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=y_pred.size(-1)).float()
        cosine_sim = torch.sum(y_pred_norm*y_one_hot, dim=1)
        sample_weights = torch.tensor([self.weights[y_i] for y_i in y], device=y.device)
        weighted_cosine_sim = sample_weights * cosine_sim
        return 1- (torch.sum(weighted_cosine_sim)/sample_weights.sum())


def get_loss_function(task: Task, config: dict) -> tuple[str, Union[clDiceLoss, SigmoidDiceBCELoss, torch.nn.CrossEntropyLoss]]:
    if task == Task.VESSEL_SEGMENTATION:
        return 'clDiceLoss', clDiceLoss(alpha=config["Train"]["lambda_cl_dice"], sigmoid=True)
    elif task == Task.AREA_SEGMENTATION:
        return 'DiceBCELoss', SigmoidDiceBCELoss()
    else:
        # return 'CrossEntropyLoss', torch.nn.CrossEntropyLoss(weights=torch.tensor([1/0.537,1/0.349,1/0.115]))
        return "CosineEmbeddingLoss", WeightedCosineLoss(weights=[1/0.537,1/0.349,1/0.115])

def get_loss_function_by_name(name: str, config: dict) -> Union[clDiceLoss, SigmoidDiceBCELoss, torch.nn.CrossEntropyLoss, WeightedCosineLoss]:
    loss_map = {
        # "clDiceLoss": clDiceLoss(alpha=config["Train"]["lambda_cl_dice"], sigmoid=True),
        "DiceBCELoss": SigmoidDiceBCELoss(),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(weight=1/torch.tensor(config["Data"]["class_balance"], device=config["General"]["device"])),
        "CosineEmbeddingLoss": WeightedCosineLoss(weights=1/torch.tensor(config["Data"]["class_balance"], device=config["General"]["device"]))
    }
    return loss_map[name]
