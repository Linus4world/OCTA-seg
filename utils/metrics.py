
import numpy as np
import torch
from monai.metrics import MeanIoU, ROCAUCMetric, MSEMetric

class Task:
    VESSEL_SEGMENTATION = "ves-seg"
    AREA_SEGMENTATION = "area-seg"
    IMAGE_QUALITY_CLASSIFICATION = "img-qual-clf"
    RETINOPATHY_CLASSIFICATION = "ret-clf"
    GAN_VESSEL_SEGMENTATION = "gan-ves-seg"
    CONSTRASTIVE_UNPAIRED_TRANSLATION = "cut"

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

class MacroDiceMetric:
    def __init__(self) -> None:
        self.preds=[]
        self.labels=[]

    def get_dice(self, gt, pred, classId=1):
        if np.sum(gt) == 0:
            return np.nan
        else:
            intersection = np.logical_and(gt == classId, pred == classId)
            dice_eff = (2. * intersection.sum()) / (gt.sum() + pred.sum())
            return dice_eff

    def __call__(self, y_pred: list[torch.Tensor], y: list[torch.Tensor]):
        for y_pred_i, y_i in zip(y_pred, y):
            for layer in range(len(y_pred_i)):
                self.preds.append(y_pred_i[layer].detach().cpu().numpy())
                self.labels.append(y_i[layer].detach().cpu().numpy())

    def aggregate(self) -> torch.Tensor:
        if len(self.preds) > 0:
            dice_list = []
            for gt_array, pred_array in zip(self.labels, self.preds):
                dice = self.get_dice(gt_array.astype(np.float32), pred_array.astype(np.float32), 1)
                dice_list.append(dice)
            mDice = np.nanmean(dice_list)
            return torch.tensor(mDice)
        else:
            return torch.tensor(0)

    def reset(self):
        self.preds = []
        self.labels = []


class MetricsManager():
    def __init__(self, task: Task):
        if task == Task.VESSEL_SEGMENTATION or task == Task.AREA_SEGMENTATION or task == Task.GAN_VESSEL_SEGMENTATION:
            self.metrics = {
                "DSC": MacroDiceMetric(),
                "IoU": MeanIoU(include_background=True, reduction="mean")
            }
            self.comp = "DSC"
        elif task == Task.IMAGE_QUALITY_CLASSIFICATION or task == Task.RETINOPATHY_CLASSIFICATION:
            self.metrics =  {
                "QwK": QuadraticWeightedKappa(),
                "ROC_AUC": ROCAUCMetric(average="macro")
            }
            self.comp = "QwK"
        else:
            self.metrics = {}
            self.comp = None

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
