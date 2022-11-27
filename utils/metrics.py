
from typing import Union
import numpy as np
import torch
from monai.metrics import MeanIoU, ROCAUCMetric, MSEMetric
from utils.cl_dice_loss import clDiceBceLoss, clDiceLoss
from monai.losses import DiceLoss
from torch.cuda.amp.grad_scaler import GradScaler
from models.noise_model import NoiseModel

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


class DiceBCELoss():
    def __init__(self, sigmoid=False):
        super().__init__()
        if sigmoid:
            self.bce = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce = torch.nn.BCELoss()
        self.dice = DiceLoss(sigmoid=sigmoid)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        return (self.dice(y_pred, y) + self.bce(y_pred, y))/2

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



def quadratic_kappa_coefficient(output, target):
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK



def quadratic_kappa_loss(output, target, scale=2.0):
    QWK = quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss

class QWKLoss(torch.nn.Module):
    def __init__(self, scale=2.0, num_classes=3):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes

    def forward(self, output, target):
        # Keep trace of output dtype for half precision training
        target = torch.nn.functional.one_hot(target.squeeze().long(), num_classes=self.num_classes).to(target.device).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return quadratic_kappa_loss(output, target, self.scale)

class WeightedMSELoss():
    def __init__(self, weights: list) -> None:
        self.weights = weights
        self.mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_per_sample = self.mse(y_pred, y)
        sample_weights = torch.tensor([self.weights[y_i] for y_i in y.long()], device=y.device)
        weighted_loss = loss_per_sample*sample_weights
        return torch.sum(weighted_loss)/sample_weights.sum()

class LSGANLoss(torch.nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0) -> None:
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = torch.nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss.mean()

class PatchNCELoss(torch.nn.Module):
    def __init__(self, batch_size: int, nce_T = 0.07):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.batch_size = batch_size
        self.nce_T = nce_T

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        batch_dim_for_bmm = self.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class NCELoss():
    def __init__(self, lambda_NCE: float, nce_layers: list[int], device: str, batch_size: int) -> None:
        self.lambda_NCE = lambda_NCE
        self.criterionNCE = []
        self.nce_layers = nce_layers
        self.device = device
        
        for nce_layer in nce_layers:
            self.criterionNCE.append(PatchNCELoss(batch_size).to(self.device))

    def __call__(self, feat_q_pool: list[torch.Tensor], feat_k_pool: list[torch.Tensor]) -> torch.Tensor:
        n_layers = len(self.nce_layers)
        total_nce_loss = 0.0
        for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss: torch.Tensor = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

class AtLoss(torch.nn.Module):
    """
    Computes an adversarial training loss by finding a small perturbation r_adv via power iteration that when applied to the
    input image maximally increases the prediction error. The prediction model will then learn to generalize to this attack.
    """
    def __init__(self, scaler: GradScaler, loss_fun: torch.nn.Module, grid_size=None, eps=1.0, ip=1, alpha=1, grad_align_cos_lambda=0) -> None:
        """
        Parameters:
            - scaler: GradScaler for amp training
            - grad_size: control points for perturbation. For pixelwise attacks set to None
            - eps: magnitude of perturbation for r_adv
            - ip: Number of power iterations. If ip=1 then Fast gradient sign method (FGSM), else Projected Gradient Descent
            - alpha: step-size during power iteration
            - init: initialization type for noise vector. Either "zero" or "random"
            - grad_align_cos_lambda: factor for GradAlign regulizer to prevent catastrophic overfitting for FGSM
        """
        super(AtLoss, self).__init__()
        self.scaler = scaler
        self.loss_fun = loss_fun
        self.grid_size = grid_size
        self.eps = eps
        self.ip = ip
        self.alpha = alpha
        self.grad_align_cos_lambda = grad_align_cos_lambda

    def l2_norm_batch(self, v: torch.Tensor):
        """
        Computes the batchwise L2-norm for a 4D tensor
        """
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms

    def compute_grad_align(self, grad1: torch.Tensor, grad2: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradient alignment regularization term as proposed by https://arxiv.org/abs/2007.02617.
        
        Parameters:
            - grad1: Gradient of r_adv when initialized with zero vector
            - grad2: Gradient of r_adv when initialized with random vector
        """
        grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
        grads_nnz_idx = (grad1_norms != 0) * (grad2_norms != 0)
        grad1, grad2 = grad1[grads_nnz_idx], grad2[grads_nnz_idx]
        grad1_norms, grad2_norms = grad1_norms[grads_nnz_idx], grad2_norms[grads_nnz_idx]
        grad1_normalized = grad1 / grad1_norms[:, None, None, None]
        grad2_normalized = grad2 / grad2_norms[:, None, None, None]
        cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
        reg = (1.0 - cos.mean())
        return reg

    def compute_r_adv(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, r_init="zero", backprop=True):
        """
        Computes an adversarial noise vector via power iteration that when applied to the input image
        maximally increases the prediction error
        
        Parameters:
            - model: predictor model
            - x: input image
            - y: input label
            - r_init: initialization type for the noise vector
            - backprob: Whether to compute the gradient graph
        """
        grid_size = self.grid_size
        if grid_size is None:
            grid_size = x.shape
        r = torch.zeros(grid_size, device=x.device, dtype=x.dtype)
        if r_init == "uniform":
            r = r.uniform_(-self.eps, self.eps)
        r.requires_grad_()
        for i in range(self.ip):
            if r.grad is not None:
                r.grad.zero_()
            with torch.cuda.amp.autocast():
                pred = model(torch.clamp(x+r,0,1))
                loss: torch.Tensor = self.loss_fun(pred, y)
            self.scaler.scale(loss).backward()
            #r.grad.div_(self.scaler.get_scale())  # reverse back the scaling
                
            grad = r.grad.detach()
            r.data = r + self.alpha * self.eps * torch.sign(grad)
            r.data = torch.clamp(x+r.data, 0, 1) - x
            r.data = torch.clamp(r.data,-self.eps,self.eps)
        return r.detach(), grad
    
    def forward(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
        if self.grad_align_cos_lambda>0:
            # Additionally compute gradient aligment regularization term
            r, grad1 = self.compute_r_adv(model, x, y, "zero", False)
            _, grad2 = self.compute_r_adv(model, x, y, "uniform", True)
            reg = self.grad_align_cos_lambda * self.compute_grad_align(grad1,grad2)
        else:
            # Only compute adversarial noise vector
            r, _ = self.compute_r_adv(model, x, y, "uniform", True)
            reg = 0

        return x+r, reg

class ANTLoss(torch.nn.Module):
    def __init__(self, scaler: GradScaler, loss_fun: torch.nn.Module, grid_size= (8,8)) -> None:
        super().__init__()
        self.noise_model = NoiseModel(
            grid_size = grid_size,
            lambda_delta = 1,
            lambda_speckle = 0.7,
            lambda_gamma = 0.3,
            alpha=0.2
        )
        self.scaler = scaler
        self.loss_fun = loss_fun

    def forward(self, model: torch.nn.Module, x: torch.Tensor, deep: torch.Tensor, y: torch.Tensor):
        model.eval()
        adv_sample = self.noise_model.forward(x, deep, False)
        loss_trajectory = []
        for i in range(3):
            with torch.cuda.amp.autocast():
                pred = model(adv_sample)
                loss: torch.Tensor = self.loss_fun(pred, y)
                loss_trajectory.append(loss.item())
            self.scaler.scale(loss).backward()
            # with torch.cuda.amp.autocast():
            adv_sample = self.noise_model.forward(x, deep, True)
        model.train()
        return adv_sample.detach(), 0


def get_loss_function(task: Task, config: dict) -> tuple[str, Union[clDiceLoss, DiceBCELoss, torch.nn.CrossEntropyLoss]]:
    if task == Task.VESSEL_SEGMENTATION:
        return 'clDiceLoss', clDiceLoss(alpha=config["Train"]["lambda_cl_dice"], sigmoid=True)
    elif task == Task.AREA_SEGMENTATION:
        return 'DiceBCELoss', DiceBCELoss()
    else:
        return "CosineEmbeddingLoss", WeightedCosineLoss(weights=[1/0.537,1/0.349,1/0.115])

def get_loss_function_by_name(name: str, config: dict, scaler: GradScaler=None, loss=None) -> Union[clDiceLoss, DiceBCELoss, torch.nn.CrossEntropyLoss, WeightedCosineLoss]:
    if "Data" in config:
        weight = 1/torch.tensor(config["Data"]["class_balance"], device=config["General"]["device"])
    else:
        weight = None
    loss_map = {
        # "AtLoss": AtLoss(scaler, loss, None, 200/255, 1, alpha=1.25 * (100/255), grad_align_cos_lambda=0),
        "AtLoss": ANTLoss(scaler, loss, (8,8)),
        "clDiceLoss": clDiceLoss(alpha=config["Train"]["lambda_cl_dice"] if "lambda_cl_dice" in config["Train"] else 0),
        "DiceBCELoss": DiceBCELoss(True),
        "clDiceBceLoss": clDiceBceLoss(lambda_dice=0.4, lambda_cldice=0.1, lambda_bce=0.5, sigmoid=True),
        "CrossEntropyLoss": torch.nn.CrossEntropyLoss(weight=weight),
        "CosineEmbeddingLoss": WeightedCosineLoss(weights=weight),
        "MSELoss": torch.nn.MSELoss(),
        "WeightedMSELoss": WeightedMSELoss(weights=weight),
        "QWKLoss": QWKLoss(),
        "LSGANLoss": LSGANLoss().to(device=config["General"]["device"]),
        "NCELoss": NCELoss(config["Train"]["lambda_NCE"] if "lambda_NCE" in config["Train"] else 0, config["Train"]["nce_layers"] if "nce_layers" in config["Train"] else [], config["General"]["device"], config["Train"]["batch_size"])
    }
    return loss_map[name]
