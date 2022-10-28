import torch
import torch.nn.functional as F
from monai.losses import DiceLoss


def soft_skel_v2(img):
    """
    Novel method by Martin Menten for skeleton extraction based on convolution. Resulting skeleton is smoother and is better at maintaining the topology.
    """
    # Kernels to calculate the maps defining the existence of the 8 neighbors
    k2 = img.new([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]]).view(1, 1, 3, 3)

    k3 = img.new([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]]).view(1, 1, 3, 3)

    k4 = img.new([[0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0]]).view(1, 1, 3, 3)

    k5 = img.new([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]]).view(1, 1, 3, 3)

    k6 = img.new([[0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0]]).view(1, 1, 3, 3)

    k7 = img.new([[0, 0, 0],
                    [0, 0, 0],
                    [1, 0, 0]]).view(1, 1, 3, 3)

    k8 = img.new([[0, 0, 0],
                    [1, 0, 0],
                    [0, 0, 0]]).view(1, 1, 3, 3)

    k9 = img.new([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]).view(1, 1, 3, 3)

    # Kernel to calculate zero-one-patterns (still needs a relu afterwards)
    zero_one_kernel = img.new([-1, 1]).view(1, 1, 1, 1, 2)

    for _ in range(10):
        # Pad input image
        img = F.pad(img, (1, 1, 1, 1))

        p2 = F.conv2d(img, k2)
        p3 = F.conv2d(img, k3)
        p4 = F.conv2d(img, k4)
        p5 = F.conv2d(img, k5)
        p6 = F.conv2d(img, k6)
        p7 = F.conv2d(img, k7)
        p8 = F.conv2d(img, k8)
        p9 = F.conv2d(img, k9)

        P = torch.stack([p2, p3, p4, p5, p6, p7, p8, p9], dim=-1)
        P_loop = torch.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], dim=-1)

        # Calculate whether each pixel fulfills four conditions for removal during first pass
        num_neighbors = P.sum(dim=-1)
        cond1a = F.hardtanh(num_neighbors - 1, min_val=0, max_val=1) # 2 or more neighbors
        cond1b = F.hardtanh(-(num_neighbors - 7), min_val=0, max_val=1) # 6 or fewer neighbors
        cond1 = cond1a * cond1b

        num_zero_one_patterns = F.relu(F.conv3d(P_loop, zero_one_kernel)).sum(dim=-1)
        cond2a = F.hardtanh(num_zero_one_patterns, min_val=0, max_val=1) # 1 or more zero-one-patterns
        cond2b = F.hardtanh(-(num_zero_one_patterns - 2), min_val=0, max_val=1) # 1 or fewer zero-one-patterns
        cond2 = cond2a * cond2b

        cond3 = 1 - p2 * p4 * p6 # One of the top, right or bottom neighbors is zero
        cond4 = 1 - p4 * p6 * p8 # One of the right, bottom or left neighbors is zero

        # Remove pixel
        img = torch.min(img[:, :, 1:-1, 1:-1], 1 - cond1 * cond2 * cond3 * cond4)

        # Pad input image
        img = F.pad(img, (1, 1, 1, 1))

        p2 = F.conv2d(img, k2)
        p3 = F.conv2d(img, k3)
        p4 = F.conv2d(img, k4)
        p5 = F.conv2d(img, k5)
        p6 = F.conv2d(img, k6)
        p7 = F.conv2d(img, k7)
        p8 = F.conv2d(img, k8)
        p9 = F.conv2d(img, k9)

        P = torch.stack([p2, p3, p4, p5, p6, p7, p8, p9], dim=-1)
        P_loop = torch.stack([p2, p3, p4, p5, p6, p7, p8, p9, p2], dim=-1)

        # Calculate whether each pixel fulfills four conditions for removal during first pass
        num_neighbors = P.sum(dim=-1)
        cond1a = F.hardtanh(num_neighbors - 1, min_val=0, max_val=1) # 2 or more neighbors
        cond1b = F.hardtanh(-(num_neighbors - 7), min_val=0, max_val=1) # 6 or fewer neighbors
        cond1 = cond1a * cond1b

        num_zero_one_patterns = F.relu(F.conv3d(P_loop, zero_one_kernel)).sum(dim=-1)
        cond2a = F.hardtanh(num_zero_one_patterns, min_val=0, max_val=1) # 1 or more zero-one-patterns
        cond2b = F.hardtanh(-(num_zero_one_patterns - 2), min_val=0, max_val=1) # 1 or fewer zero-one-patterns
        cond2 = cond2a * cond2b

        cond3 = 1 - p2 * p4 * p8 # One of the top, right or left neighbors is zero
        cond4 = 1 - p2 * p6 * p8 # One of the top, bottom or left neighbors is zero

        # Remove pixel
        img = torch.min(img[:, :, 1:-1, 1:-1], 1 - cond1 * cond2 * cond3 * cond4)
    return img

def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel_v1(img, iter_=25):
    """
    Original soft skeleton extraction based on max pooling as described in
    [clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)
    """
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

class clDiceLoss():
    """
    The clDiceLoss is a combination of the classical Dice loss and the centerline Dice loss as described in
    [clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf).
    It aims to better preserve the topology of a vessel system.
    """
    def __init__(self, alpha=0.5, sigmoid=False):
        """
        Create a clDiceLoss instance.

        Paramters:
            - alpha: How to weight the clDice to the classical dice. Alpha=0 means to only use the classical Dice loss.
        """
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.dice_loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=False)


    def __call__(self, predictions, target):

        target = torch.clip(target, 0.0, 1.0)
        if self.sigmoid:
            predictions = torch.sigmoid(predictions)
        if self.alpha > 0:
            skel_pred = soft_skel_v2(predictions)
            skel_true = soft_skel_v2(target)
            tprec = (torch.sum(torch.multiply(skel_pred, target)[:,...]) + 1e-8) / (torch.sum(skel_pred[:,...]) + 1e-8)    
            tsens = (torch.sum(torch.multiply(skel_true, predictions)[:,...]) + 1e-8) / (torch.sum(skel_true[:,...]) + 1e-8)    
            cl_dice_loss = 1.0 - 2.0 * (tprec*tsens) / (tprec+tsens)
        else:
            cl_dice_loss = 0

        dice_loss = self.dice_loss(predictions, target)
        total_loss = self.alpha * cl_dice_loss + (1 - self.alpha) * dice_loss
        return total_loss


    def __repr__(self):

        return "clDiceLoss (new)()"

class clDiceBceLoss():
    """
    The clDiceLoss is a combination of the classical Dice loss and the centerline Dice loss as described in
    [clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf).
    It aims to better preserve the topology of a vessel system.
    """
    def __init__(self, lambda_dice=0.33, lambda_cldice=0.33, lambda_bce=0.33, sigmoid=False):
        """
        Create a clDiceLoss instance.

        Paramters:
            - alpha: How to weight the clDice to the classical dice. Alpha=0 means to only use the classical Dice loss.
        """
        self.lambda_dice=lambda_dice
        self.lambda_cldice=lambda_cldice
        self.lambda_bce=lambda_bce
        self.sigmoid = sigmoid
        self.dice_loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=sigmoid)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def __call__(self, predictions, target):
        if self.lambda_dice>0:
            dice_loss = self.lambda_dice * self.dice_loss(predictions, target)
        else:
            dice_loss=0
        if self.lambda_bce>0:
            bce_loss = self.lambda_bce * self.bce(predictions, target)
        else:
            bce_loss=0

        if self.lambda_cldice > 0:
            target = torch.clip(target, 0.0, 1.0)
            if self.sigmoid:
                predictions = torch.sigmoid(predictions)
            skel_pred = soft_skel_v2(predictions)
            skel_true = soft_skel_v2(target)
            tprec = (torch.sum(torch.multiply(skel_pred, target)[:,...]) + 1e-8) / (torch.sum(skel_pred[:,...]) + 1e-8)    
            tsens = (torch.sum(torch.multiply(skel_true, predictions)[:,...]) + 1e-8) / (torch.sum(skel_true[:,...]) + 1e-8)    
            cl_dice_loss = 1.0 - 2.0 * (tprec*tsens) / (tprec+tsens)
            cl_dice_loss = self.lambda_cldice * cl_dice_loss
        else:
            cl_dice_loss = 0
        return dice_loss+cl_dice_loss+bce_loss