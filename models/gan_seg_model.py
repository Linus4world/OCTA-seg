import torch
from torch import nn

class GanSegModel(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, segmentor: nn.Module, compute_identity = True, compute_identity_seg = True):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.segmentor = segmentor
        self.compute_identity = compute_identity
        self.compute_identity_seg = compute_identity_seg
        self.inference = False

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        self.segmentor.eval()

    def train(self, *params):
        self.generator.train()
        self.discriminator.train()
        self.segmentor.train()

    def forward(self, input, _=None, complete=False):
        if not isinstance(input, tuple):
            input = input, _
        real_A, real_B = input
        fake_B, idt_B, pred_fake_B, pred_real_B = self.forward_GD(input)
        if complete:
            pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg = self.forward_GS(real_B, fake_B, idt_B)
        return fake_B


    def forward_GD(self, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        real_A, real_B = input
        fake_B = self.generator(real_A)
        if self.compute_identity_seg:
            idt_B = self.generator(real_B)
        else:
            idt_B = [None]
        
        self.discriminator.requires_grad_(True)
        pred_fake_B = self.discriminator(fake_B.detach())
        pred_real_B = self.discriminator(real_B)
        return fake_B, idt_B, pred_fake_B, pred_real_B

    def forward_GS(self, real_B, fake_B, idt_B) -> tuple[torch.Tensor]:
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        self.discriminator.requires_grad_(False)
        pred_fake_B = self.discriminator(fake_B)

        real_B_seg = self.segmentor(torch.nn.functional.interpolate(real_B, scale_factor=4, mode="bilinear"))
        if self.compute_identity_seg:
            idt_B_seg = self.segmentor(torch.nn.functional.interpolate(idt_B, scale_factor=4, mode="bilinear"))
        else:
            idt_B_seg = [None]
        fake_B_seg = self.segmentor(torch.nn.functional.interpolate(fake_B, scale_factor=4, mode="bilinear"))
        return pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg

    def apply(self, init_func):
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
        self.segmentor.apply(init_func)
