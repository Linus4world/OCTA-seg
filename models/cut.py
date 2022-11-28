import torch
from models.gan_seg_model import GanSegModel

class CUT(GanSegModel):
    def __init__(self,
        nce_layers = list[int],
        **kwargs):
        super().__init__(**kwargs)
        self.nce_layers = nce_layers

    def forward(self, input, _=None, complete=False):
        if not complete:
            return self.generator(input)
        if not isinstance(input, tuple):
            input = input, _
        real_A, real_B = input
        fake_B, idt_B, _, _ = self.forward_GD(input)
        self.forward_GS(real_B, fake_B, idt_B, real_A)
        return fake_B

    def forward_GD(self, input: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        real_A, real_B = input
        fake_B = self.generator(real_A)
        if self.compute_identity_seg:
            idt_B = self.generator(torch.cat((real_B, real_A[:,0:1]), dim=1))
        else:
            idt_B = [None]
        
        self.discriminator.requires_grad_(True)
        pred_fake_B = self.discriminator(fake_B.detach())
        pred_real_B = self.discriminator(real_B)
        return fake_B, idt_B, pred_fake_B, pred_real_B

    def forward_GS(self, real_B, fake_B, idt_B, real_A) -> tuple[torch.Tensor]:
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        self.discriminator.requires_grad_(False)
        pred_fake_B = self.discriminator(fake_B)


        feat_k = self.generator(real_A, self.nce_layers, encode_only=True)
        feat_q = self.generator(torch.cat((fake_B, real_A[:,0:1]), dim=1), self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.segmentor(feat_k, 256, None)
        feat_q_pool, _ = self.segmentor(feat_q, 256, sample_ids)

        if self.compute_identity_seg:
            feat_k_idt = self.generator(torch.cat((real_B, real_A[:,0:1]), dim=1), self.nce_layers, encode_only=True)
            feat_q_idt = self.generator(torch.cat((idt_B, real_A[:,0:1]), dim=1), self.nce_layers, encode_only=True)

            feat_k_idt_pool, sample_ids = self.segmentor(feat_k_idt, 256, None)
            feat_q_idt_pool, _ = self.segmentor(feat_q_idt, 256, sample_ids)
        else:
            feat_k_idt_pool = None
            feat_q_idt_pool = None

        return pred_fake_B, feat_q_pool, feat_k_idt_pool, feat_q_idt_pool, feat_k_pool

    def apply(self, init_func):
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
        # self.segmentor.apply(init_func)
