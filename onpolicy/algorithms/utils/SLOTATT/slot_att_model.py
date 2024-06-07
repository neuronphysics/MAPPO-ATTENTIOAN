from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from torch.nn import functional as F
import torch
from torch import Tensor, nn
from typing import Callable
from .utils import BaseModel, get_conv_output_shape, make_sequential_from_config, PositionalEmbedding
from .coord import AddCoords
from onpolicy.algorithms.utils.SLOTATT.resnet import ResNet18
from torchvision.ops import FeaturePyramidNetwork


class Encoder(nn.Module):
    def __init__(self, fpn_channel=256, img_size=44):
        super().__init__()
        self.backbone = ResNet18(input_channel=3, norm_type="group", small_inputs=True)
        self.fpn = FeaturePyramidNetwork([128, 256, 512, 1024], fpn_channel)

        self.pos_embedding = PositionalEmbedding(
            img_size, img_size, fpn_channel
        )
        self.lnorm = nn.GroupNorm(1, fpn_channel, affine=True, eps=0.001)
        self.conv_1x1 = [
            nn.Conv1d(fpn_channel, fpn_channel, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(fpn_channel, fpn_channel, kernel_size=1),
        ]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x: Tensor) -> Tensor:
        res_stages = self.backbone(x)
        fpn_res = self.fpn(res_stages)
        conv_output = self.pos_embedding(fpn_res['C0'])
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output)
        return self.conv_1x1(conv_output)


class FPNTranConv(nn.Module):
    def __init__(self):
        super(FPNTranConv, self).__init__()
        self.addcoord = AddCoords(rank=2, w=256, h=256, with_r=False, skiptile=True)
        self.tile16 = AddCoords(rank=2, w=16, h=16, with_r=False, skiptile=False)
        self.convT1 = nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(512 + 128, 128, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(256 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.convT4 = nn.ConvTranspose2d(64 + 128, 4, kernel_size=1, stride=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feature):
        # add coord for every feature
        d1 = self.act(self.convT1(feature["C3"]))
        d1 = torch.cat([d1, feature["C2"]], dim=1)
        d2 = self.act(self.convT2(d1))
        d2 = torch.cat([d2, feature["C1"]], dim=1)
        d3 = self.act(self.convT3(d2))
        d3 = torch.cat([d3, feature["C0"]], dim=1)
        output = self.convT4(d3)
        return output


class Decoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            width: int,
            height: int,

    ):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(width, height, input_channels)
        self.backbone = ResNet18(input_channel=input_channels, norm_type="group", small_inputs=True)

        self.tran_conv_fpn = FPNTranConv()

        self.width = width
        self.height = height

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pos_embedding(x)
        res_out = self.backbone(x)
        output = self.tran_conv_fpn(res_out)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask


class SlotAttentionModule(nn.Module):
    def __init__(self, num_slots, channels_enc, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.empty(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, dim))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(channels_enc, dim, bias=False)
        self.to_v = nn.Linear(channels_enc, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

        self.norm_input = nn.LayerNorm(channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(dim, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(dim, eps=0.001)
        self.dim = dim

    def forward(self, inputs: Tensor, num_slots: Optional[int] = None):
        b, n, _ = inputs.shape
        if num_slots is None:
            num_slots = self.num_slots

        mu = self.slots_mu.expand(b, num_slots, -1)
        sigma = self.slots_log_sigma.expand(b, num_slots, -1).exp()
        slots = mu + sigma * torch.randn_like(sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bjd,bij->bid", v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim), slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, attn


@dataclass(eq=False, repr=False)
class SlotAttentionAE(BaseModel):
    latent_size: int

    encoder_params: Dict
    decoder_params: Dict
    lose_fn_type: str
    input_channels: int = 3
    eps: float = 1e-8
    mlp_size: int = 128
    attention_iters: int = 3
    w_broadcast: Union[int, Literal["dataset"]] = "dataset"
    h_broadcast: Union[int, Literal["dataset"]] = "dataset"

    encoder: Encoder = field(init=False)
    decoder: Decoder = field(init=False)
    loss_fn: Callable = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.lose_fn_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.lose_fn_type == "cosine":
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif self.lose_fn_type == "l1":
            self.loss_fn = nn.functional.l1_loss

        if self.w_broadcast == "dataset":
            self.w_broadcast = self.width
        if self.h_broadcast == "dataset":
            self.h_broadcast = self.height

        self.encoder = Encoder(**self.encoder_params)
        self.slot_attention = SlotAttentionModule(
            self.num_slots,
            self.encoder_params["fpn_channel"],
            self.latent_size,
            self.attention_iters,
            self.eps,
            self.mlp_size,
        )
        self.decoder_params.update(
            width=self.w_broadcast,
            height=self.h_broadcast,
            input_channels=self.latent_size,
        )
        self.decoder = Decoder(**self.decoder_params)

    @property
    def slot_size(self) -> int:
        return self.latent_size

    def spatial_broadcast(self, slot: Tensor) -> Tensor:
        slot = slot.unsqueeze(-1).unsqueeze(-1)
        return slot.repeat(1, 1, self.w_broadcast, self.h_broadcast)

    def slot_similarity_loss(self, slots):
        """
        Calculate the similarity loss for slots with shape (batch, num_slot, hidden_size).
        """
        batch_size = slots.shape[0]
        # Normalize slot features
        slots = F.normalize(slots, dim=-1)  # Normalize along the hidden_size dimension

        # Randomly permute the slots
        perm = torch.randperm(slots.size(1)).to(slots.device)  # Permute along the num_slot dimension

        # Select a subset of n slots
        selected_slots = slots[:, perm[:self.num_slots], :]  # [batch, n, hidden_size]

        # Compute similarity matrix
        sim_matrix = torch.bmm(selected_slots, selected_slots.transpose(1, 2)) * (
                1 / np.sqrt(slots.size(2)))  # [batch, n, n]

        # Create mask to remove diagonal elements (self-similarity)
        mask = torch.eye(self.num_slots).to(slots.device).repeat(batch_size, 1, 1)  # [1, n, n]

        # Mask out the diagonal elements
        sim_matrix = sim_matrix - mask * sim_matrix

        # Compute similarity loss
        sim_loss = sim_matrix.sum(dim=(1, 2)) / (self.num_slots * (self.num_slots - 1))

        return sim_loss.mean()  # Return the mean similarity loss over the batch

    def forward(self, x: Tensor) -> dict:
        with torch.no_grad():
            x = x * 2.0 - 1.0
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)
        z, attn = self.slot_attention(encoded)
        bs = z.size(0)
        slots = z.flatten(0, 1)
        slots = self.spatial_broadcast(slots)
        img_slots, masks = self.decoder(slots)
        img_slots = img_slots.view(bs, self.num_slots, 3, self.width, self.height)
        masks = masks.view(bs, self.num_slots, 1, self.width, self.height)
        masks = masks.softmax(dim=1)

        recon_slots_masked = img_slots * masks
        recon_img = recon_slots_masked.sum(dim=1)
        loss = self.loss_fn(x, recon_img)

        if self.lose_fn_type == "mse":
            loss = loss + self.slot_similarity_loss(z)

        with torch.no_grad():
            recon_slots_output = (img_slots + 1.0) / 2.0
        return {
            "loss": loss,  # scalar
            "mask": masks,  # (B, slots, 1, H, W)
            "slot": recon_slots_output,  # (B, slots, 3, H, W)
            "representation": z,  # (B, slots, latent dim)
            #
            "reconstruction": recon_img,  # (B, 3, H, W)
            "attn": attn
        }
