import os
import copy
import sys
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, PeftMixedModel

from utils.model import (
    make_1step_sched,
    vae_encoder_fwd,
    vae_decoder_fwd,
    download_url,
)

# Legacy path hack; ideally this project should be installed as a package.
sys.path.append("..")


# -------------------------------------------------------------------------
# VAE wrappers
# -------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, vaes=None):
        super(VAE, self).__init__()
        self.vaes = vaes

    def encode(self, x):
        _vae = self.vaes[0]
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor

    def decode(self, x, direction):
        _vae = self.vaes[self.direction_list.index(direction)]
        if self.direction_list.index(direction) % 2 == 0:
            assert self.vaes[0].encoder.current_down_blocks is not None
            _vae.decoder.incoming_skip_acts = self.vaes[0].encoder.current_down_blocks
        else:
            assert self.vaes[1].encoder.current_down_blocks is not None
            _vae.decoder.incoming_skip_acts = self.vaes[1].encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded

    def forward(self, fn, x, direction, direction_list):
        if fn == 'encode':
            return self.encode(x)
        elif fn == 'decode':
            return self.decode(x, direction, direction_list)
        else:
            raise NotImplementedError


class VAE_encode(nn.Module):
    """
    Direction-aware VAE encoder wrapper.

    It routes inputs to either vae (A->B) or vae_b2a (B->A) and returns the
    scaled latent sample.
    """

    def __init__(self, vae, vae_b2a = None):
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Args:
            x: Input image tensor in [-1, 1], shape [B, C, H, W].
            direction: Either "a2b" or "b2a".

        Returns:
            Latent tensor after encoding and scaling, shape [B, C_latent, H_latent, W_latent].
        """
        assert direction in ["a2b", "b2a"], f"Invalid direction: {direction}"
        _vae = self.vae if direction == "a2b" else self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    """
    Direction-aware VAE decoder wrapper.

    It routes latent inputs to either vae (A->B) or vae_b2a (B->A) and
    restores skip connections from the encoder.
    """

    def __init__(self, vae, vae_b2a = None):
        super().__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Args:
            x: Latent tensor to decode, shape [B, C_latent, H_latent, W_latent].
            direction: Either "a2b" or "b2a".

        Returns:
            Reconstructed image tensor in [-1, 1], shape [B, C, H, W].
        """
        assert direction in ["a2b", "b2a"], f"Invalid direction: {direction}"
        _vae = self.vae if direction == "a2b" else self.vae_b2a

        # Inject encoder skip activations into decoder
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks

        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


# -------------------------------------------------------------------------
# Multimodal gating network for MoE adapters
# -------------------------------------------------------------------------


class GatingNetwork(nn.Module):
    """
    Multimodal gating network for Mixture-of-Experts.

    The gate uses both image features and text features to produce expert
    weights. Several fusion strategies are supported (concat, cross_attention,
    FiLM-based modulation).
    """

    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 1,
        image_dim: int = 1024,
        text_dim: int = 1024,
        text_seq_len: int = 77,
        hidden_dim: int = 256,
        expert_balance: float = 0.1,
        fusion_method: str = "concat",
    ):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.text_seq_len = text_seq_len
        self.num_experts = num_experts
        self.top_k = top_k
        self.fusion_method = fusion_method

        # Image feature encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Text sequence encoder (self-attention + pooling)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(text_dim)

        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Feature fusion
        if fusion_method == "concat":
            self.fusion_dim = hidden_dim * 2
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
        elif fusion_method == "cross_attention":
            self.fusion_dim = hidden_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        elif fusion_method == "film":
            # FiLM-style conditioning: image features modulated by text features
            self.fusion_dim = hidden_dim
            self.gamma_layer = nn.Linear(hidden_dim, hidden_dim)
            self.beta_layer = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

        # Decision network mapping fused features to expert logits
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts),
        )

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # Expert usage statistics for simple load balancing
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.enable_balancing = True
        self.balancing_strength = expert_balance

    def _process_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """Process image features (either latent vector or flattened CNN feature)."""
        return self.image_encoder(image_features)

    def _process_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Process text features.

        Args:
            text_features: [B, seq_len, text_dim] tensor.

        Returns:
            Encoded text features [B, hidden_dim].
        """
        attn_output, _ = self.text_attention(
            text_features,
            text_features,
            text_features,
        )
        attn_output = self.text_norm(attn_output + text_features)  # residual

        # Global pooling over sequence length
        pooled_text = attn_output.mean(dim=1)  # [B, text_dim]

        return self.text_encoder(pooled_text)

    def _fuse_features(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse image and text features using the chosen fusion strategy."""
        if self.fusion_method == "concat":
            fused = torch.cat([image_features, text_features], dim=1)
            return self.fusion_layer(fused)

        if self.fusion_method == "cross_attention":
            # Text as query, image as key/value. Both are expanded to sequence length 1.
            image_seq = image_features.unsqueeze(1)  # [B, 1, hidden_dim]
            text_seq = text_features.unsqueeze(1)  # [B, 1, hidden_dim]

            fused, _ = self.cross_attention(text_seq, image_seq, image_seq)
            fused = self.fusion_norm(fused + text_seq)
            return fused.squeeze(1)  # [B, hidden_dim]

        if self.fusion_method == "film":
            # FiLM modulation of image features with text features
            gamma = self.gamma_layer(text_features).unsqueeze(1)
            beta = self.beta_layer(text_features).unsqueeze(1)
            fused = gamma * image_features + beta
            return fused

        # Should never reach here
        raise RuntimeError("Invalid fusion method state.")

    def forward(
        self,
        gating_input: Tuple[torch.Tensor, torch.Tensor],
        hard_gating: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the gating network.

        Args:
            gating_input:
                Tuple (image_features, text_features)
                - image_features: [B, image_dim] or [B, C, H, W]
                - text_features: [B, seq_len, text_dim]
            hard_gating:
                If True, use argmax-based hard gating. Otherwise, use softmax.

        Returns:
            expert_weights: [B, num_experts] tensor of expert weights.
        """
        image_features, text_features = gating_input
        batch_size = image_features.shape[0]

        # If image features are CNN maps [B, C, H, W], flatten to [B, C*H*W]
        if len(image_features.shape) == 4:
            image_features = image_features.view(batch_size, -1)
            if image_features.shape[1] != self.image_dim:
                raise ValueError(
                    f"Image feature dim {image_features.shape[1]} "
                    f"does not match expected {self.image_dim}"
                )

        # 1) Process image features
        processed_image = self._process_image(image_features)

        # 2) Process text features
        processed_text = self._process_text(text_features)

        # 3) Fuse image/text features
        fused_features = self._fuse_features(processed_image, processed_text)

        # 4) Produce logits for each expert
        logits = self.decision_network(fused_features)

        # 5) Optional expert usage balancing (during training only)
        if self.enable_balancing and self.training:
            usage_penalty = self.expert_usage / (self.expert_usage.sum() + 1e-6)
            logits = logits - usage_penalty * self.balancing_strength

        # 6) Hard vs soft gating
        if hard_gating:
            # Hard gating: one-hot selection of the best expert per sample
            indices = torch.argmax(logits, dim=-1)
            weights = torch.zeros_like(logits).scatter_(
                -1, indices.unsqueeze(-1), 1.0
            )

            if self.training:
                for idx in indices:
                    self.expert_usage[idx] += 1
        else:
            # Soft gating: softmax over experts
            weights = F.softmax(logits / self.temperature, dim=-1)
            _, indices = torch.topk(weights, k=self.top_k, dim=-1)

            if self.training:
                for idx in indices:
                    self.expert_usage[idx] += 1
                # Alternatively: self.expert_usage += weights.sum(dim=0).detach()

        return weights


# -------------------------------------------------------------------------
# MoE adapter UNet
# -------------------------------------------------------------------------


class MoEAdapterUNet(nn.Module):
    """
    UNet wrapper with Mixture-of-Experts LoRA adapters.

    During training, a soft gating distribution over experts is learned.
    During inference, the same mechanism can be used, or a hard assignment
    can be provided via `expert_assign`.
    """

    def __init__(
        self,
        unet: PeftMixedModel,
        num_experts: int = 3,
        top_k: int = 1,
        expert_balance: float = 0.1,
        adapter_names=None,
        image_dim: int = 1024,
        text_dim: int = 1024,
        text_seq_len: int = 77,
        fusion_method: str = "concat",
    ):
        super().__init__()
        self.unet = unet
        self.num_experts = num_experts
        self.top_k = top_k  # number of experts to activate per forward
        self.adapter_names = adapter_names or []
        self._shared_adapter_name: Optional[str] = None  # shared adapter name

        # Multimodal gating network
        self.gating_network = GatingNetwork(
            num_experts=num_experts,
            top_k=top_k,
            image_dim=image_dim,
            text_dim=text_dim,
            text_seq_len=text_seq_len,
            expert_balance=expert_balance,
            fusion_method=fusion_method,
        )

    def _activate_expert(self, expert_idx: int) -> None:
        """
        Activate the adapters for a given expert, optionally including a shared adapter.
        """
        expert_adapters = self.adapter_names[expert_idx]

        if self._shared_adapter_name:
            if isinstance(expert_adapters, list):
                active_adapters = [self._shared_adapter_name] + expert_adapters
            else:
                active_adapters = [self._shared_adapter_name, expert_adapters]
        else:
            active_adapters = expert_adapters

        self.unet.set_adapter(active_adapters)

    def set_shared_adapter(self, adapter_name: str) -> None:
        """Set an adapter name that is shared by all experts."""
        self._shared_adapter_name = adapter_name

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        expert_assign: Optional[torch.Tensor] = None,
        gating_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Forward through the MoE UNet.

        Args:
            sample:
                Input latent sample [B, C, H, W].
            timestep:
                Diffusion timestep (scalar or tensor).
            encoder_hidden_states:
                Text conditioning tensor [B, seq_len, text_dim].
            expert_assign:
                Optional one-hot expert assignment tensor [1, num_experts].
                If provided, this overrides the gating network.
            gating_input:
                Optional (image_features, text_features) tuple. If None,
                (sample, encoder_hidden_states) is used.

        Returns:
            combined_output: UNet output after expert combination.
            expert_weights: Full expert weight distribution [B, num_experts].
            topk_weights: Weights for selected experts [B, top_k].
        """
        # Default gating input: use current sample and encoder_hidden_states
        if gating_input is None:
            gating_input = (sample, encoder_hidden_states)

        # Expert assignment:
        # - If expert_assign is provided, treat it as a hard one-hot weight.
        # - Otherwise, use learned gating over all experts.
        if expert_assign is not None:
            expert_weights = expert_assign.to(sample.device)
            topk_weights, topk_indices = torch.topk(expert_weights, k=1, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
        else:
            expert_weights = self.gating_network(gating_input)
            topk_weights, topk_indices = torch.topk(
                expert_weights, k=self.top_k, dim=-1
            )
            topk_weights = F.softmax(topk_weights, dim=-1)
            # For now, use experts selected for the first batch element
            topk_indices = topk_indices[0].tolist()

        expert_outputs = []

        # Run UNet separately for each selected expert
        for expert_idx in topk_indices:
            expert_idx = int(expert_idx)

            # Activate current expert adapters
            self._activate_expert(expert_idx)

            # Run UNet
            output = self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            ).sample
            expert_outputs.append(output)

        # Combine outputs with top-k weights
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            batch_size = expert_output.shape[0]
            weights = topk_weights[:batch_size, i].view(-1, 1, 1, 1)
            combined_output += expert_output * weights

        return combined_output, expert_weights, topk_weights


# -------------------------------------------------------------------------
# Initialization helpers (UNet / VAE)
# -------------------------------------------------------------------------


def initialize_unet(
    base_model_path: str = "asset/sd-turbo/unet",
    rank: int = 128,
    num_experts: int = 1,
    top_k: int = 1,
    expert_balance: float = 0.1,
    text_dim: int = 1024,
    text_seq_len: int = 77,
    fusion_method: str = "concat",
    return_lora: bool = False,
):
    """
    Initialize a UNet with LoRA-based Mixture-of-Experts adapters.

    Each expert owns its own mid and low importance LoRA adapters, while
    high-importance layers are shared across experts.
    """
    # Load pretrained UNet
    unet = UNet2DConditionModel.from_pretrained(base_model_path)
    unet.requires_grad_(False)
    unet.train()

    # Split UNet parameters into high / mid / low importance via simple patterns.
    l_layer_high, l_layer_mid, l_layer_low = [], [], []
    l_grep_high = ["conv", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in"]
    l_grep_mid = ["ff.net", "conv1", "conv2"]
    l_grep_low = ["to_out", "to_k", "to_q", "to_v", "time_embedding"]

    for n, _ in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep_high:
            if pattern in n:
                l_layer_high.append(n.replace(".weight", ""))
        for pattern in l_grep_mid:
            if pattern in n:
                l_layer_mid.append(n.replace(".weight", ""))
        for pattern in l_grep_low:
            if pattern in n:
                l_layer_low.append(n.replace(".weight", ""))

    # Shared high-importance adapter
    lora_conf_high = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_layer_high,
        lora_alpha=rank * 4,
    )

    lora_name_high = "expert_all_H-layer"
    unet = PeftMixedModel(unet, lora_conf_high, lora_name_high)

    # Per-expert mid/low adapters
    adapter_names = []
    initial_adapter_names = [lora_name_high]

    for expert_id in range(num_experts):
        scale = 4
        lora_conf_mid = LoraConfig(
            r=rank // 2,
            init_lora_weights="gaussian",
            target_modules=l_layer_mid,
            lora_alpha=rank // 2 * scale,
        )
        lora_conf_low = LoraConfig(
            r=rank // 4,
            init_lora_weights="gaussian",
            target_modules=l_layer_low,
            lora_alpha=rank // 4 * scale,
        )

        lora_name_mid = f"expert_{expert_id}_M-layer"
        lora_name_low = f"expert_{expert_id}_L-layer"

        unet.add_adapter(adapter_name=lora_name_mid, peft_config=lora_conf_mid)
        unet.add_adapter(adapter_name=lora_name_low, peft_config=lora_conf_low)

        adapter_names.append([lora_name_mid, lora_name_low])
        initial_adapter_names.append(lora_name_mid)
        initial_adapter_names.append(lora_name_low)

    # Activate all adapters initially
    unet.set_adapter(initial_adapter_names)

    # Wrap with MoE adapter layer
    moe_unet = MoEAdapterUNet(
        unet,
        num_experts=num_experts,
        top_k=top_k,
        expert_balance=expert_balance,
        adapter_names=adapter_names,
        image_dim=4096,  # 64 * 64 * 1 or other flattened size for image features
        text_dim=text_dim,
        text_seq_len=text_seq_len,
        fusion_method=fusion_method,
    )
    moe_unet.set_shared_adapter(lora_name_high)

    if return_lora:
        return moe_unet, l_layer_high, l_layer_mid, l_layer_low
    return moe_unet


def initialize_vae(rank: int = 4, return_lora_module_names: bool = False):
    """
    Initialize VAE with skip connections and LoRA adapters.
    """
    vae = AutoencoderKL.from_pretrained(r"asset/sd-turbo/vae")
    vae.requires_grad_(False)

    # Replace forward with custom functions that record skip activations
    vae.encoder.forward = vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)

    vae.requires_grad_(True)  # trainable
    vae.train()

    # Add skip connection convolutions
    vae.decoder.skip_conv_1 = nn.Conv2d(
        512, 512, kernel_size=1, stride=1, bias=False
    ).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = nn.Conv2d(
        256, 512, kernel_size=1, stride=1, bias=False
    ).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = nn.Conv2d(
        128, 512, kernel_size=1, stride=1, bias=False
    ).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = nn.Conv2d(
        128, 256, kernel_size=1, stride=1, bias=False
    ).cuda().requires_grad_(True)

    nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    l_vae_target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "skip_conv_1",
        "skip_conv_2",
        "skip_conv_3",
        "skip_conv_4",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    vae_lora_config = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_vae_target_modules,
    )
    vae = get_peft_model(vae, vae_lora_config)
    vae.add_adapter(adapter_name="vae_skip", peft_config=vae_lora_config)

    if return_lora_module_names:
        return vae, l_vae_target_modules
    return vae


# -------------------------------------------------------------------------
# High-level CycleGAN-Turbo wrapper
# -------------------------------------------------------------------------


class CycleGAN_Turbo(nn.Module):
    """
    High-level wrapper for training and inference with the Turbo-style
    CycleGAN model built on SD-Turbo components.
    """

    def __init__(self):
        super().__init__()


    def load_ckpt_from_state_dict(self, sd: dict) -> None:
        """
        Load LoRA weights for UNet & VAE from a state dict produced by training.
        """
        # UNet LoRA adapters
        lora_conf_encoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_encoder"],
            lora_alpha=sd["rank_unet"],
        )
        lora_conf_decoder = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_target_modules_decoder"],
            lora_alpha=sd["rank_unet"],
        )
        lora_conf_others = LoraConfig(
            r=sd["rank_unet"],
            init_lora_weights="gaussian",
            target_modules=sd["l_modules_others"],
            lora_alpha=sd["rank_unet"],
        )

        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")

        # Copy LoRA weights
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])

        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])

        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # VAE LoRA adapters
        vae_lora_config = LoraConfig(
            r=sd["rank_vae"],
            init_lora_weights="gaussian",
            target_modules=sd["vae_lora_target_modules"],
        )
        self.vae = get_peft_model(self.vae, vae_lora_config)
        self.vae.add_adapter(peft_config=vae_lora_config, adapter_name="vae_skip")

        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_url(self, url: str, ckpt_folder: str) -> None:
        """Download and load checkpoint from URL."""
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(
        x: torch.Tensor,
        direction: str,
        vae_enc: VAE_encode,
        unet: MoEAdapterUNet,
        vae_dec: VAE_decode,
        sched,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        expert_assign: Optional[torch.Tensor] = None,
    ):
        """
        Single-step Turbo-style forward pass used for both training and inference.

        Args:
            x: Input image tensor in [-1, 1], shape [B, C, H, W].
            direction: "a2b" or "b2a".
            vae_enc / vae_dec: Direction-aware VAE wrappers.
            unet: MoEAdapterUNet.
            sched: Diffusion scheduler with `step` method.
            timesteps: 1D tensor of timesteps [B].
            text_emb: Text embedding tensor [B, seq_len, hidden_dim].
            expert_assign: Optional expert assignment tensor.

        Returns:
            x_out_dec: Decoded output images.
            expert_weights: Full expert weights from MoE.
            topk_weights: Weights for top-k experts.
        """
        B = x.shape[0]
        assert direction in ["a2b", "b2a"], f"Invalid direction: {direction}"

        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred, expert_weights, topk_weights = unet(
            x_enc,
            timesteps,
            encoder_hidden_states=text_emb,
            expert_assign=expert_assign,
        )

        # If multiple experts are returned as a batch dimension (5D tensor)
        if len(model_pred.shape) == 5:
            x_out_dec_list = []
            for expert_idx in range(model_pred.shape[0]):
                model_pred_i = model_pred[expert_idx]
                x_out_i = torch.stack(
                    [
                        sched.step(
                            model_pred_i[b],
                            timesteps[b],
                            x_enc[b],
                            return_dict=True,
                        ).prev_sample
                        for b in range(B)
                    ]
                )
                x_out_dec_i = vae_dec(x_out_i, direction=direction)
                x_out_dec_list.append(x_out_dec_i)
            x_out_dec = torch.stack(x_out_dec_list, dim=0)
        else:
            x_out = torch.stack(
                [
                    sched.step(
                        model_pred[b],
                        timesteps[b],
                        x_enc[b],
                        return_dict=True,
                    ).prev_sample
                    for b in range(B)
                ]
            )
            x_out_dec = vae_dec(x_out, direction=direction)

        return x_out_dec, expert_weights, topk_weights

    @staticmethod
    def get_traininable_params(unet: MoEAdapterUNet, vae_a2b: AutoencoderKL, vae_b2a: AutoencoderKL):
        """
        Collect trainable parameters for generator-side optimization:
        - UNet expert LoRA parameters
        - Gating network parameters
        - VAE LoRA and skip-conv parameters for both directions
        """
        # UNet inner module
        _unet = unet.unet
        params_gen = list(_unet.conv_in.parameters())
        _unet.conv_in.requires_grad_(True)

        for n, p in _unet.named_parameters():
            if "lora" in n and "expert" in n:
                p.requires_grad_(True)
                params_gen.append(p)

        # Gating network parameters
        _gating_network = unet.gating_network
        for _, p in _gating_network.named_parameters():
            p.requires_grad_(True)
            params_gen.append(p)

        # VAE A->B LoRA + skip convs
        for n, p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                p.requires_grad_(True)
                params_gen.append(p)

        params_gen += list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen += list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen += list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen += list(vae_a2b.decoder.skip_conv_4.parameters())

        # VAE B->A LoRA + skip convs
        for n, p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                p.requires_grad_(True)
                params_gen.append(p)

        params_gen += list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen += list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen += list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen += list(vae_b2a.decoder.skip_conv_4.parameters())

        return params_gen

    def forward(
        self,
        x_t: torch.Tensor,
        direction: Optional[str] = None,
        caption: Optional[str] = None,
        caption_emb: Optional[torch.Tensor] = None,
    ):
        """
        Convenience forward for inference.

        Args:
            x_t: Input image tensor in [-1, 1].
            direction: Optional override for direction ("a2b" / "b2a").
            caption: Optional caption string.
            caption_emb: Optional precomputed caption embedding.

        Returns:
            Output of `forward_with_networks` using internal components.
        """
        if direction is None:
            assert self.direction is not None, "Direction must be provided."
            direction = self.direction

        if caption is None and caption_emb is None:
            assert self.caption is not None, "Caption must be provided."
            caption = self.caption

        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()

        return self.forward_with_networks(
            x_t,
            direction,
            self.vae_enc,
            self.unet,
            self.vae_dec,
            self.sched,
            self.timesteps,
            caption_enc,
        )

