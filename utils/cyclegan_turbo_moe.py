import os
import copy
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig, get_peft_model, PeftMixedModel
from utils.model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
from utils.Unet_Parameter_Analyzer import analyze_unet_parameters

import sys
sys.path.append("..")


class VAE(nn.Module):
    def __init__(self, vaes=None):
        super(VAE, self).__init__()
        self.vaes = vaes

    def encode(self, x):
        # assert direction in ["a2b", "b2a"]
        # if direction == "a2b":
        #     _vae = self.vae
        # else:
        #     _vae = self.vae_b2a
        # 所有转换共用唯一编码器提取共同特征
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
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


class GatingNetwork(nn.Module):
    """
    多模态专家门控网络：同时利用图像和文本特征来决定专家权重
    """

    def __init__(self, num_experts=3, top_k=1, image_dim=1024, text_dim=1024, text_seq_len=77, hidden_dim=256,
                 expert_balance=0.1, fusion_method="concat"):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.text_seq_len = text_seq_len
        self.num_experts = num_experts
        self.top_k = top_k
        self.fusion_method = fusion_method

        # 图像特征处理
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 文本序列编码
        self.text_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True
        )
        self.text_norm = nn.LayerNorm(text_dim)

        # 文本特征处理
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 特征融合
        if fusion_method == "concat":
            self.fusion_dim = hidden_dim * 2
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        elif fusion_method == "cross_attention":
            self.fusion_dim = hidden_dim
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        elif fusion_method == "film":
            # FiLM条件调制
            self.fusion_dim = hidden_dim
            self.gamma_layer = nn.Linear(hidden_dim, hidden_dim)
            self.beta_layer = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

        # 决策网络
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        # 专家使用统计
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.enable_balancing = True
        self.balancing_strength = expert_balance

    def _process_image(self, image_features):
        """处理图像特征"""
        # 假设输入是VAE编码的latent或CNN特征
        return self.image_encoder(image_features)

    def _process_text(self, text_features):
        """处理文本特征"""
        # 使用自注意力处理序列
        attn_output, _ = self.text_attention(
            text_features, text_features, text_features
        )
        attn_output = self.text_norm(attn_output + text_features)  # 残差连接

        # 池化获取全局特征
        pooled_text = attn_output.mean(dim=1)  # [batch_size, text_dim]

        # 编码文本特征
        return self.text_encoder(pooled_text)

    def _fuse_features(self, image_features, text_features):
        """融合图像和文本特征"""
        if self.fusion_method == "concat":
            # 简单拼接
            fused = torch.cat([image_features, text_features], dim=1)
            return self.fusion_layer(fused)

        elif self.fusion_method == "cross_attention":
            # 跨模态注意力，将文本作为query，图像作为key和value
            # 扩展图像特征为序列形式 [batch_size, 1, hidden_dim]
            image_seq = image_features.unsqueeze(1)
            # 扩展文本特征为序列形式 [batch_size, 1, hidden_dim]
            text_seq = text_features.unsqueeze(1)

            # 计算跨模态注意力
            fused, _ = self.cross_attention(text_seq, image_seq, image_seq)
            fused = self.fusion_norm(fused + text_seq)  # 残差连接
            return fused.squeeze(1)  # [batch_size, hidden_dim]

        elif self.fusion_method == "film":
            # 特征-语言调制 (FiLM)
            gamma = self.gamma_layer(text_features).unsqueeze(1)  # 缩放因子
            beta = self.beta_layer(text_features).unsqueeze(1)  # 偏移因子

            # 应用FiLM调制
            fused = gamma * image_features + beta
            return fused

    def forward(self, gating_input, hard_gating=False):
        """
        前向传播

        Args:
            gating_input: image_features and text_features
            image_features: 图像特征，可以是VAE latent [batch_size, image_dim]
                           或 CNN特征 [batch_size, channels, height, width]
            text_features: 文本特征 [batch_size, seq_len, text_dim]
            hard_gating: 是否使用硬门控

        Returns:
            专家权重 [batch_size, num_experts]
        """
        image_features, text_features = gating_input
        batch_size = image_features.shape[0]

        # 处理CNN特征（如果需要）
        if len(image_features.shape) == 4:
            # [B, C, H, W] -> [B, C*H*W]
            image_features = image_features.view(batch_size, -1)
            # 如果维度过大，可以添加一个投影层
            if image_features.shape[1] != self.image_dim:
                raise ValueError(f"图像特征维度 {image_features.shape[1]} 与预期的 {self.image_dim} 不匹配")

        # 1. 处理图像特征
        processed_image = self._process_image(image_features)

        # 2. 处理文本特征
        processed_text = self._process_text(text_features)

        # 3. 融合特征
        fused_features = self._fuse_features(processed_image, processed_text)

        # 4. 生成专家权重logits
        logits = self.decision_network(fused_features)

        # 5. 专家均衡（可选）
        if self.enable_balancing and self.training:
            usage_penalty = self.expert_usage / (self.expert_usage.sum() + 1e-6)
            logits = logits - usage_penalty * self.balancing_strength

        # 6. 根据模式选择软门控或硬门控
        if hard_gating:
            # 硬门控：只选择最大值，其他为0
            indices = torch.argmax(logits, dim=-1)
            weights = torch.zeros_like(logits).scatter_(-1, indices.unsqueeze(-1), 1.0)

            # 更新专家使用统计
            if self.training:
                for idx in indices:
                    self.expert_usage[idx] += 1
        else:
            # 软门控：使用softmax计算权重
            weights = F.softmax(logits / self.temperature, dim=-1)
            _, indices = torch.topk(weights, k=self.top_k, dim=-1)

            # 更新专家使用统计
            if self.training:
                for idx in indices:
                    self.expert_usage[idx] += 1
                # self.expert_usage += weights.sum(dim=0).detach()

        return weights


class MoEAdapterUNet(nn.Module):
    """
    基于MoE架构的UNet适配器微调模型，支持训练软门控和推理硬门控
    """

    def __init__(self, unet, num_experts=3, top_k=1, expert_balance=0.1, adapter_names=None,
                 image_dim=1024, text_dim=1024, text_seq_len=77, fusion_method="concat"):
        super().__init__()
        self.unet = unet
        self.num_experts = num_experts
        self.top_k = top_k  # 仅激活最相关的 top_k 个专家
        self.adapter_names = adapter_names or []
        self._shared_adapter_name = None  # 用于追踪共享适配器

        # 创建门控网络
        self.gating_network = GatingNetwork(num_experts, top_k, image_dim, text_dim, text_seq_len,
                                            expert_balance=expert_balance, fusion_method=fusion_method)

    def _activate_expert(self, expert_idx):
        """激活指定专家的适配器，并确保其他适配器被禁用"""
        # 激活指定专家的适配器
        expert_adapters = self.adapter_names[expert_idx]

        # 如果有共享适配器，添加到激活列表
        if self._shared_adapter_name:
            if isinstance(expert_adapters, list):
                active_adapters = [self._shared_adapter_name] + expert_adapters
            else:
                active_adapters = [self._shared_adapter_name, expert_adapters]
        else:
            active_adapters = expert_adapters

        self.unet.set_adapter(active_adapters)

    def set_shared_adapter(self, adapter_name):
        """设置所有专家共享的适配器"""
        self._shared_adapter_name = adapter_name

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            expert_assign: Optional[torch.Tensor] = None,
            gating_input: Optional[torch.Tensor] = None,
            **kwargs
    ):
        # 如果没有提供gating_input，则使用encoder_hidden_states的均值
        if gating_input is None:
            gating_input = [sample, encoder_hidden_states]

        # 软门控: 运行所有专家并加权组合; 硬门控: 只运行被选中的专家
        if expert_assign is not None:
            expert_weights = expert_assign.cuda()  # shape=[1, 5], e.g. tensor = [[1,0,0,0,0]]
            # 选择 top_k 个专家索引（稀疏激活）
            topk_weights, topk_indices = torch.topk(expert_weights, k=1, dim=-1)
            topk_weights = F.softmax(topk_weights)
        else:
            expert_weights = self.gating_network(gating_input)
            # 选择 top_k 个专家索引（稀疏激活）
            topk_weights, topk_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)
            topk_weights = F.softmax(topk_weights)
            topk_indices = topk_indices[0].tolist()

        expert_outputs = []
        # 对每个专家单独运行UNet，并存储输出
        for expert_idx in topk_indices:
            # 只处理被选中的专家
            expert_idx = int(expert_idx)

            # 激活当前专家的适配器
            self._activate_expert(expert_idx)

            # 运行UNet
            output = self.unet(sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states, **kwargs).sample
            expert_outputs.append(output)

        # 融合所有专家的输出
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            batch_size = expert_output.shape[0]
            weights = topk_weights[:batch_size, i].view(-1, 1, 1, 1)
            combined_output += expert_output * weights

        # 返回结果
        return combined_output, expert_weights, topk_weights


def initialize_unet(
        base_model_path="assets/sd-turbo/unet",
        rank=128,
        num_experts=1,
        top_k=1,
        expert_balance=0.1,
        text_dim=1024,
        text_seq_len=77,
        fusion_method='concat',
        return_lora=False
):
    """
    初始化基于MoE的UNet模型，每个专家有自己的encoder、decoder和其他适配器
    """
    # 加载预训练UNet
    unet = UNet2DConditionModel.from_pretrained(base_model_path)
    unet.requires_grad_(False)
    unet.train()

    # 准备LoRA目标模块，划分为high / mid / low importance from [Layer Importance Analyzer]
    l_layer_high, l_layer_mid, l_layer_low = [], [], []
    l_grep_high = ["conv", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in"]
    l_grep_mid = ["ff.net", "conv1", "conv2"]
    l_grep_low = ["to_out", "to_k", "to_q", "to_v", "time_embedding"]

    for n, p in unet.named_parameters():
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

    # 设置共同学习适配器
    # 创建LoRA配置
    lora_conf_high = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=l_layer_high,
        lora_alpha=rank * 4,
    )

    # 设置共同学习层
    lora_name_high = f"expert_all_H-layer"
    unet = PeftMixedModel(unet, lora_conf_high, lora_name_high)

    # 为每个专家创建适配器
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

        # 为每个专家添加适配器
        lora_name_mid = f"expert_{expert_id}_M-layer"
        lora_name_low = f"expert_{expert_id}_L-layer"

        # unet.add_adapter(lora_conf_high, adapter_name=lora_high_name)
        unet.add_adapter(adapter_name=lora_name_mid, peft_config=lora_conf_mid)
        unet.add_adapter(adapter_name=lora_name_low, peft_config=lora_conf_low)

        # 将这组适配器名称添加到专家适配器列表
        # adapter_names.append([lora_high_name, lora_mid_name, lora_low_name])
        adapter_names.append([lora_name_mid, lora_name_low])
        initial_adapter_names.append(lora_name_mid)
        initial_adapter_names.append(lora_name_low)

    # 初始化时set
    unet.set_adapter(initial_adapter_names)

    # 创建MoE模型
    moe_unet = MoEAdapterUNet(unet, num_experts=num_experts, top_k=top_k, expert_balance=expert_balance, adapter_names=adapter_names,
                              image_dim=4096, text_dim=text_dim, text_seq_len=text_seq_len, fusion_method=fusion_method)

    # 设置共享适配器
    moe_unet.set_shared_adapter(lora_name_high)

    if return_lora:
        return moe_unet, l_layer_high, l_layer_mid, l_layer_low
    else:
        return moe_unet


def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained(r"assets/sd-turbo/vae")
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)  # fixme (True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1),
                                              bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1),
                                              bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1),
                                              bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1),
                                              bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "skip_conv_1",
                            "skip_conv_2", "skip_conv_3", "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae = get_peft_model(vae, vae_lora_config)  # fixme
    # vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    vae.add_adapter(adapter_name="vae_skip", peft_config=vae_lora_config)   # fixme
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("assets/sd-turbo/tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("assets/sd-turbo/text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("assets/sd-turbo/vae")
        unet = UNet2DConditionModel.from_pretrained("assets/sd-turbo/unet")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
        if pretrained_name == "day_to_night":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/day2night.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the night"
            self.direction = "a2b"
        elif pretrained_name == "night_to_day":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/night2day.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "clear_to_rainy":
            sd = torch.load(r'clear2rainy.pkl')
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in heavy rain"
            self.direction = "a2b"
        elif pretrained_name == "rainy_to_clear":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/rainy2clear.pkl"
            self.load_ckpt_from_url(url, ckpt_folder)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = "driving in the day"
            self.direction = "b2a"
        elif pretrained_name == "he_to_ihc":
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = "a2b"
        elif pretrained_path is not None:
            sd = torch.load(pretrained_path)
            self.load_ckpt_from_state_dict(sd)
            self.timesteps = torch.tensor([999], device="cuda").long()
            self.caption = None
            self.direction = None

        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def load_ckpt_from_state_dict(self, sd):
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                       target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                      target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
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

        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                     target_modules=sd["vae_lora_target_modules"])
        # self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae = get_peft_model(self.vae, vae_lora_config)  # fixme
        self.vae.add_adapter(peft_config=vae_lora_config, adapter_name="vae_skip")  # fixme

        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(x, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb, expert_assign=None):
        B = x.shape[0]
        assert direction in ["a2b", "b2a"]
        x_enc = vae_enc(x, direction=direction).to(x.dtype)
        model_pred, expert_weights, topk_weights = unet(x_enc, timesteps, encoder_hidden_states=text_emb, expert_assign=expert_assign)
        if len(model_pred.shape) == 5:
            x_out_dec = []
            for i in range(model_pred.shape[0]):
                model_pred_i = model_pred[i]
                x_out_i = torch.stack([sched.step(model_pred_i[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
                x_out_dec_i = vae_dec(x_out_i, direction=direction)
                x_out_dec.append(x_out_dec_i)
            x_out_dec = torch.stack(x_out_dec, dim=0)
        else:
            x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
            x_out_dec = vae_dec(x_out, direction=direction)
        return x_out_dec, expert_weights, topk_weights

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):
        # add all unet parameters
        _unet = unet.unet
        params_gen = list(_unet.conv_in.parameters())
        _unet.conv_in.requires_grad_(True)
        for n, p in _unet.named_parameters():
            if "lora" in n and "expert" in n:
                # assert p.requires_grad
                p.requires_grad_(True)  # fixme
                params_gen.append(p)

        # add all gating network parameters
        _gating_network = unet.gating_network
        for n, p in _gating_network.named_parameters():
            p.requires_grad_(True)  # fixme
            params_gen.append(p)

        # add all vae_a2b parameters
        for n, p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                # assert p.requires_grad
                p.requires_grad_(True)  # fixme
                params_gen.append(p)

        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n, p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                # assert p.requires_grad
                p.requires_grad_(True)  # fixme
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())

        return params_gen

    def forward(self, x_t, direction=None, caption=None, caption_emb=None):
        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.to(
                x_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()
        return self.forward_with_networks(x_t, direction, self.vae_enc, self.unet, self.vae_dec, self.sched,
                                          self.timesteps, caption_enc)
