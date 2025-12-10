import os
import gc
import copy
import lpips
import torch
import wandb

import datetime
import heapq
import random
import torch.nn.functional as F
from glob import glob
from shutil import copyfile
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
import vision_aided_loss
from utils.model import make_1step_sched, make_4step_sched
from utils.cyclegan_turbo_moe import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
from utils import logger

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.simplefilter("ignore")


def compute_expert_diversity_loss(expert_weights):
    """
    计算专家多样性损失 - 促使不同专家权重差异最大化

    参数:
        expert_weights: 列表，包含不同专家的权重张量

    返回:
        多样性损失值
    """
    diversity_loss = 0.0
    num_experts = len(expert_weights)

    # 计算每对专家权重的余弦相似度，并最小化它
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            # 将专家权重展平为向量
            weights_i = expert_weights[i].reshape(expert_weights[i].shape[0], -1)
            weights_j = expert_weights[j].reshape(expert_weights[j].shape[0], -1)

            # 计算余弦相似度
            similarity = F.cosine_similarity(weights_i, weights_j, dim=1).mean()

            # 我们希望最小化相似度（促进不同）
            diversity_loss += similarity

    # 如果有多对专家，取平均值
    if num_experts > 1:
        diversity_loss /= (num_experts * (num_experts - 1) / 2)

    return diversity_loss


def find_model(ckpt_root, step=None):
    if not os.path.exists(ckpt_root):
        return None
    if step is None:
        ckpts = list(filter(lambda x: 'model_' in x, os.listdir(ckpt_root)))
        if not ckpts:
            return None
        steps = map(lambda x: int(x.split(".")[0].split("_")[-1]), ckpts)
        step = max(steps)
    ckpt_path = os.path.join(ckpt_root, f'model_{step}.pkl')
    return ckpt_path


def find_opt(ckpt_root, step=None):
    if not os.path.exists(ckpt_root):
        return None
    if step is None:
        ckpts = list(filter(lambda x: 'opt_' in x, os.listdir(ckpt_root)))
        if not ckpts:
            return None
        steps = map(lambda x: int(x.split(".")[0].split("_")[-1]), ckpts)
        step = max(steps)
    ckpt_path = os.path.join(ckpt_root, f'opt_{step}.pkl')
    return ckpt_path


def find_sched(ckpt_root, step=None):
    if not os.path.exists(ckpt_root):
        return None
    if step is None:
        ckpts = list(filter(lambda x: 'sched_' in x, os.listdir(ckpt_root)))
        if not ckpts:
            return None
        steps = map(lambda x: int(x.split(".")[0].split("_")[-1]), ckpts)
        step = max(steps)
    ckpt_path = os.path.join(ckpt_root, f'sched_{step}.pkl')
    return ckpt_path


def event_happens(prob):
    return random.random() < prob


def main(args):
    # 指定日志文件路径
    log_dir = args.output_dir

    # 创建日志目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)
    time_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    logger.configure(dir=log_dir, log_suffix=time_start)
    logger.info("Args: {}", args)

    # # save code
    # pyfiles = glob("*/*.py", recursive=True) + glob("*.py", recursive=True)
    # code_save_dir = os.path.join(log_dir, 'code')
    # os.makedirs(code_save_dir, exist_ok=True)
    # filename = os.path.basename(__file__)
    # code_save_list = [os.path.basename(__file__),
    #                   parse_args_unpaired_training.__module__.split('.')[-1] + '.py',
    #                   CycleGAN_Turbo.__module__.split('.')[-1] + '.py']
    # for py in pyfiles:
    #     if py in code_save_list:
    #         copyfile(py, os.path.join(log_dir, 'code') + "/" + py)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # From sd-turbo clip
    tokenizer = AutoTokenizer.from_pretrained(r"assets/sd-turbo/tokenizer", revision=args.revision, use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(r"assets/sd-turbo/text_encoder").cuda()
    # From local trained clip
    # tokenizer = AutoTokenizer.from_pretrained(r"clip_chn/checkpoint/acrobat", use_fast=False)
    # text_encoder = BertCLIPTextModel.from_pretrained(r"clip_chn/checkpoint/acrobat").cuda()

    noise_scheduler_step = make_1step_sched()
    text_hidden_size = text_encoder.config.hidden_size
    text_seq_len = text_encoder.config.max_position_embeddings
    expert_split = args.expert_split
    expert_assign_ratio = args.expert_assign_ratio

    unet, unet_lora_layer_high, unet_lora_layer_mid, unet_lora_layer_low = initialize_unet(
        base_model_path="assets/sd-turbo/unet",
        rank=args.lora_rank_unet,
        num_experts=args.num_experts,
        top_k=args.topk_experts,
        expert_balance=args.expert_balancing_strength,
        text_dim=text_hidden_size,
        text_seq_len=text_seq_len,
        fusion_method=args.fusion_method,
        return_lora=True)
    vae_a2b, vae_lora_layer = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_discs = [vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
                     for _ in range(args.num_classes)]
        for disc in net_discs:
            disc.cv_ensemble.requires_grad_(False)

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vae_b2a = copy.deepcopy(vae_a2b)
    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)
    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    params_disc = []
    for i in range(args.num_classes):
        params_disc += list(net_discs[i].parameters())

    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps * accelerator.num_processes,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    # load checkpoint & optimizer & schedulersched_gen
    ckpt_step = 0
    if accelerator.is_main_process:
        ckpt_model_path = find_model(ckpt_dir)
        if ckpt_model_path is not None:
            ckpt_step = int(ckpt_model_path.split('.')[0].split('_')[-1])
            checkpoint = torch.load(ckpt_model_path, map_location='cuda:0')
            vae_enc.load_state_dict(checkpoint["vae_enc"])
            vae_dec.load_state_dict(checkpoint["vae_dec"])
            unet.load_state_dict(checkpoint["unet"])
            logger.info(f'model resume from {ckpt_model_path}')
            del checkpoint
        else:
            pass

        ckpt_opt_path = find_opt(ckpt_dir)
        if ckpt_opt_path is not None:
            checkpoint = torch.load(ckpt_opt_path, map_location='cuda:0')
            optimizer_gen.load_state_dict(checkpoint["opt_gen"])
            optimizer_disc.load_state_dict(checkpoint["opt_disc"])
            logger.info(f'optimizer resume from {ckpt_opt_path}')
            del checkpoint
        else:
            pass

        ckpt_sched_path = find_sched(ckpt_dir)
        if ckpt_sched_path is not None:
            checkpoint = torch.load(ckpt_sched_path, map_location='cuda:0')
            lr_scheduler_gen.load_state_dict(checkpoint["sched_gen"])
            lr_scheduler_disc.load_state_dict(checkpoint["sched_disc"])
            logger.info(f'optimizer resume from {ckpt_sched_path}')
            del checkpoint
        else:
            pass

    dataset_train = UnpairedDataset(dataset_folder=args.train_dataset_folder, classes=args.classes,
                                    image_prep=args.train_img_prep)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)

    # fixed_captions: list 根据classes参数顺序组成的captions list
    fixed_captions = dataset_train.fixed_captions

    l_images_test = {}
    for i in range(args.num_classes):
        l_images_i_test = []
        cls = args.classes[i]
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            l_images_i_test.extend(glob(os.path.join(args.val_dataset_folder, f"{cls}", ext)))
        l_images_i_test = sorted(l_images_i_test)
        l_images_test[f'{i}'] = l_images_i_test

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_emb_base = []
    for i in range(args.num_classes):
        fixed_i_tokens = tokenizer(fixed_captions[i], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        fixed_i_emb_base = text_encoder(fixed_i_tokens.cuda().unsqueeze(0))[0].detach()
        fixed_emb_base.append(fixed_i_emb_base)
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_discs = accelerator.prepare(unet, vae_enc, vae_dec, net_discs)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = ckpt_step  # fixme
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # turn off eff. attn for the disc
    for i in range(args.num_classes):
        for name, module in net_discs[i].named_modules():
            if "attn" in name:
                module.fused_attn = False

    for i in range(args.num_classes):
        net_discs[i].to('cuda')

    # 创建专家相似度正则化损失
    diversity_weight = args.expert_diversity_weight  # 新增的超参数，建议设置为0.1-0.5
    # 专家使用记录器 - 监控每个专家的使用频率
    expert_usage_counter = torch.zeros(args.num_experts)

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            unet.train()
            l_acc = [unet]
            for i in range(args.num_classes):
                l_acc.append(net_discs[i])
            l_acc.append(vae_enc)
            l_acc.append(vae_dec)
            with accelerator.accumulate(*l_acc):
                imgs = []
                fixed_emb = []
                for i in range(args.num_classes):
                    img = batch['pixel_values_list'][i].to(dtype=weight_dtype)
                    imgs.append(img)
                    bsz = img.shape[0]
                    fixed_i_emb = fixed_emb_base[i].repeat(bsz, 1, 1).to(dtype=weight_dtype)
                    fixed_emb.append(fixed_i_emb)

                logs = {}
                real_imgs = []
                rec_imgs = []
                fake_imgs = []

                # 获取并存储当前专家的权重特征，用于后续的多样性损失计算
                expert_weights = []

                for i in range(0, args.num_classes):
                    img_a = imgs[0]
                    img_b = imgs[i]
                    bsz = img_a.shape[0]
                    fixed_a2b_emb = fixed_emb[i]
                    fixed_b2a_emb = fixed_emb[0]
                    timesteps = torch.tensor([noise_scheduler_step.config.num_train_timesteps - 1] * imgs[0].shape[0],
                                             device=imgs[0].device).long()

                    if event_happens(expert_assign_ratio):
                        expert_assign = torch.zeros(1, args.num_experts, device=img_a.device)
                        expert_assign[:, i] = 1
                    else:
                        expert_assign = None

                    """
                    Cycle Objective
                    """
                    # A -> fake B -> rec A
                    cyc_fake_b, expert_weight_a2b, _ = CycleGAN_Turbo.forward_with_networks(img_a, "a2b",
                                                                                            vae_enc, unet, vae_dec,
                                                                                            noise_scheduler_step,
                                                                                            timesteps, fixed_a2b_emb,
                                                                                            expert_assign=expert_assign)

                    cyc_rec_a, _, _ = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, "b2a",
                                                                           vae_enc, unet, vae_dec,
                                                                           noise_scheduler_step,
                                                                           timesteps, fixed_b2a_emb,
                                                                           )

                    # 获取并存储当前专家的权重特征，用于后续的多样性损失计算
                    expert_weights.append(expert_weight_a2b.detach())
                    # 更新专家使用计数
                    expert_weight = expert_weight_a2b[0].tolist()
                    expert_idx = list(map(expert_weight.index, heapq.nlargest(args.topk_experts, expert_weight)))
                    expert_usage_counter[expert_idx] += bsz

                    loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                    loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips

                    # B -> fake A -> rec B
                    cyc_fake_a, _, _ = CycleGAN_Turbo.forward_with_networks(img_b, "b2a",
                                                                            vae_enc, unet, vae_dec,
                                                                            noise_scheduler_step,
                                                                            timesteps, fixed_b2a_emb,
                                                                            )

                    cyc_rec_b, expert_weight_a2b, _ = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, "a2b",
                                                                                           vae_enc, unet, vae_dec,
                                                                                           noise_scheduler_step,
                                                                                           timesteps, fixed_a2b_emb,
                                                                                           expert_assign=expert_assign)

                    # 更新专家使用计数
                    expert_weight = expert_weight_a2b[0].tolist()
                    expert_idx = list(map(expert_weight.index, heapq.nlargest(args.topk_experts, expert_weight)))
                    expert_usage_counter[expert_idx] += bsz

                    loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                    loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips

                    # 添加专家多样性损失
                    if len(expert_weights) > 1 and i > 0:
                        diversity_loss = compute_expert_diversity_loss(expert_weights) * diversity_weight
                        total_cycle_loss = loss_cycle_a + loss_cycle_b + diversity_loss
                        logs[f"diversity_loss_{i}"] = diversity_loss.detach().item()
                    else:
                        total_cycle_loss = loss_cycle_a + loss_cycle_b

                    accelerator.backward(total_cycle_loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                    """
                    Generator Objective (GAN) for task a->b and b->a (fake inputs)
                    """
                    fake_a, expert_weight_b2a, topk_weight_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a",
                                                                                                         vae_enc, unet,
                                                                                                         vae_dec,
                                                                                                         noise_scheduler_step,
                                                                                                         timesteps,
                                                                                                         fixed_b2a_emb,
                                                                                                         expert_split,
                                                                                                         )
                    fake_b, expert_weight_a2b, topk_weight_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b",
                                                                                                         vae_enc, unet,
                                                                                                         vae_dec,
                                                                                                         noise_scheduler_step,
                                                                                                         timesteps,
                                                                                                         fixed_a2b_emb,
                                                                                                         expert_split,
                                                                                                         )

                    # 更新专家使用计数
                    expert_weight = expert_weight_a2b[0].tolist()
                    expert_idx = list(map(expert_weight.index, heapq.nlargest(args.topk_experts, expert_weight)))
                    expert_usage_counter[expert_idx] += bsz

                    if expert_split:
                        loss_gan_a = 0
                        loss_gan_b = 0
                        for j in range(fake_b.shape[0]):
                            loss_gan_a += net_discs[i](fake_b[j],
                                                       for_G=True).mean() * args.lambda_gan * topk_weight_fake_b[:bsz, j]
                            loss_gan_b += net_discs[0](fake_a[j],
                                                       for_G=True).mean() * args.lambda_gan * topk_weight_fake_a[:bsz, j]
                    else:
                        loss_gan_a = net_discs[i](fake_b, for_G=True).mean() * args.lambda_gan
                        loss_gan_b = net_discs[0](fake_a, for_G=True).mean() * args.lambda_gan

                    total_gan_loss = loss_gan_a + loss_gan_b

                    accelerator.backward(total_gan_loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                    """
                    Identity Objective
                    """
                    idt_a, expert_weight_b2a, topk_weight_idt_a = CycleGAN_Turbo.forward_with_networks(img_b, "a2b",
                                                                                                       vae_enc, unet,
                                                                                                       vae_dec,
                                                                                                       noise_scheduler_step,
                                                                                                       timesteps,
                                                                                                       fixed_a2b_emb,
                                                                                                       expert_split,
                                                                                                       expert_assign)

                    if expert_split:
                        loss_idt_a = 0
                        for j in range(idt_a.shape[0]):
                            loss_idt_a += crit_idt(idt_a[j], img_b) * args.lambda_idt * topk_weight_idt_a[:bsz, j]
                            loss_idt_a += net_lpips(idt_a[j], img_b).mean() * args.lambda_idt_lpips * topk_weight_idt_a[:bsz, j]
                    else:
                        loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                        loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips

                    # 更新专家使用计数
                    expert_weight = expert_weight_a2b[0].tolist()
                    expert_idx = list(map(expert_weight.index, heapq.nlargest(args.topk_experts, expert_weight)))
                    expert_usage_counter[expert_idx] += bsz

                    idt_b, expert_weight_a2b, topk_weight_idt_b = CycleGAN_Turbo.forward_with_networks(img_a, "b2a",
                                                                                                       vae_enc, unet,
                                                                                                       vae_dec,
                                                                                                       noise_scheduler_step,
                                                                                                       timesteps,
                                                                                                       fixed_b2a_emb,
                                                                                                       expert_split,
                                                                                                       )

                    if expert_split:
                        loss_idt_b = 0
                        for j in range(idt_b.shape[0]):
                            loss_idt_b += crit_idt(idt_b[j], img_a) * args.lambda_idt * topk_weight_idt_b[:bsz, j]
                            loss_idt_b += net_lpips(idt_b[j], img_a).mean() * args.lambda_idt_lpips * topk_weight_idt_b[:bsz, j]
                    else:
                        loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                        loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips

                    # 更新专家使用计数
                    expert_weight = expert_weight_a2b[0].tolist()
                    expert_idx = list(map(expert_weight.index, heapq.nlargest(args.topk_experts, expert_weight)))
                    expert_usage_counter[expert_idx] += bsz

                    total_idt_loss = loss_idt_a + loss_idt_b

                    accelerator.backward(total_idt_loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()

                    """
                    Discriminator for task a->b and b->a (fake inputs)
                    """
                    if expert_split:
                        loss_D_A_fake = 0
                        loss_D_B_fake = 0
                        for j in range(fake_a.shape[0]):
                            loss_D_A_fake += net_discs[i](fake_b[j].detach(),
                                                          for_real=False).mean() * args.lambda_gan * topk_weight_fake_b[
                                                                                                     :bsz, j].detach()
                            loss_D_B_fake += net_discs[0](fake_a[j].detach(),
                                                          for_real=False).mean() * args.lambda_gan * topk_weight_fake_a[
                                                                                                     :bsz, j].detach()
                    else:
                        loss_D_A_fake = net_discs[i](fake_b.detach(), for_real=False).mean() * args.lambda_gan
                        loss_D_B_fake = net_discs[0](fake_a.detach(), for_real=False).mean() * args.lambda_gan

                    loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                    accelerator.backward(loss_D_fake, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_discs[0].parameters()) + list(net_discs[i].parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()

                    """
                    Discriminator for task a->b and b->a (real inputs)
                    """
                    loss_D_A_real = net_discs[i](img_b, for_real=True).mean() * args.lambda_gan
                    loss_D_B_real = net_discs[0](img_a, for_real=True).mean() * args.lambda_gan
                    loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                    accelerator.backward(loss_D_real, retain_graph=False)
                    if accelerator.sync_gradients:
                        params_to_clip = list(net_discs[0].parameters()) + list(net_discs[i].parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad()

                    logs[f"cycle_a_{i}"] = loss_cycle_a.detach().item()
                    logs[f"cycle_b_{i}"] = loss_cycle_b.detach().item()
                    logs[f"gan_a_{i}"] = loss_gan_a.detach().item()
                    logs[f"gan_b_{i}"] = loss_gan_b.detach().item()
                    logs[f"disc_a_{i}"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
                    logs[f"disc_b_{i}"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
                    logs[f"idt_a_{i}"] = loss_idt_a.detach().item()
                    logs[f"idt_b_{i}"] = loss_idt_b.detach().item()

                    # 记录专家使用情况
                    logs["expert_usage"] = expert_usage_counter.tolist()

                    real_imgs.append(img_b)
                    rec_imgs.append(cyc_rec_b)
                    fake_imgs.append(fake_b)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_imgs = []
                                for i in range(args.num_classes):
                                    viz_img = batch['pixel_values_list'][i].to(dtype=weight_dtype)
                                    viz_imgs.append(viz_img)
                                log_dict = {}
                                log_dict[f'train/real_HE'] = [
                                    wandb.Image(viz_imgs[0][idx].float().detach().cpu(), caption=f"idx={idx}") for
                                    idx in range(bsz)]
                                for i in range(1, args.num_classes):
                                    log_dict[f'train/real_{args.classes[i]}'] = [
                                        wandb.Image(viz_imgs[i][idx].float().detach().cpu(), caption=f"idx={idx}") for
                                        idx in range(bsz)]
                                    log_dict[f"train/rec_{args.classes[i]}"] = [
                                        wandb.Image(rec_imgs[i - 1][idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)]
                                    log_dict[f"train/fake_{args.classes[i]}"] = [
                                        wandb.Image(fake_imgs[i - 1][idx].float().detach().cpu(), caption=f"idx={idx}")
                                        for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    if global_step % args.checkpointing_steps == 0 and global_step > 1:
                        # 在保存模型之前，等待所有进程到达这一点
                        accelerator.wait_for_everyone()

                        eval_unet = accelerator.unwrap_model(unet)
                        eval_vae_enc = accelerator.unwrap_model(vae_enc)
                        eval_vae_dec = accelerator.unwrap_model(vae_dec)

                        out_model = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        out_opt = os.path.join(args.output_dir, "checkpoints", f"opt_{global_step}.pkl")
                        out_sched = os.path.join(args.output_dir, "checkpoints", f"sched_{global_step}.pkl")

                        ckpt_model = {}
                        ckpt_model["vae_lora"] = vae_lora_layer
                        ckpt_model["vae_enc"] = eval_vae_enc.state_dict()
                        ckpt_model["vae_dec"] = eval_vae_dec.state_dict()
                        ckpt_model["unet"] = eval_unet.state_dict()

                        ckpt_opt = {}
                        ckpt_opt['opt_gen'] = optimizer_gen.state_dict()
                        ckpt_opt['opt_disc'] = optimizer_disc.state_dict()

                        ckpt_sched = {}
                        ckpt_sched['sched_gen'] = lr_scheduler_gen.state_dict()
                        ckpt_sched['sched_disc'] = lr_scheduler_disc.state_dict()

                        torch.save(ckpt_model, out_model)
                        torch.save(ckpt_opt, out_opt)
                        torch.save(ckpt_sched, out_sched)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # compute val FID and DINO-Struct scores
                    if global_step % args.validation_steps == 0 and global_step > 1:  # todo  验一次对于1920张 batch为1要8min30s
                        _timesteps = torch.tensor([noise_scheduler_step.config.num_train_timesteps - 1] * 1,
                                                  device="cuda").long()
                        """
                        Evaluate "A->B"
                        """
                        output_dir_m = os.path.join(args.output_dir, f"translation_step{global_step:06d}_multi-expert")
                        os.makedirs(output_dir_m, exist_ok=True)
                        output_dir_s = os.path.join(args.output_dir, f"translation_step{global_step:06d}_single-expert")
                        os.makedirs(output_dir_s, exist_ok=True)

                        # get val input images from domain a
                        for idx, input_img_path in enumerate(tqdm(l_images_test['0'])):
                            if idx > args.validation_num_images and args.validation_num_images > 0:
                                break
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                for i in range(0, args.num_classes):
                                    direction = "a2b"
                                    # Single expert output
                                    expert_assign = torch.zeros(1, args.num_experts, device=img_a.device)
                                    expert_assign[:, i] = 1
                                    eval_fake_b, _, _ = CycleGAN_Turbo.forward_with_networks(img_a, direction,
                                                                                             eval_vae_enc, eval_unet,
                                                                                             eval_vae_dec,
                                                                                             noise_scheduler_step,
                                                                                             _timesteps,
                                                                                             fixed_emb[i][0:1],
                                                                                             expert_assign=expert_assign)
                                    name = input_img_path.split('\\')[-1].split('.')[0]
                                    outf = os.path.join(output_dir_s, f"{name}_{i}.jpg")
                                    eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                    eval_fake_b_pil.save(outf)

                                    # Mixture of expert output
                                    expert_assign = None
                                    eval_fake_b, _, _ = CycleGAN_Turbo.forward_with_networks(img_a, direction,
                                                                                             eval_vae_enc, eval_unet,
                                                                                             eval_vae_dec,
                                                                                             noise_scheduler_step,
                                                                                             _timesteps,
                                                                                             fixed_emb[i][0:1],
                                                                                             expert_assign=expert_assign)
                                    name = input_img_path.split('\\')[-1].split('.')[0]
                                    outf = os.path.join(output_dir_m, f"{name}_{i}.jpg")
                                    eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                    eval_fake_b_pil.save(outf)

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    with torch.cuda.device(0):
        args = parse_args_unpaired_training()
        main(args)
