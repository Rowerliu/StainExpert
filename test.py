'''

将代码改成每次输入两类图像进行转换，moe提到diffusion部分

'''
import os
import copy
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from utils.model import make_1step_sched
from utils.cyclegan_turbo_moe_19 import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from utils.training_utils_lzy_DeepLIIF import UnpairedDataset, parse_args_unpaired_training

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(r"assets/sd-turbo/tokenizer", revision=args.revision, use_fast=False)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained(r"assets/sd-turbo/text_encoder").cuda()
    text_hidden_size = text_encoder.config.hidden_size
    text_seq_len = text_encoder.config.max_position_embeddings

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
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    vae_b2a = copy.deepcopy(vae_a2b)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    dataset_test = UnpairedDataset(dataset_folder=args.val_dataset_folder, image_prep=args.val_img_prep, classes=args.classes)
    num_testdata = dataset_test.__len__() // len(dataset_test.dataset_folders)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.train_batch_size, shuffle=False,
                                                   num_workers=args.dataloader_num_workers)
    fixed_captions = dataset_test.fixed_captions

    fixed_emb_base = []
    for i in range(args.num_classes):
        fixed_i_tokens = tokenizer(fixed_captions[i], max_length=tokenizer.model_max_length, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids[0]
        fixed_i_emb_base = text_encoder(fixed_i_tokens.cuda().unsqueeze(0))[0].detach()
        fixed_emb_base.append(fixed_i_emb_base)
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec = accelerator.prepare(unet, vae_enc, vae_dec)

    _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1,
                              device="cuda").long()

    checkpoint_path = rf'01_result/20250424_DeepLIIF-c384_expert-split-4-2/checkpoints/model_3000.pkl'
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    if accelerator.is_main_process:
        eval_unet = accelerator.unwrap_model(unet)
        eval_vae_enc = accelerator.unwrap_model(vae_enc)
        eval_vae_dec = accelerator.unwrap_model(vae_dec)
        eval_vae_enc.load_state_dict(checkpoint["vae_enc"])
        eval_vae_dec.load_state_dict(checkpoint["vae_dec"])
        eval_unet.load_state_dict(checkpoint["unet"])
    del checkpoint

    fixed_emb = []
    bsz = args.train_batch_size
    for i in range(args.num_classes):
        fixed_i_emb = fixed_emb_base[i].repeat(bsz, 1, 1).to(dtype=weight_dtype).cuda()
        fixed_emb.append(fixed_i_emb)

    model_step = int(checkpoint_path.split('.')[0].split('_')[-1])
    fid_output_dir_s = os.path.join(args.output_dir, f"translation_step{model_step:06d}_single-expert_test")
    os.makedirs(fid_output_dir_s, exist_ok=True)
    fid_output_dir_m = os.path.join(args.output_dir, f"translation_step{model_step:06d}_multi-expert_test")
    os.makedirs(fid_output_dir_m, exist_ok=True)

    # eval_unet.train()
    eval_unet.eval()

    for idx, batch in enumerate(test_dataloader):
        if idx > num_testdata:
            break
        with torch.no_grad():
            img_a = batch['pixel_values_list'][0].to(dtype=weight_dtype).cuda()
            img_a_path = batch['pixel_name_list'][0][0]
            for i in range(0, args.num_classes):
                direction = "a2b"
                eval_fake_b, _, _ = CycleGAN_Turbo.forward_with_networks(img_a, direction,
                                                                         eval_vae_enc, eval_unet,
                                                                         eval_vae_dec,
                                                                         noise_scheduler_1step,
                                                                         _timesteps, fixed_emb[i][0:1])
                outf = os.path.join(fid_output_dir_m, f"{img_a_path}_{i}.jpg")
                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                eval_fake_b_pil.save(outf)

                expert_assign = torch.zeros(1, args.num_experts, device=img_a.device)
                expert_assign[:, i] = 1
                eval_fake_b, _, _ = CycleGAN_Turbo.forward_with_networks(img_a, direction,
                                                                         eval_vae_enc, eval_unet,
                                                                         eval_vae_dec,
                                                                         noise_scheduler_1step,
                                                                         _timesteps, fixed_emb[i][0:1],
                                                                         expert_assign=expert_assign)
                outf = os.path.join(fid_output_dir_s, f"{img_a_path}_{i}.jpg")
                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                eval_fake_b_pil.save(outf)

                print(f'Translate: {img_a_path}_{i}.jpg')
    # for idx, batch in enumerate(test_dataloader):
    #     if idx > num_testdata:
    #         break
    #     with torch.no_grad():
    #         img_a = batch['pixel_values_list'][0].to(dtype=weight_dtype).cuda()
    #         img_a_path = batch['pixel_name_list'][0][0]
    #         for i in range(0, args.num_classes):
    #             for j in range(0, args.num_classes):
    #                 expert_assign = torch.zeros(1, args.num_experts, device=img_a.device)
    #                 expert_assign[:, i] = 1
    #                 eval_fake_b, _ = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
    #                                 eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_emb[j][0:1], expert_assign)
    #                 outf = os.path.join(fid_output_dir_m, f"{img_a_path}_expert-{i}_text-{j}.png")
    #                 eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
    #                 eval_fake_b_pil.save(outf)
    #                 print(f'Translate: {img_a_path}_{i}_{j}.png')

                # eval_fake_b, _ = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
                #                 eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_emb[i][0:1])
                # outf = os.path.join(fid_output_dir_m, f"{img_a_path}_st_{i}.png")
                # eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                # eval_fake_b_pil.save(outf)
                # print(f'Translate: {img_a_path}_st_{i}.png')


if __name__ == "__main__":
    with torch.cuda.device(0):
        args = parse_args_unpaired_training()
        main(args)
