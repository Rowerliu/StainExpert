import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob
import sys
sys.path.append("..")



def parse_args_training():
    """
    Parses command-line arguments used for configuring an unpaired session (CycleGAN-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """

    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")

    # fixed random seed
    parser.add_argument("--seed", type=int, default=2025, help="A seed for reproducible training.")

    # args for the data
    parser.add_argument("--num_classes", default=4, type=int)  # for ANHIR-kidney dataset
    parser.add_argument("--classes", default=['HE', 'MAS', 'PAS', 'PASM'], type=list)  # for ANHIR-kidney dataset

    # args for the loss function
    parser.add_argument("--num_experts", default=4, type=int)
    parser.add_argument("--topk_experts", default=2, type=int)
    parser.add_argument("--fusion_method", default="cross_attention", type=str, choices=["concat", "cross_attention", "film"])
    parser.add_argument("--expert_diversity_weight", default=1, type=float)
    parser.add_argument("--expert_balancing_strength", default=0.2, type=float)

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    # args for dataset and dataloader options
    parser.add_argument("--train_data_path", type=str, default=r"/path/to/anhir-kidney/train")  # fixme
    parser.add_argument("--val_data_path", type=str, default=r"/path/to/anhir-kidney/val")  # fixme
    parser.add_argument("--train_img_prep", type=str, default="random_crop_256")
    parser.add_argument("--val_img_prep", type=str, default="center_crop_256")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=100001)

    # args for the model
    parser.add_argument("--pretrained_model_name_or_path", default=r"asset\sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=8, type=int)

    # args for validation and logging
    parser.add_argument("--viz_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="train")  # track the training project
    parser.add_argument("--validation_steps", type=int, default=20000)  # 20000
    parser.add_argument("--validation_num_images", type=int, default=-1, help="Number of images to use for validation. -1 to use all images.")
    parser.add_argument("--checkpointing_steps", type=int, default=20000)  # 20000

    # args for the optimization options
    parser.add_argument("--learning_rate", type=float, default=1e-5,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # memory saving options
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--gradient_checkpointing", action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")

    args = parser.parse_args()
    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == 'crop_128':
        T = transforms.Compose([
            transforms.CenterCrop(128),
        ])
    elif image_prep == 'random_crop_256':
        T = transforms.Compose([
            transforms.RandomCrop(256),
        ])
    elif image_prep == 'center_crop_256':
        T = transforms.Compose([
            transforms.CenterCrop(256),
        ])
    elif image_prep == 'resize_crop_128':
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.RandomCrop((128, 128)),
        ])
    elif image_prep == 'test_resize_crop_128':
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.CenterCrop((128, 128)),
        ])
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        """
        Itialize the paired dataset object for loading and transforming paired data samples
        from specified dataset folders.

        This constructor sets up the paths to input and output folders based on the specified 'split',
        loads the captions (or prompts) for the input images, and prepares the transformations and
        tokenizer to be applied on the data.

        Parameters:
        - dataset_folder (str): The root folder containing the dataset, expected to include
                                sub-folders for different splits (e.g., 'train_A', 'train_B').
        - split (str): The dataset split to use ('train' or 'test'), used to select the appropriate
                       sub-folders and caption files within the dataset folder.
        - image_prep (str): The image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a dataset item given its index. Each item consists of an input image,
        its corresponding output image, the captions associated with the input image,
        and the tokenized form of this caption.

        This method performs the necessary preprocessing on both the input and output images,
        including scaling and normalization, as well as tokenizing the caption using a provided tokenizer.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        dict: A dictionary containing the following key-value pairs:
            - "output_pixel_values": a tensor of the preprocessed output image with pixel values
            scaled to [-1, 1].
            - "conditioning_pixel_values": a tensor of the preprocessed input image with pixel values
            scaled to [0, 1].
            - "caption": the text caption.
            - "input_ids": a tensor of the tokenized caption.

        Note:
        The actual preprocessing steps (scaling and normalization) for images are defined externally
        and passed to this class through the `image_prep` parameter during initialization. The
        tokenization process relies on the `tokenizer` also provided at initialization, which
        should be compatible with the models intended to be used with this dataset.
        """
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, classes, image_prep):
        """
        A dataset class for loading unpaired data samples from two distinct domains (source and target),
        typically used in unsupervised learning tasks like image-to-image translation.

        The class supports loading images from specified dataset folders, applying predefined image
        preprocessing transformations, and utilizing fixed textual prompts (captions) for each domain,
        tokenized using a provided tokenizer.

        Parameters:
        - dataset_folder (str): Base directory of the dataset containing subdirectories (train_A, train_B, test_A, test_B)
        - split (str): Indicates the dataset split to use. Expected values are 'train' or 'test'.
        - image_prep (str): he image preprocessing transformation to apply to each image.
        - tokenizer: The tokenizer used for tokenizing the captions (or prompts).
        """
        super().__init__()
        self.dataset_folders = {}
        for i in range(len(classes)):
            self.dataset_folders[f'{i}'] = os.path.join(dataset_folder, f"{classes[i]}")

        self.fixed_captions = []
        self.input_ids = []
        for cls in classes:
            with open(os.path.join(dataset_folder, f"fixed_prompt_{cls}.txt"), "r") as f:
                fixed_caption_cls = f.read().strip()
                self.fixed_captions.append(fixed_caption_cls)

        # find all images in the source and target folders with all IMG extensions
        self.l_imgs = {}
        for i in range(len(classes)):
            l_imgs_cls = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
                l_imgs_cls.extend(glob(os.path.join(self.dataset_folders[f'{i}'], ext)))
            self.l_imgs[f'{i}'] = l_imgs_cls

        self.T = build_transform(image_prep)

    def __len__(self):
        """
        Returns:
        int: The total number of items in the dataset.
        """
        length = 0
        for i in range(len(self.l_imgs)):
            length += len(self.l_imgs[f'{i}'])
        return length

    def __getitem__(self, index):
        """
        Fetches a pair of unaligned images from the source and target domains along with their
        corresponding tokenized captions.

        For the source domain, if the requested index is within the range of available images,
        the specific image at that index is chosen. If the index exceeds the number of source
        images, a random source image is selected. For the target domain,
        an image is always randomly selected, irrespective of the index, to maintain the
        unpaired nature of the dataset.

        Both images are preprocessed according to the specified image transformation `T`, and normalized.
        The fixed captions for both domains
        are included along with their tokenized forms.

        Parameters:
        - index (int): The index of the source image to retrieve.

        Returns:
        dict: A dictionary containing processed data for a single training example, with the following keys:
            - "pixel_values_src": The processed source image
            - "pixel_values_tgt": The processed target image
            - "caption_src": The fixed caption of the source domain.
            - "caption_tgt": The fixed caption of the target domain.
            - "input_ids_src": The source domain's fixed caption tokenized.
            - "input_ids_tgt": The target domain's fixed caption tokenized.
        """
        img_path = []
        for i in range(len(self.l_imgs)):
            if index < len(self.l_imgs[f'{i}']):
                img_path_i = self.l_imgs[f'{i}'][index]
            else:
                img_path_i = random.choice(self.l_imgs[f'{i}'])
            img_path.append(img_path_i)
        img_ts = []
        img_path_ts = []
        for i in range(len(img_path)):
            img_pil_i = Image.open(img_path[i]).convert("RGB")
            img_t_i = F.to_tensor(self.T(img_pil_i))
            img_t_i = F.normalize(img_t_i, mean=[0.5], std=[0.5])
            img_ts.append(img_t_i)
            img_path_ts.append(img_path[i].split('\\')[-1].split('.')[0])


        return {
            "pixel_values_list": img_ts,  # image tensor list
            "caption_list": self.fixed_captions,  # captions list
            "input_ids_list": self.input_ids,  # captions embedding list
            "pixel_name_list": img_path_ts,  # image path list
        }
