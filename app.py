from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything
from PIL import Image
import random
import argparse
import functools
import os
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import streamlit as st



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device).float()


@torch.inference_mode()
def generate_image(seed: int, truncation_psi: float, model: nn.Module,
                   device: torch.device) -> np.ndarray:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)

    out = model(z, label, truncation_psi=truncation_psi, force_fp32=True)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = r'C:\Users\onlym\GAN_DIPLOMA\stylegan_human_v2_1024.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label, force_fp32=True)
    return model

def load_model1(file_name: str, device: torch.device) -> nn.Module:
    path = r'C:\Users\onlym\GAN_DIPLOMA\stylegan3\network-snapshot-000560.pkl'
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label, force_fp32=True)
    return model



def main():
    st.title('')
    menu = ['Исследовать латентное пространство', 'Исследовать латентное пространство2', 'Сгенерировать изображение по тексту', 'О проекте']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Исследовать латентное пространство':
        st.subheader('Пример исследования латентного пространства 1024x512 на StyleGAN2')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model('stylegan_human_v2_1024.pkl', device)

        func = functools.partial(generate_image, model=model, device=device)
        func = functools.update_wrapper(func, generate_image)
        st.sidebar.title("Features")
        st.subheader('Slider1')
        seed = st.number_input(min_value=0, label='Введите SEED')
        psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Truncation psi')
        test1 = func(seed,psi)
        st.image(test1, width=600)
    if choice == 'Исследовать латентное пространство2':
        st.subheader('Пример исследования латентного пространства 1024x512 на StyleGAN2')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model1('network-snapshot-000560.pkl', device)

        func = functools.partial(generate_image, model=model, device=device)
        func = functools.update_wrapper(func, generate_image)
        st.sidebar.title("Features")
        st.subheader('Slider1')
        seed = st.number_input(min_value=0, label='Введите SEED')
        psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Truncation psi')
        test1 = func(seed,psi)
        st.image(test1, width=600)

    if choice == 'Сгенерировать изображение по тексту':
        st.subheader('ruDALL-E tuned by SokolovAV')
        device = 'cuda'
        dalle = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=device)
        tokenizer = get_tokenizer()
        vae = get_vae().to(device)

        def dalle_wrapper(prompt: str):
            pil_images = []

            top_k, top_p = random.choice([
                (1024, 0.98),
                (512, 0.97),
                (384, 0.96),

            ])

            _images, _ = generate_images(
                prompt,
                tokenizer,
                dalle,
                vae,
                top_k=top_k,
                images_num=2,
                top_p=top_p
            )
            pil_images += _images

            return pil_images

        with st.form(key='123'):
            raw_text = st.text_input('Введите что-нибудь')
            submit_text = st.form_submit_button(label='Сгенерировать!')
            if submit_text!='':
                st.image(dalle_wrapper(raw_text), width=600)



if __name__ == '__main__':
    main()