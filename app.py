from __future__ import annotations
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



st.set_page_config(
     page_title="SokolovAVapp",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded")

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
def generate_interpolated_images(
        seed0: int, psi0: float, seed1: int, psi1: float,
        num_intermediate: int, model: nn.Module,
        device: torch.device) -> tuple[list[np.ndarray], np.ndarray]:
    seed0 = int(np.clip(seed0, 0, np.iinfo(np.uint32).max))
    seed1 = int(np.clip(seed1, 0, np.iinfo(np.uint32).max))

    z0 = generate_z(model.z_dim, seed0, device)
    z1 = generate_z(model.z_dim, seed1, device)
    vec = z1 - z0
    dvec = vec / (num_intermediate + 1)
    zs = [z0 + dvec * i for i in range(num_intermediate + 2)]
    dpsi = (psi1 - psi0) / (num_intermediate + 1)
    psis = [psi0 + dpsi * i for i in range(num_intermediate + 2)]

    label = torch.zeros([1, model.c_dim], device=device)

    res = []
    for z, psi in zip(zs, psis):
        out = model(z, label, truncation_psi=psi, force_fp32=True)
        out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        out = out[0].cpu().numpy()
        res.append(out)
    concatenated = np.hstack(res)
    return res, concatenated


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = 'models/stylegan_human_v2_1024.pkl'
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
    path = './models/network-snapshot-000560.pkl'
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
    menu = ['О Проекте','Исследовать [1] латентное пространство', 'Создание моделей и манипулирование [1] стилем','Исследовать [2] латентное пространство',
            'Создание моделей и манипулирование [2] стилем','Сгенерировать изображение на основе текста']
    choice = st.sidebar.selectbox('Меню', menu)

    if choice == 'О Проекте':
        st.markdown("<h1 style='text-align: center; color: black; font-size: 32px;'>Магистерская диссертация на тему <br> «Использование StyleGANs в рекомендательных моделях в e-commerce»</h1>", unsafe_allow_html = True)

        st.success(
            '''Автор данного исследования: **Соколов Александр Владиславович, ВШЭ**  \n Научный руководитель: **Просветов Артем Владимирович, Кандидат физико-математических наук,  \n Руководитель поведенческих технологий в Райдтех, Яндекс**''')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("https://media.giphy.com/media/h2mwamAvxIXXYKRXty/giphy.gif")

        with col2:
            st.image("https://media.giphy.com/media/wcRQGfE7rR5MzqOauo/giphy.gif")

        with col3:
            st.image("https://media.giphy.com/media/crTs54iF2E8dFTfSdN/giphy.gif")
        col4, col5, col6 = st.columns(3)

        with col4:
            st.image("https://media.giphy.com/media/GUSLVKr1fCNJUwgW2q/giphy.gif")

        with col5:
            st.image("https://media.giphy.com/media/eyehti0n6xxxIuMvIz/giphy.gif")

        with col6:
            st.image("https://media.giphy.com/media/rwPlpxsORgEf5tRiIU/giphy.gif")

        st.markdown(''' Для создания онлайн-примерочной использовалась предобученная 1024x512 модель на основе 40k изображений моделей, а также самостоятельно обученная модель в разрешении 512х512на основе StyleGAN2-ada-pytorch на данных DeepFashion.  \n **Результаты исследования в дальшейнем будут улучшаться и применяться для улучшения пользовательского опыта клиентов!**''')
        st.markdown('''**Что можно и нужно реализовать**:  \n  
        1. В качестве интересного примера можно использовать генерацию изображений моделей, а также онлайн-примерочную в AR-технологиях.  
    2. Можно извлекать сущности из входящих изображений с помощью SMPL-X моделей для получения 3D модели человеческого тела 
       для последующего преобразования его стиля. 
    3. В данном исследовании не реализован метод FaceInput, чтобы можно было выбрать  входящее изображение лица и вставить его 
       в необходимый стиль модели. Загвоздка в том, что на данный момент FaceInput использует лица из 
       предобученного латентного пространства, поэтому при желании нужно было бы менять архитектуру и создавать подходящий формат, 
       на который изначально данная дипломная работа не была нацелена''')



    if choice == 'Исследовать [1] латентное пространство':
        st.subheader('Исследование [1] латентного пространства')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model('stylegan_human_v2_1024.pkl', device)

        func = functools.partial(generate_image, model=model, device=device)
        func = functools.update_wrapper(func, generate_image)
        with st.form(key='123'):
            with st.sidebar:
                seed = st.number_input(min_value=0, label='Выбор модели из латентного пространства')
                psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси')
                test1 = func(seed,psi)
                submit_text = st.form_submit_button(label='Сгенерировать модель!')
        st.image(test1, width=600)

    if choice == 'Создание моделей и манипулирование [1] стилем':
        st.subheader('Создание моделей и манипулирование [1] стилем')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model('models/stylegan_human_v2_1024.pkl', device)
        func = functools.partial(generate_interpolated_images,
                                 model=model,
                                 device=device)
        func = functools.update_wrapper(func, generate_interpolated_images)
        with st.form(key='123'):
            with st.sidebar:
                seed = st.number_input(min_value=0, label='Выбор [1] модели из латентного пространства', key=0)
                psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси', key=0)

                seed1 = st.number_input(min_value=0, label='Выбор [2] модели из латентного пространства', key=1)
                psi1 = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси', key=1)
                slid = st.slider(min_value=0, max_value=21, step=1, value=7, label='Количество трансформаций')
                test11 = func(seed, psi, seed1, psi1, slid)
                submit_text = st.form_submit_button(label='Сгенерировать модель!')
        st.image(test11[1], width=650)
        st.image(test11[0], width=600)




    if choice == 'Исследовать [2] латентное пространство':
        st.subheader('Исследование [2] латентного пространства')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model1('models/network-snapshot-000560.pkl', device)

        func = functools.partial(generate_image, model=model, device=device)
        func = functools.update_wrapper(func, generate_image)
        with st.form(key='123'):
            with st.sidebar:
                seed = st.number_input(min_value=0, label='Выбор модели из латентного пространства')
                psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси')
                test1 = func(seed,psi)
                submit_text = st.form_submit_button(label='Сгенерировать модель!')
        st.image(test1, width=600)

    if choice == 'Создание моделей и манипулирование [2] стилем':
        st.subheader('Создание моделей и манипулирование [2] стилем')
        args = parse_args()
        device = torch.device(args.device)

        model = load_model1('models/network-snapshot-000560.pkl', device)
        func = functools.partial(generate_interpolated_images,
                                 model=model,
                                 device=device)
        func = functools.update_wrapper(func, generate_interpolated_images)
        with st.form(key='123'):
            with st.sidebar:
                seed = st.number_input(min_value=0, label='Выбор [1] модели из латентного пространства', key=0)
                psi = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси', key=0)

                seed1 = st.number_input(min_value=0, label='Выбор [2] модели из латентного пространства', key=1)
                psi1 = st.slider(min_value=0.0, max_value=2.0, step=0.05, value=0.7, label='Усеченное пси', key=1)
                slid = st.slider(min_value=0, max_value=21, step=1, value=7, label='Количество трансформаций')
                test11 = func(seed, psi, seed1, psi1, slid)
                submit_text = st.form_submit_button(label='Сгенерировать модель!')
        st.image(test11[1], width=650)
        st.image(test11[0], width=600)




    if choice == 'Сгенерировать изображение на основе текста':
        st.subheader('Использование tuned модели ruDALL-E для генерации изображений по текстовому описанию')
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
            submit_text = st.form_submit_button(label='Сгенерировать изображение!')
            if submit_text!='':
                st.image(dalle_wrapper(raw_text), width=600)



if __name__ == '__main__':
    main()
