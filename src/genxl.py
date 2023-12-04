import re
import os
import time
import random
import torch
import traceback
import tqdm

from secretics import cache_dir, root, basic_negation, special_negation, quality_requirements

os.environ['XDG_CACHE_HOME'] = cache_dir
os.environ['HF_HOME'] = f'{cache_dir}/huggingface'
os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/huggingface/transformers'

from diffusers import DiffusionPipeline, AutoencoderKL
from pytorch_lightning import seed_everything
from transformers import logging as tl

from prompts import get_prompt


outdir = f'{root}/New'

tl.set_verbosity_error()
torch.cuda.empty_cache()
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16)
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16"
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    variant="fp16"
)
refiner.to("cuda")

limits = {'total': random.randint(3000, 5000)}
rolls = {'seed': 0, 'prompt': 0}


def soft_clamp_tensor(input_tensor, threshold=3.5, boundary=4):
    if max(abs(input_tensor.max()), abs(input_tensor.min())) < 4:
        return input_tensor
    channel_dim = 1

    max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
    over_mask = (input_tensor > threshold)

    min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
    under_mask = (input_tensor < -threshold)

    return torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor))

def center_tensor(input_tensor, channel_shift=1, full_shift=1, channels=[0, 1, 2, 3]):
    for channel in channels:
        input_tensor[0, channel] -= input_tensor[0, channel].mean() * channel_shift
    return input_tensor - input_tensor.mean() * full_shift

def maximize_tensor(input_tensor, boundary=4, channels=[0, 1, 2]):
    min_val = input_tensor.min()
    max_val = input_tensor.max()

    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    input_tensor[0, channels] *= normalization_factor

    return input_tensor

def callback(pipe, step_index, timestep, cbk):
    if timestep > 950:
        threshold = max(cbk["latents"].max(), abs(cbk["latents"].min())) * 0.998
        cbk["latents"] = soft_clamp_tensor(cbk["latents"], threshold * 0.998, threshold)
    if timestep > 700:
        cbk["latents"] = center_tensor(cbk["latents"], 0.8, 0.8)
    if timestep > 1 and timestep < 100:
        cbk["latents"] = center_tensor(cbk["latents"], 0.6, 1.0)
        cbk["latents"] = maximize_tensor(cbk["latents"])
    return cbk


for roll in range(limits['total']):
    torch.cuda.empty_cache()

    if not rolls['seed']:
        limits['seed'] = random.randint(20, 500)
        rolls['seed'] = 1
        seed = random.randint(1, 1000000)
        seed_everything(seed)

    if not rolls['prompt']:
        torch.cuda.empty_cache()
        raw_prompt = get_prompt()
        if raw_prompt.startswith('+'):
            raw_prompt = raw_prompt[1:]
            selectors = re.findall(r'\[[a-zA-Z0-9.|,-]+\]', raw_prompt)
            minimum = 5 * (len(selectors) + 1)
        else:
            selectors = None
            minimum = 12
        limits['prompt'] = random.randint(minimum, int(minimum * 1.2))
        rolls['prompt'] = 1

    steps = random.randint(70, 200)
    size = random.randint(15000, 40000)
    ratio = random.uniform(8/20, 355/144)
    if random.random() > 0.4:
        ratio = ratio ** 0.5
    height = round((size/ratio)**0.5)
    width = round(size/height)
    height = height*8
    width = width*8
    noise = random.uniform(0.75, 0.9)
    scale = random.uniform(6, 20)
    gen_prompt = raw_prompt
    if selectors:
        for selector in selectors:
            options = selector[1:-1].split('|')
            gen_prompt = gen_prompt.replace(selector, random.choice(options))
    gen_prompt = ' '.join(gen_prompt.replace('.', ' ').split())

    if any([x in gen_prompt for x in quality_requirements]):
        negation_threshold = 0.95
        negation = basic_negation + special_negation
    else:
        negation_threshold = 0.25
        negation = basic_negation

    coin = random.random()
    if coin < negation_threshold:
        random.shuffle(negation)
        negative_prompt = ', '.join(negation[:random.randint(round(len(negation)*0.6), len(negation))])
    else:
        negative_prompt = None

    file_prompt = f'{gen_prompt} ({negative_prompt})' if negative_prompt else gen_prompt
    file_prompt = file_prompt[:180] + '...' if len(file_prompt) > 180 else file_prompt

    try:
        stopwatch = time.time()
        print(f'\n\n\n'
              f'     {roll+1:>4} of {limits["total"]:>4}{"" if roll else " — R E S T A R T E D + R E I N I T I A L I Z E D"}\n'
              f'{rolls["prompt"]:>3} of {limits["prompt"] - 1:>3} — {"NEG " if negative_prompt else ""}{gen_prompt[:70] + "..." if len(gen_prompt) > 70 else gen_prompt}{" — NEW PROMPT" if rolls["prompt"] == 1 else ""}\n'
              f'{steps:>7} STEPS, {scale:>7.2f} SCALE, {noise:>7.2f} NOISE\n'
              f'{width:>2} × {height:>2} DIMENSIONS\n')

        image = base(
            prompt=gen_prompt,
            negative_prompt=negative_prompt,
            height=height, width=width, num_inference_steps=steps,
            denoising_end=noise, guidance_scale=scale,
            callback_on_step_end=callback,
            callback_on_step_end_inputs=["latents"],
            output_type="latent"
        ).images
        image = refiner(
            prompt=gen_prompt,
            negative_prompt=negative_prompt,
            height=height, width=width, num_inference_steps=steps,
            denoising_start=noise,
            image=image
        ).images[0]
        image.save(f'{outdir}/{file_prompt} — ST{steps} TM{int(time.time() - stopwatch)} NS{noise:.2f} {int(time.time()) % 10000}.jpg')

        for key in ['prompt', 'seed']:
            rolls[key] += 1

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
        print('F A I L E D')

    for key in ['prompt', 'seed']:
        if rolls[key] >= limits[key]:
            rolls[key] = 0

    if rolls['prompt']:
        try:
            print('Continue with prompt?')
            for i in tqdm.trange(100):
                time.sleep(.01)
            print('Prompt continues')
        except KeyboardInterrupt:
            rolls['prompt'] = 0
            print('Prompt dropped')
