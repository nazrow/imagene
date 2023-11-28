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
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
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

for roll in range(limits['total']):
    torch.cuda.empty_cache()

    if not rolls['seed']:
        limits['seed'] = random.randint(20, 500)
        rolls['seed'] = 1
        seed = random.randint(1, 1000000)
        seed_everything(seed)

    if not rolls['prompt']:
        raw_prompt = get_prompt()
        if raw_prompt.startswith('+'):
            raw_prompt = raw_prompt[1:]
            selectors = re.findall(r'\[[a-zA-Z0-9.|,-]+\]', raw_prompt)
            minimum = 25 * (len(selectors) + 1)
        else:
            selectors = None
            minimum = 25
        limits['prompt'] = random.randint(minimum, minimum * 2)
        rolls['prompt'] = 1

    steps = random.randint(20, 200)
    # size = random.randint(130, 135)
    # ratio = random.uniform(.5, 2.2)
    # if random.random() > 0.4:
    #     ratio = ratio ** 0.5
    # height = round((size/ratio)**0.5)
    # width = round(size/height)
    height = 1280
    width = 1800
    noise = random.uniform(0.3, 0.99)
    gen_prompt = raw_prompt
    if selectors:
        for selector in selectors:
            options = selector[1:-1].split('|')
            gen_prompt = gen_prompt.replace(selector, random.choice(options))
    gen_prompt = ' '.join(gen_prompt.replace('.', ' ').split())

    if any([x in gen_prompt for x in quality_requirements]):
        negation_threshold = 0.65
        negation = basic_negation + special_negation
    else:
        negation_threshold = 0.25
        negation = basic_negation

    coin = random.random()
    if coin < negation_threshold:
        random.shuffle(negation)
        negative_prompt = ', '.join(negation[:random.randint(round(len(negation)*0.2), len(negation))])
    else:
        negative_prompt = None

    file_prompt = f'{gen_prompt} ({negative_prompt})' if negative_prompt else gen_prompt
    file_prompt = file_prompt[:180] + '...' if len(file_prompt) > 180 else file_prompt

    try:
        stopwatch = time.time()
        print(f'\n\n\n'
              f'     {roll+1:>4} of {limits["total"]:>4}{"" if roll else " — R E S T A R T E D + R E I N I T I A L I Z E D"}\n'
              f'{rolls["prompt"]:>3} of {limits["prompt"] - 1:>3} — {"NEG " if negative_prompt else ""}{gen_prompt[:70] + "..." if len(gen_prompt) > 70 else gen_prompt}{" — NEW PROMPT" if rolls["prompt"] == 1 else ""}\n'
              f'{steps:>7} STEPS, {noise:>7.2f} NOISE\n'
              f'{width:>2} × {height:>2} DIMENSIONS\n')

        image = base(
            prompt=gen_prompt,
            negative_prompt=negative_prompt,
            height=height, width=width, num_inference_steps=steps,
            denoising_end=noise,
            output_type="latent",
        ).images
        image = refiner(
            prompt=gen_prompt,
            negative_prompt=negative_prompt,
            height=height, width=width, num_inference_steps=steps,
            denoising_start=noise,
            image=image,
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
