import re
import os
import time
import random
import torch
import traceback

from secretics import cache_dir, root, basic_negation, special_negation, quality_requirements

os.environ['XDG_CACHE_HOME'] = cache_dir
os.environ['HF_HOME'] = f'{cache_dir}/huggingface'
os.environ['TRANSFORMERS_CACHE'] = f'{cache_dir}/huggingface/transformers'

from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, \
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, \
    KDPM2DiscreteScheduler, PNDMScheduler, StableDiffusionLatentUpscalePipeline
from pytorch_lightning import seed_everything
from transformers import logging as tl

from prompts import get_prompt


outdir = f'{root}/New'

tl.set_verbosity_error()
torch.cuda.empty_cache()
generator = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, safety_checker=None)
schedulers = [
    ('EADS', 0.164, EulerAncestralDiscreteScheduler),
    ('DPMS', 0.119, DPMSolverSinglestepScheduler),
    ('KDPA', 0.106, KDPM2AncestralDiscreteScheduler),
    ('DDIM', 0.084, DDIMScheduler),
    ('EUDS', 0.082, EulerDiscreteScheduler),
    ('HEDS', 0.080, HeunDiscreteScheduler),
    ('DPMM', 0.079, DPMSolverMultistepScheduler),
    ('KDPM', 0.053, KDPM2DiscreteScheduler),
    ('PNDM', 0.043, PNDMScheduler)
]
schedulers_limit = sum(entry[1] for entry in schedulers)
generator = generator.to("cuda")
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16)
upscaler = upscaler.to("cuda")
upscaler.enable_attention_slicing()

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
            minimum = 15 * (len(selectors) + 1)
        else:
            selectors = None
            minimum = 5
        limits['prompt'] = random.randint(minimum, minimum * 2)
        rolls['prompt'] = 1

    steps = random.randint(10, 160)
    size = random.randint(90, 135)
    ratio = random.uniform(.5, 2.2)
    if random.random() > 0.4:
        ratio = ratio ** 0.5
    height = round((size/ratio)**0.5)
    width = round(size/height)
    eta = random.uniform(0.5, 0.95)
    scale = random.uniform(5, 15)
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

    file_prompt += f' ({negative_prompt})' if negative_prompt else ''
    file_prompt = file_prompt[:180] + '...' if len(file_prompt) > 180 else file_prompt

    try:
        stopwatch = time.time()
        coin = random.uniform(0, schedulers_limit*0.99)
        random.shuffle(schedulers)
        for entry in schedulers:
            coin -= entry[1]
            if coin < 0:
                scheduler = entry[0]
                generator.scheduler = entry[2].from_config(generator.scheduler.config)
                break
        print(f'\n\n\n'
              f'     {roll+1:>4} of {limits["total"]:>4}{"" if roll else " — R E S T A R T E D + R E I N I T I A L I Z E D"}\n'
              f'{rolls["prompt"]:>3} of {limits["prompt"] - 1:>3} — {"NEG " if negative_prompt else ""}{gen_prompt[:70] + "..." if len(gen_prompt) > 70 else gen_prompt}{" — NEW PROMPT" if rolls["prompt"] == 1 else ""}\n'
              f'{steps:>7} STEPS, {scale:>7.2f} SCALE, {eta:>7.2f} ETA, {scheduler}\n'
              f'{width:>2} × {height:>2} DIMENSIONS\n')
        latents = generator(gen_prompt, height * 64, width * 64, steps, scale, eta=eta, negative_prompt=negative_prompt, output_type='latent').images
        hires = upscaler(prompt=gen_prompt, image=latents, num_inference_steps=20, guidance_scale=0).images[0]
        hires.save(f'{outdir}/{file_prompt} — {scheduler} ST{steps} TM{int(time.time() - stopwatch)} {int(time.time()) % 10000}.jpg')

        for key in ['prompt', 'seed']:
            rolls[key] += 1

    except Exception as e:
        # print(str(e))
        # print(traceback.format_exc())
        print('F A I L E D')

    for key in ['prompt', 'seed']:
        if rolls[key] >= limits[key]:
            rolls[key] = 0

    if rolls['prompt']:
        try:
            print('Continue with prompt?')
            for i in range(5):
                print(f'{2.5 - i * 0.5} seconds left...')
                time.sleep(.25)
            print('Prompt continues')
        except KeyboardInterrupt:
            rolls['prompt'] = 0
            print('Prompt dropped')
