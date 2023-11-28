import os
import re
import traceback

from tqdm import tqdm
from PIL import Image

from secretics import root, vocab_location, subdirs
from prompts import Word


files = []
vocabulary = {}
all_prompts = {True: '', False: ''}
schedulers = {}


class Tech:
    def __init__(self, good, string=''):
        self.version = '1.5'
        self.scheduler = 'DDIM'
        self.steps = None
        self.time = None
        self.pixels = None
        self.workload = None
        self.effort = None
        self.pixel_speed = None
        self.workload_speed = None

        parts = string.split()
        for part in parts:
            if part.startswith('ST'):
                try:
                    self.steps = int(part[2:])
                except:
                    pass
            if part.startswith('TM'):
                try:
                    self.time = int(part[2:])
                except:
                    pass
            if part in ['1.4', '1.5', '2.0', '2.1']:
                self.version = part
            if not re.search(r'[0-9]', part):
                self.scheduler = part

        scheduler = schedulers.get(self.scheduler, [0, 0, self.scheduler])
        schedulers[self.scheduler] = [scheduler[0] + 1, scheduler[1] + good, self.scheduler]

    def __str__(self):
        return f'{self.version} {self.scheduler}'


class Prompt:
    def __init__(self, string, good):
        if string.endswith(','):
            string = string[:-1]
        string = ' '.join(string.replace(',', ', ').split())
        parts = string.split('(')
        if len(parts) > 1:
            self.negative_prompt = parts[1].replace(')', '')
        else:
            self.negative_prompt = '...' if 'NEG' in parts[0] else '———'
        self.prompt = ' '.join(parts[0].replace('NEG', '').split())
        all_prompts[good] += f" {self.prompt.lower().replace(',', '')} "

    def __str__(self):
        return f'{self.prompt}\n{self.negative_prompt}'


class File:
    def __init__(self, subdir, filename):
        self.filename = filename
        self.good = subdir[0] == 'G'
        self.path = f'{root}/{subdir}/{filename}'
        self.tech = None
        self.prompt = None
        clean_filename = ' '.join(filename.replace('POSTED', '').replace('SHRT', '').replace('FULL', '').replace('CPU', '').replace('.jpg', '').split())
        clean_filename = re.sub(r' [0-9a-z-,]+ ?\.\.\.', '', clean_filename)
        for part in clean_filename.split(' — '):
            if part.upper() != part:
                self.prompt = Prompt(part, self.good)
            else:
                self.tech = Tech(self.good, part)
        if self.tech is None:
            self.tech = Tech(self.good)
        # with Image.open(self.path) as im:
        #     self.tech.pixels = im.size[0] * im.size[1]
        #     if self.tech.steps is not None and self.tech.pixels is not None:
        #         self.tech.workload = self.tech.steps * self.tech.pixels
        #     if self.tech.time is not None and self.tech.pixels is not None:
        #         self.tech.pixel_speed = self.tech.pixels / self.tech.time
        #     if self.tech.time is not None and self.tech.workload is not None:
        #         self.tech.workload_speed = self.tech.workload / self.tech.time

    def __str__(self):
        return f'{self.path}\n{self.tech}\n{self.prompt}\n{"GOOD" if self.good else "REJECT"}\n'


class StatWord(Word):
    def __init__(self, string):
        super().__init__(string)
        self.clean = self.value.replace('_', '').replace('.', ' ').lower()
        self.tags = sorted(self.tags)
        self.occurences = 7
        self.good = 1

    def count(self, good, number=1):
        self.occurences += number
        if good:
            self.good += number


vocab = []
with open(vocab_location, 'r') as file:
    for line in file.readlines():
        vocab.append(StatWord(line))
print('Vocab loaded')

for subdir in subdirs:
    for filename in tqdm(os.listdir(f'{root}/{subdir}')):
        if filename.endswith('.jpg'):
            files.append(File(subdir, filename))
print('Files parsed')

# for good in [True, False]:
#     for word in tqdm(vocab):
#         for variation in [word.clean, word.multi(word.clean)]:
#             occurences = all_prompts[good].count(f' {variation} ')
#             word.count(good, occurences)
# print('Vocab occurences counted')

# vocab = sorted(vocab, key=lambda x:( x.tags, (-1)*x.good/x.occurences))
# with open(f'{vocab_location}.new', 'w') as file:
#     for entry in vocab:
#         file.write(f'{" ".join(entry.tags)}\t{entry.good/entry.occurences:.4f}\t{entry.value}\t{" ".join(entry.joiners)}\t{" ".join(entry.qualities)}\n')
#     file.write('chim\t0.0500\t?\t\tbeautychar bodyparts wearables handheld furniture vehicle action mood agesmall agebig armoredness wealth mood myst nationality burn')
# print('Vocabulary exported')

schedulers_list = sorted(list(schedulers.values()), key=lambda x: (-1)*x[1]/x[0] if x[0] else 999)
for entry in schedulers_list:
    print(f'{entry[2]} ({entry[1]/entry[0] if entry[0] else -1:.3f})')

step_ranges = {10*i: [0] for i in range(20)}
for file in files:
    if file.tech.steps is not None:
        steps = int(file.tech.steps / 10) * 10
        step_ranges[steps].append(1 if file.good else 0)
for steps, values in step_ranges.items():
    print(f'{steps}...{steps+9}: {sum(values)/len(values)} ({sum(values)}/{len(values)})')