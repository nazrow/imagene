import random
import re
import traceback

from typing import List, Optional, Union

from secretics import excluded_words, excluded_constructions, excluded_additions, vocab_location


class Word:
    def __init__(self, string):
        parts = string.split('\t')
        self.tags = parts[0].split()
        self.weight = float(parts[1]) if len(parts[1]) else 1
        self.value = parts[2]
        self.joiners = parts[3].split()
        self.qualities = parts[4].split()

    def multi(self, override=None):
        if override:
            target = override
        else:
            target = self.value
        if target.endswith('s') and not target.endswith('ss'):
            return target
        if target.endswith('man'):
            return f'{target[:-3]}men'
        elif target.endswith('y') and not target.endswith('oy'):
            return f'{target[:-1]}ies'
        elif target.endswith('s') or target.endswith('ch') or target.endswith('o'):
            return f'{target}es'
        elif target.endswith('ife'):
            return f'{target[:-3]}ives'
        elif target.endswith('ff'):
            return f'{target[:-2]}ves'
        elif target.endswith('lf'):
            return f'{target[:-2]}lves'
        elif target.endswith('sh'):
            return f'{target}es'
        elif target == 'mouse':
            return 'mice'
        else:
            return f'{target}s'


vocab = []
with open(vocab_location, 'r') as file:
    for line in file.readlines():
        vocab.append(Word(line))


def get_from_vocab(tags: Union[str, List[str]]) -> Optional[Word]:
    if not len(tags):
        return None
    if isinstance(tags, str):
        tags = [tags]
    scope = [word for word in vocab if any(tag in word.tags for tag in tags)]
    random.shuffle(scope)
    limit = 0
    for word in scope:
        limit += word.weight
    coin = random.uniform(0, limit)
    for word in scope:
        coin -= word.weight
        if coin < 0:
            return word
    return scope[-1] if len(scope) else None


def get_from_probability_array(config: List[float]) -> int:
    if len(config) == 1:
        return 0
    else:
        coin = random.uniform(0, sum(config))
        for index in range(len(config)):
            coin -= config[index]
            if coin <= 0:
                return index


def article(word):
    if word[0].isupper():
        return ''
    elif word[0] in 'aioue':
        return 'an'
    else:
        return 'a'


def chimera():
    parts = [get_from_vocab(['animal', 'char']) for _ in range(2)]
    return random.choice([
        f'half {parts[0].value}, half {parts[1].value}',
        f'{random.choice(["beast", "creature", "monster", "chimera"])} with the head of {article(parts[0].value)} {parts[0].value} and the body of {article(parts[1].value)} {parts[1].value}',
        f'{parts[0].value} with the head of {article(parts[1].value)} {parts[1].value}'
    ])


def clean(string):
    return ' '.join(string.replace('_', ' ').replace('.', ' ').replace(' ,', ',').strip().split())


non_adjective_tags = [
    'animal', 'arch', 'bodyparts', 'char', 'content', 'furniture', 'handheld', 'nature', 'scene', 'thing', 'vehicle', 'wearables', 'weather'
]


def qualify(target, scope=[], limit=0, qualities=[], multiple=False, adjective=False):
    if not qualities:
        random.shuffle(scope)
        qualities = list(set(get_from_vocab(tag) for tag in scope[:limit]))
    qualities = [item for item in qualities if item is not None]
    if len(qualities):
        joiners = {}
        for quality in qualities:
            quality_value = qualify(quality.value, scope=quality.qualities, limit=random.randint(0, 3), multiple=quality.value == quality.multi(), adjective=not any(tag in non_adjective_tags for tag in quality.tags))
            quality_joiner = random.choice(quality.joiners) if len(quality.joiners) else ''
            if quality_value.startswith('_') and not quality_joiner:
                quality_joiner = '_'
                quality_value = quality_value[1:]
            joiner_list = joiners.get(quality_joiner, [])
            joiner_list.append(quality_value)
            joiners[quality_joiner] = joiner_list
        pre_post = {True: [], False: []}
        for joiner, joiner_list in joiners.items():
            string = f'{joiner + " " if joiner else ""}{" and ".join(joiner_list) if len(joiner) > 1 else " ".join(joiner_list)}'
            pre_post[string.startswith('_')].append(string)
        result = clean(f'{" ".join(pre_post[False])} {target} {" ".join(pre_post[True])}')
    else:
        result = target
    if multiple or result.lower() != result or result.split()[0] in ['a', 'an'] or result.startswith('_') or adjective:
        pass
    else:
        result = f'{article(result)} {result}'
    return result


subject_qualities = {
    'some': [0, 2, 1, .5, .1],
    'char': [.5, 3, 1.5, .5, .1],
    'chim': [1, 1, .1],
    'scene': [1, 1, .5, .2],
    'arch': [2, 2, .5, .1],
    'thing': [1, 2, .5, .3, .1]
}


def subject_gen(theme):
    multiple_limits = {
        'some': .9,
        'char': .9,
        'chim': .99,
        'scene': 1,
        'arch': .95,
        'thing': .9
    }
    subject = get_from_vocab(theme)
    if subject.value.lower() != subject.value:
        multiple = False
    elif subject.value == subject.multi():
        multiple = True
    else:
        multiple = random.random() > multiple_limits[theme]
    if multiple and random.random() > 0.4:
        multiplier = get_from_vocab('multiplier')
    else:
        multiplier = None
    if subject.value.startswith('?'):
        subject_value = chimera()
    elif multiple:
        subject_value = subject.multi()
    else:
        subject_value = subject.value
    qualities_num = get_from_probability_array(subject_qualities[theme])
    subject_value = qualify(subject_value, scope=subject.qualities, limit=qualities_num, multiple=multiple)
    if multiplier:
        if multiplier.value.startswith('_'):
            subject_value = f'{subject_value} {multiplier.value}'
        else:
            subject_value = f'{multiplier.value} {subject_value}'
    return subject_value


def scene_gen():
    scene = get_from_vocab('scene')
    qualities_num = get_from_probability_array(subject_qualities['scene'])
    scene_value = qualify(scene.value, scope=scene.qualities, limit=qualities_num)
    if random.random() > 0.6:
        cond_tags = ['lightingscene']
        if 'weather' not in scene.tags:
            cond_tags.append('weather')
        scene_value = qualify(scene_value, scope=cond_tags, limit=random.randint(0, 3))
    return [Word(f'\t\t{scene_value}\t{random.choice(scene.joiners) if len(scene.joiners) else ""}\t')]


def content_gen():
    result = []
    if random.random() > 0.4:
        content_type = get_from_vocab('content')
        result.append(f'{qualify(content_type.value, scope=content_type.qualities, limit=random.randint(0, 2))} of')
    if random.random() > 0.85:
        author_tags = ['author']
        try:
            author_tags.append(f'author{content_type.value}')
        except:
            pass
        author = get_from_vocab(author_tags)
        result.append(f'_, by {author.value}')
    if random.random() > 0.85:
        result.append(f'_, {get_from_vocab("filter").value}')
    if random.random() > 0.6:
        result.append(f'_, {get_from_vocab("style").value}')
    if random.random() > 0.15:
        result.append(f'_, trending on {get_from_vocab("platform").value}')
    return [Word(f'\t\t{item}\t\t') for item in list(set(result))]


def prompt_gen():
    subjects_num = {
        'some': [0, 1, .05],
        'char': [0, 1, .1],
        'chim': [0, 1, .03],
        'scene': [0, 1],
        'arch': [0, 1, .05],
        'thing': [0, 1, .1]
    }
    scene_limits = {
        'some': .8,
        'char': .75,
        'chim': .95,
        'scene': 1,
        'arch': .95,
        'thing': .85
    }
    theme = get_from_vocab('theme').value
    result = ' and '.join([subject_gen(theme) for _ in range(get_from_probability_array(subjects_num[theme]))])
    if random.random() > scene_limits[theme]:
        result = qualify(result, qualities=scene_gen())
    if random.random() > 0.1:
        result = qualify(result, qualities=content_gen())
    return clean(result)


def get_prompt():
    with open('C:/Users/nazro/YandexDisk/StableDiffusion/Txts/Prompts.txt', 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        file.truncate()
        result = None
        new_lines = []
        if len(lines):
            for line in lines:
                if len(line):
                    if line.startswith('-'):
                        new_lines.append(line)
                    else:
                        if result:
                            new_lines.append(line)
                        else:
                            result = line
                            new_lines.append(f'-{line}')
            file.write('\n'.join([line.strip() for line in new_lines if len(line.strip())]))
        if result:
            return f'+{result.strip()}'
    return prompt_gen()


def clean_filename(src):
    prompt = None
    for piece in src.replace('.jpg', '').split('—'):
        if piece.upper() != piece:
            prompt = piece.replace('NEG', '').split('(')[0]
    prompt = ', '.join([piece for piece in prompt.split(',') if not any(word in piece for word in excluded_constructions)])
    for word in excluded_words:
        prompt = prompt.replace(word, '')
    for word in excluded_additions:
        finds = re.findall(rf'(with|and) ([a-z\-]* ?)?({word})( and)?', prompt)
        for find in finds:
            _find = ' '.join(find)
            _find = ' '.join(_find.split())
            prompt = prompt.replace(_find, find[0] if find[-1] == ' and' else '')
    prompt = f'_{" ".join(prompt.split()).replace(" ,", ",").strip()}_'
    return prompt


if __name__ == "__main__":
    for i in range(100):
        print(prompt_gen())
