import clip 
import json
import torch
import pprint
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from models.clip_context_prompts import select_template

TEMPLATES = [
    'cupl', # CuPL
    'vcd',  # VCD
    'hiec', # HieC (clustering)
    'hiet', # HieT (tree)
    'ours-LP', # ours, leaf-level peers
    'ours-AP', # ours, ancestor-level peers
    'ours-path', # ours, path-based Generic prompts (G)
    'ours-comp', # ours, comparative prompts: LP + AP
    'ours-full', # ours, LP + AP + G
    'merge', # merge multiple image prompts
    
    # the above image prompts are all generated with ChatGPT (gpt-3.5-turbo-0125)
    'ours-full-claude', # ours, LP + AP + G generated by Claude-3.5-sonnet
    'ours-full-gemini', # ours, LP + AP + G generated by Gemini-1.5-flash
    'clip', # CLIP ensemble and its CRM variant
]


def get_index(name_tag):
    return name_tag.split('@')[1]


def get_name(name_tag):
    return name_tag.replace('_', ' ').lower().split('@')[0]


def name_preprocessing(classnames):
    classnames = [get_name(name) for name in classnames]
    return  classnames


def flat_ensemble_embedding(embedding_rep, classnames):
    """
        classnames, from ImageFolder to provide proper order of classes
        embedding_rep, a dictionary of ndarrays indexed by classname
        aggregate potentially multiple raw class embeddigns to one ensemble
    """
    classnames = name_preprocessing(classnames)

    with torch.no_grad():
        zeroshot_weights = []
        for name in tqdm(classnames):
            class_embeddings = torch.from_numpy(embedding_rep[name]).cuda()
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            embedding = class_embeddings.mean(dim=0, keepdim=True)
            embedding /= embedding.norm()
            zeroshot_weights.append(embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        zeroshot_weights = torch.squeeze(zeroshot_weights, dim=0)

    return zeroshot_weights


def vcd_ensemble_embedding(embedding_rep, classnames):
    """
        classnames, from ImageFolder to provide proper order of classes
        embedding_rep, a dictionary of ndarrays (embedding of vcd descriptors) indexed by classname
        L2 normalize each vcd descriptor, no average of vcd descriptor
    """
    classnames = name_preprocessing(classnames)

    with torch.no_grad():
        # ordered dict itself is not on cuda, but its values are on cuda
        zeroshot_weights = OrderedDict()
        for name in tqdm(classnames):
            class_embeddings = torch.from_numpy(embedding_rep[name]).cuda()
            # L2 norm
            class_embeddings /= class_embeddings.norm(dim=-1, p=2, keepdim=True)
            zeroshot_weights[name] = class_embeddings

    return zeroshot_weights


def hiecomp_ensemble_embedding(embedding_rep, classnames='ignore'):
    """
        embedding_rep: raw embeddings derived from HieComp prompts with clip.tokenizer
    """
    with torch.no_grad():
        zeroshot_weights = OrderedDict()
        for k, v in tqdm(embedding_rep.items(), total=len(embedding_rep)):
            for x in v:
                if k not in zeroshot_weights: zeroshot_weights[k] = []
                class_embeddings = torch.from_numpy(x).cuda()
                class_embeddings /= class_embeddings.norm(dim=-1, p=2, keepdim=True)
                zeroshot_weights[k].append(class_embeddings)
    return zeroshot_weights


def zeroshot_classifier(embedding_rep, classnames, tree, opts):
    if opts.inference == 'flat':
        if opts.prompt not in ['hiec', 'hiet']:
            return flat_ensemble_embedding(embedding_rep, classnames)
        else:
            flatten_embedding_rep = dict()
            for k, v in embedding_rep.items():
                ls = [x for x in v] if len(v)>1 else v
                flatten_embedding_rep[k] = np.concatenate(ls, axis=0)
            return flat_ensemble_embedding(flatten_embedding_rep, classnames)
    elif opts.inference == 'vcd':
        return vcd_ensemble_embedding(embedding_rep, classnames)
    elif opts.inference == 'hie':
        return hiecomp_ensemble_embedding(embedding_rep, classnames)
    else:
        raise NotImplementedError(f'opts.embedding {opts.embedding} not supported')


def load_img_prompts(opts):
    img_prompts_path  = opts.data_paths[f'{opts.dataset}-{opts.prompt}-prompts']
    with open(img_prompts_path, 'r') as f:
        img_prompts = json.load(f)
    return img_prompts


def class_textual_representations(classnames, opts):
    """
        convert class images load from ImageFolder to 
        natural language descriptions of the class, i.e.,
        the textual representations of the class.

        one class may have multiple textual representations.
    """
    # preprocess special symbols in class names
    # 'Class_Name' -> 'class name'
    # 'Class_Name@n123456' -> 'class name'
    classnames = name_preprocessing(classnames)

    class_textual_representation = dict()


    # ensemble
    assert opts.prompt in TEMPLATES, \
        f'''opts.prompt: {opts.prompt} not 
            supported when opts.embedding = flat'''
    
    
    if opts.prompt != 'clip': # 
        class_textual_representation = load_img_prompts(opts)
    else: # clip ensemble
        templates = select_template(opts)
        for name in tqdm(classnames):
            # insert additional info if needed
            texts = [tmp.format(name) for tmp in templates]
            class_textual_representation[name] = texts
    
    return class_textual_representation


def merge_template_names(template_names):
    ls = template_names
    tp = []
    for n in ls:
        if '.' in n: tp.append(n.split('.')[0][:-2])
        else: tp.append(n)
    name = "-".join(sorted(tp))
    return name


class Dummy(object):
    def __init__(self, dataset, prompt, data_paths, partition):
        self.dataset = dataset
        self.prompt = prompt
        self.data_paths = data_paths
        self.partition = partition


def merging_textual_representations(template_names, classnames, opts):
    print(f'merging the following prompt templates:')
    pprint.pprint(template_names)
    classnames = name_preprocessing(classnames)
    textual_rep = {n:[] for n in classnames}
    for tname in template_names: 
        dummy_opts = Dummy(
            opts.dataset, tname, 
            opts.data_paths, opts.partition, 
        )
        # get target image prompts
        rep = class_textual_representations(classnames, dummy_opts)
        # merging/appending image prompts
        for n in classnames:
            textual_rep[n] = textual_rep[n] + rep[n]
        
    return textual_rep


def textual_to_raw_embedding(clip_model, textual_rep, opts):
    """
        generic conversion of textual representation to embeddings
        each textual representation, i.e., nlp text description gets
        converted to its raw text-embeddings without:
        1) L2 normalization
        2) ensemble of text embeddings
    """
    embedding_rep = dict()
    with torch.no_grad():
        for name, text_list in textual_rep.items():

            if opts.prompt not in ['hiec', 'hiet']:

                texts = clip.tokenize(texts=text_list, context_length=77, truncate=True).cuda()
                class_embeddings = clip_model.encode_text(texts) # on cuda if available
                embedding_rep[name] = class_embeddings.detach().cpu().numpy()

            else: # HIE textual_rep: {name: [[text-rep-level-1,..., [text-rep-level-x]]}
                
                for text_sublist in text_list:

                    texts = clip.tokenize(texts=text_sublist, context_length=77, truncate=True).cuda()
                    class_embeddings = clip_model.encode_text(texts) # on cuda if available
                    if name not in embedding_rep:
                        embedding_rep[name] = [class_embeddings.detach().cpu().numpy()]
                    else:
                        embedding_rep[name].append(class_embeddings.detach().cpu().numpy())

    return embedding_rep    

# EOF