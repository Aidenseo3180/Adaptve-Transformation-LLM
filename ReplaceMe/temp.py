import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

# Use llama2-7B instead for faster computation
llama_13B_path = 'Your Path'
llama_13B = AutoModelForCausalLM.from_pretrained(llama_13B_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(llama_13B_path, trust_remote_code=True)

INTERVAL = 1
MERGE_LAYERS = 7
HIGHEST_LAY = 39
LOWEST_LAY = 0
THRESHOLD = 0.45

from copy import deepcopy
def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
   
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    
    model_copy = deepcopy(model)
    for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):      
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        )
        

        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )

        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        ) 
    
        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )
    
        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )        
                       
    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):

        del(model_copy.model.layers[diff_lay])
    return model_copy

import copy
llama_13B_copy_to_compress = copy.deepcopy(llama_13B)

import numpy as np
def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    sim_ls = []
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt')
        with torch.no_grad():
            outputs1 = model1(**encoded_inputs, output_hidden_states=True)
        hidden_states1 = outputs1.hidden_states[-1] # (1, seq_len, hidden)
        with torch.no_grad():
            outputs2 = model2(**encoded_inputs, output_hidden_states=True)
        hidden_states2 = outputs2.hidden_states[-1] # (1, seq_len, hidden)
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
    sim_ls = [i.item() for i in sim_ls]
    print(sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)

lay = HIGHEST_LAY - MERGE_LAYERS
last_merge_flag = False

sents = []
en_wiki_selected = ['Mouron () is a commune in the Arde',
 'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
 'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
 'Torreorgaz is a municipality in the',
 'Copa Libertadores 1973 was won by defending champions Independiente of A']


sents.extend(en_wiki_selected)


while lay >= LOWEST_LAY:
    print(lay)
    print('current model layer', len(llama_13B_copy_to_compress.model.layers))
    tmp_merged_model = merge_layers_return_model(llama_13B_copy_to_compress, lay, MERGE_LAYERS-1)
    sim_value = cal_last_hidden_sim(llama_13B, tmp_merged_model, tokenizer, sents)
    if sim_value > THRESHOLD:
        llama_13B_copy_to_compress = tmp_merged_model
        lay -= INTERVAL
        if lay >= len(llama_13B_copy_to_compress.model.layers):
            lay = len(llama_13B_copy_to_compress.model.layers) - 1 - MERGE_LAYERS
    else:
        lay -= 1
    

llama_13B_copy_to_compress.config.num_hidden_layers = len(llama_13B_copy_to_compress.model.layers)
llama_13B_copy_to_compress

