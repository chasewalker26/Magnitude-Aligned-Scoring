# https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py

import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class InterpretTransformer(object):
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
    # https://github.com/XianrenYty/Transition_Attention_Maps
    def transition_attention_maps(self, input, index=None, start_layer=4, steps=20, with_integral=True, first_state=False):
        input.requires_grad = True
        b = input.shape[0]
        output = self.model(input.to(self.device), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        # take mean across all attention heads, take the first row (class token values), and all columns
        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            states += states_

        total_gradients = torch.zeros(b, h, s, s).to(self.device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled.to(self.device), register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(self.device) * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients

        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state

        input.requires_grad = False

        return states[:, 0, 1:]
    
    def raw_attn(self, input, index=None):
        input.requires_grad = True
        b = input.shape[0]
        output = self.model(input.to(self.device), register_hook=True)

        attr = self.model.blocks[-1].attn.get_attention_map().mean(dim=1)
        input.requires_grad = False
    
        return attr[:, 0, 1:]