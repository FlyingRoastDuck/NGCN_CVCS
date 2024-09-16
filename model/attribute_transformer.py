import math
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import trunc_normal_
import copy
from model.vision_transformer import vit_small, vit_base
from model.meta_graph import GraphConvolution

class vit_backbone(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        
        self.cls_token = copy.deepcopy(vit_backbone_model.cls_token)
        self.patch_embed = copy.deepcopy(vit_backbone_model.patch_embed)
        self.pos_embed = copy.deepcopy(vit_backbone_model.pos_embed)
        self.num_prompts = getattr(vit_backbone_model, 'num_prompts', None)
        if self.num_prompts:
            self.prompt_tokens = vit_backbone_model.prompt_tokens
            self.n_shallow_prompts = vit_backbone_model.n_shallow_prompts
            self.vpt_drop = vit_backbone_model.vpt_drop

        self.pos_drop = copy.deepcopy(vit_backbone_model.pos_drop)
        bottom_blocks = vit_backbone_model.blocks[:grad_from_block]
        self.bottom_blocks = copy.deepcopy(bottom_blocks)
        self.out_feat_dim = bottom_blocks[-1].norm1.weight.shape[0]
        del bottom_blocks

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if self.num_prompts:
            prompt_tokens = self.prompt_tokens[0].expand(B, -1, -1)
            prompt_tokens = self.vpt_drop[0](prompt_tokens)
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, return_all_patches=False):
        x = self.prepare_tokens(x)  # 64 197 768
        B = x.size(0)
        if self.num_prompts:
            n_vpt_layer = self.prompt_tokens.size(0)-1 
        
        for idx_layer, blk in enumerate(self.bottom_blocks):
            x = blk(x)
            if self.num_prompts and idx_layer < n_vpt_layer:
                ### exclude precedent prompts
                a = x[:, 0, :].unsqueeze(1) if self.n_shallow_prompts==0 else x[:, :1+self.n_shallow_prompts, :]
                c = x[:, self.num_prompts+1:, :]
                ### generate prompt input
                b = self.prompt_tokens[idx_layer+1, self.n_shallow_prompts:, :].expand(B, -1, -1) # corrected by i+1, origical i
                b = self.vpt_drop[idx_layer+1](b)
                x = torch.cat([a, b, c], dim=1)
        if return_all_patches:
            return x
        else:
            return x[:, 0]


class vit_branch(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        top_blocks = vit_backbone_model.blocks[grad_from_block:]
        self.top_blocks = copy.deepcopy(top_blocks)
        self.norm = copy.deepcopy(vit_backbone_model.norm)

    def forward(self, x, return_all_patches=False):
        for blk in self.top_blocks:
            x = blk(x)
        x = self.norm(x)
        if return_all_patches:
            return x
        else:
            return x[:, 0]


class AttributeTransformer(nn.Module):
    def __init__(self, vit_backbone_model, grad_from_block=11):
        super().__init__()
        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False

        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        self.attribute_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                           grad_from_block=grad_from_block)

        del vit_backbone_model
        torch.cuda.empty_cache()

    def forward(self, x):
        z = self.feature_extractor(x, return_all_patches=True)
        at_embedding = self.attribute_branch(z, return_all_patches=False)
        co_embedding = self.contrastive_branch(z, return_all_patches=False)
        if self.training:
            return co_embedding, at_embedding
        else:
            return torch.cat((co_embedding, at_embedding), dim=1)


from torch.nn import MaxPool1d, AvgPool1d
import torch.nn.functional as F
from einops import rearrange


class ChannelMaxPoolFlat(MaxPool1d):
    def forward(self, input):
        if len(input.size()) == 4:
            n, c, w, h = input.size()
            pool = lambda x: F.max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
            out = rearrange(
                pool(rearrange(input, "n c w h -> n (w h) c")),
                "n (w h) c -> n c w h",
                n=n,
                w=w,
                h=h,
            )
            return out.squeeze()
        elif len(input.size()) == 3:
            n, c, l = input.size()
            pool = lambda x: F.max_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
            out = rearrange(
                pool(rearrange(input, "n c l -> n l c")),
                "n l c -> n c l",
                n=n,
                l=l
            )
            return out.squeeze()
        else:
            raise NotImplementedError


class ChannelAvgPoolFlat(AvgPool1d):
    def forward(self, input):
        if len(input.size()) == 4:
            n, c, w, h = input.size()
            pool = lambda x: F.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
            )
            out = rearrange(
                pool(rearrange(input, "n c w h -> n (w h) c")),
                "n (w h) c -> n c w h",
                n=n,
                w=w,
                h=h,
            )
            return out.squeeze()
        elif len(input.size()) == 3:
            n, c, l = input.size()
            pool = lambda x: F.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
            )
            out = rearrange(
                pool(rearrange(input, "n c l -> n l c")),
                "n l c -> n c l",
                n=n,
                l=l
            )
            return out.squeeze()
        else:
            raise NotImplementedError


class attribute_subnet(nn.Module):
    def __init__(self, input_feature_dim, norm_type='bn'):
        super().__init__()
        self.conv_1_1 = nn.Conv1d(input_feature_dim, input_feature_dim, 1)
        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm1d(input_feature_dim)
        elif norm_type == 'ln':
            self.norm1 = nn.LayerNorm(input_feature_dim)
        elif norm_type == 'none' or norm_type is None:
            self.norm1 == nn.Identity()
        else:
            raise NotImplementedError
        self.activation = nn.GELU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.norm2 = nn.LayerNorm(input_feature_dim)

    def forward(self, x):
        out = self.flatten(self.pool(self.activation(self.norm1(self.conv_1_1(x)))))
        out = self.norm2(out)
        return out


class AttributeTransformer13(nn.Module):
    def __init__(self, vit_backbone_model, 
                 num_classes=0, grad_from_block=11):
        super().__init__()

        self.feature_extractor = vit_backbone(vit_backbone_model=vit_backbone_model,
                                              grad_from_block=grad_from_block)
        for m in self.feature_extractor.parameters():
            m.requires_grad = False
        self.contrastive_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                             grad_from_block=grad_from_block)
        backbone_feat_dim = self.feature_extractor.out_feat_dim
        self.num_features = backbone_feat_dim

        del vit_backbone_model
        torch.cuda.empty_cache()
        
    def forward(self, x):
        z = self.feature_extractor(x, return_all_patches=True)
        co_embedding = self.contrastive_branch(z, return_all_patches=True)
        co_embedding_out = co_embedding[:, 0]
        att_embedding = torch.transpose(co_embedding[:, 1:], 1, 2)

        if self.training:
            return co_embedding_out, att_embedding, co_embedding
        else:
            return co_embedding_out



class Meta_Graph1(nn.Module):
    def __init__(self, hidden_dim, device=torch.cuda.device('cuda')):
        super().__init__()


        self.device = device
        self.gcn = GraphConvolution(device=device,
                                    hidden_dim=hidden_dim,
                                    sparse_inputs=False,
                                    act=nn.Tanh(),
                                    bias=True, dropout=0.6).to(device=device)

        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, attribute_feat=None, attribute_label=None):
        if attribute_label is not None:
            x_out = []
            adj = self.create_compositional_graph(attribute_label)
            attribute_feat_tensor = torch.stack(attribute_feat, dim=1)
            for _x, _att_f, _adj in zip(x, attribute_feat_tensor, adj):
                _vertex = torch.cat((_att_f, _x.unsqueeze(0)), dim=0)
                after_vertex = self.gcn(_vertex, _adj)
                x_out.append(after_vertex[-1])
            x_out = torch.stack(x_out, dim=0)
            return x_out
        else:
            l2norm_head_embedding_list = []
            for _att_f in attribute_feat:
                l2norm_head_embedding_list.append(F.normalize(_att_f))
            a = torch.stack(l2norm_head_embedding_list, dim=1)  # 64*28*384
            l2_x = F.normalize(x)
            b = torch.unsqueeze(l2_x, dim=2)  # 64*384*1
            ab = F.softmax(torch.bmm(a, b), dim=1)  # 64*28*1
            a_t = torch.transpose(a, dim0=1, dim1=2)  # 64*384*28
            a_t_ab = torch.bmm(a_t, ab)
            a_t_ab = a_t_ab.squeeze()
            return a_t_ab

    def create_compositional_graph(self, attribute_label):
        att_num = attribute_label.size(1)
        copy_attribute_label = attribute_label.detach()
        adj_list = []
        for row in copy_attribute_label:
            adj = torch.zeros((att_num + 1, att_num + 1))
            non_zero_positions = torch.nonzero(row)
            for p in non_zero_positions:
                adj[p, att_num] = 1
                adj[att_num, p] = 1
            adj_list.append(adj)
        adj_matrix = torch.stack(adj_list, dim=0).to(device=self.device)

        return adj_matrix


class Meta_Attribute_Generator1(nn.Module):
    def __init__(self, vit_backbone_model, dict_attribute, grad_from_block=11):
        super().__init__()
        backbone_feat_dim = vit_backbone_model.num_features
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.attribute_generator_list = nn.ModuleList()

        self.meta_branch = vit_branch(vit_backbone_model=vit_backbone_model,
                                      grad_from_block=grad_from_block)

        for key in dict_attribute.keys():
            _conv = attribute_subnet(backbone_feat_dim)
            _classifier = nn.Linear(backbone_feat_dim, len(dict_attribute[key]) + 1)  # 1 for no present
            _softmax = nn.Softmax(dim=1)
            self.attribute_generator_list.append(nn.Sequential(_conv, _classifier, _softmax))

        del vit_backbone_model
        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fake_prob_list = []
        meta_embedding = self.meta_branch(x, return_all_patches=True)
        meta_embedding = torch.transpose(meta_embedding, 1, 2)
        for att_head in self.attribute_generator_list:
            fake_prob = att_head(meta_embedding)
            fake_prob_list.append(fake_prob)
        return fake_prob_list


class Meta_Attribute_Generator2(nn.Module):
    def __init__(self, vit_backbone_model, dict_attribute, grad_from_block=11):
        super().__init__()
        backbone_feat_dim = vit_backbone_model.num_features
        self.num_attribute_class = len(dict_attribute.keys())
        self.num_attribute_all = sum([len(v) for v in dict_attribute.values()])
        self.attribute_generator_list = nn.ModuleList()

        for key in dict_attribute.keys():
            _conv = attribute_subnet(backbone_feat_dim)
            _classifier = nn.Linear(backbone_feat_dim, len(dict_attribute[key]) + 1)  # 1 for no present
            _softmax = nn.Softmax(dim=1)
            self.attribute_generator_list.append(nn.Sequential(_conv, _classifier, _softmax))

        del vit_backbone_model
        torch.cuda.empty_cache()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fake_prob_list = []
        meta_embedding = torch.transpose(x, 1, 2)
        for att_head in self.attribute_generator_list:
            fake_prob = att_head(meta_embedding)
            fake_prob_list.append(fake_prob)
        return fake_prob_list


def at_small(pretrain_path):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer(student)
    return model


def at_base(pretrain_path):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer(student)
    return model



def at13_small(pretrain_path, num_attribute=28, 
               attribute_feat_channal=8, grad_from_block=11
               ):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = AttributeTransformer13(
        student, num_attribute, attribute_feat_channal, grad_from_block
    )
    return model


def at13_base(pretrain_path, num_attribute=28, attribute_feat_channal=8, 
              grad_from_block=11, pre_train=True, num_prt=0):
    student = vit_base(num_prt=num_prt)
    weight = torch.load(pretrain_path, map_location='cpu')
    if 'model' in weight:
        weight = weight['model']
    if pre_train:
        msg = student.load_state_dict(weight, strict=True)
        print(msg)
    model = AttributeTransformer13(
        student, attribute_feat_channal, 
        grad_from_block
    )
    return model 

def meta1_small(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator1(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta1_base(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator1(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta2_small(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_small()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator2(student, dict_attribute, grad_from_block=grad_from_block)
    return model


def meta2_base(pretrain_path, dict_attribute, grad_from_block=11):
    student = vit_base()
    weight = torch.load(pretrain_path, map_location='cpu')
    msg = student.load_state_dict(weight, strict=False)
    print(msg)
    model = Meta_Attribute_Generator2(student, dict_attribute, grad_from_block=grad_from_block)
    return model



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return x_proj, logits

