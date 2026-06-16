import copy
import torch
from torch import nn
from copy import deepcopy
import timm
from lora import NullSpaceViT
from models.basic_lora import PlainLoRAViT
from models.sgp_lora import SGPLoRAViT, SGPLoRACLIPVisionTransformer

def get_vit(args, pretrained=False):
    name = args['vit_type']
    name = name.lower()
    rank = args['lora_rank']

    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained, num_classes=0)

    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=pretrained, num_classes=0)
    
    elif name == 'vit-b-p16-clip':
        vit = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0)

    else:
        raise ValueError(f'Model {name} not supported')
    
    vit.head = nn.Identity()
    vit.norm = nn.LayerNorm(768, elementwise_affine=False)

    lora_type = args['lora_type']

    if lora_type == "full" or lora_type == "joint_full":
        from models.full_finetune import FullFinetuneViT
        return FullFinetuneViT(vit)
    
    elif lora_type == "full_nsp":
        from models.full_finetune import NullSpaceViT
        return NullSpaceViT(vit, use_projection=True)
    
    elif lora_type == "basic_lora" or lora_type == "joint_lora":
        return PlainLoRAViT(vit, r=rank, include_norm=False)

    elif lora_type == "sgp_lora":
        return SGPLoRAViT(vit, r=rank, weight_temp=args['weight_temp'], use_soft_projection=True, weight_kind=args['weight_kind'], weight_p=args['weight_p'])
    
    elif lora_type == "nsp_lora":
        return SGPLoRAViT(vit, r=rank, weight_temp=args['weight_temp'], use_soft_projection=False, nsp_eps=args['nsp_eps'], nsp_weight=args['nsp_weight'])

    else:
        raise ValueError(f"LoRA type {lora_type} not supported")


class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes, bias=False)])
        self.head_weights = nn.Parameter(torch.ones(nb_classes))
        self.current_output_size = nb_classes

    def update(self, nb_classes):
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=False)
        self.heads.append(new_head)
        new_head_weights = nn.Parameter(torch.ones(self.current_output_size + nb_classes))

        with torch.no_grad():
            new_head_weights[:self.current_output_size] = self.head_weights
            new_head_weights[self.current_output_size:] = 1.0
        
        self.head_weights = new_head_weights
        self.current_output_size += nb_classes

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        combined = torch.cat(outputs, dim=1)
        return combined * self.head_weights


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.vit = get_vit(args, pretrained)
        self.fc = None

    def extract_vector(self, x):
        return self.vit(x)

    def forward(self, x):
        feat = self.vit(x)
        logits = self.fc(feat)
        return feat, logits
    
    def forward_features(self, x):
        return self.vit(x)
    
    def update_projection_matrices(self, covariances):
        if hasattr(self.vit, 'update_projection_matrices') and callable(getattr(self.vit, 'update_projection_matrices', None)):
            self.vit.update_projection_matrices(covariances)
    
    @property
    def feature_dim(self):
        return self.vit.feature_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = ContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes)

    def copy(self):
        return copy.deepcopy(self)


class RandomNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.vit = get_vit(args, pretrained)
        if args['random_projection_dim'] <= 0:
            self.random_projector = nn.Identity()
            self.random_projection_dim = self.vit.feature_dim

        elif args['random_projection_dim'] > 0:
            self.random_projection_dim = args['random_projection_dim']
            
            self.random_projector = nn.Sequential(
                nn.Linear(getattr(self.vit, 'feature_dim', 768), self.random_projection_dim, bias=False),
                nn.ReLU())
            
            # 对线性层进行高斯正态分布初始化
            for module in self.random_projector:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=1.0)
        
        self.fc = None

    def extract_vector(self, x):
        return self.vit(x)

    def forward(self, x):
        return self.forward_features(x)
    
    def forward_features(self, x, random_projection=True):
        feat = self.vit(x)
        if random_projection:
            feat = self.random_projector(feat)
        return feat
    
    @property
    def feature_dim(self):
        return self.random_projection_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = ContinualLinear(self.vit.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes)
    