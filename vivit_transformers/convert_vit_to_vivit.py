import argparse
import torch
from collections import OrderedDict
from vivit_transformer import ViViTConfig, ViViTModel
from transformers import ViTModel
from pathlib import Path
import collections.abc

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


@torch.no_grad()
def conv_uniform_frame_sampling(weights, num_frames):
    shape = weights.shape
    new_weights = torch.zeros((shape[0], shape[1], num_frames, shape[2], shape[3]))

    new_weights[:, :, (num_frames//2), :, :] = weights
    
    return new_weights

@torch.no_grad()
def transform_pos_embeddings(weights, num_frames, video_len):
    # calculate dim of the new pos embeddings
    n = (video_len // num_frames) * (weights.shape[1] - 1)
    
    #extract cls_token
    cls_token_weights = weights[:, 0]
    cls_token_weights = cls_token_weights.unsqueeze(1)
    
    pos_tokens_weights = weights[:, 1:]
    
    # number of times the dim should be repeated
    repeat_n = n//pos_tokens_weights.shape[1]
    
    new_weights = pos_tokens_weights.repeat(1, repeat_n, 1)
    
    return torch.cat((cls_token_weights, new_weights), 1)

def transform_state_dict(old_state_dict, tubelet_n, video_length):
    new_state_dict = OrderedDict()
    
    for key, value in old_state_dict.items():
        if 'vit' in key:
            key = key.replace('vit', 'vivit')

        new_state_dict[key] = value
        
        if 'embeddings.patch_embeddings.projection.weight' in key:
            new_state_dict[key] = conv_uniform_frame_sampling(value, tubelet_n)
            
        if 'embeddings.position_embeddings' in key:
            new_state_dict[key] = transform_pos_embeddings(value, tubelet_n, video_length)
    
    return new_state_dict


def convert_vit(vit_path, output_path, tubelet_n, video_length):
    vit_model = ViTModel.from_pretrained(vit_path)
    
    
    vit_state_dict = vit_model.state_dict()
    new_state_dict = transform_state_dict(vit_state_dict, tubelet_n, video_length)
    
    video_size = (video_length, ) + to_2tuple(vit_model.config.image_size)
    tubelet_size = (tubelet_n, ) + to_2tuple(vit_model.config.patch_size)
    
    vivit_config = ViViTConfig(
        video_size=video_size, 
        tubelet_size=tubelet_size,
        hidden_size=vit_model.config.hidden_size,
        num_hidden_layers=vit_model.config.num_hidden_layers,
        num_attention_heads=vit_model.config.num_attention_heads,
        intermediate_size=vit_model.config.intermediate_size,
        hidden_act=vit_model.config.hidden_act,
        hidden_dropout_prob=vit_model.config.hidden_dropout_prob,
        attention_probs_dropout_prob=vit_model.config.attention_probs_dropout_prob,
        initializer_range=vit_model.config.initializer_range,
        layer_norm_eps=vit_model.config.layer_norm_eps,
        is_encoder_decoder=vit_model.config.is_encoder_decoder,
        num_channels=vit_model.config.num_channels,
        qkv_bias=vit_model.config.qkv_bias
    )

    vivit_model = ViViTModel(vivit_config)
    vivit_model.load_state_dict(new_state_dict)
    
    Path(output_path).mkdir(exist_ok=True)
    
    vivit_model.save_pretrained(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ViT-based model weights into ViViT model weights')
    
    parser.add_argument(
        '--vit_model_path',
        type=str,
        help='Path to the ViT model'
    )
    
    parser.add_argument(
        '--tubelet_n',
        default=2,
        type=int,
        help='Number of frames used in the tubelet embedder (e.g. N x 16 x 16)'
    )
    
    parser.add_argument(
        '--video_length',
        default=10,
        type=int,
        help='Number of frames in the input videos'
    )
    
    parser.add_argument(
        '--output_path',
        type=str
    )
    
    args = parser.parse_args()
    
    convert_vit(args.vit_model_path, args.output_path, args.tubelet_n, args. video_length)
    
    
    
    