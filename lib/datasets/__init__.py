import torch
import torch.nn as nn
from lib.core.config import config
from  transformers import RobertaModel, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

        
def collate_fn(batch):
    texts = [b['text'] for b in batch]
    # batch_txt_mask = [b['txt_mask'] for b in batch]
    text_tokens = tokenizer.batch_encode_plus(texts, padding='longest', return_tensors='pt')
    textual_tokens = text_tokens['input_ids']
    textual_mask = text_tokens['attention_mask']


   
    visual_mask = [b['visual_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]

    max_num_clips = max([map_gt.shape[-1] for map_gt in batch_map_gt])
    padded_batch_map_gt = torch.zeros(len(batch_map_gt), 1, max_num_clips, max_num_clips)
    for i, map_gt in enumerate(batch_map_gt):
        num_clips = map_gt.shape[-1]
        padded_batch_map_gt[i][0,:num_clips,:num_clips] = map_gt

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_text_tokens': textual_tokens,
        'batch_txt_mask': textual_mask,
        'batch_visual_mask': torch.stack(visual_mask),
        
        'batch_map_gt': padded_batch_map_gt,
        'batch_vis_input': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
        'batch_duration': batch_duration,
    }

    return batch_data

def average_to_fixed_length(visual_input):
    #video_frame_len > num_sample_clips : uniformly sparse and avg
    #video_frame_len < num_sample_clips : same frame insert 
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1)) #uniformly sampling
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0)) #frame avg (start ~ end-1 )
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input 

from lib.datasets.activitynet import ActivityNet
from lib.datasets.charades import Charades
from lib.datasets.tacos import TACoS
