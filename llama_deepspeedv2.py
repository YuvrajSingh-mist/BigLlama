#Based on Llama from Meta (https://github.com/meta-llama/llama/blob/main/llama/model.py) 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass
from tokenizers import Tokenizer
from pathlib import Path
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm


# Load model directly
from transformers import AutoTokenizer

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

import deepspeed  # <-- NEW IMPORT
import torch
import wandb
from deepspeed.accelerator import get_accelerator  # <-- NEW
from liger_kernel.transformers import LigerRMSNorm
from liger_kernel.transformers import LigerSwiGLUMLP
from liger_kernel.transformers import liger_rotary_pos_emb
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from termcolor import colored


# import wandb
# wandb.login()


# from torch.utils.tensorboard import SummaryWriter


from datasets import load_dataset, concatenate_datasets

# from google.colab import userdata
HF_TOKEN = '...'

tinystories = False
fw = True
fw_train = None
fw_test = None
if(tinystories):
    fw_train = load_dataset("roneneldan/TinyStories", split="train")
    fw_test = load_dataset("roneneldan/TinyStories", split="validation")
    print(fw_train)
    print(fw_test)
if(fw):   
    fw_train = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=False)
    fw_train = fw_train.train_test_split(test_size=0.01)
    print(fw_train)
    print(fw_train)

def setup(rank=None, world_size=None):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl")
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
def cleanup():
    destroy_process_group()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", hf_token = HF_TOKEN)

# tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.pad_token is None:
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

@dataclass
class ModelArgs:
    #Hyperparameters
    
    epochs = 4
    block_size = 1024
    batch_size = 4
    embeddings_dims = 1024
    attn_dropout = 0.1
    no_of_heads = 8
    dropout = 0.1
    # epochs = 100
    val_epochs = 2
    max_lr = 4e-4
    no_of_decoder_layers = 12 #IMP needs to be thoroughly calculated
    weight_decay_optim = 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    clip = 1.0
    # device = 'cuda'
    no_kv_heads = 4
    vocab_size = len(tokenizer)  #powers of 2 so nice!
    eps = 1e-8
    dtype = 'float16'  # Force float16 precision
    mqa_heads = 2
    use_flash_attn = False #Not working
    use_compile = True
    use_liger = True
    checkpoint_dir = './'
    rms_norm_eps: float  = 1e-6

def dataloader_to_step(data_loader, target_step):
    """
    Advance the data loader to a specific step by consuming batches.
    This is used when resuming training from a checkpoint.
    """
    print(f"Advancing dataloader to step {target_step}...")
    for step in range(target_step):
        try:
            next(iter(data_loader))
        except StopIteration:
            # If we reach the end, break and let training continue normally
            break
    print(f"Dataloader advanced to step {target_step}")


def load_deepspeed_checkpoint(model_engine, load_dir, ckpt_id):
    """
    Load DeepSpeed checkpoint using the official API.
    Returns the step from client state.
    """
    try:
        print(f"Loading checkpoint from {load_dir}, checkpoint ID: {ckpt_id}")
        _, client_sd = model_engine.load_checkpoint(load_dir, ckpt_id)
        step = client_sd.get('step', 0)
        print(f"Successfully loaded checkpoint. Resuming from step: {step}")
        return step, client_sd
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 0, {'step': 0}


def save_deepspeed_checkpoint(model_engine, save_dir, step, loss_value):
    """
    Save DeepSpeed checkpoint using the official API.
    All processes must call this method, not just rank 0.
    """
    client_sd = {'step': step}
    ckpt_id = str(step)  # Use step as string tag
    
    print(f"Saving checkpoint at step {step} to {save_dir}")
    model_engine.save_checkpoint(save_dir, ckpt_id, client_state=client_sd)
    print(f"Checkpoint saved successfully at step {step}")




def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        max_length=ModelArgs.block_size,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )




def prepare_dataset(split, device, batch_size):
    print("Device is: ", device)
 
    def collate_fn(batch):
        # Extract text data
        texts = [item ["text"] for item in batch]

        input_encodings = tokenizer(texts, max_length = ModelArgs.block_size, padding='max_length', truncation=True, return_tensors="pt")
        
        input_encodings["labels"] = input_encodings["input_ids"].clone()  # Use `input_ids` as labels
        
        input_encodings["labels"][:, :-1] = input_encodings["input_ids"][:, 1:]  # Shift right
        input_encodings["labels"][:, -1] = tokenizer.eos_token_id  # Let the last token be end 
       
        return input_encodings

  
    dataloader = None
    if(tinystories):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train,
            # generator=generator,
            batch_size=batch_size,
             
            sampler=DistributedSampler(fw_train, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            # shuffle=False
        )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_test,
              
            
            batch_size=batch_size,
            sampler=DistributedSampler(fw_test, shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            # shuffle=False
        )
    elif(fw):
        if(split == 'train'):
            data_loader = DataLoader(
            fw_train['train'],
            batch_size=batch_size,
            
            
            sampler=DistributedSampler(fw_train['train'], shuffle=True),
            collate_fn=collate_fn,
            drop_last=True,
            # shuffle=False
    )
        elif(split == 'val'):
            data_loader = DataLoader(
            fw_train['test'],
            batch_size=batch_size,
                # generator=generator,
            sampler=DistributedSampler(fw_train["test"]),
            collate_fn=collate_fn,
              
            drop_last=True,
            # shuffle=False
        )
    return data_loader




    

class RMSNormalization(nn.Module):
    def __init__(
        self,
        embeddings_dims = ModelArgs.embeddings_dims
    ):
        super().__init__()
        if(ModelArgs.use_liger == False):
            self.norm = nn.RMSNorm(embeddings_dims, eps=ModelArgs.rms_norm_eps)
        else:
            self.norm = LigerRMSNorm(embeddings_dims, eps=ModelArgs.rms_norm_eps)

    def forward(self, x):

        return self.norm(x)


# import numpy as np
class RotaryEmbeddings(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        batch_size: int = ModelArgs.batch_size
    ):
        super().__init__()

        self.embeddings_dims = embeddings_dims
        self.block_size = block_size
        self.batch_size = batch_size
        self.theta = 0
        self.device=device

    
    def apply_rope(self, seq, q=None, k=None):
        batch_size, seq_len, embeds_dims = seq.shape
        device = seq.device  # Get the device from the input tensor

        # if(ModelArgs.use_liger):
        #   token_idx = torch.arange(0, seq_len, device=device).unsqueeze(1)
        #   positions = torch.arange(0, embeds_dims, device=device).unsqueeze(0)
        #   # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        #   theta = 10000 ** (-2 * (positions) / embeds_dims)
        #   angles = token_idx * theta
        #   angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved

        #   cos = torch.cos(angles).to(device)
        #   sin = torch.sin(angles).to(device)
        #   cos = cos.unsqueeze(0)
        #   sin = sin.unsqueeze(0)
        #   # print(cos.shape)
        #   # print(sin.shape)
        #   out = liger_rotary_pos_emb(q, k, cos, sin)

        # else:

        # print(seq.shape)
        # print(self.embeddings_dims)
        # self.matrix = torch.zeros((seq_len, self.embeddings_dims, self.embeddings_dims), dtype=torch.float32,  requires_grad=False,  device = self.device)
        device = seq.device  # Get the device from the input tensor
        token_idx = torch.arange(0, seq_len, device=device).unsqueeze(1)
        positions = torch.arange(0, embeds_dims, 2, device=device).unsqueeze(0)
        # dims = torch.arange(1, self.embeddings_dims // 2,  dtype=torch.float32)
        theta = 10000 ** (-2 * (positions) / embeds_dims)
        angles = token_idx * theta
        angles = angles.expand(seq_len, -1) # because this thing needs to be applied to every sequence in the batch but with embeds dims halved
        x_reshaped = seq.view(batch_size, seq_len, embeds_dims // 2, 2)
        
        cos_angles = torch.cos(angles).to(device)
        sin_angles = torch.sin(angles).to(device)
        # print(cos_angles.shape)
        # print(sin_angles.shape)
        # print(x_reshaped.shape)
        # indices = torch.arange(self.embeddings_dims,  dtype=torch.int64,  device = self.device)

        out = torch.stack([
            x_reshaped[..., 0]*cos_angles - (x_reshaped[...,1] * sin_angles), 
            x_reshaped[...,1] * cos_angles + x_reshaped[..., 0] * sin_angles
        ], dim=-1).to(device)
        out = out.view(batch_size, seq_len, embeds_dims)
        return out

    def forward(self, x, q=None, k=None):
        # print("X shape: ", x.shape)
        # print("X is: ", x)
        # B,T,C = x.shape
        # print("MATRIX:",x)
        # if(x > self.block_size or x < self.block_size):
        #     matrix = self.init_matrix(x)
        #     return matrix
        # else:
        #     matrix = self.init_matrix(self.block_size)

        #     return matrix
        # if(ModelArgs.inference):
        res = self.apply_rope(x, q, k)
        return res 
        # else:
            # return self.x_re
    



class MQA(nn.Module):
    def __init__(
        self,
        device,
        no_of_q_heads: int,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
    ):
        super().__init__()

        self.no_of_heads = ModelArgs.no_of_heads
        self.no_of_kv_heads = 2  # I want to have a kv for each pair of query heads
        self.head_size = embeddings_dims // no_of_q_heads
        self.device = device
        
        # Create different initialization paths based on whether flash attention is used
        if ModelArgs.use_flash_attn:
            # Combined linear projections for K, V
            self.kv_proj = nn.Linear(embeddings_dims, 2 * self.head_size, bias=False, device=device)
            self.rotary = RotaryEmbeddings(embeddings_dims=embeddings_dims, device=device)
        else:
            # Separate projections for K, V
            self.key = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, bias=False, device=device)
            self.value = nn.Linear(in_features=embeddings_dims, out_features=self.head_size, bias=False, device=device)
            self.rotary = RotaryEmbeddings(embeddings_dims=self.head_size, device=device)
            
        # Multi-query projections
        self.multi_query = nn.ModuleList([
            nn.Linear(in_features=embeddings_dims, out_features=self.head_size, bias=False, device=device) 
            for _ in range(self.no_of_kv_heads)
        ])
        
        # Output projection
        self.linear_layer = nn.Linear(
            in_features=self.head_size * self.no_of_kv_heads, 
            out_features=embeddings_dims, 
            bias=False, 
            device=device
        )
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(p=ModelArgs.attn_dropout)
        
        # Specific setup for Liger
        if ModelArgs.use_liger:
            self.rope = RotaryEmbeddings(embeddings_dims=embeddings_dims, device=device)

    def scaled_dot_product(self, q, k, v, block_size):
        device = q.device
        
        # Apply rotary embeddings
        q = self.rotary(q)
        k = self.rotary(k)
        
        # Create causal mask
        masked_table = torch.tril(torch.ones((block_size, block_size), requires_grad=False, device=device))
        
        # Compute attention scores
        weights = q @ torch.transpose(k, dim0=-2, dim1=-1) * (k.shape[-1] ** -0.5)
        masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
        
        # Apply softmax and dropout
        weights_normalized = nn.functional.softmax(masked_values, dim=-1)
        weights_normalized = self.dropout(weights_normalized)
        
        # Apply attention to values
        out = weights_normalized @ v
        return out

    def flash_attention(self, x, q, k, v, batch_size, block_size):
        # Reshape for multi-head attention
        q = q.view(batch_size, block_size, self.no_of_heads, self.head_size // ModelArgs.no_kv_heads).transpose(1, 2)
        k = k.view(batch_size, block_size, self.no_of_heads, self.head_size // ModelArgs.no_kv_heads).transpose(1, 2)
        v = v.view(batch_size, block_size, self.no_of_heads, self.head_size // ModelArgs.no_kv_heads).transpose(1, 2)
        
        # Apply rotary embeddings if needed
        q, k = self.rotary(x, q, k)
        
        # Use PyTorch's efficient scaled dot-product attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=ModelArgs.dropout, is_causal=True
        )
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, block_size, -1)
        return out

    def forward(self, x):
        batch_size, block_size, embeddings_dims = x.shape
        device = x.device

        if ModelArgs.use_liger:
            # Liger path
            # Prepare position embeddings for rotary
            token_idx = torch.arange(0, block_size, device=device).unsqueeze(1)
            positions = torch.arange(0, embeddings_dims, device=device).unsqueeze(0)
            theta = 10000 ** (-2 * (positions) / embeddings_dims)
            angles = token_idx * theta
            angles = angles.expand(block_size, -1)

            # Create cos/sin matrices
            cos = torch.cos(angles).to(device)
            sin = torch.sin(angles).to(device)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

            # Standard attention path but with Liger rotary
            k = self.key(x)
            v = self.value(x)

            multi_query_outputs = []
            for query in self.multi_query:
                q = query(x)
                
                # Reshape tensors for liger_rotary_pos_emb (batch_size, seq_len, n_q_head, head_dim)
                q_reshaped = q.view(batch_size, block_size, 1, self.head_size)
                k_reshaped = k.view(batch_size, block_size, 1, self.head_size)
                
                # Apply Liger rotary position embeddings
                q_rotated, k_rotated = liger_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)
                
                # Reshape back to 3D
                q_rotated = q_rotated.view(batch_size, block_size, -1)
                k_rotated = k_rotated.view(batch_size, block_size, -1)
                
                # Create causal mask
                masked_table = torch.tril(torch.ones((block_size, block_size), requires_grad=False, device=device))
                
                # Compute attention scores
                weights = q_rotated @ torch.transpose(k_rotated, dim0=-2, dim1=-1) * (k_rotated.shape[-1] ** -0.5)
                masked_values = weights.masked_fill(masked_table[: block_size, : block_size] == 0, float('-inf'))
                
                # Apply softmax and dropout
                weights_normalized = nn.functional.softmax(masked_values, dim=-1)
                weights_normalized = self.dropout(weights_normalized)
                
                # Apply attention to values
                out = weights_normalized @ v
                multi_query_outputs.append(out)
            
            multi_query_concat = torch.cat(multi_query_outputs, dim=-1)
            
        elif ModelArgs.use_flash_attn:
            # Flash attention path
            kv = self.kv_proj(x)
            k, v = kv.chunk(2, dim=-1)
            
            # Process each query head and concatenate results
            multi_query_concat = torch.cat([
                self.flash_attention(x, query(x), k, v, batch_size, block_size) 
                for query in self.multi_query
            ], dim=-1)
        else:
            # Standard attention path
            k = self.key(x)
            v = self.value(x)
            
            # Process each query head and concatenate results
            multi_query_concat = torch.cat([
                self.scaled_dot_product(query(x), k, v, block_size) 
                for query in self.multi_query
            ], dim=-1)
            
        # Apply output projection and dropout
        linear_layer = self.linear_layer(multi_query_concat)
        return linear_layer
     

class GQA(nn.Module):
    def __init__(
        self,
         device,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        block_size: int = ModelArgs.block_size,
        # no_of_q_heads: int = ModelArgs.no_of_heads,
        mqa_heads: int = ModelArgs.no_kv_heads
    ):
        super().__init__()

        # self.no_of_kv_heads = no_of_kv_heads
        self.no_of_q_heads = ModelArgs.no_of_heads // mqa_heads
        # self.head_dim = embeddings_dims // self.no_kv_heads
        self.dropout = nn.Dropout(p = ModelArgs.attn_dropout)
        self.linear_layer = nn.Linear(in_features=embeddings_dims * self.no_of_q_heads, out_features=embeddings_dims ,  bias=False,  device = device)
        self.device = device
        self.mqa = nn.ModuleList([MQA(no_of_q_heads=self.no_of_q_heads, embeddings_dims=embeddings_dims, device = self.device, block_size=block_size) for _ in range(self.no_of_q_heads)])
        # self.mqa = MQA(no_of_q_heads=self.no_of_q_heads, device=self.device, embeddings_dims=embeddings_dims, block_size=block_size)
    def forward(self,x):

        batch, block_size, embeddings_dims = x.shape

        # res = self.mqa(x)
        grouped_query_concat = torch.cat([group(x) for group in self.mqa], dim=-1)

        linear_layer= self.linear_layer(grouped_query_concat) #Basically MQA is made into GQA with no_of_q_heads and this class right here is just to consolidate everything into one
        out = self.dropout(linear_layer)
        return out


class Swish(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims
    ):
        super().__init__()

        self.sig = torch.nn.Sigmoid()


    def forward(self, x):
        swish = x * self.sig(x)

        return swish




class SWiGLUExpert(nn.Module):
    def __init__(
        self,
        device,
        block_size: int = ModelArgs.block_size,
        embeddings_dims: int = ModelArgs.embeddings_dims,
        
    ):
        super().__init__()

        self.hidden_dims = embeddings_dims * 2  #Apply this when memory permits

        if(ModelArgs.use_liger):

          @dataclass
          class config:

              hidden_size = embeddings_dims
              intermediate_size = self.hidden_dims
              hidden_act = 'swish'

          conf = config()

          self.swiglu = LigerSwiGLUMLP(conf)
        else:
          self.swish = Swish(block_size=block_size, embeddings_dims=embeddings_dims, device=device)
          self.linear_layer1 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
          self.linear_layer2 = nn.Linear(in_features=embeddings_dims, out_features=self.hidden_dims,  bias=False, device = device)
          self.linear_layer3 = nn.Linear(in_features=self.hidden_dims, out_features=embeddings_dims,  bias=False, device = device)




    def forward(self, x):
        if(ModelArgs.use_liger == False):
          swish_res = self.swish(self.linear_layer1(x))
          x_V = self.linear_layer2(x)
          res = torch.mul(swish_res, x_V)
          out = self.linear_layer3(res)

        else:
          out = self.swiglu(x)
          # out = self.linear_layer2(out)
          # out = self.linear_layer3(out)
        return out




class FFN(nn.Module):
    def __init__(self,
                  device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                   dropout = ModelArgs.dropout

                 ):
        super().__init__()

        # self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=embeddings_dims,  dtype=torch.float32,  device = device)
        self.swiglue = SWiGLUExpert(block_size=block_size, embeddings_dims=embeddings_dims,  device = device)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = self.swiglue(x)
        # x = self.linear_layer(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                  device,
                embeddings_dims: int = ModelArgs.embeddings_dims,
                dropout = ModelArgs.dropout,
                block_size: int = ModelArgs.block_size,
                vocab_size: int = ModelArgs.vocab_size,

                 ) :
        super().__init__()


        self.feedforward_network = FFN(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size,  device = device)
        self.gqa = GQA(embeddings_dims=embeddings_dims, block_size=block_size, mqa_heads=ModelArgs.mqa_heads,  device = device)
        # self.norm = Normalization(embeddings_dims=embeddings_dims)
        self.norm1 = RMSNormalization(embeddings_dims=embeddings_dims)
        self.norm2 = RMSNormalization(embeddings_dims=embeddings_dims)
        self.norm3 = RMSNormalization(embeddings_dims=embeddings_dims)
        self.norm4 = RMSNormalization(embeddings_dims=embeddings_dims)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self, x):

        x = x + self.norm2(self.gqa(self.norm1(x)))
        x = x + self.norm4(self.feedforward_network(self.norm3(x)))
        return x


class Llama(nn.Module):
    def __init__(self,
                device,
                  embeddings_dims: int = ModelArgs.embeddings_dims,
                  no_of_decoder_layers: int = ModelArgs.no_of_decoder_layers,
                  block_size: int = ModelArgs.block_size,
                  vocab_size: int = ModelArgs.vocab_size,
                  dropout = ModelArgs.dropout

                 ) :
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeddings_dims,  device = device)
        self.decoder = nn.Sequential(*[DecoderLayer(embeddings_dims=embeddings_dims, block_size=block_size, vocab_size=vocab_size, dropout=dropout,  device = device) for _ in range(no_of_decoder_layers)])
        self.linear_layer = nn.Linear(in_features=embeddings_dims, out_features=vocab_size,   device = device)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = RMSNormalization(embeddings_dims)
        
        
        #weight tying
        # self.embeddings.weight = self.linear_layer.weight
    
        self.apply(self._init_weights)
        if(ModelArgs.use_liger):
          self.le_loss = LigerFusedLinearCrossEntropyLoss(
              ignore_index=tokenizer.pad_token_id
          )

    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
               
                     
    
           
    def forward(self, x, actual_labels=None, inference=False):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.decoder(x)
        x = 2 * (1/math.sqrt((ModelArgs.no_of_decoder_layers))) * x
        x = self.norm(x)
        if(inference):
            out = self.linear_layer(x)
            return out
        if(ModelArgs.use_liger):  
            # print("yo")
            y = x.contiguous().view(-1, ModelArgs.embeddings_dims)
            if(actual_labels is not None):
                labels = actual_labels.contiguous().view(-1)
                
                # Pass linear layer weights FIRST as required [2][5]
                loss = self.le_loss(self.linear_layer.weight, y, labels)
                return loss
        else:
            # print("Hi")
            out = self.linear_layer(x)
            return out
        # x = self.linear_layer(x)
        # out = self.norm(x)
        # return x


def print_model_gradients(model):
   
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            print(f"Parameter: {name}")
            print(f"  Norm: {torch.norm(grad).item():.6f}")
        else:
            print(f"Parameter: {name} - No gradients")
        print("-----------------------------------")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    accelerator = get_accelerator()
    trainable_params = 0
    all_param = 0
    
    print("\n" + "="*80)
    print(colored("🔥 TRAINABLE PARAMETERS ANALYSIS 🔥", "red", attrs=["bold"]))
    print("="*80)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(colored(f'✅ "{name}" = {param.requires_grad} ({num_params:,} params)', "green", attrs=["bold"]))
        else:
            print(colored(f'❌ "{name}" = {param.requires_grad} ({num_params:,} params)', "red"))
    
    print("\n" + "="*80)
    if all_param > 0:
        trainable_percentage = 100 * trainable_params / all_param
        print(colored(f"🎯 SUMMARY:", "cyan", attrs=["bold"]))
        print(colored(f"   📊 Trainable params: {trainable_params:,}", "yellow", attrs=["bold"]))
        print(colored(f"   📊 All params: {all_param:,}", "yellow", attrs=["bold"]))
        print(colored(f"   📊 Trainable%: {trainable_percentage:.2f}%", "yellow", attrs=["bold"]))
    print("="*80 + "\n")


def log_model_parameters_to_file(model, filepath='model_parameters.log'):
    """
    Logs detailed model parameters to a file with size analysis by component.
    Helps verify the model size (e.g., if it's ~210M parameters).
    """
    import datetime
    import os
    import math
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Write header
        f.write(f"=== MODEL PARAMETER ANALYSIS ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model configuration:\n")
        f.write(f"  - Embedding dimensions: {ModelArgs.embeddings_dims}\n")
        f.write(f"  - Number of decoder layers: {ModelArgs.no_of_decoder_layers}\n")
        f.write(f"  - Number of attention heads: {ModelArgs.no_of_heads}\n")
        f.write(f"  - Number of KV heads: {ModelArgs.no_kv_heads}\n")
        f.write(f"  - Block size (sequence length): {ModelArgs.block_size}\n")
        f.write(f"  - Vocabulary size: {ModelArgs.vocab_size}\n\n")
        
        # Initialize counters
        total_params = 0
        total_trainable_params = 0
        param_counts_by_layer = {}
        param_counts_by_type = {
            "embedding": 0,
            "attention": 0,
            "ffn": 0,
            "layer_norm": 0,
            "output": 0,
            "other": 0
        }
        
        # Analyze parameters
        f.write("=== PARAMETER BREAKDOWN ===\n")
        f.write(f"{'Parameter Name':<60} {'Shape':<20} {'Params':<15} {'Trainable':<10}\n")
        f.write("-" * 105 + "\n")
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
                
            total_params += num_params
            if param.requires_grad:
                total_trainable_params += num_params
            
            # Classify parameter by layer
            if "decoder" in name:
                layer_match = name.split("decoder.")[1].split(".")[0]
                if layer_match.isdigit():
                    layer_num = int(layer_match)
                    if layer_num not in param_counts_by_layer:
                        param_counts_by_layer[layer_num] = 0
                    param_counts_by_layer[layer_num] += num_params
            
            # Classify parameter by type
            if "embedding" in name:
                param_counts_by_type["embedding"] += num_params
            elif any(x in name for x in ["gqa", "mqa", "attention", "key", "value", "query"]):
                param_counts_by_type["attention"] += num_params
            elif any(x in name for x in ["ffn", "feedforward", "swi", "swiglue"]):
                param_counts_by_type["ffn"] += num_params
            elif "norm" in name:
                param_counts_by_type["layer_norm"] += num_params
            elif "linear_layer" in name and "output" in name:
                param_counts_by_type["output"] += num_params
            else:
                param_counts_by_type["other"] += num_params
            
            # Write parameter details
            shape_str = str(tuple(param.shape))
            f.write(f"{name:<60} {shape_str:<20} {num_params:<15,} {'Yes' if param.requires_grad else 'No':<10}\n")
        
        # Write summary by layer
        f.write("\n=== PARAMETERS BY LAYER ===\n")
        total_layer_params = 0
        for layer_num in sorted(param_counts_by_layer.keys()):
            params = param_counts_by_layer[layer_num]
            total_layer_params += params
            f.write(f"Layer {layer_num:<3}: {params:,} parameters ({params/total_params*100:.2f}%)\n")
        
        f.write(f"\nTotal parameters in layers: {total_layer_params:,}\n")
        f.write(f"Parameters outside layers: {total_params - total_layer_params:,}\n")
        
        # Write summary by type
        f.write("\n=== PARAMETERS BY TYPE ===\n")
        for type_name, count in param_counts_by_type.items():
            f.write(f"{type_name.capitalize():<15}: {count:,} parameters ({count/total_params*100:.2f}%)\n")
        
        # Write final summary
        f.write("\n=== SUMMARY ===\n")
        f.write(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")
        f.write(f"Trainable parameters: {total_trainable_params:,} ({total_trainable_params/1e6:.2f}M)\n")
        f.write(f"Non-trainable parameters: {total_params - total_trainable_params:,} ({(total_params - total_trainable_params)/1e6:.2f}M)\n")
        f.write(f"Trainable parameters percentage: {total_trainable_params/total_params*100:.2f}%\n\n")
        
        # Write theoretical parameter count
        embedding_params = ModelArgs.embeddings_dims * ModelArgs.vocab_size
        attention_params_per_layer = 3 * ModelArgs.embeddings_dims * ModelArgs.embeddings_dims  # Q, K, V projections
        ffn_params_per_layer = 4 * ModelArgs.embeddings_dims * ModelArgs.embeddings_dims  # Up and down projections with 4x hidden dim
        layer_norm_params_per_layer = 2 * ModelArgs.embeddings_dims  # 2 layer norms per transformer block
        
        theoretical_params = embedding_params + ModelArgs.no_of_decoder_layers * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer) + ModelArgs.embeddings_dims * ModelArgs.vocab_size  # Final projection
        
        f.write(f"Theoretical parameter count: ~{theoretical_params/1e6:.2f}M\n")
        f.write(f"Difference from actual: {abs(theoretical_params - total_params)/1e6:.2f}M parameters\n")
        
        # Memory estimation (very rough)
        bytes_per_param = 2  # for fp16
        model_size_mb = total_params * bytes_per_param / (1024 * 1024)
        f.write(f"\nEstimated model size in memory (FP16): {model_size_mb:.2f} MB\n")
        
        f.write("\nAnalysis complete!\n")
    
    print(f"Model parameters logged to {filepath}")
    return total_params


# from andrej karapathy github
def topk_sampling(model, prompt, device, max_length=50, top_k=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = []
    # ModelArgs.inference=True
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model.module(input_ids, inference=True)
            logits = outputs[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            # Top-k filtering
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            
            
            # Apply temperature scaling
            probs = probs / temperature
            
            # Sample from top-k
            next_token = torch.multinomial(top_k_probs, num_samples=1)
           
            
            # generated_tokens.append(next_token.item())
            
            xcol = torch.gather(top_k_indices, -1, next_token)
            # generated_tokens.append(xcol)
            input_ids = torch.cat([input_ids, xcol], dim=1) #1 because is it the dimension of the sequence
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def find_unused_parameters(model):
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused.append(name)
    with open('unused_parameters.txt', 'w') as f:
        for name in unused:
            f.write(f"{name}\n")
    return unused



def save_to_file(text, step):
    with open(f'generated_texts/generations_{step}.txt', 'w') as f:
        f.writelines(text + "\n\n")


#Train the  model



torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True  # Better performance on newer GPUs
torch.backends.cudnn.allow_tf32 = True
# scaler = torch.amp.GradScaler(enabled=(ModelArgs.dtype == 'float16'))

save_chechpoint_iter = 1000
total_iters = 20000 * ModelArgs.epochs
eval_iters = 200
eval_check = 200
warmup_iters = 700 * ModelArgs.epochs  
min_lr = 0.1 * ModelArgs.max_lr
lr_decay_iters = 20000 * ModelArgs.epochs
total_batch_size = 524288
micro_batch_size = ModelArgs.batch_size
gradient_accumulation_steps = 32



def train():
    # Default checkpoint settings
    save_dir = './checkpoints'
    save_interval = 2000
    load_dir = None
    ckpt_id = None
    
    # Initialize distributed training
    deepspeed.init_distributed(dist_backend='nccl')  
    
    # Get local rank for device identification
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # print(f"Local rank: {local_rank}")
    if local_rank == 0:  # Only rank 0 initializes WandB
        wandb.init(project='Llama-DeepSpeed-Pretrain')

    model = Llama(device=None, embeddings_dims=ModelArgs.embeddings_dims, block_size=ModelArgs.block_size, vocab_size=ModelArgs.vocab_size, dropout=ModelArgs.dropout)
    
    print(f"Start running DDP on local rank {local_rank}.")
    
    print("wandb initialized")
    
    print(f"Model initialization complete")
    
    # Log model parameters to file (only on rank 0)
    if local_rank == 0:
        total_params = log_model_parameters_to_file(model, filepath='model_parameters.log')
        print(f"Model has {total_params:,} parameters (~{total_params/1e6:.2f}M)")
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config='./ds_config.json'  
        
    )
    model = model.half()  # Convert model to half precision if using float16
    print(f"Model device: {model.device}")

    # # Set device for operations outside of model
    # torch.cuda.set_device(model.device)
    
    # Print detailed parameter analysis
    if local_rank == 0:  # Only print on rank 0 to avoid duplicate output
        print_trainable_parameters(model)
    
    # Load checkpoint if specified
    start_step = 0
    client_sd = {'step': 0}
    if load_dir and ckpt_id:
        start_step, client_sd = load_deepspeed_checkpoint(model, load_dir, ckpt_id)
        print(f"Resuming training from step {start_step}")
    
    # Prepare data loaders
    train_dataloader = prepare_dataset('train', model.device, ModelArgs.batch_size)
    val_loader = prepare_dataset('val', model.device, ModelArgs.batch_size)
    
    # Advance data loader to checkpoint step if resuming
    if start_step > 0:
        train_data_iterator = iter(train_dataloader)
        dataloader_to_step(train_data_iterator, start_step)
    else:
        train_data_iterator = iter(train_dataloader)
    
    # model = torch.compile(model)
    # model = model.to(device)
    
    # model = DDP(model, device_ids=[device])

    # model.eval()
    world_size = torch.cuda.device_count()
    @torch.no_grad()
    def estimate_loss(val_loader, val_iterator, device):
        out = {}
        # train_loader = prepare_dataset('train', ModelArgs.batch_size)
        
        # val_loader_iterator = iter(val_loader)
        loader = None
        epoch_loss = None
        epoch_losses = []
        # print("Starting the eval...")
        for split in ['val']:
            print(f"Starting with {split} evaluation...")
            # losses = torch.zeros(ModelArgs.val_epochs)
            # if(split == 'train'):
            #         loader = train_loader
            # if(split == 'val'):
            #         loader = val_loader
            eval_step_iterator = range(eval_check)
            if local_rank == 0:  # Only show progress bar on main process
                eval_step_iterator = tqdm(eval_step_iterator, desc=f"{split.capitalize()} Evaluation", leave=False)
            
            for step in eval_step_iterator:  
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    val_loader_iterator = iter(val_loader)
                    batch = next(val_loader_iterator)
                
                total_loss = 0  
                # loader.sampler.set_epoch(step)
                total_batches = 0 
                # batch = next(val_loader_iterator)
                # for batch in loader:  # Loop through DataLoader batches
                idx = batch['input_ids']
                targets = batch['labels']
                idx = idx.to(device)
                targets = targets.to(device)
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                if(ModelArgs.use_liger == True):
                    loss = model(idx, targets)

                else:
                    
                    logits = model(idx)
                    batch_size, block_size, embeddings_dims = logits.shape
                    logits = logits.view(batch_size * block_size, embeddings_dims)
                    targets = targets.view(batch_size * block_size)

                    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)

                total_loss += loss.item()
                total_batches += 1

            # Compute mean loss for this epoch
            epoch_loss = total_loss / total_batches if total_batches > 0 else 0.0
            epoch_losses.append(epoch_loss)

                # print(f"Epoch {epoch + 1}/{ModelArgs.val_epochs}: Loss = {epoch_loss:.4f}")

            # Compute mean loss across all evaluation epochs
            out[split] = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            epoch_loss = None
            epoch_losses = []

        model.train()
        return out

    # model = model.to(rank)
    model.train()
    count = 0
    
    # for p in model.parameters():
    #     p.requires_grad = True

    
    train_dataloader = prepare_dataset('train', model.device, ModelArgs.batch_size)
    val_loader= prepare_dataset('val', model.device, ModelArgs.batch_size)
    # for step in tqdm(range(total_iters)):
    # for epoch in range(ModelArgs.epochs):
        # torch.cuda.synchronize() 
    
    # train_dataloader.sampler.set_epoch(epoch)
    
    # val_loader.sampler.set_epoch(epoch)
    print("Loaders ready both")
    epochs = ModelArgs.epochs

    # train_step_iterator = range(len(train_dataloader))
    # if device == 0:  # Only create progress bar on rank 0
    #   train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)

        # Print progress on rank 0
    train_loader_length = 0
    train_data_iterator = iter(train_dataloader)
    val_data_iterator = iter(val_loader)
    token_count = 0
    if local_rank == 0:
        train_loader_length = len(train_dataloader)
        # print("Total batches: ", train_loader_length)
    # print("Length of : ", len(train_dataloader))
    # print("Length of val: ", len(val_loader))
    # for  step, batch in enumerate(train_dataloader):
    # Create progress bar only on rank 0 for main training loop
    train_step_iterator = range(start_step, total_iters)
    if local_rank == 0:  # Only show progress bar on main process
        train_step_iterator = tqdm(train_step_iterator, desc="Training Progress", position=0, leave=True)
    
    for epoch in range(ModelArgs.epochs):
        for step in train_step_iterator:
            # print("Dataloader things: ", batch)
            # print("Total batches: ", len(train_dataloader))
            
            # model.train()
            
            if local_rank == 0:
                # if(step % 100 == 0):
            #     if(step == train_loader_length):
            #       break
                    print("Step : ", step, "/", total_iters)
                    print('Total batches: ', len(train_dataloader))
                    print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                    print("Total tokens processed: ", token_count)
                    
            # all_gpus_avg_train_loss = None
            # all_gpus_avg_val_loss = None
            # every once in a while evaluate the loss on train and val sets
            if (step  % eval_iters == 0 and step != 0) or step == total_iters - 1:
                    losses = estimate_loss( val_loader, val_data_iterator, model.device)
                    # avg_train_loss = losses['train']
                    avg_val_loss = losses['val']
                    # print(avg_val_loss)
                    avg_val_loss = torch.tensor([avg_val_loss]).to(model.device)
                    # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                    # if device == 0:  # Only print on main process
                    print(f"[GPU {local_rank}] | Step: {step} / {total_iters} | Val Loss: {losses['val']:.4f}")
                    # print(f"[GPU {local_rank}] | Epoch {epoch}/{ModelArgs.epochs}| |Step: {step} | Train Loss: {losses['train']:.4f}")
                        # print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                        # Log training loss more frequently
                        # Aggregate average loss across all GPUs
                    # avg_train_loss = torch.Tensor([losses['train']]).to(model.device)
                    avg_val_loss = torch.Tensor([losses['val']]).to(model.device)
                    torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.AVG)
                    # torch.distributed.reduce(avg_val_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    
                    # if device == 'cuda:0':
                    # all_gpus_avg_train_loss = avg_val_loss / world_size
                        # print(f"All_GPUs_Train_losses: {all_gpus_avg_train_loss.item():.4f}")
                    # all_gpus_avg_val_loss = avg_val_loss / world_size
                    print(f"Val Loss: {avg_val_loss.item():.4f}")
        
                    perplexity = torch.exp(torch.tensor(avg_val_loss.item()))  # Calculate perplexity

                    # if device == 0:
                    if local_rank == 0:
                        wandb.log({
                                "All GPU Val_Loss": avg_val_loss.item(),
                                "Val Perplexity": perplexity.item(),
                                "Total Tokens Processed": token_count,
                                "Step": step,
                            })
                

            if step % save_interval == 0 and step != 0:
                print(f"Saving the model checkpoint for step: {step}")
                # Use DeepSpeed checkpoint saving - ALL processes must call this
                client_sd['step'] = step
                ckpt_id = str(step)  # Use step as checkpoint ID
                model.save_checkpoint(save_dir, ckpt_id, client_state=client_sd)
            
            accumulated_loss = 0.0
            
            
            # optimizer.zero_grad(set_to_none=True)
            # for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                batch = next(train_data_iterator)
                # print("Here: ", model.device)
            # print("Before: ", batch['input_ids'].device)
            # print("Before: ", batch['labels'].device)
            idx = batch['input_ids'].to(model.device)
            targets = batch['labels'].to(model.device)
            # token_count += len(idx)
            # idx = idx.to(device)
            # targets = targets.to(device)
            # print("After: ", idx.device)
            # print("After: ", targets.device)
            # optimizer.zero_grad(set_to_none=True)  # Zero gradients before backward pass
            token_count += idx.numel() * gradient_accumulation_steps * world_size
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            if(ModelArgs.use_liger == True):
                loss = model(idx, targets)

            else:
                logits = model(idx)
                batch_size, block_size, embeddings_dims = logits.shape
                # print(logits.shape)
                # print(targets)
                logits = logits.view(batch_size*block_size, embeddings_dims)
                # print("OK")
                targets = targets.view(batch_size * block_size)
                # print("OK2")
                loss = nn.functional.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
                
                    # loss = loss / gradient_accumulation_steps #IDK why div is done here specifically? Maybe think of it in terms of a very big batch being processed and there is need for equal important of each mini batch for the overall big batch
            
            # Verify loss properties (not gradients!)
            print(f"Loss value: {loss.item():.6f}")
            print(f"Loss requires_grad: {loss.requires_grad}")
            print(f"Loss grad_fn: {loss.grad_fn}")


           
            model.backward(loss)
            
            # # Calculate gradient norm after backward pass but before optimizer step
            # total_grad_norm = None
            # if local_rank == 0:  # Only calculate on rank 0 to avoid redundant computation
            #     # Compute total gradient norm across all model parameters
            #     total_grad_norm = torch.norm(
            #         torch.stack([
            #             torch.norm(p.grad.detach(), 2) 
            #             for p in model.parameters() 
            #             if p.grad is not None
            #         ]), 
            #         2
            #     )
            #     print(f"Gradient Norm: {total_grad_norm.item():.6f}")
            if(local_rank == 0):
                unused = find_unused_parameters(model)
                if local_rank == 0:
                    if unused:
                        print(f"Unused parameters found: {unused}")
                    else:
                        print("No unused parameters found.")
                
            # model.lr_scheduler.step()  # Update learning rate
            # weight update
            model.step()
            model.lr_scheduler.step()
            accumulated_loss += loss.item()
            
            if local_rank == 0:
                # if(micro_step % 10 == 0):
            #     if(step == train_loader_length):
            #       break
                    
                    # print("Micro Batch : ", micro_step)
                    print("Step : ", step, "/", total_iters)
                    print('Total batches: ', len(train_dataloader))
                    print("Total gradient accumulation steps: ", gradient_accumulation_steps)
                    print("Total tokens processed: ", token_count)
                # count += 1
    
            # scaler.update()

            # model.all_reduce(accumulated_loss)
            torch.distributed.reduce(torch.tensor(accumulated_loss).to(model.device), dst=0, op=torch.distributed.ReduceOp.AVG)
            # accumulated_loss /= world_size

            if local_rank == 0:

                perplexity = torch.exp(torch.tensor(accumulated_loss))  # Calculate perplexity
                # if(device == 0):
                wandb.log({
                            "Learning Rate": model.optimizer.param_groups[0]['lr'],
                            "All GPU Train_Loss": accumulated_loss,
                            # "Train loss": loss.item(),
                            "Train Perplexity": perplexity.item(),
                            "Total Tokens Processed": token_count,
                            "Step": step,
                            # "Gradient Norm": total_grad_norm.item() if total_grad_norm is not None else 0.0,
                            # "Epoch": epoch
                            
                })

            if local_rank == 0 and step % 200 == 0 and step != 0:
                count = 1
                while(count):  
                    prompt = "Hello Myself a LLM and "
                    generated_text = topk_sampling(model, prompt, max_length=50, top_k=50, temperature=1.0, device=model.device)
        
        
                    print(f" Step: {step} | Generated Text: {generated_text}")
                    save_to_file(generated_text, step)
                    count -= 1
        
        if local_rank == 0:
            
            wandb.finish()
        cleanup()


world_size = torch.cuda.device_count()
print(f"World size: {world_size}")
train()

