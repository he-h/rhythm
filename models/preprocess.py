import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(f"device: {self.device}")
        
        self.llama = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(configs.llm_ckp_dir,  use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.vocab_size = self.llama_tokenizer.vocab_size
        # self.hidden_dim_of_llama = 4096
        self.hidden_dim_of_llama = self.llama.config.hidden_size
        print(f"hidden_dim_of_llama: {self.hidden_dim_of_llama}")
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
    def tokenizer(self, x):
        output = self.llama_tokenizer(x, padding=True, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc):        
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc = [self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))]
        max_length = max(i.shape[1] for i in x_mark_enc)
        x_mark_enc = [
            F.pad(i, (0, 0, 0, max_length - i.shape[1]), value=0)  # Pad along the second dimension
            for i in x_mark_enc
        ]
        x_mark_enc = torch.cat(x_mark_enc, 0)
        text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
        text_outputs = text_outputs[:, -1, :]
        return text_outputs
    
    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)