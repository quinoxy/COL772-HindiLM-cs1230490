import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any, Dict, List
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, config : Dict[str, Any]):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_model = config["d_model"]
        self.d_head = config["d_head"]
        self.tanh_mode = True if (config["mode"] == "tanh-clipped") else False
        self.tau = None
        if (self.tanh_mode):
            self.tau = config["tau"]
        self.q_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.k_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.v_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.concat_mat = nn.Linear(self.d_model, self.d_model, bias= False)

        init.xavier_normal_(self.q_mat.weight)
        init.xavier_normal_(self.k_mat.weight)
        init.xavier_normal_(self.v_mat.weight)
        init.xavier_normal_(self.concat_mat.weight)

    def set_weights(self, weights : Dict[str, Any], layer_no):
        self.concat_mat.weight.data = weights[f"W_{layer_no}_O"].T
        for i in range(1,self.n_heads + 1):
            self.q_mat.weight.data[:,(i-1) * self.d_head : i * self.d_head] = weights[f"W_{layer_no}_Q_{i}"].T
            self.k_mat.weight.data[:,(i-1) * self.d_head : i * self.d_head] = weights[f"W_{layer_no}_K_{i}"].T
            self.v_mat.weight.data[:,(i-1) * self.d_head : i * self.d_head] = weights[f"W_{layer_no}_V_{i}"].T

    def forward(self, inputs, attention_mask, causal_attention_mask):

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        #making all matrices (batch_size, n_heads, seq_len, d_head)
        Q_mat = self.q_mat(inputs).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        K_mat = self.k_mat(inputs).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        V_mat = self.v_mat(inputs).reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)

        K_mat = K_mat.transpose(-2,-1)

        # S matrix has dim (batch_size, n_heads, seq_len, seq_len)
        S_mat = torch.matmul(Q_mat, K_mat) / math.sqrt(self.d_head)


        if (self.tanh_mode):
            S_mat = self.tau * torch.tanh(S_mat)

        S_mat = S_mat + causal_attention_mask


        # attention_mask = (1-attention_mask).float() * -1e9
        # attention_mask = attention_mask[:,None, None, :]

        S_mat = S_mat + attention_mask

        
        
        attn = torch.softmax(S_mat, dim = -1)

        # attn * V will be (batch_size, n_heads, seq_len, d_head)
        # we will first transpose to (batch_size, seq_len, n_heads, d_head)
        # then reshape to (batch_size, seq_len, d_model)
        concatenated_result = torch.matmul(attn, V_mat).transpose(1,2).reshape(batch_size, seq_len, self.d_model)
        return self.concat_mat(concatenated_result)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

        init.xavier_normal_(self.w1.weight)
        init.xavier_normal_(self.w2.weight)
        init.xavier_normal_(self.out.weight)

    def forward(self, x):
        return self.out(nn.functional.silu(self.w1(x)) * self.w2(x))
    
class TransformerBlock(nn.Module):

    def __init__(self, config : Dict[str, Any]):
        super().__init__()
        self.d_model = config["d_model"]
        self.ln = nn.LayerNorm(self.d_model, elementwise_affine = True)
        self.mha = MultiHeadAttention(config)
        
        self.seqblock = nn.Sequential(
            nn.LayerNorm(self.d_model, elementwise_affine = True),
            SwiGLU(self.d_model, 4*self.d_model)
        )

        for layer in self.seqblock:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)

    def set_weights(self, weights : Dict[str, Any], layer_no):
        self.seqblock[1].weight.data = weights[f"W_{layer_no}_up"].T
        self.seqblock[1].bias.data = weights[f"b_{layer_no}_up"]
        self.seqblock[3].weight.data = weights[f"W_{layer_no}_down"].T
        self.seqblock[3].bias.data = weights[f"b_{layer_no}_down"]
        self.mha.set_weights(weights, layer_no)
        self.ln.weight.data = weights[f"gamma_{layer_no}_1"]
        self.ln.bias.data = weights[f"beta_{layer_no}_1"]
        self.seqblock[0].weight.data = weights[f"gamma_{layer_no}_2"]
        self.seqblock[0].bias.data = weights[f"beta_{layer_no}_2"]

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor, causal_attention : torch.Tensor) -> torch.Tensor :
        intermediate1 = self.ln(input)
        intermediate2 = self.mha(intermediate1, attention_mask, causal_attention)
        intermediate3 = input + intermediate2
        intermediate4 = self.seqblock(intermediate3)
        return intermediate3 + intermediate4
        


class PositionalEncodingBlock(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.d_model = config["d_model"]

        pos = torch.arange(512).unsqueeze(1)
        i = torch.arange(0, self.d_model, 2)
        power = torch.exp(-math.log(10000.0) * i/self.d_model)
        angles = pos*power
        
        encoding = torch.zeros(512, self.d_model)

        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        self.encoding = encoding
    
    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        new_encoding = self.encoding[:seq_len, :self.d_model].unsqueeze(0)
        new_encoding = new_encoding.to(inputs.device)
        return new_encoding

        

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()

        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        
        self.pe = PositionalEncodingBlock(config)

        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )

        self.finalLayerNorm = nn.LayerNorm(config["d_model"], elementwise_affine = True)
        
        self.devocab_and_softmax = nn.Sequential(
            nn.Linear(config["d_model"], config["vocab_size"], bias = False)
        )

        self.devocab_and_softmax[0].weight = self.embed.weight

        pos = torch.arange(512)
        mask = pos[None, :] > pos[:, None]
        causal_attention = torch.zeros(512, 512)
        causal_attention = causal_attention.masked_fill(mask, -1e9)
        self.register_buffer(
            "causal_attention",
            causal_attention[None, None, :, :]
        ) 

        

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        self.embed.weight.data = weights["W_vocab"].T
        self.devocab_and_softmax[0].weight.data = weights["W_devocab"].T
        self.finalLayerNorm.weight.data = weights["gamma_final"]
        self.finalLayerNorm.bias.data = weights["beta_final"]

        for layer_no, block in enumerate(self.transformerBlocks):
            block.set_weights(weights, layer_no+1)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        _, seq_len = input_ids.shape
        
        

        causal_attention = self.causal_attention[:,:,:seq_len,:seq_len]

        attn_mask = (1-attention_mask).float() * -1e9
        attn_mask = attn_mask[:,None,None,:]


        intermediate1 = self.embed(input_ids)
        intermediate2 = self.pe(input_ids)
        intermediate3 = intermediate1 + intermediate2

        for blk in self.transformerBlocks:
            intermediate3 = blk(intermediate3, attn_mask, causal_attention)
        
        intermediate4 = self.finalLayerNorm(intermediate3)
        return self.devocab_and_softmax(intermediate4)




def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    print(config)
    model.set_weights(weights)
    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0

    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    maxlen = max(x.shape[0] for x in input_ids)

    input_id_tensor = torch.full((len(input_ids), maxlen), PAD_ID, dtype=torch.long)
    att_mask_tensor = torch.zeros(len(input_ids), maxlen, dtype=torch.long)

    for i in range(len(input_ids)):
        size = input_ids[i].shape[0]
        input_id_tensor[i, :size] = input_ids[i]
        att_mask_tensor[i, :size] = attention_mask[i]

    return {
        "input_ids": input_id_tensor,
        "attention_mask": att_mask_tensor
    }

