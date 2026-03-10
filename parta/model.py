import torch
import torch.nn as nn
from typing import Any, Dict, List
import math


class SingleAttentionHead(nn.Module):
    def __init__(self, config : Dict[str, Any]):
        super().__init__()
        d_model = config["d_model"]
        self.d_head = config["d_head"]
        self.tanh_mode = True if (config["mode"] == "tanh_clipped") else False
        self.tau = config["tau"]
        self.query = nn.Linear(d_model, self.d_head, bias = False)
        self.value = nn.Linear(d_model, self.d_head, bias = False)
        self.key = nn.Linear(d_model, self.d_head, bias = False)

    def set_weights(self, weights : Dict[str, Any], layer_no, head_no):
        query_string = f"W_{layer_no}_Q_{head_no}"
        key_string = f"W_{layer_no}_K_{head_no}"
        value_string = f"W_{layer_no}_V_{head_no}"
        self.query.weight.data = weights[query_string]
        self.key.weight.data = weights[key_string]
        self.value.weight.data = weights[value_string]

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor :
        query = self.query(input)
        key = self.key(input)
        key = key.transpose(-2,-1)
        value = self.value(input)
        mat_S = torch.matmul(query, key) / math.sqrt(self.d_head)
        mat_S = mat_S + attention_mask
        if (self.tanh_mode):
            mat_S.tanh_()
            mat_S.mul_(self.tau)
        attn = torch.softmax(mat_S, dim = -1)
        return torch.matmul(attn, value)



class MultiHeadAttention(nn.Module):
    def __init__(self, config : Dict[str, Any]):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.d_model = config["d_model"]
        self.d_head = config["d_head"]
        self.tanh_mode = True if (config["mode"] == "tanh_clipped") else False
        self.tau = config["tau"]
        self.q_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.k_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.v_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        self.concat_mat = nn.Linear(self.d_model, self.d_model, bias= False)
        

    def set_weights(self, weights : Dict[str, Any], layer_no):
        self.concat_mat.weight.data = weights[f"W_{layer_no}_O"]
        for i in range(1,self.n_heads + 1):
            self.q_mat.weight.data[(i-1) * self.d_head : i * self.d_head, :] = weights[f"W_{layer_no}_Q_{i}"]
            self.k_mat.weight.data[(i-1) * self.d_head : i * self.d_head, :] = weights[f"W_{layer_no}_K_{i}"]
            self.v_mat.weight.data[(i-1) * self.d_head : i * self.d_head, :] = weights[f"W_{layer_no}_V_{i}"]

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

        S_mat = S_mat + causal_attention_mask
        S_mat = S_mat + attention_mask

        if (self.tanh_mode):
            S_mat.tanh_()
            S_mat.mul_(self.tau)
        
        attn = torch.softmax(S_mat, dim = -1)

        # attn * V will be (batch_size, n_heads, seq_len, d_head)
        # we will first transpose to (batch_size, seq_len, n_heads, d_head)
        # then reshape to (batch_size, seq_len, d_model)
        concatenated_result = torch.matmul(attn, V_mat).transpose(1,2).reshape(batch_size, seq_len, self.d_model)
        return self.concat_mat(concatenated_result)



        

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor :
        pass

class TransformerBlock(nn.Module):

    def __init__(self, config : Dict[str, Any]):
        self.config = config
        super().__init__()

    def set_weights(self, weights : Dict[str, Any]):
        pass

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor :
        pass

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        raise NotImplementedError("Implement set_weights as described in assignment document")

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
        raise NotImplementedError("Implement forward as described in assignment document")



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    raise NotImplementedError("Implement collate_fn as described in assignment document")

def main():
    pass

if __name__ == "__main__":
    main()