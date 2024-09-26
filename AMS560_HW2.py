from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

path = "..."
# modify the path first (the path from Q3)
# For Q6, explore this path
# For Q7, explore 'params.json' file

tokenizer_path = path + "/tokenizer.model"
special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
)

print(tokenizer.decode(tokenizer.encode("hello world!")))
# For Q8, print the tokenizer.encode() output

model = torch.load(path+"/consolidated.00.pth", weights_only=True)
print('This is the first 20 matrices weights:')
print(json.dumps(list(model.keys())[:20], indent=4))
# This show the first 20 matrices weights. 
# For Q9, show last 5 or them all.

print('This is the shape for feed forward weight W_2 of layer 0, ')
print(model["layers.0.feed_forward.w2.weight"].shape)
# For Q10, print the shape for attention W_o of layer 21.

print('Note: Please update this file to complete your responses for questions 6 through 10.')