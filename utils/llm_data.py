# learned yield makes the function return a generator, which is why only. one batch lives in
# memory at a time. 2. GPT style pretraining works soley on eos_id and squences can be cut_off
# mid sentence. 3. NJTs are efficient for differently long sequences but not necessary
# for how we are training llms in the notebooks + sdpa does not even support NJT for "mps"

import os

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '.cache'))

def load_data(name="stanfordnlp/imdb", tokenizer_name="bert-base-uncased"):
    """
    Returns (tokenizer, train_ids, val_ids)
    """

    tknz = Tokenizer.from_pretrained(tokenizer_name)
    eos_id = tknz.token_to_id("<|endoftext|>")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
                              

    tag = f"{name}-{tokenizer_name}".replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{tag}.pt")


    # def to_nested(blob):
    #     # soon supported for my mps!
    #     return torch.nested.nested_tensor_from_jagged(blob["ids"], blob["offsets"])
    
    if os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return tknz, blob["train"], blob["val"]
    
    dataset = load_dataset(name) # Dict: train, test unsupervised: 'text', 'label'

    def encode_split(split):
        texts = dataset[split]["text"]
        ids = []
        for start in range(0, len(texts), 1000):
            for encoding in tknz.encode_batch(texts[start:start+1000]):
                ids.extend(encoding.ids)
                if eos_id is not None:
                    ids.append(eos_id)
                

        return torch.tensor(ids, dtype=torch.long)
      
    
    train_ids = encode_split("train")
    test_ids = encode_split("test")


    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save({'train': train_ids, 'val': test_ids}, cache_path)

    # load from cache
    blob = torch.load(cache_path)
    return tknz, blob["train"], blob["val"]

def batches(ids, block_size=512, batch_size=128, device=None, shuffle=True):
    # (ids,)
    ids = ids.view(-1)
    n_blocks = (ids.numel() - 1) // block_size
    if n_blocks < batch_size:
        raise ValueError(
            f"need at least {batch_size * block_size + 1} tokens for "
            f"block_size={block_size}, batch_size={batch_size}, got {ids.numel()}"
        )

    # (n_blocks,): [0, block_size, 2*block_size, ...]
    start = torch.arange(n_blocks) * block_size
    if shuffle:
        start = start[torch.randperm(n_blocks)]


    for i in range(0, n_blocks - batch_size + 1, batch_size):
        offset = start[i:i+batch_size][:, None] + torch.arange(block_size)
        x = ids[offset]
        y = ids[offset + 1]
        if device:
            x, y = x.to(device=device), y.to(device=device)
        yield x, y

tknz, train_ids, val = load_data()
print(train_ids.shape)
for x, _ in batches(train_ids):
    print(x.shape)
    break
