# %%
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = 7
MODEL_NAME = "google/gemma-3-1b-pt"
SAE_MODEL_NAME = "google/gemma-scope-2-1b-pt"

prompt_comparison = (
    "Under the terms of this contract, the service provider agrees to deliver "
    "the specified services within the agreed-upon timeframe."
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs_comparison = tokenizer.encode(
    prompt_comparison, return_tensors="pt", add_special_tokens=True
).to(model.device)

print(inputs_comparison)
print(inputs_comparison.shape)

# Token id and token text table
ids = inputs_comparison[0].tolist()
tokens_raw = tokenizer.convert_ids_to_tokens(ids)

print("\nidx | token_id | token_raw                | token_decoded")
print("-" * 80)
for idx, (token_id, token_raw) in enumerate(zip(ids, tokens_raw)):
    token_decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    print(f"{idx:>3} | {token_id:>8} | {repr(token_raw):<24} | {repr(token_decoded)}")

print("total tokens:", len(ids))


WIDTH = "16k"
L0 = "medium"  # options are {small, medium, big}

path_to_params = hf_hub_download(
    repo_id=SAE_MODEL_NAME,
    filename=f"resid_post/layer_{LAYER}_width_{WIDTH}_l0_{L0}/params.safetensors",
)

params = load_file(path_to_params)


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in, d_sae, affine_skip_connection=False):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None

    def encode(self, input_acts):
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.w_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        if self.affine_skip_connection is not None:
            return recon + x @ self.affine_skip_connection
        return recon


d_model, d_sae = params["w_enc"].shape
sae = JumpReLUSAE(d_model, d_sae)
sae.load_state_dict(params)
sae.cuda()


def gather_acts_hook(mod, inputs, outputs, cache: dict, key: str, use_input: bool):
    """Generic hook function whic stores activations (either input or output of a particular PyTorch module)."""
    acts = (
        inputs[0].squeeze(0) if use_input else outputs[0]
    )  # inputs usually have a batch dim
    cache[key] = acts
    return outputs


def gather_residual_activations(model, target_layer, inputs):

    cache = {}

    # Add a hook function to store the output of this layer of the model
    handle = model.model.layers[target_layer].register_forward_hook(
        partial(gather_acts_hook, cache=cache, key="resid_post", use_input=False)
    )

    # Forward pass inside a try/except/finally block (useful just in case our hook breaks
    # and we can't remove it!)
    try:
        _ = model.forward(inputs)
    finally:
        handle.remove()

    return cache["resid_post"]


target_act = gather_residual_activations(model, LAYER, inputs_comparison)
sae_acts = sae.encode(target_act.to(torch.float32))
recon = sae.decode(sae_acts)
# We need to drop BOS tokens since it looks like BOS has no signal.
sae_acts = sae_acts[1:, :]
print(sae_acts.shape)

# TODO(davidnet): This is our apple of discord
# In a long text, and in short text, how to argue what is happening:
top_vals, top_idx = sae_acts.sum(dim=0).topk(5)
for act, idx in zip(top_vals, top_idx):
    print(f"{act:>6.1f} | {idx}")
    # 1198.2 | 344 <- Most important feature (as shown in neuropedia)
    # 1048.5 | 72
    #  669.2 | 208
    #  653.9 | 570
    #  623.2 | 302
feature_id = 344

##### ----- #####
# Get the SAE Examples (metadata, to find the documents and themes)
repo_id = SAE_MODEL_NAME
filename = f"resid_post/layer_{LAYER}_width_{WIDTH}_l0_{L0}/examples.safetensors"


example_data = load_file(hf_hub_download(repo_id=repo_id, filename=filename))
tokens = example_data["tokens"]
activations = example_data["activations"]  # torch.Size([16384, 1000])
positions = example_data["positions"]
seq_ids = example_data["seq_ids"]
feature_frequencies = example_data["feature_frequencies"]
logit_effects = example_data["logit_effects"]

# Get activations, cropped past the point where it's not actually active (we
# just padded the array out to the same length as the other features)
activations = activations[feature_id]
n_acts = (activations > 0).sum().item()

# print(seq_ids.shape)
seq_ids = seq_ids[feature_id][:n_acts]
# print(positions.shape)
positions = positions[feature_id][:n_acts]
# print(logit_effects.shape)
logit_effects = logit_effects[feature_id][:n_acts]

max_logit_effect = np.abs(logit_effects).max().item()
print(f"Inspecting feature {feature_id}")
print(f"Frequency: {feature_frequencies[feature_id]:.2e}")


def to_str_tokens(tokens: list[int]) -> list[str]:
    str_tokens = tokenizer.convert_ids_to_tokens(tokens)
    for i, t in enumerate(str_tokens):
        if t.startswith("▁"):
            str_tokens[i] = " " + t[1:]
    return str_tokens


str_tokens = to_str_tokens(example_data["top_tokens"][feature_id])
print(f"Top tokens: {str_tokens}")

op_activations = []
formatted_sequences = []
position_tuples = []
max_act = max(activations[0], 1e-12)
max_examples = 10
top_activations = []
formatted_sequences = []
position_tuples = []
buf: tuple[int, int] = (25, 25)
max_act = max(activations[0], 1e-12)


def span(str_tok, act, logit, max_logit):
    # Activtion determines background colour
    bg_color = f"rgba(0,255,0,{act:.3f})"
    # Logit effect determines underline color: blue (+ve), red (-ve)
    logit_normed = logit / max_logit if max_logit > 1e-9 else 0.0
    logit_normed = max(-1.0, min(1.0, logit_normed))
    u_color = (
        f"rgba(0,0,255,{logit_normed:.3f})"
        if logit_normed >= 0.0
        else f"rgba(0,0,255,{-logit_normed:.3f})"
    )
    # Use thick bottom border for underline
    style = f"background-color: {bg_color}; border-bottom: 3px solid {u_color};"
    return f'<span style="{style}">{str_tok}</span>'


def join_str_tok_list(str_toks, acts, logits, max_logit):
    str_toks = [x.replace("\n", "⏎") for x in str_toks]
    logits = [0.0] + logits.tolist()[:-1]  # logit effect is for the *next* token
    return "".join(
        [span(*args, max_logit) for args in zip(str_toks, acts, logits, strict=True)]
    )


escape_html = lambda x: x.replace("<", "&lt;").replace(">", "&gt;")
while len(formatted_sequences) < max_examples:
    # Finish if we don't have any more nonzero examples we can take
    if not seq_ids.shape[0]:
        break

    # Pick the max-activation sequence not yet chosen
    idx = np.argmax(activations).item()
    seq_id = seq_ids[idx].item()
    position = positions[idx].item()
    activation = activations[idx].item()
    position_tuples.append(f"{seq_id},{position}")
    top_activations.append(activation)

    # Get the string tokens, maybe adjusting the buffer if this token is too
    # close to the start or end of the sequence
    true_buf = (
        min(buf[0], position),
        min(buf[1], tokens.shape[1] - 1 - position),
    )
    str_toks = to_str_tokens(
        tokens[seq_id, position - true_buf[0] : position + true_buf[1] + 1]
    )

    # Initialize buffers for activations and logit effects
    seq_len_window = true_buf[1] + true_buf[0] + 1
    acts = np.zeros((seq_len_window,))
    logits = np.zeros((seq_len_window,))
    str_toks = list(map(escape_html, str_toks))

    # Get the tokens & activations in a padded region around that sequence
    seq_id_mask = seq_ids == seq_id
    pos_diff = positions - position
    position_mask = (-pos_diff < true_buf[0]) & (pos_diff < true_buf[1])
    full_mask = seq_id_mask & position_mask

    # Apply mask to both activations and logit effects
    for pos, act, logit in zip(
        positions[full_mask], activations[full_mask], logit_effects[full_mask]
    ):
        offset = pos - position + true_buf[0]
        acts[offset] = act / max_act
        logits[offset] = logit
    formatted_sequences.append(
        join_str_tok_list(str_toks, acts, logits, max_logit_effect)
    )

    # Filter out all other activations with the same sequence
    filter_mask = ~(full_mask if True else seq_id_mask)
    activations = activations[filter_mask]
    seq_ids = seq_ids[filter_mask]
    positions = positions[filter_mask]
    logit_effects = logit_effects[filter_mask]
