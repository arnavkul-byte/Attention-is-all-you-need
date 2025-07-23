import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
  def __init__(self, ds, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len):
    super().__init__()
    self.seq_len = seq_len
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_target = tokenizer_target
    self.src_lang = src_lang
    self.target_lang = target_lang

    print("BilingualDataset constructor called - checking if this print appears!")

    # Debugging print statements
    print(f"tokenizer_src.token_to_id('[SOS]'): {type(tokenizer_src.token_to_id('[SOS]'))}, {tokenizer_src.token_to_id('[SOS]')}")
    print(f"tokenizer_src.token_to_id('[EOS]'): {type(tokenizer_src.token_to_id('[EOS]'))}, {tokenizer_src.token_to_id('[EOS]')}")
    print(f"tokenizer_src.token_to_id('[PAD]'): {type(tokenizer_src.token_to_id('[PAD]'))}, {tokenizer_src.token_to_id('[PAD]')}")


    # Corrected lines: Pass token as a string, not a list
    self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    # Ensure these tokens exist in the tokenizer vocabulary
    if self.sos_token.item() is None:
        raise ValueError("Token '[SOS]' not found in source tokenizer vocabulary.")
    if self.eos_token.item() is None:
        raise ValueError("Token '[EOS]' not found in source tokenizer vocabulary.")
    if self.pad_token.item() is None:
        raise ValueError("Token '[PAD]' not found in source tokenizer vocabulary.")


  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index):
    src_target_pair = self.ds[index]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.target_lang]

    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_target.encode(tgt_text).ids

    encoder_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    decoder_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

    if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
      raise ValueError('Sentence is too long')

    encoder_input = torch.cat([
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
    ], dim=0)

    decoder_input = torch.cat([
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
    ], dim=0)

    label = torch.cat([
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
    ], dim=0)

    encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
    decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "label": label,
        "src_text": src_text,
        "tgt_text": tgt_text
    }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0