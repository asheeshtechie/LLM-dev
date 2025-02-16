import argparse
import re
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
from tokenizers import Tokenizer
import sentencepiece as spm
import tiktoken 

# Top-level dictionary mapping tokenizer types to their encode/decode functions
TOKENIZERS = {
    "B": {
        "encode": lambda text: BasicTokenizer().encode(text),
        "decode": lambda token_ids: BasicTokenizer().decode(token_ids),
    },
    "TIKTOKEN": {
        "encode": lambda text: tiktoken.get_encoding("cl100k_base").encode(text),
        "decode": lambda token_ids: tiktoken.get_encoding("cl100k_base").decode(token_ids),
    },
    "BPE": {
        "encode": lambda text: Tokenizer.from_pretrained("gpt2").encode(text).ids,
        "decode": lambda token_ids: Tokenizer.from_pretrained("gpt2").decode(token_ids),
    },
    "WP": {
        "encode": lambda text: AutoTokenizer.from_pretrained("bert-base-uncased")(text)["input_ids"],
        "decode": lambda token_ids: AutoTokenizer.from_pretrained("bert-base-uncased").decode(token_ids),
    },
    "SP": {
        "encode": lambda text: spm.SentencePieceProcessor(model_file="spm.model").encode_as_ids(text),
        "decode": lambda token_ids: spm.SentencePieceProcessor(model_file="spm.model").decode_ids(token_ids),
    },
    "ULM": {
        "encode": lambda text: spm.SentencePieceProcessor(model_file="ulm.model").encode_as_ids(text),
        "decode": lambda token_ids: spm.SentencePieceProcessor(model_file="ulm.model").decode_ids(token_ids),
    },
    "BL-BPE": {
        "encode": lambda text: Tokenizer.from_pretrained("gpt2").encode(text).ids,
        "decode": lambda token_ids: Tokenizer.from_pretrained("gpt2").decode(token_ids),
    },
    "CHAR": {
        "encode": lambda text: [ord(char) for char in text],  # Use ASCII values as token IDs
        "decode": lambda token_ids: "".join([chr(token_id) for token_id in token_ids]),
    },
    "T5": {
        "encode": lambda text: AutoTokenizer.from_pretrained("t5-small")(text)["input_ids"],
        "decode": lambda token_ids: AutoTokenizer.from_pretrained("t5-small").decode(token_ids),
    },
}

# Basic Tokenizer Class
class BasicTokenizer:
    def __init__(self):
        self.vocab = []  # Vocabulary (list of tokens)
        self.token_to_id = {}  # Token to ID mapping
        self.id_to_token = {}  # ID to token mapping
        self.vocab_file = "./data/vocabulary/basictokenizer_vocab.json"
        
        # Add special tokens to the vocabulary
        self.special_tokens = {
            "<|BOS|>": "Beginning of Sequence",
            "<|EOS|>": "End of Sequence",
        }
        
        # Ensure the directory exists
        vocab_dir = os.path.dirname(self.vocab_file)
        os.makedirs(vocab_dir, exist_ok=True)
        
        # Load vocabulary from file if it exists
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, "r") as f:
                self.vocab = json.load(f)
                self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
                self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        else:
            # Initialize vocabulary with special tokens
            for token in self.special_tokens:
                self.vocab.append(token)
                self.token_to_id[token] = len(self.vocab) - 1
                self.id_to_token[len(self.vocab) - 1] = token
            self.save_vocab()  # Save the initial vocabulary

    def save_vocab(self):
        """
        Save the current vocabulary to the vocabulary file.
        """
        with open(self.vocab_file, "w") as f:
            json.dump(self.vocab, f)

    def encode(self, text, vocab_update=True):
        """
        Encode text into token IDs using a basic regex-based tokenizer.
        If vocab_update is True, update the vocabulary file with new tokens.
        """
        # Add <|BOS|> at the start and <|EOS|> at the end
        tokens = ["<|BOS|>"]
        
        # Split text using regex
        text_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove empty strings and strip whitespace
        text_tokens = [item.strip() for item in text_tokens if item.strip()]
        tokens.extend(text_tokens)
        
        # Add <|EOS|> at the end
        tokens.append("<|EOS|>")
        
        # Update vocabulary and token-to-ID mappings
        token_ids = []
        new_tokens = []
        for token in tokens:
            if token not in self.token_to_id:
                self.vocab.append(token)
                self.token_to_id[token] = len(self.vocab) - 1
                self.id_to_token[len(self.vocab) - 1] = token
                new_tokens.append(token)
            token_ids.append(self.token_to_id[token])
        
        # Update vocabulary file if new tokens were added
        if vocab_update and new_tokens:
            self.save_vocab()
        
        return token_ids

    def decode(self, token_ids):
        """
        Decode token IDs back into text using the vocabulary.
        Special tokens (<|BOS|> and <|EOS|>) are preserved in the output.
        """
        tokens = [self.id_to_token.get(token_id, "[UNK]") for token_id in token_ids]
        return "".join(tokens)




def encode_text(text, tokenizer_type):
    """
    Encode the input text using the specified tokenizer.
    """
    if tokenizer_type not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available options: {list(TOKENIZERS.keys())}")
    
    encode_func = TOKENIZERS[tokenizer_type]["encode"]
    return encode_func(text)

def decode_text(token_ids, tokenizer_type):
    """
    Decode the input token IDs using the specified tokenizer.
    """
    if tokenizer_type not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Available options: {list(TOKENIZERS.keys())}")
    
    decode_func = TOKENIZERS[tokenizer_type]["decode"]
    return decode_func(token_ids)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Tokenize text using different tokenizers.")
    parser.add_argument("-T", "--tokenizer", type=str, required=True,
                        help="Tokenizer type (B, BPE, WP, SP, ULM, BL-BPE, CHAR, T5)")
    parser.add_argument("-e", "--encode", action="store_true", help="Encode the input text")
    parser.add_argument("-d", "--decode", action="store_true", help="Decode the input token IDs")
    parser.add_argument("-s", "--text", type=str, help="Input text to encode (used with -e)")
    parser.add_argument("-t", "--token_ids", nargs="+", type=int, help="List of token IDs to decode (used with -d)")
    args = parser.parse_args()

    # Validate arguments
    if args.encode and args.decode:
        raise ValueError("Cannot use both -e and -d at the same time.")
    if not args.encode and not args.decode:
        raise ValueError("Must specify either -e (encode) or -d (decode).")
    if args.encode and not args.text:
        raise ValueError("Text to encode (-s) is required for encoding.")
    if args.decode and not args.token_ids:
        raise ValueError("Token IDs to decode (-t) are required for decoding.")

    # Perform encoding or decoding
    try:
        if args.encode:
            token_ids = encode_text(args.text, args.tokenizer)
            print(f"Tokenizer: {args.tokenizer}")
            print(f"Token IDs: {token_ids}")
        elif args.decode:
            decoded_text = decode_text(args.token_ids, args.tokenizer)
            print(f"Tokenizer: {args.tokenizer}")
            print(f"Decoded Text: {decoded_text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()