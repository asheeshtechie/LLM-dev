# LLM-dev
### Project structure:
```sh
llm-tokenizer-driver/
├── data/
│   └── original_text/          Directory for input text files
│   └── vocabulary/             Directory for keeping the vocabulary of basic tokenizer
├── do_tokenize.py              Tokenizer functions
├── do_download.py              book (text) download functions
├── LLM_driver.py               Main LLM driver 
├── README.md                   Project documentation
└── requirements.txt            Python dependencies
```

## llm_driver.py
Reads lines from a text file, uses a tokenizer to encode and/or decode tokens

### Usage:
```sh
llm_driver.py [-h] -f DATA_FILE -sl START_LINE -el END_LINE [-T TOKENIZER] [-e] [-d] [-pretty]

Tokenize text using different tokenizers.

options:
  -h, --help            show this help message and exit
  -f DATA_FILE, --data_file DATA_FILE
                        Name of the text data file in ./data/original_text/
  -sl START_LINE, --start_line START_LINE
                        Start line number (0-based index)
  -el END_LINE, --end_line END_LINE
                        End line number (0-based index, -1 to read until the end)
  -T, --tokenizer Tokenizer type (B,TIKTOKEN, BPE, WP, SP, ULM, BL-BPE, CHAR, T5)
  -e, --encode          Encode the input text
  -d, --decode          Decode the input token IDs
  -pretty, --pretty     Display line numbers with encoded/decoded text
```

### Example:
```sh
>python llm_driver.py -f book_1.txt -sl 100 -el 125 -T TIKTOKEN -e -d
```

## Tokenizer 
### do_tokenize.py 

Tool for tokenizing text using various tokenizers.
```sh
Usage: do_tokenize.py [-h] -T TOKENIZER [-e] [-d] [-s TEXT] [-t TOKEN_IDS [TOKEN_IDS ...]]

Tokenize text using different tokenizers.

options:
  -h, --help            show this help message and exit
  -T, --tokenizer       Tokenizer type (B,TIKTOKEN, BPE, WP, SP, ULM, BL-BPE, CHAR, T5)
  -e, --encode          Encode the input text
  -d, --decode          Decode the input token IDs
  -s TEXT, --text TEXT  Input text to encode (used with -e)
  -t TOKEN_IDS [TOKEN_IDS ...], --token_ids TOKEN_IDS [TOKEN_IDS ...]
                        List of token IDs to decode (used with -d)
```
### Example:
```sh
>python do_tokenize.py -T TIKTOKEN -e -s "Hello, world! This is a test"
Tokenizer: TIKTOKEN
Token IDs: [9906, 11, 1917, 0, 1115, 374, 264, 1296]

> python do_tokenize.py -T TIKTOKEN  -d -t 9906 11 1917 0 1115 374 264 1296 
Tokenizer: TIKTOKEN
Decoded Text: Hello, world! This is a test
```
