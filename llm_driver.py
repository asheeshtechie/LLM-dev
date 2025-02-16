import argparse
import os
from pathlib import Path

# Import tokenizer functions from the previous script
from do_tokenize import encode_text, decode_text

def read_lines_from_file(file_path, start_line, end_line):
    """
    Read lines from a text file between start_line and end_line.
    If end_line is -1, read until the end of the file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Adjust start_line and end_line based on file length
    if start_line < 0:
        start_line = 0
    if end_line == -1 or end_line >= len(lines):
        end_line = len(lines)
    
    # Ensure start_line is less than end_line
    if start_line >= end_line:
        return []
    
    # Read the specified lines
    selected_lines = lines[start_line:end_line]
    return [line.strip() for line in selected_lines if line.strip()]

def encode_data(lines, tokenizer_type):
    """
    Encode a list of text lines using the specified tokenizer.
    """
    encoded_data = []
    for line in lines:
        token_ids = encode_text(line, tokenizer_type)
        encoded_data.append(token_ids)
    return encoded_data

def decode_data(encoded_data, tokenizer_type):
    """
    Decode a list of token IDs using the specified tokenizer.
    """
    decoded_data = []
    for token_ids in encoded_data:
        text = decode_text(token_ids, tokenizer_type)
        decoded_data.append(text)
    return decoded_data

def display(data, data_type, start_line, pretty=False):
    """
    Display data (encoded, decoded, or lines) with optional pretty formatting.
    """
    if pretty:
        print(f"{data_type}:")
        for i, item in enumerate(data):
            print(f"Line {i + start_line}: {item}")
    else:
        print(f"{data_type}: {data}")

def command_line_parsing():
    """
    Parse command-line arguments.
    If no arguments are passed, display the help message and exit.
    """
    parser = argparse.ArgumentParser(description="Tokenize text using different tokenizers.")
    parser.add_argument("-f", "--data_file", type=str, required=True,
                        help="Name of the text data file in ./data/original_text/")
    parser.add_argument("-sl", "--start_line", type=int, required=True,
                        help="Start line number (0-based index)")
    parser.add_argument("-el", "--end_line", type=int, required=True,
                        help="End line number (0-based index, -1 to read until the end)")
    parser.add_argument("-T", "--tokenizer", type=str, required=False,
                        help="Tokenizer type (B, TIKTOKEN)")
    parser.add_argument("-e", "--encode", action="store_true", help="Encode the input text")
    parser.add_argument("-d", "--decode", action="store_true", help="Decode the input token IDs")
    parser.add_argument("-pretty", "--pretty", action="store_true", help="Display line numbers with encoded/decoded text")

    # If no arguments are passed, show the help message and exit
    if len(os.sys.argv) == 1:
        parser.print_help()
        os.sys.exit(0)

    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = command_line_parsing()

    # Construct the full file path
    data_dir = "./data/original_text/"
    file_path = os.path.join(data_dir, args.data_file)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    # Read lines from the file
    lines = read_lines_from_file(file_path, args.start_line, args.end_line)
    print(f"Read {len(lines)} lines from the file.")

    # If no -T, -e, or -d options are provided, just display the lines
    if not args.tokenizer and not args.encode and not args.decode:
        display(lines, "Lines read", args.start_line, args.pretty)
    else:
        # Perform encoding if requested
        if args.encode:
            encoded_data = encode_data(lines, args.tokenizer)
            display(encoded_data, "Encoded Data", args.start_line, args.pretty)

            # Perform decoding if requested
            if args.decode:
                decoded_data = decode_data(encoded_data, args.tokenizer)
                display(decoded_data, "Decoded Data", args.start_line, args.pretty)
        elif args.decode:
            # If only decoding is requested, assume the input is already encoded
            encoded_data = [list(map(int, line.split())) for line in lines]
            decoded_data = decode_data(encoded_data, args.tokenizer)
            display(decoded_data, "Decoded Data", args.start_line, args.pretty)

if __name__ == "__main__":
    main()