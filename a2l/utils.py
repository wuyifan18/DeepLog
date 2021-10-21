# utils.py

def read_vocab(file):
    # Function to read a text file of log keys and returns a dict
    # Args:
    #   - file: str

    # read vocab
    with open(file) as file:
        vocabs = file.read().split('\n')

    # split pair of enumerated_event_id and raw_event_id
    # parse to dict
    vocab_dict = {}
    for pair in vocabs:
        pair = pair.split(' ') # split by space
        vocab_dict[pair[0]] : pair[-1]
    return vocab_dict