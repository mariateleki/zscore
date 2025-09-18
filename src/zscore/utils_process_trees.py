import re
import os
import string
from icecream import ic

from zscore import tb # For parsing Penn Treebank format
from zscore.utils_dirs import * 

def get_tree_file_path(file_id, base_dir="data/treebank_3/parsed/mrg/swbd"):
    subdir = file_id[2]  # e.g., 'sw2005' → '2'
    return os.path.join(base_dir, subdir, file_id.replace('.txt','.mrg'))


# Extract terminal tokens from preterminal nodes, while skipping disfluency, speaker codes, and mumbles
def get_leaves_from_preterminals(tree):
    words = []

    # Identify and skip speaker label subtrees like (CODE (SYM SpeakerA1))
    def is_speaker_code(subtree):
        if isinstance(subtree, list) and len(subtree) >= 2 and subtree[0] == "CODE":
            for child in subtree[1:]:
                if isinstance(child, list) and len(child) == 2 and child[0] == "SYM" and child[1].startswith("Speaker"):
                    return True
        return False

    # Identify and skip unintelligible speech markers like (X (XX MUMBLEx))
    def is_mumble(subtree):
        if isinstance(subtree, list):
            if len(subtree) == 2 and subtree[1] == "MUMBLEx":
                return True
            if subtree[0] in ("XX", "X"):
                for child in subtree[1:]:
                    if isinstance(child, str) and child == "MUMBLEx":
                        return True
        return False

    # Traverse the tree recursively
    if isinstance(tree, list):
        if is_speaker_code(tree) or is_mumble(tree):
            return []  # Skip unwanted metadata or noise nodes
        # If it's a preterminal node with a real word, collect the word
        if len(tree) == 2 and isinstance(tree[1], str) and tree[0] not in ("-NONE-", "-DFL-"):
            words.append(tree[1])
        else:
            # Recursively process children
            for subtree in tree[1:]:
                words.extend(get_leaves_from_preterminals(subtree))

    return words

# Clean punctuation spacing and collapse extra whitespace
def clean_sentence(tokens):
    sentence = " ".join(tokens)

    # Remove unnecessary space before punctuation
    sentence = re.sub(r'\s+([.,!?])', r'\1', sentence)

    # Add space after punctuation if it's missing
    sentence = re.sub(r'([.,!?])(?=[^\s.,!?])', r'\1 ', sentence)

    # Replace multiple spaces with a single space
    sentence = re.sub(r'\s{2,}', ' ', sentence)

    return sentence.strip()

# Fix tokenized contractions into natural English form
def fix_contractions(sentence):
    fixes = {
        " n't": "n't", " 're": "'re", " 've": "'ve",
        " 'll": "'ll", " 'd": "'d", " 'm": "'m", " 's": "'s"
    }
    for k, v in fixes.items():
        sentence = sentence.replace(k, v)
    return sentence

# Run all postprocessing on a list of tokens
def postprocess_sentence(tokens):
    sentence = clean_sentence(tokens)
    sentence = fix_contractions(sentence)
    return sentence

def is_disfluent_node(label):
    return label in ("EDITED", "INTJ", "PRN")

def extract_tokens(tree, return_tags=False):
    # Lists to hold fluent and disfluent token outputs
    fluent_tokens = []
    disfluent_tokens = []
    token_tag_pairs = []  # Optional: (token, tag)

    # Utility to check if a label indicates a metadata node
    def is_metadata_node(label):
        return label in ("CODE", "SYM")

    # Utility to check if a token represents unintelligible speech
    def is_mumble_token(token):
        return token == "MUMBLEx"

    # Recursive helper to traverse the tree and collect tokens
    # Now also tracks the highest-level disfluent node label (EDITED, INTJ, PRN)
    def recurse(subtree, under_disfluent=False, disfluent_label=None):

        if isinstance(subtree, list):
            label = subtree[0] if subtree else ""

            # Skip entire subtree if it is a metadata node
            if is_metadata_node(label):
                return

            # Determine if current node is disfluent and capture top-level label
            if is_disfluent_node(label) and not under_disfluent:
                under_disfluent = True
                disfluent_label = label  # store the top-most disfluent label

            # Check if this is a preterminal node (label and a word)
            if len(subtree) == 2 and isinstance(subtree[1], str):
                token = subtree[1]
                if subtree[0] not in ("-NONE-", "-DFL-") and not is_mumble_token(token):

                    # Include in fluent output only if not under disfluent context
                    if not under_disfluent:
                        fluent_tokens.append(token)

                    # Always include in disfluent output
                    disfluent_tokens.append(token)

                    # Append tag for disfluent output if requested
                    if return_tags:
                        tag = disfluent_label if under_disfluent else "NONE"
                        if token not in string.punctuation:
                            token_tag_pairs.append((token, tag))
            else:
                # Recurse into children with updated disfluent status and top-level label
                for child in subtree[1:]:
                    recurse(child, under_disfluent, disfluent_label)

    recurse(tree)

    if return_tags:
        return fluent_tokens, disfluent_tokens, token_tag_pairs

    return fluent_tokens, disfluent_tokens


def correct_final_punctuation(text):
    # Remove leading punctuation errors like ",." or ",,"
    text = re.sub(r'([.,!?]){2,}', lambda m: m.group(0)[-1], text)  # Reduce repeated punctuations to the last one
    text = re.sub(r'([.,!?])\s*([.,!?])', r'\2', text)              # Fix sequences like ", ." or ",." to "."

    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)

    # Remove extra spaces
    text = text.strip()

    return text

def get_text_dual(trees):
    fluent_sentences = []
    disfluent_sentences = []

    sep_counter = 1
    for i, tree in enumerate(trees):
        fluent_tokens, disfluent_tokens = extract_tokens(tree)

        fluent_sentence = postprocess_sentence(fluent_tokens).replace('\n','')
        disfluent_sentence = postprocess_sentence(disfluent_tokens).replace('\n','')

        fluent_sentences.append(fluent_sentence)
        disfluent_sentences.append(disfluent_sentence)

        # Add special token every 4 trees (after the 4th, 8th, etc.)
        if (i + 1) % 4 == 0:
            sep_token = f"<SEP{sep_counter}>"
            fluent_sentences.append(sep_token)
            disfluent_sentences.append(sep_token)
            sep_counter += 1

    # form paragraph from sentences and perform post-processing
    fluent_text = ' '.join(fluent_sentences)
    fluent_text = correct_final_punctuation(fluent_text)

    disfluent_text = ' '.join(disfluent_sentences)
    disfluent_text = correct_final_punctuation(disfluent_text)

    return fluent_text, disfluent_text

def get_text_dual_from_file(tree_file):
    trees = tb.read_file(tree_file)
    return get_text_dual(trees)

def get_text_dual_from_string(tree_string):
    trees = tb.string_trees(tree_string)
    return get_text_dual(trees)
