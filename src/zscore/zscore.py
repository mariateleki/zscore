# python -m zscore.zscore
import re
import os
import pandas as pd
from pathlib import Path
from icecream import ic
import fnmatch

from zscore.utils_evaluate import *
from zscore import tb
from zscore.utils_process_trees import extract_tokens, get_tree_file_path

def evaluate_file(file_path):
    df = pd.read_csv(file_path)

    metrics = {"e_p": [], "e_r": [], "e_f": [], "z_e": [], "z_i": [], "z_p": []}

    for _, row in df.iterrows():
        try:
            generated_text = str(row["generated-text"])
            file_id = row["filename"]  # e.g., 'sw2005.mrg'
            tree_file = get_tree_file_path(file_id)

            # Parse trees and extract tokens and tags
            trees = tb.read_file(tree_file)
            disfluent_tokens = []
            disfluent_tags = []

            for tree in trees:
                _, _, token_tag_pairs = extract_tokens(tree, return_tags=True)
                if token_tag_pairs:
                    tokens, tags = zip(*token_tag_pairs)
                    disfluent_tokens.extend(tokens)
                    disfluent_tags.extend(tags)

            # Run alignment and metric computation
            alignment = align(disfluent_tokens, disfluent_tags, generated_text)
            e_p, e_r, e_f = e_prf(alignment)
            z_e, z_i, z_p = z_eip(alignment)

        except Exception as e:
            print(f"Error processing row ({row.get('filename', 'unknown')}): {e}")
            e_p, e_r, e_f, z_e, z_i, z_p = [float("nan")] * 6

        metrics["e_p"].append(e_p)
        metrics["e_r"].append(e_r)
        metrics["e_f"].append(e_f)
        metrics["z_e"].append(z_e)
        metrics["z_i"].append(z_i)
        metrics["z_p"].append(z_p)

    for k, v in metrics.items():
        df[k] = v

    eval_path = os.path.join(os.path.dirname(file_path), "eval__" + os.path.basename(file_path))
    df.to_csv(eval_path, index=False)
    print(f"Saved evaluation to {eval_path}")
