from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple

import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

#  constants 
TOKENIZER = TreebankWordTokenizer()
DISFLUENCY_CLASSES = ("EDITED", "INTJ", "PRN")  # order matters for z_eip

def build_alignment_df(d_tok, tags, g_tok):
    """
    Return a DataFrame with aligned tokens and masks.

    Columns:
        w_d, w_t, w_g : original/disfluent token, its tag, generated token
        gt_mask   : 1 if token *should* be removed, 0 if kept, "*" padding
        pred_mask     : 1 if model *removed* token, 0 if kept,  "*" padding
    """
    # Alter disfluent tokens so SequenceMatcher aligns them late

    # special token in g_tok_prime forces disfluent (ONLY disfluent) g_tokens into "replace" block (instead of "equal" block) for late matching
    g_tok_prime = [w if t == "NONE" else f"{w}§{t}" for w, t in zip(d_tok, tags, strict=True)]

    # match g_tok_prime & g_tok, but actually write with d_tok & g_tok
    sm = SequenceMatcher(None, g_tok_prime, g_tok, autojunk=False)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":  #  exact match with non-disfluent g_tokens
            for k in range(i2 - i1):
                rows.append((d_tok[i1 + k], tags[i1 + k], g_tok[j1 + k]))

        elif tag == "delete":  # missing g_tokens
            for k in range(i2 - i1):
                rows.append((d_tok[i1 + k], tags[i1 + k], ""))

        elif tag == "insert":  # hallucinated g_tokens
            for k in range(j2 - j1):
                rows.append(("", "", g_tok[j1 + k]))

        elif tag == "replace":  # TWO CASES:

            # 1ST CASE IN REPLACE: disfluent g_tokens
            # happens because SequenceMatching occurs between g_tok_prime & g_tok, not d_tok & g_tok

            # a dictionary that counts how many times each g_tok appears
            inserted = {tok: g_tok[j1:j2].count(tok) for tok in set(g_tok[j1:j2])}
            for k in range(i2 - i1):
                tok, lab = d_tok[i1 + k], tags[i1 + k]
                if inserted.get(tok, 0):  # match non-extras to DISFLUENT_TYPEs
                    rows.append((tok, lab, tok))
                    inserted[tok] -= 1
                else:                     # match extras to NONEs
                    rows.append((tok, lab, ""))

            # 2ND CASE IN REPLACE: hallucinated g_tokens
            for tok, cnt in inserted.items():
                rows.extend([("", "", tok)] * cnt)

    df = pd.DataFrame(rows, columns=["w_d", "w_t", "w_g"])

    # mask columns 
    gt_mask = []
    pred_mask = []
    tp_mask = []
    tn_mask = []
    fp_mask = []
    fn_mask = []

    for _, row in df.iterrows():
        w_d = row["w_d"]
        w_t = row["w_t"]
        w_g = row["w_g"]
        
        # gt_mask: should we remove it or not?
        gt_mask_current = ""
        if w_t in {"EDITED","PRN","INTJ"}:
            gt_mask_current = 1
        elif w_t == "NONE":
            gt_mask_current = 0
        else:
            gt_mask_current = "*"
        gt_mask.append(gt_mask_current)
        
        # pred_mask: did we remove it or not?
        pred_mask_current = ""
        if (w_d == ""):
            pred_mask_current = "*"  # hallucinated token
        elif w_d != w_g:
            pred_mask_current = 1
        else:
            pred_mask_current = 0
        pred_mask.append(pred_mask_current)
        
        
        if (pred_mask_current == "*") and (gt_mask_current == "*"):
            tp_mask.append("*")
            tn_mask.append("*")
            fp_mask.append("*")
            fn_mask.append("*")
        else: 

            # tp
            if (pred_mask_current == 1) and (gt_mask_current == 1):
                tp_mask.append(1)
            else:
                tp_mask.append(0)
            
            # tn
            if (pred_mask_current == 0) and (gt_mask_current == 0):
                tn_mask.append(1)
            else:
                tn_mask.append(0)
            
            # fp
            if (pred_mask_current == 1) and (gt_mask_current == 0):
                fp_mask.append(1)
            else:
                fp_mask.append(0)
            
            # fn
            if (pred_mask_current == 0) and (gt_mask_current == 1):
                fn_mask.append(1)
            else:
                fn_mask.append(0)
        
    df["gt_mask"] = gt_mask
    df["pred_mask"] = pred_mask
    df["tp_mask"] = tp_mask
    df["tn_mask"] = tn_mask
    df["fp_mask"] = fp_mask
    df["fn_mask"] = fn_mask
    return df

def align(disfluent_tokens, disfluent_tags, generated_text):
    if len(disfluent_tokens) != len(disfluent_tags):
        raise ValueError(
            f"tag_list length {len(disfluent_tokens)} ≠ token count {len(disfluent_tags)}"
        )
    
    # clean and tokenize generated_text
    def remove_selected_punctuation(text, punctuation=",."):
        return text.translate(str.maketrans('', '', punctuation))
    generated_text = remove_selected_punctuation(generated_text, punctuation=",.!?")
    g_tok = TOKENIZER.tokenize(generated_text)

    # lower both token lists
    g_tok = [w.lower() for w in g_tok]
    disfluent_tokens = [w.lower() for w in disfluent_tokens]

    # build the alignment df
    alignment_df = build_alignment_df(disfluent_tokens, disfluent_tags, g_tok)

    return alignment_df


def e_prf(alignment_df):
    df = alignment_df

    tp = float((df["tp_mask"] == 1).sum())
    tn = float((df["tn_mask"] == 1).sum())
    fp = float((df["fp_mask"] == 1).sum())
    fn = float((df["fn_mask"] == 1).sum())

    e_p = tp / (tp + fp) if tp + fp else float("nan")
    e_r = tp / (tp + fn) if tp + fn else float("nan")
    e_f = 2 * e_p * e_r / (e_p + e_r) if e_p + e_r else float("nan")
    return e_p, e_r, e_f


def z_eip(alignment_df):
    """
    Per-class removal *rate* for EDITED, INTJ, PRN on this example.

    z_e : EDITED removal rate
    z_i : INTJ removal rate
    z_p : PRN removal rate
    """
    df = alignment_df
    rates = []
    for lab in DISFLUENCY_CLASSES:
        total = int((df["w_t"] == lab).sum())
        removed = int(((df["w_t"] == lab) & (df["pred_mask"] == 1)).sum())
        rates.append(removed / total if total else float("nan"))
    # Order: EDITED → INTJ → PRN
    return tuple(rates)  # type: ignore[return-value]
