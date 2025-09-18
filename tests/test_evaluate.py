# python -m unittest tests.test_evaluate

import unittest
import os
import tempfile
import pandas as pd

from zscore.utils_evaluate import align, e_prf, z_eip
from zscore import tb
from zscore.utils_process_trees import extract_tokens

class TestEvaluateVariantClasses(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = self.tmpdir.name
        self.tree_path = os.path.join(self.base, "sw9999.mrg")

        self.tree_text = """((S
  (PRN
    (S (NP-SBJ (PRP I))
       (VP (VBP mean)))
    (, ,))
  (CC but)
  (EDITED
    (RM (-DFL- \\[))
    (S
      (NP-SBJ (PRP she))
      (VP-UNF (VBD was)
              (ADVP (RB truly))))
    (, ,)
    (IP (-DFL- \\+)))
  (NP-SBJ (PRP she))
  (VP (VBD was)
      (ADJP-PRD (RB truly)
                (RS (-DFL- -\\]))
                (JJ aware)))
  (. .)
  (-DFL- E_S)))"""
        with open(self.tree_path, "w") as f:
            f.write(self.tree_text)

        trees = tb.read_file(self.tree_path)
        _, _, token_tag_pairs = extract_tokens(trees[0], return_tags=True)
        self.disfluent_tokens = [token for token, _ in token_tag_pairs]
        self.tags = [tag for _, tag in token_tag_pairs]

        self.variant_classes = {
            "class_most_important": {
                "variants": [
                    "I mean but cats was truly aware.",
                    "I mean but cats was truly aware!",
                    "I mean but... cats was truly aware",
                    "I mean but. cats was truly aware.",
                    "I mean but cats, was truly aware.",
                    "I mean BUT cats, was truly aware."
                ],
                "skip_alignment_check": False,
                "expected_alignment": {
                    "gt_mask": [1,1,0,1,1,1,0,'*',0,0,0],
                    "pred_mask": [0,0,0,1,1,1,1,'*',0,0,0]
                }
            },
            "class_exact_same": {
                "variants": [
                    "I mean but she was truly she was truly aware."
                ],
                "skip_alignment_check": True
            },
            "class_repeats": {
                "variants": [
                    "I mean but she was truly she was truly she was truly aware."
                ],
                "skip_alignment_check": True
            },
            "class_halluciantions_after": {
                "variants": [
                    "I was truly aware.",
                    "I was truly aware of the situation.",
                    "I was truly aware of just how many cats there were in the situation.",
                ],
                "skip_alignment_check": True
            },
            "class_nonsense_strings": {
                "variants": [
                    ",,, idk idk idk idk idk,,, ",
                    "abc cd idk idk idk whatevs bruh",
                    "Error: unrecognized string!",
                    "how many software engineers does it take to change a lightbulb? none, that's a hardware problem"
                ],
                "skip_alignment_check": True
            },
            "class_2_strings": {
                "variants": [
                    "i mean but she was truly aware"
                ],
                "skip_alignment_check": True
            }
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def assert_mask_equal(self, expected, actual, label, class_name):
        if expected != actual:
            diffs = []
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    diffs.append(f"  Index {i}: expected {e}, got {a}")
            mismatch_msg = (
                f"\n{class_name} {label} mismatch:\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}\n"
                f"Differences:\n" + "\n".join(diffs)
            )
            raise AssertionError(mismatch_msg)

    def test_variant_class_consistency(self):
        for class_name, class_config in self.variant_classes.items():
            print(f"\nTesting {class_name}:")
            variants = class_config["variants"]
            skip_align_check = class_config.get("skip_alignment_check", False)
            expected_alignment = class_config.get("expected_alignment", None)

            metrics_list = []
            alignments = []
            rows = []
            first_alignment_printed = False

            for idx, generated in enumerate(variants):
                alignment = align(self.disfluent_tokens, self.tags, generated)

                if not first_alignment_printed:
                    print(f"\nAlignment from first sample in {class_name}: {generated}")
                    print(alignment.to_string(index=False))
                    first_alignment_printed = True

                e_p, e_r, e_f = e_prf(alignment)
                z_e, z_i, z_p = z_eip(alignment)

                alignments.append(alignment.reset_index(drop=True))
                metric_series = pd.Series({
                    "e_p": e_p, "e_r": e_r, "e_f": e_f,
                    "z_e": z_e, "z_i": z_i, "z_p": z_p
                })
                metrics_list.append(metric_series)
                rows.append({
                    "generated-text": generated,
                    **metric_series.to_dict()
                })

            df = pd.DataFrame(rows)
            print(f"\nSimulated eval__test.csv for {class_name}:")
            print(df.to_string(index=False))

            baseline_metrics = metrics_list[0]
            baseline_alignment = alignments[0]

            for idx, (metrics, alignment) in enumerate(zip(metrics_list, alignments)):
                pd.testing.assert_series_equal(
                    metrics, baseline_metrics,
                    check_exact=True,
                    check_names=False,
                    obj=f"Metrics differ in {class_name}, variant index {idx}"
                )

                if not skip_align_check:
                    pd.testing.assert_frame_equal(
                        alignment, baseline_alignment,
                        check_exact=True,
                        obj=f"Alignment differs in {class_name}, variant index {idx}"
                    )

                    # Optional expected mask check on first sample
                    if expected_alignment and idx == 0:
                        for col in ["gt_mask", "pred_mask"]:
                            if col in expected_alignment:
                                expected = expected_alignment[col]
                                actual = list(alignment[col])
                                self.assert_mask_equal(expected, actual, col, class_name)

if __name__ == "__main__":
    unittest.main()



