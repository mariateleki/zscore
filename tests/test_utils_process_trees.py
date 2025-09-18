import unittest
import re

from zscore.utils_process_trees import *
from zscore.utils_dirs import *

class TestProcessingPipeline(unittest.TestCase):

    def test_get_sentences_from_tree_end_of_text(self):
        tree_string_1 = """
                    ( (CODE (SYM SpeakerA53) (. .) ))
                    ( (S 
                        (INTJ (UH Uh) )
                        (, ,) 
                        (NP-SBJ (PRP I) )
                        (VP (VBD was) (RB n't) 
                        (ADVP (RB really) )
                        (VP (VBG keeping) 
                            (NP (NN count) )))
                        (. .) (-DFL- E_S) ))
                    ( (S (CC But) 
                        (NP-SBJ (PRP I) )
                        (VP (VBP guess) )
                        (, ,) (-DFL- E_S) ))
                    ( (INTJ (UH good-bye) (. .) (-DFL- E_S) ))
                    ( (CODE (SYM SpeakerB54) (. .) ))
                    ( (INTJ (UH Yeah) 
                        (, ,)
                        (-DFL- E_S) ))
                    ( (INTJ (UH okay) 
                        (, ,)
                        (-DFL- E_S) ))
                    ( (INTJ (UH bye-bye) (. .) (-DFL- E_S) ))
                    ( (CODE (SYM SpeakerA55) (. .) ))
                    ( (INTJ (UH Bye) (. .) (-DFL- E_S) ))
                    """
        fluent_result, disfluent_result = get_text_dual_from_string(tree_string_1)
        self.assertEqual(fluent_result, ", I wasn't really keeping count. But I guess, <SEP1> <SEP2>")
        self.assertEqual(disfluent_result, "Uh, I wasn't really keeping count. But I guess, good-bye. <SEP1> Yeah, okay, bye-bye. <SEP2> Bye.")

    def test_get_sentences_from_tree_weird_T_tokens(self):
        tree_string_2 = """
                ( (CODE (SYM SpeakerA1) (. .) ))
                ( (SBARQ 
                    (WHADVP-1 (WRB How) )
                    (SQ (VBP do) 
                    (NP-SBJ (PRP you) )
                    (VP (VB feel) 
                        (PP (IN about) 
                        (NP (DT the) (NNP Viet) (NNP Nam) (NN war) ))
                        (ADVP-MNR (-NONE- *T*-1) )))
                    (. ?) (-DFL- E_S) ))
                ( (CODE (SYM SpeakerB2) (. .) ))
                """
        fluent_result, disfluent_result = get_text_dual_from_string(tree_string_2)
        self.assertEqual(fluent_result, "How do you feel about the Viet Nam war?")
        self.assertEqual(disfluent_result, "How do you feel about the Viet Nam war?")

    def test_get_leaves_from_preterminals_regular(self):
        tree = ['S', ['NP', ['PRP', 'I']], ['VP', ['VBP', 'go']]]
        result = get_leaves_from_preterminals(tree)
        self.assertEqual(result, ['I', 'go'])

    def test_get_leaves_skips_speaker(self):
        tree = ['CODE', ['SYM', 'SpeakerA1'], ['.', '.']]
        result = get_leaves_from_preterminals(tree)
        self.assertEqual(result, [])

    def test_exclude_mumble(self):
        tree = ['S', ['NP', ['PRP', 'I']], ['VP', ['VBP', 'saw'], ['NP', ['XX', 'MUMBLEx']]]]
        fluent, disfluent = extract_tokens(tree)
        self.assertNotIn('MUMBLEx', fluent)
        self.assertNotIn('MUMBLEx', disfluent)
        self.assertEqual(fluent, ['I', 'saw'])
        self.assertEqual(disfluent, ['I', 'saw'])

    def test_get_leaves_skips_mumble(self):
        tree = ['X', ['XX', 'MUMBLEx']]
        result = get_leaves_from_preterminals(tree)
        self.assertEqual(result, [])

    def test_clean_sentence_spacing(self):
        tokens = ['I', 'saw', 'it', ',', 'okay', '.', 'Bye', '!']
        cleaned = clean_sentence(tokens)
        self.assertEqual(cleaned, 'I saw it, okay. Bye!')

    def test_fix_contractions(self):
        text = "I 'm happy you 're here and I ca n't wait."
        fixed = fix_contractions(text)
        self.assertEqual(fixed, "I'm happy you're here and I can't wait.")

    def test_postprocess_sentence_combined(self):
        tokens = ['I', 'ca', "n't", 'do', 'that', ',']
        result = postprocess_sentence(tokens)
        self.assertEqual(result, "I can't do that,")

    def test_dual_output_with_edited(self):  # too short to have <SEPn> inserted
            tree_string = """
                ( (S
                    (NP (PRP I))
                    (VP (VBP think)
                        (EDITED (NP (PRP she)))
                        (SBAR (IN that)
                            (S (NP (PRP he)) (VP (VBD left)))))
                    (. .)))
            """
            fluent, disfluent = get_text_dual_from_string(tree_string)
            self.assertEqual(fluent, "I think that he left.")
            self.assertEqual(disfluent, "I think she that he left.")

    def test_dual_output_with_intj_prn(self):
        tree_string = """
            ( (S
                (INTJ (UH well))
                (NP (PRP I))
                (VP (VBP guess)
                    (PRN (S (NP (PRP you)) (VP (VBP know))))
                    (SBAR (IN that) (S (NP (PRP he)) (VP (VBD left)))))
                (. .)))
        """
        fluent, disfluent = get_text_dual_from_string(tree_string)
        self.assertEqual(fluent, "I guess that he left.")
        self.assertEqual(disfluent, "well I guess you know that he left.")

    def test_sep_token_alignment(self):
        # Sample input: 8 trees (enough for 2 separators if inserted every 4 trees)
        example_trees = "(S (NP (NN Hello))) (S (NP (NN world))) (S (NP (NN test))) (S (NP (NN trees))) (S (NP (NN more))) (S (NP (NN data))) (S (NP (NN for))) (S (NP (NN checking)))"

        fluent_text, disfluent_text = get_text_dual_from_string(example_trees)

        # Match all <SEPn> tokens
        fluent_seps = re.findall(r"<SEP\d+>", fluent_text)
        disfluent_seps = re.findall(r"<SEP\d+>", disfluent_text)

        # Assert same number of sep tokens
        self.assertEqual(len(fluent_seps), len(disfluent_seps), "Mismatch in number of SEP tokens between fluent and disfluent")

        # Optionally: assert the token values match exactly
        self.assertEqual(fluent_seps, disfluent_seps, "SEP tokens differ in order or naming")

    def test_get_text_dual_from_file_sample_mrg(self):
        fluent_text, disfluent_text = get_text_dual_from_file(TREEBANK_SAMPLE_MRG_FILE)
        self.maxDiff = None

        sw4004_fluent = """How do you feel about the Viet Nam war, I guess it's pretty deep feelings, <SEP1> I just, went back and rented, the movie VIET NAM <SEP2> and. I saw that as well. <SEP3> Got, some insight there, to kind of help me put together the feelings. I really appreciated the whole, English class where, the, fellow just wouldn't do it, the guy's gouging your eyes out, <SEP4> what are you going to do? <SEP5> what for him to finish me off. And, it was, good to remember that kind of Asian philosophy that. <SEP6>, were you ever in Viet Nam <SEP7> or, <SEP8> I was kind of an in-between, finally drew a high draft number, and you? <SEP9>, I was much too young, I was born in sixty-seven, so. <SEP10>, both my brothers were, draft age, <SEP11> but neither of them wound up going over, which, I think they were very happy for, personally, I just went in limbo. I had a passport and was ready to go, out of the country or join special forces, <SEP12> either one. <SEP13> <SEP14> I just didn't know. So. <SEP15>, so do you feel that it was worth what we did over there? <SEP16> just a second. <SEP17>, Mark, what was that again? <SEP18>, do you think, the investment in lives and money was worth it? <SEP19> I totally agree with that. <SEP20> what effects do you think it's had on our country? <SEP21> Downside, the says we should, go into the grief that's there and presidents have always avoided that as a country. <SEP22> So it's pretty serious, really, lot of things that aren't being addressed. <SEP23> I think that's pretty typical of the entire involvement over that nothing was really addressed, it was never <SEP24> we announced that we were going to war, it was such a gradual and subtle increasement of force that. <SEP25> Gulf of Tonkin, resolution and was it a dolphin or a torpedo. <SEP26> You remember that? I vaguely remember we had a, spy ship torpedoed or something. <SEP27> only it was foggy and finally President Johnson said, they weren't really sure whether it was a dolphin or a torpedo. <SEP28> Isn't that something? <SEP29> so, do you think, for example in the Persian Gulf war, that it seemed to me that Bush was going to extraordinary lengths to, prepare the country for war. <SEP30> <SEP31> Mark I've got to go. <SEP32> We'll see you. <SEP33> I guess our five minutes are up according to me. Are they to you, I wasn't really keeping count. <SEP34> But I guess, <SEP35> <SEP36>"""
        sw4004_disfluent = """How do you feel about the Viet Nam war? Huh, well, um, you know, I guess it's pretty deep feelings, uh, <SEP1> I just, uh, went back and rented, uh, the movie, what is it, GOOD MORNING VIET NAM Uh-huh. <SEP2> and, uh. Yeah, I saw that as well. <SEP3> Got that, uh, uh, some insight there, to, to kind of help me put together the feelings. I really appreciated the, the whole, uh, English class where the, uh, the, uh, fellow just wouldn't do it, you know, the guy's gouging, gouging your eyes out, <SEP4> what are you going to do? Uh-huh. <SEP5> You know, what for him to finish me off. And, uh, it, it was, uh, good to remember the, uh, that, that kind of Asian philosophy that, uh. <SEP6> Uh-huh. Well, were you ever in Viet Nam <SEP7> or, No, <SEP8> no, I was kind of an in-between, uh, finally drew a high draft number, and you? <SEP9> Um, I was much too young, I was born in sixty-seven, so. <SEP10> Oh. Um, you know, both my, well, both my brothers were, um, draft age, <SEP11> but neither of them wound up going over, which, I think they were very happy for. Well, I, personally, uh, you know, uh, I just went in limbo. I had a passport and was ready to go or um, out of the country or join special forces, <SEP12> either one. Uh-huh. <SEP13> Yeah. <SEP14> Uh-huh. I mean, I just didn't know. So, uh. <SEP15> Well, um, so well do you, do you feel that it was worth what we did over there? Um, yeah, <SEP16> just a second. Okay. <SEP17> Okay. Sure, now. Well, Mark, um, what was that again? <SEP18> Um, do you think, I mean, do you think, our invest-, the investment in lives and money was worth it? No, not, not really. <SEP19> I totally agree with that. Um. <SEP20> Um. What, what effects do you think it's had on our country? <SEP21> Downside. Um, uh, well, the says we should, uh, go into the grief that, that's there and, you know, presidents have always avoided that as a country. <SEP22> Uh-huh. So it's pretty serious, really, you know, lot of things that aren't being addressed. <SEP23> Uh-huh. I think you know, that's pretty typical that, of the entire, entire involvement over that, you know, that nothing was really addressed, it was-, it wasn't, you know, it was never <SEP24> we, we announced that we were going to war, it was such a gradual and subtle, you know thi-, um, you know increasement of, of force that. Yeah. <SEP25> Gulf of Tonkin, uh, resolution and was it a dolphin or a torpedo. <SEP26> You remember that? I vaguely remember we, um, we had a, we had a, um, spy ship torpedoed or something. <SEP27> Yeah, yeah, only only it was foggy and finally President Johnson said, well, they're weren't really sure whether it was a dolphin or a torpedo. <SEP28> Oh. Isn't that something? <SEP29> Uh-huh. Um, so, um, do, do, do you think that like, uh, um, for example like in, in this past war, in the Persian Gulf war that, uh, that you see, it seemed to me that, that Bush was going, going to extraordinary lengths to, um, you know, prepare the country for war. <SEP30> Uh-huh. <SEP31> Hey Mark I've got to go, um. Yeah. <SEP32> Okay. We'll see you. <SEP33> I guess our five minutes are up according to me. Are they to you? Uh, I wasn't really keeping count. <SEP34> But I guess, good-bye. Yeah, <SEP35> okay, bye-bye. Bye. <SEP36>"""

        self.assertEqual(fluent_text, sw4004_fluent)
        self.assertEqual(disfluent_text, sw4004_disfluent)

    def test_expected_mrg_file_splits(self):
        mrg_root = TREEBANK_MRG_DIR
        assert mrg_root.exists(), f"Missing treebank MRG path: {mrg_root}"

        split_patterns = {
            "train": re.compile(r"^sw[23]\d+\.mrg$"),
            "dev": re.compile(r"^sw4[5-9]\d+\.mrg$"),
            "test": re.compile(r"^sw4[0-1]\d+\.mrg$"),
        }

        split_counts = {"train": 0, "dev": 0, "test": 0}
        unmatched_files = []

        for f in mrg_root.rglob("sw*.mrg"):
            name = f.name
            matched = False
            for split, pattern in split_patterns.items():
                if pattern.match(name):
                    split_counts[split] += 1
                    matched = True
                    break
            if not matched:
                unmatched_files.append(name)

        print("\n[test_utils_process_trees]")
        for split, count in split_counts.items():
            print(f"{split.capitalize()}: {count}")

        print(f"\nUnmatched files: {len(unmatched_files)} (not assigned to any split)")
        print(f"650 total – 496 train – 52 dev – 63 test = 39 unmatched is correct number")
        if unmatched_files:
            print("Examples:", unmatched_files[:5])

if __name__ == '__main__':
    unittest.main()
