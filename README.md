<div align="center">

# Coding practice 2024-2025:<br>Python, LeetCode and transformers

[![Applied in](https://img.shields.io/badge/Applied%20in-PDAC__paper-4E81BE?logo=github&logoColor=white&style=flat-square)](https://github.com/kahkengwong/PDAC_paper)
[![Applied in](https://img.shields.io/badge/Applied%20in-CEP--IP%20with%20CMRF%20in%20SEA--AD-4E81BE?logo=github&logoColor=white&style=flat-square)](https://github.com/kahkengwong/CEP-IP_with_CMRF_in_SEA-AD)
[![Pages](https://img.shields.io/badge/handwritten%20pages-45-555555?style=flat-square)](#contents)
[![Last commit](https://img.shields.io/github/last-commit/kahkengwong/Coding_practice_2024-2025?style=flat-square)](https://github.com/kahkengwong/Coding_practice_2024-2025/commits/main)

### Author: Kah Keng Wong [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7359--6202-A6CE39?logo=orcid&logoColor=white&style=flat-square)](https://orcid.org/0000-0001-7359-6202)

</div>

---

- These are photos of my handwritten Python coding practice in 2024 and 2025, covering LeetCode exercises and the QKV transformers.
- In 2024-2025, I wanted to increase my Python coding proficiency by working through coding exercises by hand.
- I applied this practice to my subsequent machine and statistical learning projects such as [PDAC_paper](https://github.com/kahkengwong/PDAC_paper) and [CEP-IP_with_CMRF_in_SEA-AD](https://github.com/kahkengwong/CEP-IP_with_CMRF_in_SEA-AD).
- I am depositing the pages here so that I also have a softcopy for my own reference.

---

## Contents

| Folder | Pages | Topics |
|---|---|---|
| [`python_leetcode/`](python_leetcode) | L00 to L12 | Python basics, prefix sums, Kadane's algorithm, hash maps and dictionaries (2024 notebook) |
| [`transformers/`](transformers) | T00 to T33 | A single-layer attention-only transformer in numpy, its QK and OV circuits, and skip-trigram analysis (2025 notebook) |

- L00 and T00 are the notebook covers. The other pages are numbered in the order they were written.
- [`page_order.csv`](page_order.csv) lists the original photo for each page.
- Each photo was brightened by 5%, and pages photographed sideways or upside down were rotated upright. The transformer photos were resized to 2000 px on the long side. No other changes were made.

---

## Python and LeetCode

<p align="center"><img src="python_leetcode/L00_cover_python_2024.jpg" width="40%"></p>

### 1. Python basics

`range`, loops with conditions and indices, and LeetCode 2706 Buy Two Chocolates by brute force (L01). Nested loops to build a pattern, `if`, `elif` and `while` with a password-attempt example, and a reminder to check edge cases (L02).

<img src="python_leetcode/L01_python_basics_loops_and_2706_buy_two_chocolates.jpg" width="49%"> <img src="python_leetcode/L02_python_basics_patterns_conditions_and_while_loops.jpg" width="49%">

### 2. Maximum subarray: prefix sums and Kadane's algorithm

LeetCode 53 Maximum Subarray solved with prefix sums, by tracking the smallest prefix sum seen so far and walking through nums = [1, 2, 3] step by step (L03). The prefix-sum definitions, including sum(i..j) = prefix_sum[j+1] &minus; prefix_sum[i] (L04).

<img src="python_leetcode/L03_53_maximum_subarray_prefix_sum.jpg" width="49%"> <img src="python_leetcode/L04_prefix_sum_definitions.jpg" width="49%">

The same problem with Kadane's algorithm. At each position the subarray either starts again at the current number or extends the previous subarray, whichever gives the larger sum (L05).

<p align="center"><img src="python_leetcode/L05_53_maximum_subarray_kadane.jpg" width="49%"></p>

### 3. Hash maps and dictionaries

LeetCode 1 Two Sum by brute force and with a hash map (L06), and notes on adding to and retrieving from a Python dictionary, as used in the hash map solution (L07). A modified Two Sum that returns all unique index pairs (L08), and LeetCode 217 Contains Duplicate with a set (L09).

<img src="python_leetcode/L06_1_two_sum.jpg" width="49%"> <img src="python_leetcode/L07_python_dictionary.jpg" width="49%">
<img src="python_leetcode/L08_modified_two_sum.jpg" width="49%"> <img src="python_leetcode/L09_217_contains_duplicate.jpg" width="49%">

LeetCode 242 Valid Anagram with a character-count dictionary (L10), and a step-by-step trace of `char_count` for s = "too" and t = "oto" (L11).

<img src="python_leetcode/L10_242_valid_anagram.jpg" width="49%"> <img src="python_leetcode/L11_242_valid_anagram_char_count.jpg" width="49%">

LeetCode 49 Group Anagrams, using the sorted letters of each word as the dictionary key (L12).

<p align="center"><img src="python_leetcode/L12_49_group_anagrams.jpg" width="49%"></p>

---

## Transformers

The concepts in this notebook follow Elhage et al. (2021), [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): the one-layer attention-only transformer, the QK and OV circuits, skip-trigrams and the direct path. The code is written in numpy.

<p align="center"><img src="transformers/T00_cover_attention_only_transformer_2025.jpg" width="40%"></p>

### 1. Worked example in numpy

Input embeddings for the three tokens "Quick", "brown" and "fox", and the Q and K matrices (T01). The scaled scores QK<sup>T</sup>/&radic;d<sub>k</sub> and the softmax, with the maximum subtracted first for numerical stability (T02). The V matrix, the attention output OV and the final output through W<sub>O</sub> (T03). Interpretation of the result: every token attends fully to "fox", so every output row is built from the value vector of "fox", and W<sub>O</sub> projects the result back to d<sub>model</sub> (T04).

<img src="transformers/T01_worked_example_steps_1_to_3_embeddings_Q_and_K.jpg" width="49%"> <img src="transformers/T02_worked_example_steps_4_5_QK_scores_and_softmax.jpg" width="49%">
<img src="transformers/T03_worked_example_steps_6_to_8_V_OV_and_output.jpg" width="49%"> <img src="transformers/T04_worked_example_interpretation.jpg" width="49%">

### 2. First modular version

The same steps as three functions: the QK matrix, the attention weights and the output (T05).

<p align="center"><img src="transformers/T05_modular_v1_levels_1_to_3.jpg" width="49%"></p>

### 3. QK and OV circuits and skip-trigram analysis

Notes on the shapes and indices of each matrix, the direction of influence in a skip-trigram, and the combined W<sub>QK</sub> and W<sub>OV</sub> matrices written with `np.einsum`. The notes pages face the code they explain.

> [!NOTE]
> **About `np.einsum`:** `np.einsum` (Einstein summation) writes a matrix operation by giving each dimension a letter. For example, `np.einsum('bsm,mk->bsk', x, W_Q)` gives the same result as `x @ W_Q`. Here b is the batch, s is the token position, m is the embedding feature and k is the query feature. The letter m appears in both inputs but not in the output, so it is summed over, and the other letters are kept.
>
> It is widely used in transformer interpretability because every dimension is named, so the code shows which ones are summed over without transposing or reshaping. It also combines weight matrices directly into the QK and OV circuits, and keeps the source and destination positions apart, so each token-to-token contribution can be read out (T13 and T15).

<img src="transformers/T06_skip_trigram_dimensions_and_direction_of_influence.jpg" width="49%"> <img src="transformers/T07_skip_trigram_transformer_circuit_code.jpg" width="49%">
<img src="transformers/T08_QK_circuit_and_embedding_notes.jpg" width="49%"> <img src="transformers/T09_tokenisation_and_index_notes.jpg" width="49%">

The einsum operations one by one, the embedding and unembedding matrices W<sub>E</sub> and W<sub>U</sub>, the direct path, and the causal mask that blocks attention to future tokens, alongside the Level 4 circuit code and the Level 5 skip-trigram code (T10 to T15).

<img src="transformers/T10_einsum_operations_1_to_4.jpg" width="49%"> <img src="transformers/T11_level_4_QK_and_OV_circuits_code.jpg" width="49%">
<img src="transformers/T12_einsum_operations_5_to_7_and_interpretability.jpg" width="49%"> <img src="transformers/T13_level_4_output_and_level_5_skip_trigram_code.jpg" width="49%">

<img src="transformers/T14_OV_circuit_direct_path_and_causal_mask_notes.jpg" width="49%"> <img src="transformers/T15_level_5_skip_trigram_code_and_notes.jpg" width="49%">

Explanations of three lines of the code: tuple unpacking with `_`, adding a dimension with `None` for broadcasting, and `np.nan_to_num` for numerical stability (T16).

<p align="center"><img src="transformers/T16_code_explanations.jpg" width="49%"></p>

### 4. Modular code, finalised

Levels 1 to 5 rewritten with type hints and documented arguments and return shapes (T17 to T20).

<img src="transformers/T17_modular_finalised_levels_1_2.jpg" width="49%"> <img src="transformers/T18_modular_finalised_level_3.jpg" width="49%">
<img src="transformers/T19_modular_finalised_level_4.jpg" width="49%"> <img src="transformers/T20_modular_finalised_levels_4_5.jpg" width="49%">

Level 6, which measures the influence of each source token as the norm of its skip-trigram contributions (T21).

<p align="center"><img src="transformers/T21_modular_finalised_level_6.jpg" width="49%"></p>

### 5. Final code

Levels 1 to 4 with shape checks, handling of 2D input and empty input, positional embeddings and the causal mask (T22 to T25).

<img src="transformers/T22_final_level_1_QK_matrix.jpg" width="49%"> <img src="transformers/T23_final_level_2_attention_weights.jpg" width="49%">
<img src="transformers/T24_final_level_3_attention_output.jpg" width="49%"> <img src="transformers/T25_final_level_4_transformer_circuit.jpg" width="49%">

The end of Level 4, Level 5 skip-trigram analysis, and the start of Level 6, which follows the paper's approach (T26 to T29).

<img src="transformers/T26_final_level_4_continued.jpg" width="49%"> <img src="transformers/T27_final_level_5_skip_trigram.jpg" width="49%">
<img src="transformers/T28_final_level_5_continued.jpg" width="49%"> <img src="transformers/T29_final_level_6_circuit_paper_approach.jpg" width="49%">

> Note. Some steps on T29 and T32 are crossed out and redone on the next page (T30 and T33).

Level 6 computes the vocabulary-level QK circuit W<sub>E</sub><sup>T</sup>W<sub>QK</sub>W<sub>E</sub> and OV circuit W<sub>E</sub>W<sub>OV</sub>W<sub>U</sub>, and the attention scores from the factorised QK circuit. Level 7 is the skip-trigram analysis in the paper's approach (T30 to T33).

<img src="transformers/T30_final_level_6_vocab_level_circuits.jpg" width="49%"> <img src="transformers/T31_final_level_6_continued.jpg" width="49%">
<img src="transformers/T32_final_level_7_skip_trigram_paper_approach.jpg" width="49%"> <img src="transformers/T33_final_level_7_continued.jpg" width="49%">

---

## Reference
Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html

---

## License
The pages in this repository are photos of my own notebooks. They are shared for reading. Please ask before reusing them.

---

## Contact
Kah Keng Wong: [kahkeng@usm.my](mailto:kahkeng@usm.my)
