from torcheval.metrics import BLEUScore
from nltk.translate import meteor_score
import re
import string
import numpy as np
from collections import Counter

# nltk.download("wordnet")



def word_overlap(ground_truth: list[str] | str, predicted: list[str] | str) -> float:
    if isinstance(ground_truth, str) and isinstance(predicted, str):
        ground_truth = set(ground_truth.split(" "))
        predicted = set(predicted.split(" "))
        assert len(ground_truth) > 0, "Ground truth set is empty"
        return len(ground_truth.intersection(predicted)) / len(ground_truth)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        avg_word_overlap = 0
        n_words = 0
        for gt_text, pred_text in zip(ground_truth, predicted):
            gt_text = set(gt_text.split(" "))
            pred_text = set(pred_text.split(" "))
            assert len(gt_text) > 0, "Ground truth set is empty"
            n_words += len(gt_text)
            avg_word_overlap += len(gt_text.intersection(pred_text))
        return avg_word_overlap / n_words


def get_bleu_score(
    ground_truth: list[str] | str,
    predicted: list[str] | str,
    avg: bool = False,
    trunc: bool = False,
) -> float:
    if not avg:
        metric = BLEUScore(n_gram=4)
        if isinstance(ground_truth, str) and isinstance(predicted, str):
            assert len(ground_truth) > 0, "Ground truth set is empty"
            predicted = predicted if not trunc else predicted[: len(ground_truth)]
            metric.update(predicted, [ground_truth])
            return metric.compute().item()
        elif isinstance(ground_truth, list) and isinstance(predicted, list):
            for gt_text, pred_text in zip(ground_truth, predicted):
                assert len(gt_text) > 0, "Ground truth set is empty"
                try:
                    pred_text = pred_text if not trunc else pred_text[: len(gt_text)]
                    metric.update(pred_text, [gt_text])
                except:
                    print(
                        "Error with update:",
                        "\nGround-Truth: ",
                        gt_text,
                        "\nPred: ",
                        pred_text,
                    )
            return metric.compute().item()
    else:
        metrics = [BLEUScore(n_gram=i) for i in range(1, 5)]
        if isinstance(ground_truth, str) and isinstance(predicted, str):
            assert len(ground_truth) > 0, "Ground truth set is empty"
            for metric in metrics:
                predicted = predicted if not trunc else predicted[: len(ground_truth)]
                metric.update(predicted, [ground_truth])
            result = np.array([metric.compute().item() for metric in metrics])
            return result.mean()
        elif isinstance(ground_truth, list) and isinstance(predicted, list):
            for gt_text, pred_text in zip(ground_truth, predicted):
                assert len(gt_text) > 0, "Ground truth set is empty"
                try:
                    for metric in metrics:
                        pred_text = (
                            pred_text if not trunc else pred_text[: len(gt_text)]
                        )
                        metric.update(pred_text, [gt_text])
                except:
                    print(
                        "Error with update:",
                        "\nGround-Truth: ",
                        gt_text,
                        "\nPred: ",
                        pred_text,
                    )
            result = np.array([metric.compute().item() for metric in metrics])
            return result.mean()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_accuracy(pred: str, ground_truth: str) -> int:
    return int(ground_truth == pred)


def get_em(pred: str, ground_truth: str) -> int:
    return int(normalize_answer(ground_truth) == normalize_answer(pred))


def get_approx_em(pred: str, ground_truth: str) -> int:
    l_normed_pred = normalize_answer(pred).split(" ")
    l_normed_gt = normalize_answer(ground_truth).split(" ")
    len_gt = len(l_normed_gt)
    if len(l_normed_pred) < len_gt:
        return 0

    for i in range(len(l_normed_pred) - len_gt + 1):
        if l_normed_pred[i : i + len_gt] == l_normed_gt:
            return 1
    return 0


def get_meteor(ground_truth: list[str] | str, predicted: list[str] | str) -> float:

    if isinstance(ground_truth, str) and isinstance(predicted, str):
        assert len(ground_truth) > 0, "Ground truth set is empty"
        l_ground_truth = ground_truth.split(" ")
        l_predicted = predicted.split(" ")
        return meteor_score.single_meteor_score(l_ground_truth, l_predicted)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        meteor_avg_score = 0
        for gt_text, pred_text in zip(ground_truth, predicted):
            assert len(gt_text) > 0, "Ground truth set is empty"
            l_ground_truth = gt_text.split(" ")
            l_predicted = pred_text.split(" ")
            meteor_avg_score += meteor_score.single_meteor_score(
                l_ground_truth, l_predicted
            )
        return meteor_avg_score / len(ground_truth)



# import regex
# import unicodedata
# import nltk

# class SimpleTokenizer(object):
#     ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
#     NON_WS = r"[^\p{Z}\p{C}]"

#     def __init__(self):
#         """
#         Args:
#             annotators: None or empty set (only tokenizes).
#         """
#         self._regexp = regex.compile(
#             "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
#         )

#     def tokenize(self, text, uncased=False):
#         matches = [m for m in self._regexp.finditer(text)]
#         if uncased:
#             tokens = [m.group().lower() for m in matches]
#         else:
#             tokens = [m.group() for m in matches]
#         return tokens


# def check_answer(example, tokenizer) -> list[bool]:
#     """Search through all the top docs to see if they have any of the answers."""
#     answers = example["answer"]
#     ctxs = example["passage"]

#     hits = []

#     for _, text in enumerate(ctxs):

#         if text is None:  # cannot find the document for some reason
#             hits.append(False)
#             continue

#         hits.append(has_answer(answers, text, tokenizer))

#     return hits


# def _normalize(text):
#     return unicodedata.normalize("NFD", text)


# def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
#     """Check if a document contains an answer string."""
#     text = _normalize(text)
#     text = tokenizer.tokenize(text, uncased=True)

#     for answer in answers:
#         answer = _normalize(answer)
#         answer = tokenizer.tokenize(answer, uncased=True)
#         for i in range(0, len(text) - len(answer) + 1):
#             if answer == text[i : i + len(answer)]:
#                 return True
#     return False


# def get_unigram_f1(text: str, answers: list[str]) -> float:
#     """Calculate unigram f1 score between the text and reference answers."""

#     def _get_unigram_f1(text, answers):
#         if isinstance(answers, str):
#             answers = [answers]
#         norm_pred = normalize_answer(text)
#         norm_answers = [normalize_answer(ans) for ans in answers]
#         common_tokens = [
#             Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
#         ]
#         num_same = [sum(common.values()) for common in common_tokens]

#         score_list = []
#         for i, num in enumerate(num_same):
#             if num == 0:
#                 score_list.append(0.0)
#             else:
#                 p = 1.0 * num / len(norm_pred)
#                 r = 1.0 * num / len(norm_answers[i])
#                 f1 = 2 * p * r / (p + r)
#                 score_list.append(f1)
#         return max(score_list)

#     unigram_f1 = [_get_unigram_f1(t, a) for t, a in zip(text, answers)]

#     return sum(unigram_f1) / len(unigram_f1), unigram_f1


# def eval_multiple_choice(generated_answers, answers):
#     ret = []
#     assert len(generated_answers) == len(answers)
#     for g_answer, answer in zip(generated_answers, answers):
#         ret.append(float(g_answer == answer))
#     return round(sum(ret) / len(ret), 3), ret


# def get_substring_match_score(outputs, answers):
#     """
#     outputs: [string1,string2]
#     answers: [
#                 [string1_1,string1_2],
#                 [string2_1,string2_2]
#              ]
#     """
#     import numpy as np

#     assert len(outputs) == len(answers)
#     if not isinstance(answers[0], list):
#         answers = [[x] for x in answers]
#     substring_match_scores = []
#     answer_lengths = []
#     for output, answer in zip(outputs, answers):
#         if has_answer(answer, output):  # EM evaluation
#             substring_match_scores.append(1.0)
#         else:
#             substring_match_scores.append(0.0)

#         answer_lengths.append(len(output.split()))

#     substring_match = round(sum(substring_match_scores) / len(outputs), 4)
#     lens = round(np.mean(answer_lengths), 4)

#     return substring_match, substring_match_scores



# def eval_fact_checking(outputs, answers):

#     tokenizer = SimpleTokenizer()

#     results = []
#     acc_count = 0
#     answer_lengths = []
#     for output, answer in zip(outputs, answers):

#         if answer == "False":
#             answer = ["refutes", "no", "false"]
#         if answer == "True":
#             answer = ["supports", "yes", "true"]
#         assert answer == ["refutes", "no", "false"] or answer == [
#             "supports",
#             "yes",
#             "true",
#         ]

#         if has_answer(answer, output, tokenizer):
#             acc_count += 1
#             results.append(1.0)
#         else:
#             results.append(0.0)

#         answer_lengths.append(len(output.split()))

#     acc = round(sum(results) / len(results), 4)
#     return acc, results


# def get_rougel_score(prediction, ground_truth):
#     from rouge import Rouge

#     rouge = Rouge()
#     # no normalization
#     try:
#         scores = rouge.get_scores(prediction, ground_truth, avg=True)
#     except ValueError:  # "Hypothesis is empty."
#         return 0.0
#     return scores["rouge-l"]["f"]