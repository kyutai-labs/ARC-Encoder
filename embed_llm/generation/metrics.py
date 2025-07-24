import re
import string
import unicodedata
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer

import regex
from sacrebleu.metrics import BLEU as SacreBLEU
from torcheval.metrics import BLEUScore
from nltk.translate import meteor_score
# nltk.download("wordnet")

ROUGE_SCORER = None


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


def get_rouge_score(predicted: str, ground_truth: str) -> float:
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        # init RougeScorer once (https://github.com/EleutherAI/lm-evaluation-harness/issues/1692)--rouge_types are constant
        ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    r_scorer = ROUGE_SCORER
    return r_scorer.score(predicted, ground_truth)["rougeL"].fmeasure


class SimpleTokenizer(object):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i : i + len(answer)]:
                return True
    return False


def get_substring_match_score(outputs, answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """

    assert len(outputs) == len(answers)
    if not isinstance(answers[0], list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output, answer in zip(outputs, answers):
        if has_answer(answer, output):  # EM evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)

        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores) / len(outputs), 4)

    return substring_match, substring_match_scores


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
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.
    
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_accuracy(pred: str, ground_truth: str) -> int:
    return int(ground_truth == pred)


def get_em(pred: str, ground_truth: str) -> int:
    return int(normalize_answer(ground_truth) == normalize_answer(pred))


def get_approx_em(pred: str, ground_truth: str) -> int:
    l_normed_pred = normalize_answer(str(pred)).split(" ")
    l_normed_gt = normalize_answer(str(ground_truth)).split(" ")
    len_gt = len(l_normed_gt)
    if len(l_normed_pred) < len_gt:
        return 0

    for i in range(len(l_normed_pred) - len_gt + 1):
        if l_normed_pred[i : i + len_gt] == l_normed_gt:
            return 1
    return 0


def get_acc_factchecking(pred: str, ground_truth: str) -> int:
    if str(ground_truth).lower() == "false":
        answer = ["refutes", "no", "false"]
    if str(ground_truth).lower() == "true":
        answer = ["supports", "yes", "true"]

    assert answer == ["refutes", "no", "false"] or answer == ["supports", "yes", "true"]
    if pred.lower() in answer:
        return 1
    return 0


def get_bleu_score(
    ground_truth: list[str] | str,
    predicted: list[str] | str,
    avg: bool = False,
    trunc: bool = False,
    sacrebleu: bool = True,
) -> float:
    if sacrebleu:
        _, _, bleu_fn = [], [], SacreBLEU(tokenize="13a", lowercase=True)
        return bleu_fn.corpus_score(
            [predicted] if isinstance(predicted, str) else predicted,
            [[ground_truth]] if isinstance(ground_truth, str) else [ground_truth],
        ).score
    else:
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
                        pred_text = (
                            pred_text if not trunc else pred_text[: len(gt_text)]
                        )
                        metric.update(pred_text, [gt_text])
                    except Exception as e:
                        print(
                            "Error with update:",
                            "\nGround-Truth: ",
                            gt_text,
                            "\nPred: ",
                            pred_text,
                            e,
                        )
                return metric.compute().item()
        else:
            metrics = [BLEUScore(n_gram=i) for i in range(1, 5)]
            if isinstance(ground_truth, str) and isinstance(predicted, str):
                assert len(ground_truth) > 0, "Ground truth set is empty"
                for metric in metrics:
                    try:
                        predicted = (
                            predicted if not trunc else predicted[: len(ground_truth)]
                        )
                        metric.update(predicted, [ground_truth])
                    except Exception as e:
                        print(
                            "Error with update:",
                            "\nGround-Truth: ",
                            ground_truth,
                            "\nPred: ",
                            predicted,
                            e,
                        )
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
                    except Exception as e:
                        print(
                            "Error with update:",
                            "\nGround-Truth: ",
                            gt_text,
                            "\nPred: ",
                            pred_text,
                            e,
                        )
                result = np.array([metric.compute().item() for metric in metrics])
                return result.mean()


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
