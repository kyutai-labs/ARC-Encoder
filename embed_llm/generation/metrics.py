import re
import string
import unicodedata
from collections import Counter
from rouge_score import rouge_scorer

import regex
import nltk
from sacrebleu.metrics import BLEU as SacreBLEU

nltk.download("wordnet")

ROUGE_SCORER = None


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

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
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


def get_bleu_score(
    ground_truth: list[str] | str,
    predicted: list[str] | str,
) -> float:
    _, _, bleu_fn = [], [], SacreBLEU(tokenize="13a", lowercase=True)
    return bleu_fn.corpus_score(
        [predicted] if isinstance(predicted, str) else predicted,
        [[ground_truth]] if isinstance(ground_truth, str) else [ground_truth],
    ).score
