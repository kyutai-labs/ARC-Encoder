import re
import string
from collections import Counter
import unicodedata
import regex


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
