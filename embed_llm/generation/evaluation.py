from torcheval.metrics import BLEUScore
 

def word_overlap(ground_truth: list[str] | str, predicted: list[str] | str) -> float:
    if isinstance(ground_truth, str) and isinstance(predicted, str):
        ground_truth = set(ground_truth.split(' '))
        predicted = set(predicted.split(' '))
        assert len(ground_truth) > 0, "Ground truth set is empty"
        return len(ground_truth.intersection(predicted)) / len(ground_truth)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        avg_word_overlap = 0
        n_words = 0
        for gt_text, pred_text in zip(ground_truth, predicted):
            gt_text = set(gt_text.split(' '))
            pred_text = set(pred_text.split(' '))
            assert len(gt_text) > 0, "Ground truth set is empty"
            n_words += len(gt_text)
            avg_word_overlap += len(gt_text.intersection(pred_text))
        return avg_word_overlap / n_words
    

def get_bleu_score(ground_truth: list[str] | str, predicted: list[str] | str) -> float:
    metric = BLEUScore(n_gram = 4)
    if isinstance(ground_truth, str) and isinstance(predicted, str):
        assert len(ground_truth) > 0, "Ground truth set is empty"
        metric.update(predicted, [ground_truth])
        return metric.compute().item()
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        for gt_text, pred_text in zip(ground_truth, predicted):
            assert len(gt_text) > 0, "Ground truth set is empty"
            try:
                metric.update(pred_text, [gt_text])
            except:
                print('Error with update:', '\nGround-Truth: ',gt_text, '\nPred: ', pred_text)
        return metric.compute().item()

# def get_ppl(ground_truth: torch.Tensor | list[torch.Tensor], logprobs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
#     if isinstance(ground_truth, torch.Tensor) and isinstance(logprobs, torch.Tensor):
#         cross_entropy = F.cross_entropy(logprobs, ground_truth, reduction="mean")
#         return 2**cross_entropy.item()
#     elif isinstance(ground_truth, list) and isinstance(logprobs, list):
#         ground_truth = torch.cat(ground_truth, dim=0)
#         # TODO: check if this is correct
#         logprobs = torch.cat(logprobs, dim=0)
#         cross_entropy = F.cross_entropy(logprobs, ground_truth, reduction="mean")
#         return 2**cross_entropy.item()
    
    