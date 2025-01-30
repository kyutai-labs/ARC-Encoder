import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from embed_llm.training.args import TrainArgs, WandbArgs
from embed_llm.training.utils import TrainState

logger = logging.getLogger("metrics_logger")

GB = 1024**3


def get_train_logs(
    state: TrainState,
    loss: float,
    avg_grad_norm: float,
    ppl: float,
    lr: float,
    peak_allocated_mem: float,
    allocated_mem: float,
    train_args: TrainArgs,
    cross_entropy: float | None = None,
    kl: float | None = None,
    batch_type: str = "reconstruction",
) -> dict[str, float | int]:
    metrics = {
        "lr": lr,
        "step": state.step,
        "loss": loss,
        "perplexity": ppl,
        "avg_grad_norm": avg_grad_norm,
        "percent_done": 100 * state.step / train_args.max_steps,
        "peak_allocated_mem": peak_allocated_mem / GB,
        "allocated_mem": allocated_mem / GB,
        "wps": state.wps,
        "avg_wps": state.avg_wps,
        "eta_in_seconds": state.eta,
        "Cross_Entropy": cross_entropy,
        "KL": kl,
        "batch_type": batch_type,
    }

    return metrics


def get_eval_logs(
    step: int,
    train_loss: float,
    perplexity_rec: float | None = None,
    eval_loss_rec: float | None = None,
    perplexity_textcont: float | None = None,
    eval_loss_textcont: float | None = None,
    perplexity_embcont: float | None = None,
    eval_loss_embcont: float | None = None,
    instruct_cross_entropy: float | None = None,
    instruct_kl: float | None = None,
    eval_kl_loss: float | None = None,
    eval_loss_nocontext: float | None = None,
    eval_perplexity_nocontext: float | None = None,
) -> dict[str, float | int]:
    eval_dict = {"step": step, "train_loss": train_loss}

    if perplexity_rec is not None:
        eval_dict["perplexity_rec"] = perplexity_rec

    if eval_loss_rec is not None:
        eval_dict["eval_loss_rec"] = eval_loss_rec

    if perplexity_textcont is not None:
        eval_dict["perplexity_textcont"] = perplexity_textcont

    if eval_loss_textcont is not None:
        eval_dict["eval_loss_textcont"] = eval_loss_textcont

    if perplexity_embcont is not None:
        eval_dict["perplexity_embcont"] = perplexity_embcont

    if eval_loss_embcont is not None:
        eval_dict["eval_loss_embcont"] = eval_loss_embcont

    if instruct_cross_entropy is not None:
        eval_dict["instruct_cross_entropy"] = instruct_cross_entropy

    if instruct_kl is not None:
        eval_dict["instruct_kl"] = instruct_kl

    if eval_kl_loss is not None:
        eval_dict["eval_kl_loss"] = eval_kl_loss

    if eval_loss_nocontext is not None:
        eval_dict["eval_loss_nocontext"] = eval_loss_nocontext

    if eval_perplexity_nocontext is not None:
        eval_dict["eval_perplexity_nocontext"] = eval_perplexity_nocontext

    return eval_dict


def train_log_msg(
    state: TrainState, logs: dict[str, float | int], loss: float, seen_tokens: int
) -> str:
    metrics: dict[str, float | int | datetime] = dict(logs)  # shallow copy
    metrics.pop("eta_in_seconds")

    metrics["eta"] = datetime.now() + timedelta(seconds=state.eta)
    metrics["step"] = state.step
    metrics["loss"] = loss
    metrics["seen_tokens"] = seen_tokens

    parts = []
    for key, fmt, new_name in [
        ("step", "06", None),
        ("percent_done", "03.1f", "done (%)"),
        ("loss", ".3f", None),
        ("avg_grad_norm", ".2e", "Grad Norm"),
        ("perplexity", ".3f", None),
        ("lr", ".1e", None),
        ("peak_allocated_mem", ".1f", "peak_alloc_mem (GB)"),
        ("allocated_mem", ".1f", "alloc_mem (GB)"),
        ("wps", ".1f", "words_per_second"),
        ("avg_wps", ".1f", "avg_words_per_second"),
        ("eta", "%Y-%m-%d %H:%M:%S", "ETA"),
        ("Cross_Entropy", ".3f", None),
        ("KL", ".3f", "KL"),
        ("batch_type", "s", "Batch Type"),
        ("seen_tokens", "d", "Seen Tokens"),
    ]:
        name = key if new_name is None else new_name
        if metrics[key] is None:
            continue
        try:
            parts.append(f"{name}: {metrics[key]:>{fmt}}")
        except KeyError:
            logger.error(f"{key} not found in {sorted(metrics.keys())}")
            raise

    return " - ".join(parts)


def eval_log_msg(logs: dict[str, float | int]) -> str:
    parts = []
    for key, fmt, new_name in [
        ("step", "06", None),
        ("perplexity_rec", ".3f", "Eval Reconstruction PPL"),
        ("eval_loss_rec", ".3f", "Eval Reconstruction Loss"),
        ("train_loss", ".3f", "Train Loss"),
        ("perplexity_textcont", ".3f", "Eval Textcont PPL"),
        ("eval_loss_textcont", ".3f", "Eval Textcont Loss"),
        ("perplexity_embcont", ".3f", "Eval Embcont PPL"),
        ("eval_loss_embcont", ".3f", "Eval Embcont Loss"),
        ("instruct_cross_entropy", ".3f", "instruct_cross_entropy"),
        ("instruct_kl", ".3f", "instruct_kl"),
        ("eval_kl_loss", ".3f", "eval_kl_loss"),
        ("eval_loss_nocontext", ".3f", "eval_loss_nocontext"),
        ("eval_perplexity_nocontext", ".3f", "PPL continuation wo context"),
    ]:
        name = key if new_name is None else new_name
        if key in logs:
            if logs[key] is not None:
                parts.append(f"{name}: {logs[key]:>{fmt}}")

    return " - ".join(parts)


class MetricsLogger:
    def __init__(
        self,
        dst_dir: Path,
        tag: str,
        is_master: bool,
        wandb_args: WandbArgs,
        config: dict[str, object] | None = None,
    ):
        self.dst_dir = dst_dir
        self.tag = tag
        self.is_master = is_master
        self.jsonl_path = dst_dir / f"metrics.{tag}.jsonl"
        self.tb_dir = dst_dir / "tb"
        self.summary_writer: SummaryWriter | None = None

        if not self.is_master:
            return

        filename_suffix = f".{tag}"
        self.tb_dir.mkdir(exist_ok=True)
        self.summary_writer = SummaryWriter(
            log_dir=str(self.tb_dir),
            max_queue=1000,
            filename_suffix=filename_suffix,
        )
        self.is_wandb = wandb_args.project is not None

        if self.is_wandb:
            import wandb

            if wandb_args.key is not None:
                wandb.login(key=wandb_args.key)
            if wandb_args.offline:
                os.environ["WANDB_MODE"] = "offline"
            if wandb.run is None:
                logger.info("initializing wandb")
                wandb.init(
                    config=config,
                    dir=dst_dir,
                    project=wandb_args.project,
                    job_type="training",
                    name=wandb_args.run_name or dst_dir.name,
                    resume=False,
                )

            self.wandb_log = wandb.log

    def log(self, metrics: dict[str, float | int], step: int):
        if not self.is_master:
            return

        metrics_to_ignore = {"step", "batch_type"}
        assert self.summary_writer is not None
        for key, value in metrics.items():
            if key in metrics_to_ignore or value is None:
                continue
            assert isinstance(value, (int, float)), (key, value)
            self.summary_writer.add_scalar(
                tag=f"{self.tag}.{key}", scalar_value=value, global_step=step
            )

        if self.is_wandb:
            # grouping in wandb is done with /
            self.wandb_log(
                {
                    f"{self.tag}/{key}": value
                    for key, value in metrics.items()
                    if key not in metrics_to_ignore
                },
                step=step,
            )

        metrics_: dict[str, object] = dict(metrics)  # shallow copy
        if "step" in metrics_:
            assert step == metrics_["step"]
        else:
            metrics_["step"] = step
        metrics_["at"] = datetime.utcnow().isoformat()
        with self.jsonl_path.open("a") as fp:
            fp.write(f"{json.dumps(metrics_)}\n")

    def close(self):
        if not self.is_master:
            return

        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None

        if self.is_wandb:
            import wandb

            # to be sure we are not hanging while finishing
            wandb.finish()

    def __del__(self):
        if self.summary_writer is not None:
            raise RuntimeError(
                "MetricsLogger not closed properly! You should "
                "make sure the close() method is called!"
            )
