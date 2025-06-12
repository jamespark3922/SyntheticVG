import os
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from tqdm import tqdm
from transformers import Trainer, TrainerCallback
from transformers.trainer import (
    get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, is_torch_xla_available
)
from typing import Dict, List, Optional

from svg.utils import rank0_print
from accelerate import Accelerator,  InitProcessGroupKwargs
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from datetime import timedelta

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                rank0_print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class SVGTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            rank0_print(f"Using optimizer with mapping: {lr_mapper}")
        rank0_print(f"Optimizer: {self.optimizer}")

        return self.optimizer
    
    def _get_learning_rates(self):
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                # Order of learning_rates is [model,mm_projector,vision_tower]
                # Note: learning rates are not provided for modules with disabled gradients 
                last_lr = self.lr_scheduler.get_last_lr()[0]
                vit_lr = self.lr_scheduler.get_last_lr()[-1]
            except AssertionError as e:
                if "need to call step" in str(e):
                    rank0_print("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    return 0,0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
                vit_lr =  self.optimizer.param_groups[-1]["lr"]
            else:
                last_lr = self.lr_scheduler.get_last_lr()[0]
                vit_lr = self.lr_scheduler.get_last_lr()[-1]
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
            if torch.is_tensor(vit_lr):
                vit_lr = vit_lr.item()
        return last_lr, vit_lr
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            learning_rate, vit_learning_rate = self._get_learning_rates()
            logs["learning_rate"] = learning_rate
            logs["vit_learning_rate"] = vit_learning_rate

            # Compute average batch load and processing times since the last log
            timing_callback = None
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, TimingCallback):
                    timing_callback = callback
                    break

            if timing_callback and timing_callback.num_batches > 0:
                avg_batch_load_time = timing_callback.total_batch_load_time / timing_callback.num_batches
                avg_batch_processing_time = timing_callback.total_batch_processing_time / timing_callback.num_batches
                logs["avg_batch_load_time"] = round(avg_batch_load_time, 4)
                logs["avg_batch_processing_time"] = round(avg_batch_processing_time, 4)
                # Reset accumulators in the callback
                timing_callback.total_batch_load_time = 0.0
                timing_callback.total_batch_processing_time = 0.0
                timing_callback.num_batches = 0

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(SVGTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(SVGTrainer, self)._save(output_dir, state_dict)
    
class TimingCallback(TrainerCallback):
    def __init__(self):
        self.batch_start_time = None
        self.batch_end_time = None

        # Initialize accumulators
        self.total_batch_load_time = 0.0
        self.total_batch_processing_time = 0.0
        self.num_batches = 0

    def on_step_begin(self, args, state, control, **kwargs):
        # If it's not the first batch, calculate the loading time of the previous batch
        if self.batch_end_time:
            batch_load_time = time.time() - self.batch_end_time
            self.total_batch_load_time += batch_load_time
            self.num_batches += 1  # Increment the batch count

        self.batch_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.batch_end_time = time.time()
        batch_processing_time = self.batch_end_time - self.batch_start_time
        self.total_batch_processing_time += batch_processing_time

        # Optionally, print individual batch times for debugging
        # print(f"Batch processing time: {batch_processing_time:.4f}s")
        # print(f"Batch load time: {batch_load_time:.4f}s")

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    #     # want to save except FullyShardedDDP.
    #     # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    #     PREFIX_CHECKPOINT_DIR="checkpoint"
    #     TRAINER_STATE_NAME = "trainer_state.json"

    #     # Save model checkpoint
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()

    #     run_dir = self._get_output_dir(trial=trial)
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     self.save_model(output_dir, _internal_call=True)

    #     # save vision model checkpoint

    #     if not self.args.save_only_model:
    #         # Save optimizer and scheduler
    #         self._save_optimizer_and_scheduler(output_dir)
    #         # Save RNG state
    #         self._save_rng_state(output_dir)

    #     # Determine the new best metric / best model checkpoint
    #     if metrics is not None and self.args.metric_for_best_model is not None:
    #         metric_to_check = self.args.metric_for_best_model
    #         if not metric_to_check.startswith("eval_"):
    #             metric_to_check = f"eval_{metric_to_check}"
    #         try:
    #             metric_value = metrics[metric_to_check]
    #         except KeyError as exc:
    #             raise KeyError(
    #                 f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
    #                 f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
    #             ) from exc

    #         operator = np.greater if self.args.greater_is_better else np.less
    #         if (
    #             self.state.best_metric is None
    #             or self.state.best_model_checkpoint is None
    #             or operator(metric_value, self.state.best_metric)
    #         ):
    #             self.state.best_metric = metric_value
    #             self.state.best_model_checkpoint = output_dir

    #     # Save the Trainer state
    #     if self.args.should_save:
    #         # Update the `TrainerControl` state to where we are currently
    #         self.state.stateful_callbacks["TrainerControl"] = self.control.state()
    #         self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    #     if self.args.push_to_hub:
    #         self._push_from_checkpoint(output_dir)

    #     # Maybe delete some older checkpoints.
    #     if self.args.should_save:
    #         # Solely rely on numerical checkpoint id for rotation.
    #         # mtime is not reliable especially on some fuse fs in cloud environments.
    #         self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    # def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
    #     if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
    #         return

    #     # Check if we should delete older checkpoint(s)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    #     if len(checkpoints_sorted) <= self.args.save_total_limit:
    #         return

    #     # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
    #     # we don't do to allow resuming.
    #     save_total_limit = self.args.save_total_limit
    #     if (
    #         self.state.best_model_checkpoint is not None
    #         and self.args.save_total_limit == 1
    #         and checkpoints_sorted[-1] != self.state.best_model_checkpoint
    #     ):
    #         save_total_limit = 2

    #     number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    #     checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    #     for checkpoint in checkpoints_to_be_deleted:
    #         logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #         shutil.rmtree(checkpoint, ignore_errors=True)
