"""Resumable tokenizer training on one GPU or with torchrun across GPUs.

Example: torchrun --standalone --nproc_per_node=2 train_tokenizer.py --batch-size 2
Batch size is per GPU. Checkpoints preserve the next sample in the shuffled epoch.
"""

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from config import CONTENT_SIZE, TOKENIZER_TRAINING, build_optimizer
from dataloader import ArrayRecordClipDataset
from vae import VAE
from tokenizer_validation import evaluate, initialize_codebook


class EpochBatches(Sampler):
    """Partition global batches without dropping or weighting padding as data.

    Empty ranks in the final batch execute one dummy sample with zero loss weight,
    keeping DDP collectives aligned. Each real sample belongs to exactly one rank.
    """

    def __init__(self, size, batch_size, rank, world_size, seed, epoch, start=0):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size
        self.seed, self.epoch, self.start = seed, epoch, start
        self.global_batch = batch_size * world_size

    def __len__(self):
        return math.ceil((self.size - self.start) / self.global_batch)

    def counts(self, step):
        count = min(self.global_batch, self.size - self.start - step * self.global_batch)
        local = max(0, min(self.batch_size, count - self.rank * self.batch_size))
        return local, count

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(
            self.seed + self.epoch)).tolist()
        for offset in range(self.start, self.size, self.global_batch):
            batch = order[offset:offset + self.global_batch]
            local = batch[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
            yield local if local else [batch[0]]


def train(*, data_dir=None, epochs=1, batch_size=2, device="cuda", num_workers=4,
          learning_rate=1e-4, warmup_steps=1000, seed=42, checkpoint=None,
          checkpoint_every=250, log_every=10, resume=None, max_steps=None, model=None,
          grad_clip=1.0, validation_every=250, validation_dir=None, health_after=500):
    if min(epochs, batch_size, checkpoint_every, log_every) < 1:
        raise ValueError("epochs, batch size, checkpoint/log intervals must be positive")
    if num_workers < 0 or warmup_steps < 0 or learning_rate <= 0:
        raise ValueError("Invalid workers, warmup or learning rate")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if grad_clip <= 0 or validation_every < 0 or health_after < 0:
        raise ValueError("Invalid gradient clip or validation settings")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(device)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    old_handlers = {}
    stopping = [False]

    def request_stop(signum, frame):
        stopping[0] = True

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
        torch.manual_seed(seed)
        root = Path(__file__).resolve().parent
        checkpoint = Path(checkpoint or root / "data/checkpoints/vae_latest.pt")
        saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
        if saved and saved.get("format_version") != 1:
            raise ValueError("Resume requires a train_tokenizer.py checkpoint")
        if saved and saved["seed"] != seed:
            raise ValueError("Resume seed must match the checkpoint")
        constructor = {"stabilize": False, **saved["model_config"]} if saved else None
        model = model if model is not None else VAE(**constructor) if saved else VAE()
        if saved:
            if model.model_config != constructor:
                raise ValueError("Model configuration differs from checkpoint")
            model.load_state_dict(saved["model_state_dict"])
        model = model.to(device)
        if model.temporal_pos.shape[0] < 16 or model.channels != 3:
            raise ValueError("Tokenizer must accept 16-frame RGB sequences")
        if rank == 0:
            print(f"Indexing {data_dir or root / 'data/train'} ...", flush=True)
        dataset = ArrayRecordClipDataset(data_dir or root / "data/train", T=16,
            image_size=model.image_size,
            content_size=tuple(min(a, b) for a, b in zip(CONTENT_SIZE, model.image_size)))
        sources = [(str(p), Path(p).stat().st_size) for p in dataset.paths]
        if saved and (sources != saved["sources"] or len(dataset) != saved["dataset_size"]):
            raise ValueError("Dataset differs from the checkpoint")
        validation_clips = None
        validation_output = checkpoint.parent / "validation"
        if rank == 0:
            if not saved:
                initialize_codebook(model, dataset, device)
                print("Initialized codebook from encoder features across training clips", flush=True)
            if validation_every:
                val_path = Path(validation_dir) if validation_dir else Path(data_dir or root / "data/train").parent / "val"
                validation_dataset = ArrayRecordClipDataset(val_path, T=16,
                    image_size=model.image_size,
                    content_size=tuple(min(a,b) for a,b in zip(CONTENT_SIZE,model.image_size)))
                positions = torch.linspace(0, len(validation_dataset)-1, min(4,len(validation_dataset))).long().tolist()
                validation_clips = [validation_dataset[i] for i in positions]
        global_batch = batch_size * world
        config = replace(TOKENIZER_TRAINING, max_lr=learning_rate, min_lr=learning_rate,
                         warmup_steps=warmup_steps, steps=epochs * math.ceil(len(dataset) / global_batch),
                         global_batch_size=global_batch)
        if saved:
            for key in ("max_lr", "warmup_steps"):
                if getattr(config, key) != saved["training_config"][key]:
                    raise ValueError(f"Resume {key} must match the checkpoint")
        optimizer, scheduler = build_optimizer(model, config)
        use_amp = device.type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        if saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            scaler.load_state_dict(saved["scaler_state_dict"])
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if use_amp else None,
                                         gradient_as_bucket_view=True) if world > 1 else model
        history = saved["history"] if saved else []
        best_validation = saved.get("best_validation_mse", float("inf")) if saved else float("inf")
        global_step = saved["global_step"] if saved else 0
        first_epoch = saved["epoch"] + int(saved["epoch_complete"]) if saved else 1
        first_sample = saved["next_sample"] if saved and not saved["epoch_complete"] else 0
        base_totals = saved["totals"] if first_sample else [0.0] * 3
        invocation_steps = 0
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(f"{len(dataset):,} sequences; {world} ranks; batch/GPU={batch_size}; "
                  f"global batch={global_batch}; precision={amp_dtype if use_amp else torch.float32}; "
                  f"resume step={global_step}, epoch={first_epoch}, sample={first_sample}", flush=True)
        # Release CPU copies of model and optimizer tensors loaded from disk.
        del saved

        for epoch in range(first_epoch, epochs + 1):
            sampler = EpochBatches(len(dataset), batch_size, rank, world, seed, epoch, first_sample)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                pin_memory=use_amp, generator=torch.Generator().manual_seed(seed + epoch + rank),
                **({"multiprocessing_context": "spawn", "persistent_workers": True} if num_workers else {}))
            model.train()
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            started, timed_samples = time.perf_counter(), 0
            for step, video in enumerate(loader):
                local_count, count = sampler.counts(step)
                video = video.to(device, non_blocking=use_amp)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    result = wrapped(video, return_details=True)
                    loss = result["loss"]
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f"Nonfinite loss at step {global_step + 1}")
                scale = scaler.get_scale()
                # DDP averages rank gradients; this recovers the global sample mean
                # even when the final local batches differ in size.
                scaler.scale(loss * (world * local_count / count)).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                global_step += 1
                invocation_steps += 1
                next_sample = min(len(dataset), first_sample + (step + 1) * global_batch)
                totals += torch.stack([result[k].detach().double() for k in
                                       ("loss", "reconstruction_loss", "vq_loss")]) * local_count
                timed_samples += count
                stop = torch.tensor(int(stopping[0] or
                    (max_steps is not None and invocation_steps >= max_steps)), device=device)
                if world > 1:
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                stop = bool(stop.item())
                complete = next_sample == len(dataset)
                validation = None
                improved = False
                health_failure = False
                if validation_every and (global_step % validation_every == 0 or complete or stop):
                    message = [evaluate(model, validation_clips, device, step=global_step,
                                        output_dir=validation_output) if rank == 0 else None]
                    if world > 1:
                        dist.broadcast_object_list(message, src=0)
                    validation = message[0]
                    health_failure = global_step >= health_after and bool(validation["collapse_reasons"])
                    improved = not health_failure and validation["mse"] < best_validation
                    if improved:
                        best_validation = validation["mse"]
                    if rank == 0:
                        print(f"VALIDATION step={global_step} mse={validation['mse']:.6f} "
                              f"codes={validation['active_codes']}/{model.cb.K} "
                              f"perplexity={validation['perplexity']:.2f} "
                              f"saturated={validation['saturated_fraction']:.4f} "
                              f"output_difference={validation['output_difference']:.6f}", flush=True)
                        if health_failure:
                            print("HEALTH STOP: " + "; ".join(validation["collapse_reasons"]), flush=True)
                    stop = stop or health_failure
                save = complete or stop or global_step % checkpoint_every == 0
                save = save or improved
                log = step == 0 or global_step % log_every == 0 or save
                if log:
                    combined = totals.clone()
                    if world > 1:
                        dist.all_reduce(combined)
                    combined += torch.tensor(base_totals, device=device, dtype=torch.float64)
                    combined = combined.tolist()
                    if rank == 0:
                        rate = timed_samples / max(time.perf_counter() - started, 1e-9)
                        print(f"epoch {epoch}/{epochs} step={global_step} "
                              f"samples={next_sample}/{len(dataset)} "
                              f"loss={combined[0]/next_sample:.6f} recon={combined[1]/next_sample:.6f} "
                              f"vq={combined[2]/next_sample:.6f} seq/s={rate:.2f} "
                              f"eta_min={(len(dataset)-next_sample)/max(rate, 1e-9)/60:.1f} "
                              f"lr={optimizer.param_groups[0]['lr']:.2g} grad_norm={grad_norm.item():.3f}", flush=True)
                    if complete:
                        history.append(dict(epoch=epoch, sequences=next_sample,
                            loss=combined[0]/next_sample, reconstruction_loss=combined[1]/next_sample,
                            vq_loss=combined[2]/next_sample))
                    if save:
                        if rank == 0:
                            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
                            torch.save(dict(format_version=1, model_config=model.model_config,
                                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
                                epoch=epoch, next_sample=next_sample, epoch_complete=complete,
                                global_step=global_step, history=history, totals=combined, seed=seed,
                                dataset_size=len(dataset), sources=sources, training_config=vars(config),
                                world_size=world, batch_size=batch_size, grad_clip=grad_clip,
                                implementation_version="stabilized-v2",
                                best_validation_mse=best_validation, validation=validation,
                                health_failure=health_failure), temporary)
                            temporary.replace(checkpoint)
                            if improved:
                                # Checkpoints are atomically replaced, never modified
                                # in place, so a hard link retains the best snapshot.
                                best = checkpoint.with_name(checkpoint.stem + "_best.pt")
                                best_temp = best.with_suffix(".pt.tmp")
                                best_temp.unlink(missing_ok=True)
                                os.link(checkpoint, best_temp)
                                best_temp.replace(best)
                            print(f"Saved {checkpoint} at step {global_step}", flush=True)
                        if world > 1:
                            dist.barrier()
                if stop:
                    if health_failure:
                        raise RuntimeError("Tokenizer collapsed; checkpoint and validation previews saved")
                    model.eval()
                    return model, history
            first_sample, base_totals = 0, [0.0] * 3
        model.eval()
        return model, history
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--epochs", type=int, default=1, help="Target total epoch count, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Sequences per GPU")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-every", type=int, default=250, help="0 disables validation (tests only)")
    parser.add_argument("--validation-dir", help="Default: val alongside the training directory")
    parser.add_argument("--health-after", type=int, default=500, help="Stop on validation collapse after this step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Stop and checkpoint after this many steps in this invocation")
    train(**vars(parser.parse_args()))
