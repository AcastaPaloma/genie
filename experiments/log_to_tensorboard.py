"""Mirror a train_dynamics stdout log into TensorBoard scalars, following the file.

    python3 experiments/log_to_tensorboard.py data/experiments/dynamics_a100.log data/experiments/tb/dynamics_a100

Parses `epoch E/N batch B/M loss=... masked=... grad_norm=... lr=...` and
`[eval @ step S] masked_ce=... next_frame_ce=... next_frame_acc=... action_delta_ce=...`.
Global step for batch lines is (E-1)*M + B. Re-reads only new lines every 30 s.
"""
import re, sys, time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

BATCH = re.compile(r"^epoch (\d+)/\d+ batch (\d+)/(\d+) loss=([\d.]+) masked=([\d.]+) grad_norm=([\d.naninf]+) lr=([\d.e+-]+)")
EVAL = re.compile(r"^\[eval @ step (\d+)\] (.*)$")

def main(log_path, tb_dir, follow=True):
    writer = SummaryWriter(tb_dir)
    seen = 0
    while True:
        lines = Path(log_path).read_text().splitlines() if Path(log_path).exists() else []
        for line in lines[seen:]:
            m = BATCH.match(line)
            if m:
                epoch, batch, per_epoch, loss, masked, grad_norm, lr = m.groups()
                step = (int(epoch) - 1) * int(per_epoch) + int(batch)
                writer.add_scalar("train/loss", float(loss), step)
                writer.add_scalar("train/mask_rate", float(masked), step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                writer.add_scalar("train/lr", float(lr), step)
                continue
            m = EVAL.match(line)
            if m:
                step = int(m.group(1))
                for key, value in re.findall(r"(\w+)=([+-]?[\d.]+)", m.group(2)):
                    writer.add_scalar(f"eval/{key}", float(value), step)
        if len(lines) > seen:
            writer.flush()
        seen = len(lines)
        if not follow or (lines and lines[-1].startswith("dynamics run finished")):
            break  # stop only when the finished line is the LAST line; a resumed run appends after it
        time.sleep(30)
    writer.close()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], follow="--once" not in sys.argv)
