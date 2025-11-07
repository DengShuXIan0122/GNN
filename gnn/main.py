import argparse

from .utils import create_logger
import torch
import numpy as np
import os
import time
#from Models.ReaRev.rearev import 
from train_model import Trainer_KBQA
from parsing import add_parse_args

parser = argparse.ArgumentParser()
add_parse_args(parser)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

# 处理GPU IDs参数
if hasattr(args, 'gpu_ids') and args.gpu_ids:
    args.gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
else:
    args.gpu_ids = [0]

def maybe_cap_gpu_memory(limit_gb):
    if limit_gb is None:
        return
    if not torch.cuda.is_available():
        print("CUDA not available; skip gpu_memory_limit_gb.")
        return
    device_count = torch.cuda.device_count()
    for device_idx in range(device_count):
        try:
            total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        except AssertionError:
            continue
        if total_bytes <= 0:
            continue
        limit_bytes = limit_gb * (1024 ** 3)
        fraction = min(1.0, float(limit_bytes) / float(total_bytes))
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
            total_gb = total_bytes / (1024 ** 3)
            applied_gb = min(limit_gb, total_gb * fraction)
            print(f"Set CUDA memory cap on device {device_idx} to {applied_gb:.2f} GB (fraction {fraction:.4f} of {total_gb:.2f} GB)")
        except AttributeError:
            print("torch.cuda.set_per_process_memory_fraction unavailable; skip gpu_memory_limit_gb.")
            break
        except RuntimeError as exc:
            print(f"Unable to set memory cap on cuda:{device_idx}: {exc}")

maybe_cap_gpu_memory(args.gpu_memory_limit_gb)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=logger)
    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        assert args.load_experiment is not None
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            print("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
