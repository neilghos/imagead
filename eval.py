from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch

from dataloader.datamodule import DATASETS_PATH, DECOMPOSITION_CACHE_ROOT
from determinism import make_deterministic
from perturbation.precompute_stage2_cache import STAGE2_CACHE_ROOT
from stage2_module import Stage2DataModule, Stage2LightningModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-class", type=str, required=True)
    parser.add_argument("--stage2-checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=str, default=str(STAGE2_CACHE_ROOT))
    parser.add_argument("--plot-dir", type=str, default="/data/stepdown vision/plots/eval")
    return parser.parse_args()


def main():
    args = parse_args()
    make_deterministic(args.seed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = Path(args.plot_dir) / args.mvtec_class / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    datamodule = Stage2DataModule(
        mvtec_class=args.mvtec_class,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
        data_root=args.data_root,
        clean_cache_root=args.clean_cache_root,
        stage2_cache_root=args.stage2_cache_root,
        seed=args.seed,
    )
    module = Stage2LightningModule.load_from_checkpoint(
        args.stage2_checkpoint,
        stage1_checkpoint=None,
        artifact_dir=str(artifact_dir),
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.test(module, datamodule=datamodule)
    print(f"[eval] artifact_dir={artifact_dir}")


if __name__ == "__main__":
    main()
