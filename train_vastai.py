# Gunakan skrip ini untuk melakukan training dengan cara:
# Type command: 
    # tmux new -s train
# Setelah berada pada sesi tmux train, lalu eksekusi skrip dengan cara:
    # /venv/main/bin/python3.12 train_vastai.py (kalau ngga salah)
# Dengan cara ini, kita bebas menutup laptop lokal kita, dan dapat kembali memonitor training jika diperlukan atau sudah selesai.

import os
import torch
from rfdetr import RFDETRSmall

ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
OUTPUT_DIR = os.path.join(ROOT_DIR, "weights")
RESUME_WEIGHT = os.path.join(OUTPUT_DIR, "checkpoint.pth")

# Training Parameters
EPOCHS = 30
LR = 5e-5
LR_ENCODER = 1e-5
LR_SCHEDULER = "cosine"
LR_MIN_FACTOR = 0.01
WARMUP_EPOCHS=3.0

# RESOLUTION = 672
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
GRADIENT_CHECKPOINTING = False # Change to True if Out Of Memory, tapi ini jarang jika menggunakan batch_size=2 dan grad_accum_steps=8

EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
USE_EMA = True
EARLY_STOPPING_USE_EMA = True

EVAL_MAX_DETS = 500
SEED = 42
DEVICE = "cuda"
ACCELERATOR = "gpu"
TENSORBOARD = True
PROGRESS_BAR = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device.upper()}\n")

model = RFDETRSmall(num_queries=500, num_select=500) # Ubah model yang diinginkan, misal RFDETRMedium

model.train(
    dataset_dir=DATASET_DIR,
    output_dir=OUTPUT_DIR,
    # resume=RESUME_WEIGHT, UNCOMMENT IF RESUMING

    epochs=EPOCHS,
    lr=LR,
    lr_encoder=LR_ENCODER,
    lr_scheduler=LR_SCHEDULER,
    lr_min_factor=LR_MIN_FACTOR,
    warmup_epochs=WARMUP_EPOCHS,

    # resolution=RESOLUTION,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,

    early_stopping=EARLY_STOPPING,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA, 
    use_ema=USE_EMA,
    early_stopping_use_ema=EARLY_STOPPING_USE_EMA,

    eval_max_dets=EVAL_MAX_DETS,
    seed=SEED,
    device=DEVICE,
    accelerator=ACCELERATOR,
    tensorboard=TENSORBOARD,
    progress_bar=PROGRESS_BAR,
)
