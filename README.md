# dreambooth-hackathon
Simple repository for training dreambooth models largely based off the huggingface diffusion course.

Training usage:
```
usage: Dreambooth training script [-h] [--files FILES] [--output-dir OUTPUT_DIR] [--instance-prompt INSTANCE_PROMPT] [--lr LR] [--steps STEPS]
                                  [--accumulation ACCUMULATION] [--seed SEED] [--batch-size BATCH_SIZE] [--max-grad-norm MAX_GRAD_NORM] [--gradient-checkpoint]
                                  [--use-8bit-adam]

optional arguments:
  -h, --help            show this help message and exit
  --files FILES         Path to training images
  --output-dir OUTPUT_DIR
                        Path to save dreambooth models.
  --instance-prompt INSTANCE_PROMPT
                        Instance prompt to use
  --lr LR               Learning rate
  --steps STEPS         Number of training steps
  --accumulation ACCUMULATION
                        Number of gradient accumulation steps
  --seed SEED           RNG seed
  --batch-size BATCH_SIZE
                        Batch size
  --max-grad-norm MAX_GRAD_NORM
                        Gradient clip bound
  --gradient-checkpoint
                        Apply gradient checkpointing
  --use-8bit-adam       Use 8-bit adam optimizer
```