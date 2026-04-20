import wandb
import os

print("Testing WandB connection...")
try:
    run = wandb.init(
        entity="sihang-personal",
        project="debug-test",
        name="test-run"
    )
    print("Successfully initialized WandB run!")
    run.finish()
except Exception as e:
    print(f"FAILED to initialize WandB run: {e}")
