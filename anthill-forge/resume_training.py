# resume_training.py
import torch
from transformers import Trainer, TrainingArguments

def resume_from_checkpoint(checkpoint_path, output_dir, steps_to_skip=0):
    """Resume training from a checkpoint"""
    # Load model, tokenizer, datasets...
    # Adjust training arguments to start from step
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        # ... other args
        skip_memory_metrics=True,  # Saves time
        dataloader_drop_last=True,  # Avoids partial batches
    )
    
    trainer = Trainer(...)
    trainer.train(resume_from_checkpoint=checkpoint_path)
