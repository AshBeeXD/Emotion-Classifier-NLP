from transformers import Trainer, TrainingArguments

def get_training_args(output_dir="outputs/model"):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="outputs/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

def train_model(model, args, train_dataset, val_dataset, compute_metrics):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

