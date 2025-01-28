import os
import random
from argparse import ArgumentParser
import logging

import torch
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

from utils import (
    read_yaml,
    get_model_config,
    get_tokenizer_config,
    get_split_config,
    get_dataset_config,
    get_trainer_config,
    get_prediction_config,
    get_lora_config,
    _handle_seed,
    DEVICE,
)
from sentiment_dataset import SentimentDataset
from model import (
    load_model,
    load_tokenizer,
    load_lora_model,
    predict,
    compute_metrics,
    METRICS_DICT,
)


logging.basicConfig(
    filename="./logs.txt",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger(__name__)  # Logger for the main script


TRAINING_CONFIG = dict(
    {
        "output_dir": "./training_logs/",
        "overwrite_output_dir": True,
        "do_train": True,
        "run_name": "test_run1",
        "num_train_epochs": 2,
        "per_device_train_batch_size": 128,
        "prediction_loss_only": True,
        "logging_steps": 50,
        "save_steps": 0,
        "seed": _handle_seed(seed_val=None),
        "learning_rate": 0.001,
        "report_to": "tensorboard",
    }
)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        default="./train_config.yaml",
        help="Path to the configuration file (YAML or JSON format).",
    )
    arg_parser.add_argument(
        "--validate",
        type=bool,
        required=False,
        default=True,
        help="Validate on val split.",
    )
    arg_parser.add_argument(
        "--test",
        type=bool,
        required=False,
        default=True,
        help="Validate on val split.",

    )
    arg_parser.add_argument(
        "--predict",
        type=bool,
        required=False,
        default=True,
        help="Predict on samples provided in samples.",
    )
    arg_parser.add_argument(
        "--use_lora",
        type=bool,
        required=False,
        default=False,
        help="Lora for model.",
    )

    
    args = arg_parser.parse_args()
    config = read_yaml(args.config)

    tokenizer_name, tokenizer_path, tokenizer_config = get_tokenizer_config(config)
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_path=tokenizer_path,
        tokenizer_config=tokenizer_config,
    )

    dataset_desc, (train_split_config, val_split_config, test_split_config) = (
        get_split_config(config)
    )

    train_dataset_path, train_dataset_language, train_dataset_config = (
        get_dataset_config(train_split_config)
    )
    train_dataset = SentimentDataset(
        dataset_path=train_dataset_path,
        dataset_language=train_dataset_language,
        split_type="train",
        tokenizer=tokenizer,
        **train_dataset_config,
    )
    sub_split_size = train_dataset_config.pop("sub_split_size", None)
    logger.info(
        f"Loaded Train Dataset: {dataset_desc}, Dataset Length: {len(train_dataset)} with sub_split_size {sub_split_size if sub_split_size else None}."
    )

    if args.validate:
        if not val_split_config:
            raise ValueError(
                "Validation split is missing while the `--validate` argument is set to `True`. Please provide a validation split or disable validation."
            )

        val_dataset_path, val_dataset_language, val_dataset_config = get_dataset_config(
            val_split_config
        )
        val_dataset = SentimentDataset(
            dataset_path=val_dataset_path,
            dataset_language=val_dataset_language,
            split_type="validation",
            tokenizer=tokenizer,
            **val_dataset_config,
        )
        logger.info(
            f"Loaded Val Dataset: {dataset_desc}, Dataset Length: {len(val_dataset)}."
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=train_dataset_config.get("padding", True),
        return_tensors="pt",
    )

    model_name, model_path, base_model_path, model_config = get_model_config(config)
    model_config.update(dict(pad_token_id=tokenizer.pad_token_id))
    model = load_model(
        model_string=model_name,
        model_path=model_path,
        base_model_path=base_model_path,
        model_config=model_config,
    )
    model = model.to(device=DEVICE)

    if args.use_lora:
        lora_config = get_lora_config(config)
        model = load_lora_model(model, lora_config)

    trainer_config = get_trainer_config(config) or TRAINING_CONFIG
    trainer_config["logging_dir"] = os.path.join(
        trainer_config["output_dir"], "runs", trainer_config["run_name"]
    )
    save_trained_model = trainer_config.pop("save_trained_model", True)
    trainer_args = TrainingArguments(**trainer_config)
    trainer = Trainer(
        model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.validate else None,
        data_collator=data_collator,
        compute_metrics=lambda x: (compute_metrics(x, METRICS_DICT)),
    )
    logger.info("Training started.")
    training_outs = trainer.train()
    logger.info("Training finished.")

    if save_trained_model:
        logger.info(f'Saving model at {trainer_config["logging_dir"]}')
        model.save_pretrained(os.path.join(trainer_config["logging_dir"], "model"))

    if args.predict:
        prediction_config, prompt_samples = get_prediction_config(
            config, pop_samples=True
        )
        predcition_classes, labels = predict(
            model, tokenizer, prompt_samples, device=DEVICE, **prediction_config
        )

        logger.info("Logging predicted outputs for sample prompts:")
        for i, (prompt, label) in enumerate(zip(prompt_samples, labels)):
            logger.info(f"Sample {i+1}:")
            logger.info(f'  Prompt: "{prompt}"')
            logger.info(f'  Predicted Output: "{label}"')
