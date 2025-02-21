import os
import random
import logging
from typing import Union, List, Optional
import html

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
    BitsAndBytesConfig,
    LlamaForSequenceClassification,
    LlamaTokenizer,
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
)
import evaluate

from utils import DEVICE
from sentiment_dataset import tokenize_text, _INDEX2LABEL

METRICS_DICT = dict(
    accuracy=evaluate.load("accuracy"), auc=evaluate.load("roc_auc", "multiclass")
)

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


def load_pretrained_gpt2_tokenizer(tokenizer_path="openai-community/gpt2", **kwargs):
    """
    Loads a pre-trained GPT-2 tokenizer and adds a special padding token.
    """
    tokenizer_config = kwargs.pop("config", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        tokenizer.add_special_tokens(dict(pad_token="<|endoftext|>"))
        logger.debug("Special tokens added to the tokenizer.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load GPT-2 tokenizer from {tokenizer_path}: {e}")
        raise


def load_pretrained_gpt2_model(model_path="openai-community/gpt2", **kwargs):
    """
    Loads a pre-trained GPT-2 model with an optional configuration.
    """
    config = kwargs.pop("config", {})

    try:
        model = GPT2ForSequenceClassification.from_pretrained(model_path, **config)
        logger.debug("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load GPT-2 model from {model_path}: {e}")
        raise


def load_pretrained_llama2_tokenizer(
    tokenizer_path="meta-llama/Llama-2-7b-hf", **kwargs
):
    """
    Loads a pre-trained Llama2 tokenizer and adds a special padding token.
    """
    tokenizer_config = kwargs.pop("config", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )
    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Special tokens added to the tokenizer.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load Llama2 tokenizer from {tokenizer_path}: {e}")
        raise


def load_pretrained_base_llama2_model(
    base_model_path="meta-llama/Llama-2-7b-hf",
    **kwargs,
):
    config = kwargs.pop("config", {})
    bnb_config = config.pop("bnb_config", {})

    if isinstance(bnb_config, dict):
        bnb_config = BitsAndBytesConfig(**bnb_config)

    elif isinstance(bnb_config, BitsAndBytesConfig):
        pass

    else:
        raise ValueError(
            f"Expected type dict() or BitsAndBytesConfig() for bnb_config found {type(bnb_config)}"
        )

    if not isinstance(kwargs, dict):
        logger.error(
            f"Found model_config of type: {type(kwargs)}; expected config of type: Dict"
        )
    try:
        model = LlamaForSequenceClassification.from_pretrained(
            base_model_path, quantization_config=bnb_config, **config
        )
        logger.debug("Base model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load base Llama2 model from {base_model_path}: {e}")
        raise


def load_finetuned_llama2_model(
    model_path,
    **kwargs,
):
    if "base_model_path" not in kwargs:
        kwargs.update("base_model_path", "meta-llama/Llama-2-7b-hf")

    try:
        base_model = load_pretrained_base_llama2_model(**kwargs)
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.debug("Finetuned Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Llama2 model from {model_path}: {e}")
        raise


TOKENIZER_DICT = {
    "gpt2": (
        load_pretrained_gpt2_tokenizer,
        {"tokenizer_path": "openai-community/gpt2", "config": {}},
    ),
    "gpt2-xl": (
        load_pretrained_gpt2_tokenizer,
        {"tokenizer_path": "openai-community/gpt2-xl", "config": {}},
    ),
    "llama2": (
        load_pretrained_llama2_tokenizer,
        {"tokenizer_path": "meta-llama/Llama-2-7b-hf", "config": {}},
    ),
}

MODEL_DICT = {
    "gpt2": (
        load_pretrained_gpt2_model,
        {"model_path": "openai-community/gpt2", "config": {}},
    ),
    "gpt2-xl": (
        load_pretrained_gpt2_model,
        {"model_path": "openai-community/gpt2-xl", "config": {}},
    ),
    "llama2-base": (
        load_pretrained_base_llama2_model,
        {
            "base_model_path": "meta-llama/Llama-2-7b-hf",
            "config": {
                "bnb_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, "float16"),
                    bnb_4bit_use_double_quant=False,
                ),
            },
        },
    ),
    "llama2-finetuned": (
        load_finetuned_llama2_model,
        {
            "model_path": None,
            "base_model_path": "meta-llama/Llama-2-7b-hf",
            "config": {
                "bnb_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, "float16"),
                    bnb_4bit_use_double_quant=False,
                ),
            },
        },
    ),
}


def load_tokenizer(tokenizer_name, tokenizer_path=None, tokenizer_config=None):
    """
    Dynamically fetch and initialize a tokenizer based on the tokenizer string.

    Args:
        tokenizer_name (str): The key corresponding to the desired tokenizer in TOKENIZER_DICT.
        tokenizer_path (str, optional): Custom tokenizer path to override the default path in TOKENIZER_DICT.
        tokenizer_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT
    Returns:
        Tokenizer object initialized with the specified parameters.

    Raises:
        ValueError: If the tokenizer string is not registered in TOKENIZER_DICT.
    """
    if tokenizer_name in TOKENIZER_DICT:
        func, kwargs = TOKENIZER_DICT[tokenizer_name]

        # Dynamically update kwargs based on provided arguments
        if tokenizer_path is not None:
            kwargs["tokenizer_path"] = tokenizer_path

        if tokenizer_config is not None:
            kwargs["config"] = tokenizer_config

        logger.info(
            f"Initializing tokenizer '{tokenizer_name}' with arguments: {kwargs}"
        )
        return func(**kwargs)
    else:
        logger.error(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )


def load_model(
    model_string,
    model_path=None,
    base_model_path=None,
    model_config=None,
    device=DEVICE,
):
    """
    Dynamically fetch and initialize a model based on the model string.

    Args:
        model_string (str): The key corresponding to the desired model in MODEL_DICT.
        model_path (str, optional): Custom model path to override the default path in MODEL_DICT.
        model_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT.

    Returns:
        Model object initialized with the specified parameters.

    Raises:
        ValueError: If the model string is not registered in MODEL_DICT.
    """
    if model_string in MODEL_DICT:
        func, kwargs = MODEL_DICT[model_string]

        # Dynamically update kwargs based on provided arguments
        if model_path is not None:
            kwargs["model_path"] = model_path

        if base_model_path is not None:
            kwargs["base_model_path"] = base_model_path

        if model_config is not None:
            kwargs["config"] = model_config

        logger.info(f"Initializing model '{model_string}' with arguments: {kwargs}")
        return func(**kwargs).to(device=device)
    else:
        logger.error(f"Model '{model_string}' is not registered in MODEL_DICT.")
        raise ValueError(f"Model '{model_string}' is not registered in MODEL_DICT.")


def load_lora_model(model, lora_config: dict):
    """
    Load a model with LoRA (Low-Rank Adaptation) configurations for efficient fine-tuning.

    This function prepares the model for low-bit (k-bit) training and applies LoRA configurations
    to integrate parameter-efficient fine-tuning (PEFT) capabilities.

    Args:
        model (PreTrainedModel):
            The base Hugging Face model to which LoRA configurations will be applied.
        lora_config (dict):
            A dictionary containing LoRA configuration parameters. Example keys include:
                - `r` (int): Rank of the low-rank decomposition.
                - `lora_alpha` (float): Scaling factor for LoRA updates.
                - `lora_dropout` (float): Dropout rate for LoRA layers.
                - `bias` (str): Bias handling strategy, e.g., "none", "all", or "lora_only".

    Returns:
        PreTrainedModel:
            The input model modified to support LoRA-based fine-tuning.

    """
    # Step 1: Create LoRA configuration using the provided dictionary
    peft_config = LoraConfig(**lora_config)

    # Step 2: Prepare the model for k-bit training (quantized training support)
    model = prepare_model_for_kbit_training(model)

    # Step 3: Apply the LoRA configuration to the model
    model = get_peft_model(model, peft_config)

    # Return the LoRA-modified model
    return model


def predict(
    model,
    tokenizer,
    prompt_samples,
    use_encode=False,
    device=DEVICE,
):
    """
    Predicts using a pre-trained LLM model based on a given prompt.
    """
    if (
        not prompt_samples
        or isinstance(prompt_samples, str)
        or isinstance(prompt_samples, list)
    ):
        if isinstance(prompt_samples, str):
            prompt_samples = [prompt_samples]
    else:
        logger.error(
            "Invalid prompt provided. Prompt must be a non-empty string or list or strings."
        )
        raise ValueError("Prompt must be a non-empty string.")

    logger.info(f"Generating text for prompt: {prompt_samples}")
    try:
        # Tokenize the input prompt
        logger.debug("Tokenizing the input prompt.")
        tokenized_prompt = tokenize_text(
            prompt_samples,
            tokenizer,
            padding=True,
            use_encode=use_encode,
            return_tensors="pt",
        ).to(device=device)

        # Perform text generation
        logger.debug("Starting sentiment inference.")
        with torch.no_grad():
            logits = model(**tokenized_prompt).logits

        # Decode the generated tokens
        output_classes = logits.argmax(dim=-1).detach().cpu().numpy()
        logger.info("Sentiment classification completed successfully.")

        return output_classes, list(map(_INDEX2LABEL.get, output_classes.tolist()))

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise RuntimeError(f"An error occurred during prediction: {e}")


def compute_metrics(data, metrics):
    predictions, labels = data
    predictions = predictions.astype(dtype=np.float32)
    predict_scores = torch.nn.functional.softmax(
        torch.from_numpy(predictions), dim=-1
    ).numpy()
    predict_labels = torch.from_numpy(predictions).argmax(dim=-1).numpy()

    results = {}
    if "accuracy" in metrics:
        acc_metric = metrics["accuracy"].compute(
            predictions=predict_labels, references=labels
        )
        results["accuracy"] = round(acc_metric["accuracy"], 5)

    if "auc" in metrics:
        auc_metric = metrics["auc"].compute(
            prediction_scores=predict_scores, references=labels, multi_class="ovr"
        )
        results["auc"] = round(auc_metric["roc_auc"], 5)

    return results
