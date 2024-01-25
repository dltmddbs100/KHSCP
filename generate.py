import re
import json
import pandas as pd
from tqdm import tqdm

import torch

from konlpy.tag import Mecab

from utils import Argument
from dataloader import CNDataset
from torch.utils.data import DataLoader
from datasets import load_dataset

from rouge_score import Rouge
from evaluate import load
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    T5Tokenizer,
)


def generation(args):
    references = list(pd.read_csv(args.path_to_test_data)["CN"])

    if "gpt" in args.test_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.test_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.test_model_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    elif "bart" in args.test_model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.test_model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.test_model_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    elif "t5" in args.test_model_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.test_model_path)
        tokenizer = T5Tokenizer.from_pretrained(args.test_model_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    dataset = load_dataset("csv", data_files={"test": args.path_to_test_data})

    test_dataset = CNDataset(dataset, args, "test").return_dataset()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=data_collator,
    )

    model.cuda()
    model.eval()

    test_sum = []

    for ind, batch in tqdm(enumerate(test_dataloader)):
        sets = batch["input_ids"].cuda()

        with torch.no_grad():
            batch_sum = model.generate(
                input_ids=sets,
                max_length=512,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )

        test_sum = [*test_sum, *batch_sum]

    # decode outputs
    if "gpt" in args.test_model_path:
        test_sum_sent = [
            tokenizer.decode(
                g, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for g in test_sum
        ]
        predictions = [
            re.sub("<pad>|</s>|<pad>", "", i.split("<sep>")[-1]).strip()
            for i in test_sum_sent
        ]

    elif "bart" in args.test_model_path:
        predictions = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for g in test_sum
        ]

    elif "t5" in args.test_model_path:
        predictions = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for g in test_sum
        ]

    elif "m2m" in args.test_model_path:
        predictions = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            for g in test_sum
        ]

    # Load
    mecab = Mecab()
    bleu = load("bleu")
    rouge = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
    )

    # BLEU
    bl1 = bleu.compute(
        predictions=predictions,
        references=references,
        tokenizer=mecab.morphs,
        max_order=1,
    )
    bl3 = bleu.compute(
        predictions=predictions,
        references=references,
        tokenizer=mecab.morphs,
        max_order=3,
    )
    bl4 = bleu.compute(
        predictions=predictions,
        references=references,
        tokenizer=mecab.morphs,
        max_order=4,
    )

    # ROUGE
    rouge_score = rouge.get_scores(predictions, references)
    rouge_1 = rouge_score["rouge-1"]["f"]
    rouge_2 = rouge_score["rouge-2"]["f"]
    rouge_l = rouge_score["rouge-l"]["f"]

    results = {
        "prediction": predictions,
        "bleu_1": bl1["bleu"] * 100,
        "bleu_3": bl3["bleu"] * 100,
        "bleu_4": bl4["bleu"] * 100,
        "rouge-1": rouge_1 * 100,
        "rouge-2": rouge_2 * 100,
        "rouge-l": rouge_l * 100,
    }

    with open(
        f"./results/{args.run_name}_results.json", "w", encoding="utf-8"
    ) as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

    print("\nDone!")

    return results


if __name__ == "__main__":
    args = Argument().add_args()
    generation(args)
