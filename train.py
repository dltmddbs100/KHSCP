import wandb, os

from datasets import load_dataset
from dataloader import CNDataset
from utils import Argument

import torch
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
)


def main(args):
    if "gpt" in args.model_path:
        model_type = "gpt"
    elif "bart" in args.model_path:
        model_type = "bart"
    elif "t5" in args.model_path:
        model_type = "t5"

    if model_type == "bart":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    elif model_type == "gpt":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            bos_token="</s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="<sep>",
            pad_token="<pad>",
        )
    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": args.path_to_train_data,
            "validation": args.path_to_valid_data,
        },
    )

    train_dataset = CNDataset(dataset, args, "train").return_dataset()
    eval_dataset = CNDataset(dataset, args, "eval").return_dataset()

    if (model_type == "bart") | (model_type == "t5"):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    elif model_type == "gpt":
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if args.local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(torch.distributed.get_rank())
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        if dist.get_rank() == 0:
            wandb.init(project="Korean_CN", name=args.run_name)
    else:
        wandb.init(project="KHSCP", name=args.run_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, args.run_name),
        report_to="wandb",
        run_name=args.run_name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        do_train=True,
        do_eval=True,
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing_factor,
        fp16=args.fp16,
        load_best_model_at_end=True,
        predict_with_generate=True,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()

    wandb.finish()


if __name__ == "__main__":
    args = Argument().add_args()
    main(args)
