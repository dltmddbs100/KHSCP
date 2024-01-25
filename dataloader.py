from transformers import AutoTokenizer, T5Tokenizer


class CNDataset(object):
    def __init__(self, dataset, args, mode):
        self.args = args
        self.mode = mode

        if mode == "train":
            self.dataset = dataset["train"]
        elif mode == "eval":
            self.dataset = dataset["validation"]
        elif mode == "test":
            self.dataset = dataset["test"]

        if "gpt" in self.args.model_path:
            self.model_type = "gpt"

        elif "bart" in self.args.model_path:
            self.model_type = "bart"

        elif "t5" in self.args.model_path:
            self.model_type = "t5"

        if self.model_type == "gpt":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_path,
                bos_token="</s>",
                eos_token="</s>",
                unk_token="<unk>",
                sep_token="<sep>",
                pad_token="<pad>",
            )
            self.bos_token = "</s>"
            self.eos_token = "</s>"
            self.sep_token = "<sep>"

        elif self.model_type == "bart":
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
            self.bos_token = "<s>"
            self.eos_token = "</s>"

        elif self.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_path)

    def tokenize_function(self, examples):
        if self.model_type == "gpt":
            model_inputs = self.tokenizer(
                [
                    self.bos_token + hs + self.sep_token + cn + self.eos_token
                    for hs, cn in zip(examples["HS"], examples["CN"])
                ],
                max_length=self.args.HS_CN_len,
                truncation=True,
            )

            if self.mode == "test":
                model_inputs = self.tokenizer(
                    [self.bos_token + hs + self.sep_token for hs in examples["HS"]],
                    max_length=self.args.HS_max_len,
                    truncation=True,
                )

        elif self.model_type == "bart":
            model_inputs = self.tokenizer(
                [i + self.eos_token for i in examples["HS"]],
                max_length=self.args.HS_max_len,
                truncation=True,
            )
            model_inputs["labels"] = self.tokenizer(
                [i + self.eos_token for i in examples["CN"]],
                max_length=self.args.CN_max_len,
                truncation=True,
            )["input_ids"]

        elif self.model_type == "t5":
            model_inputs = self.tokenizer(
                ["hate speech: " + i for i in examples["HS"]],
                max_length=self.args.HS_max_len,
                truncation=True,
            )
            model_inputs["labels"] = self.tokenizer(
                [i for i in examples["CN"]],
                max_length=self.args.CN_max_len,
                truncation=True,
            )["input_ids"]

        return model_inputs

    def return_dataset(self):
        column_names = self.dataset.column_names

        tokenized_datasets = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on Dataset",
            load_from_cache_file=False,
        )

        return tokenized_datasets
