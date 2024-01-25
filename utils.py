import logging, sys
from argparse import ArgumentParser
from transformers import GPT2Tokenizer


def add_special_tokens(model_path):
    """Returns GPT2 tokenizer after adding separator and padding tokens"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    special_tokens = {"pad_token": "<|pad|>", "sep_token": "<|sep|>"}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer


class Argument:
    def __init__(self):
        self.parser = ArgumentParser()

    def add_args(self):
        self.parser.add_argument(
            "--model_path", type=str, default="gogamza/kobart-base-v1"
        )
        self.parser.add_argument(
            "--test_model_path", type=str, default="gogamza/kobart-base-v1"
        )
        self.parser.add_argument(
            "--path_to_train_data", type=str, default="./data/train.csv"
        )
        self.parser.add_argument(
            "--path_to_valid_data", type=str, default="./data/valid.csv"
        )
        self.parser.add_argument(
            "--path_to_test_data", type=str, default="./data/test.csv"
        )

        self.parser.add_argument("--output_dir", type=str, default="./runs")
        self.parser.add_argument("--run_name", type=str, default="baseline")

        self.parser.add_argument("--batch_size", type=int, default="64")
        self.parser.add_argument("--max_epochs", type=int, default="5")

        self.parser.add_argument("--HS_max_len", type=int, default="80")
        self.parser.add_argument("--CN_max_len", type=int, default="80")
        self.parser.add_argument("--HS_CN_len", type=int, default="160")

        self.parser.add_argument("--learning_rate", type=float, default="5e-5")
        self.parser.add_argument("--weight_decay", type=float, default="1e-3")
        self.parser.add_argument("--label_smoothing_factor", type=float, default="0.1")

        self.parser.add_argument("--fp16", action="store_true")
        self.parser.add_argument("--local_rank", type=int, default="-1")
        args = self.parser.parse_args()

        self.print_args(args)

        return args

    def print_args(self, args):
        print("====== Input arguments ======")

        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:
                print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:
                print("\t", key, ":", value, "\n}")
            else:
                print("\t", key, ":", value)


def make_logger(args, name=None):
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(filename=f"logs/{args.run_name}.log")

    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
