import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, models

from argparse import ArgumentParser
from tqdm import tqdm

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocessing(data_path):
    if "unsmile" in data_path:
        data_type = "unsmile"
        unsmile = pd.read_csv(data_path)

        unsmile = unsmile[(unsmile["clean"] != 1)].reset_index(drop=True)

        unsmile = unsmile.rename(columns={"문장": "HS"})

        return list(unsmile["HS"]), data_type

    elif "apeach" in data_path:
        data_type = "apeach"
        apeach = pd.read_csv(data_path)

        apeach = apeach.rename(columns={"text": "HS"})

        return list(apeach["HS"]), data_type

    elif "beep" in data_path:
        data_type = "beep"
        beep = pd.read_csv(data_path)

        beep = beep[beep.hate != "none"].reset_index(drop=True)

        beep = beep.rename(columns={"comments": "HS"})

        return list(beep["HS"]), data_type

    elif "kold" in data_path:
        data_type = "kold"
        kold = pd.read_json(data_path)

        kold = kold[(kold["OFF"] == True)].reset_index(drop=True)

        kold = kold.rename(columns={"comment": "HS"})

        return list(kold["HS"]), data_type


def main(args):
    word_embedding_model = models.Transformer(
        "BM-K/KoSimCSE-roberta-multitask", max_seq_length=128
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True
    )

    embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model]).cuda()

    train = pd.read_csv("./data/train.csv")

    hs = list(train["HS"])
    cn = list(train["CN"])
    corpus_embeddings = embedder.encode(hs, convert_to_tensor=True)

    queries, data_type = preprocessing(args.data_path)

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 1
    matched_CN, matched_score = [], []

    for query in tqdm(queries):
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        matched_CN.append(cn[top_results[1]])
        matched_score.append(top_results[0].item())

    matched = pd.DataFrame({"HS": queries, "CN": matched_CN, "score": matched_score})

    matched.to_csv(f"./data/matched/{data_type}_matched_unfiltered.csv", index=False)

    print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/mono_hs_data/unsmile/unsmile_full.csv")
    args = parser.parse_args()
    main(args)
