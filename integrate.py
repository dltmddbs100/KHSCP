import pandas as pd
from argparse import ArgumentParser

train = pd.read_csv("./data/train.csv")


def main(args):
    # APEACH
    apeach = pd.read_csv("./data/matched/apeach_matched_unfiltered.csv")
    apeach = apeach[apeach["score"] > args.threshold].reset_index(drop=True)
    train.append(apeach[["HS", "CN"]]).to_csv(
        "./data/integrated/apeach_integrated_unfiltered.csv", index=False
    )

    # BEEP
    beep = pd.read_csv("./data/matched/beep_matched_unfiltered.csv")
    beep = beep[beep["score"] > args.threshold].reset_index(drop=True)
    train.append(beep[["HS", "CN"]]).to_csv(
        "./data/integrated/beep_integrated_unfiltered.csv", index=False
    )

    # Unsmile
    unsmile = pd.read_csv("./data/matched/unsmile_matched_unfiltered.csv")
    unsmile = unsmile[unsmile["score"] > args.threshold].reset_index(drop=True)
    train.append(unsmile[["HS", "CN"]]).to_csv(
        "./data/integrated/unsmile_integrated_unfiltered.csv", index=False
    )

    # KOLD
    kold = pd.read_csv("./data/matched/kold_matched_unfiltered.csv")
    kold = kold[kold["score"] > args.threshold].reset_index(drop=True)
    train.append(kold[["HS", "CN"]]).to_csv(
        "./data/integrated/kold_integrated_unfiltered.csv", index=False
    )

    #####################################################################

    # Threshold Filtered
    unsmile = pd.read_csv("./data/matched/unsmile_matched_unfiltered.csv")
    for i in [0, 0.4, 0.5, 0.6, 0.7, 0.8]:
        unsmile = unsmile[unsmile["score"] > i].reset_index(drop=True)
        train.append(unsmile[["HS", "CN"]]).to_csv(
            f"./data/threshold_integrated/unsmile_{i}_integrated.csv", index=False
        )

    # Threshold Filtered
    kold = pd.read_csv("./data/matched/kold_matched_unfiltered.csv")
    for i in [0, 0.4, 0.5, 0.7, 0.8]:
        kold = kold[kold["score"] > i].reset_index(drop=True)
        train.append(kold[["HS", "CN"]]).to_csv(
            f"./data/threshold_integrated/kold_{i}_integrated.csv", index=False
        )

    print('Sucessfully save integrated dataset')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()
    main(args)