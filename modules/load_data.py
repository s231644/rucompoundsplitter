import pandas as pd


def get_gold_data():
    df = pd.read_csv("../data/compounds_gold.csv")
    df = df[df["pos_d"] == "adj"]
    df = df[pd.isna(df["rules_h"])]
    df1 = df[df["pos_h"] == "adj"]
    df2 = df[df["pos_h"] == "part"]
    df = df1.append(df2).reset_index().drop(
        columns=["index", "order", "rules_h", "rules_am"],
        axis=1
    )

    gold_data = df.values
    return gold_data


def get_fsm_data():
    df = pd.read_csv("../data/generated_comp_adj.txt", header=None, sep="\t")
    fsm_data = df.values
    return fsm_data


def get_data():
    gold_data = get_gold_data()
    gold_compounds = dict()
    for l in gold_data:
        l = l.tolist()
        word_c, pos_c, word_m, pos_m, word_h, pos_h = l
        gold_compounds[(word_c, pos_c)] = (word_m, pos_m, word_h, pos_h)

    fsm_data = get_fsm_data()
    examples = []
    for l in fsm_data:
        l = l.tolist()
        word_c, pos_c, rule_c, word_m, pos_m, rule_m, word_h, pos_h, rule_h = l
        if (word_c, pos_c) in gold_compounds:
            if gold_compounds[(word_c, pos_c)] == (word_m, pos_m, word_h, pos_h):
                examples.append((l, 1))
            else:
                examples.append((l, 0))

    return examples


examples = get_data()
with open("../data/data_adj.txt", "w") as f:
    for l, score in examples:
        f.write("\t".join(l))
        f.write(f"\t{score}\n")
