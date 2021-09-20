import pandas as pd
from sklearn.model_selection import train_test_split


data_df = pd.read_csv("../../data/gold_analyses/full.csv")

# selecting only the examples having rule id
# (otherwise compound types are considered occasional)
columns = ['word_d', 'pos_d', 'words_m', 'poss_m', 'word_h', 'pos_h']
no_nan_data = []
for i in range(len(data_df)):
    record = data_df.iloc[i]
    rule_d = record['rule_d']
    rule_h = record['rules_h']
    if isinstance(rule_d, str):
        rule = rule_d
    elif isinstance(rule_h, str):
        rule = rule_h
    else:
        continue
    no_nan_data.append([rule] + record[columns].tolist())
sub_df = pd.DataFrame(data=no_nan_data, columns=["rule"] + columns)

# splitting data into train, val and test
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

val_part = 0.1  # 10% of non-test data
test_part = 0.2

rules_d = set()
rules_h = set()

for r in sub_df['rule'].unique():
    grouped = sub_df[sub_df['rule'] == r]
    try:
        data_trainval, data_test = train_test_split(
            grouped,
            train_size=1 - test_part,
            test_size=test_part,
            random_state=42
        )
    except ValueError:
        # 1 sample goes to trainval
        data_trainval = grouped
        data_test = pd.DataFrame()
    try:
        data_train, data_val = train_test_split(
            data_trainval,
            train_size=1 - val_part,
            test_size=val_part,
            random_state=42
        )
    except ValueError:
        # 1 sample goes to train
        data_train = data_trainval
        data_val = pd.DataFrame()

    train_df = pd.concat([train_df, data_train])
    val_df = pd.concat([val_df, data_val])
    test_df = pd.concat([test_df, data_test])

print(
    f"train_size={len(train_df)}, "
    f"val_size={len(val_df)}, "
    f"test_size={len(test_df)}"
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv("../../data/gold_analyses/train.csv", index=False)
val_df.to_csv("../../data/gold_analyses/val.csv", index=False)
test_df.to_csv("../../data/gold_analyses/test.csv", index=False)
