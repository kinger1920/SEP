import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('images/labels.csv')


train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['label'] 
)

df['split'] = 'train'
df.loc[val_df.index, 'split'] = 'val'

print("--- Verteilung im Trainingsset ---")
print(df[df['split'] == 'train']['label'].value_counts())

print("\n--- Verteilung im Validierungsset ---")
print(df[df['split'] == 'val']['label'].value_counts())

df.to_csv('dataset_final_split.csv', index=False)