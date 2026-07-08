import pandas as pd
from sklearn.model_selection import train_test_split
# 1. CSV-Datei einlesen
# Passe 'deine_daten.csv' an den echten Dateinamen an
df = pd.read_csv('images/labels.csv')

# 2. Stratified Split durchführen
# 'stratify=df["label"]' zwingt den Algorithmus, aus jedem Rassen-Block
# exakt das gleiche Verhältnis (80 zu 20) zu ziehen.
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42, # Garantiert, dass der Split immer identisch bleibt
    stratify=df['label'] # HIER wird die Block-Sortierung aufgehoben!
)
# 3. Kennzeichnung in der Original-Tabelle eintragen
df['split'] = 'train'
df.loc[val_df.index, 'split'] = 'val'

# 4. Qualitätskontrolle: Verteilung prüfen
print("--- Verteilung im Trainingsset ---")
print(df[df['split'] == 'train']['label'].value_counts())

print("\n--- Verteilung im Validierungsset ---")
print(df[df['split'] == 'val']['label'].value_counts())

# 5. Speichern der neuen Zuweisung
df.to_csv('dataset_final_split.csv', index=False)
print("\nErfolgreich gespeichert in 'dataset_final_split.csv'!")