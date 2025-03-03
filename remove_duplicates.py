import pandas as pd
from datetime import datetime

current_date = datetime.now().strftime("%d_%m_%y")
CSV_PATH = f"results_{current_date}.csv"
CSV_PATH = "results_12_02_25.csv"
NEW_CSV_PATH = "preprocessed_"+CSV_PATH
df = pd.read_csv(CSV_PATH)

df = df.groupby(['instance', 'solver', 'users', 'neighbors', 'groups']).agg(
    score=('score', 'max'),          # Take the maximum score
    runtime=('runtime', 'mean'),     # Average the runtime
    buildtime=('buildtime', 'mean')  # Average the buildtime
).reset_index()

df.to_csv(NEW_CSV_PATH, index=False)