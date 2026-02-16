# import pandas as pd
# from datetime import datetime

# current_date = datetime.now().strftime("%d_%m_%y")
# CSV_PATH = f"results_{current_date}.csv"
# CSV_PATH = "results_12_02_25.csv"
# NEW_CSV_PATH = "preprocessed_"+CSV_PATH
# df = pd.read_csv(CSV_PATH)

# df = df.groupby(['instance', 'solver', 'users', 'neighbors', 'groups']).agg(
#     score=('score', 'max'),          # Take the maximum score
#     runtime=('runtime', 'mean'),     # Average the runtime
#     buildtime=('buildtime', 'mean')  # Average the buildtime
# ).reset_index()

# df.to_csv(NEW_CSV_PATH, index=False)

import pandas as pd

# Load the CSV file
df = pd.read_csv("final.csv")

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Save the cleaned data back to CSV
df_cleaned.to_csv("final_cleaned.csv", index=False)

print("Duplicate rows removed. Cleaned file saved as 'final_cleaned.csv'.")
