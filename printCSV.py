import pandas as pd



# New row as a dictionary or list
new_row = {'id': 4, 'Label': 'w'}  # Must match CSV column names

# Convert to DataFrame
new_row_df = pd.DataFrame([new_row])

# Append to CSV
new_row_df.to_csv('file.csv', mode='a', header=False, index=False) 