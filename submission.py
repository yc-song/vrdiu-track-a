import pandas as pd
# Step 1: Read the contents of 'test_predictions.txt' into a dataframe
predictions = []
with open(os.path.join(file_path, 'test_predictions_corrected.txt'), 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        predictions.append({
            'file': parts[0],
            'key_fix_text': parts[1] if parts[1] != 'NULL' else None,
            'label': int(int(parts[2]))
        })

predictions_df = pd.DataFrame(predictions)
# Step 2: Read 'test_dataframe.csv' into a dataframe
test_df = pd.read_csv('./data/test_printed_dataframe.csv')

# Step 3: Check each row in 'test_dataframe.csv' and find corresponding row in 'predictions_df'
def find_label(row):
    prediction_row = predictions_df[
        (predictions_df['file'] == row['file']) &
        (predictions_df['key_fix_text'] == row['key_fix_text'])
    ]
    if not prediction_row.empty:
        return prediction_row.iloc[0]['label']
    else:
        return -1

test_df['label(global_id)'] = test_df.apply(find_label, axis=1)

# Step 4: Write the resulting ID and label to 'submission.csv'
submission_df = test_df[['ID','label(global_id)']]
submission_df.to_csv(os.path.join(file_path, 'submission.csv'), index=False)
print(f'file saved to {file_path}/submission.csv')
