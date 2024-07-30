import pandas as pd

# Function to check for overlapping keys in the same image file and fix common mistakes
def check_and_fix_overlapping_keys(file_path):
    # Read the file into a pandas DataFrame, specifying column names
    df = pd.read_csv(file_path, header=None, names=['image_file_name', 'key', 'label'])

    def get_duplicate_rows(df):
        duplicates = df[df.duplicated(subset=['image_file_name', 'key'], keep=False) & (df['key'] != ' NULL')]
        return duplicates

    # Get the duplicate rows
    duplicates = get_duplicate_rows(df)

    if not duplicates.empty:
        for index, row in duplicates.iterrows():
            print(f"Image File: {row['image_file_name']}, Key: {row['key']}, Label: {row['label']}")

        for image_file_name in duplicates['image_file_name'].unique():
            for key in duplicates[duplicates['image_file_name'] == image_file_name]['key'].unique():
                duplicate_rows = df[(df['image_file_name'] == image_file_name) & (df['key'] == key)]
                first_index = duplicate_rows.index[0]
                for i in duplicate_rows.index[1:]:
                    used_queries = df[df['image_file_name'] == image_file_name]['key'].unique()
                    available_queries = set([
                        ' company name', ' substantial holder name', ' holder ACN/ARSN',
                        ' There was a change in the interests of the substantial holder on',
                        ' The previous notice was dated', ' The previous notice was given to the company on',
                        ' class of securities', ' Previous notice Person\'s notes', ' Previous notice Voting power',
                        ' Present notice Person\'s votes', ' Present notice Voting power', ' company ACN/ARSN'
                    ]) - set(used_queries)
                    print(available_queries, used_queries)
                    if available_queries:
                        new_key = find_most_appropriate_key(available_queries, key)
                        df.at[i, 'key'] = new_key
                        print(f"Changed Key: {key} to {new_key} for Image File: {image_file_name}")
    else:
        print("No overlapping keys found.")




    # Apply category rule
    categories = {
        'company name': [5, 7],
        'substantial holder name': [5, 7],
        'holder ACN/ARSN': [5, 7],
        'There was a change in the interests of the substantial holder on': [5, 7],
        'The previous notice was dated': [5, 7],
        'The previous notice was given to the company on': [5, 7],
        'class of securities': [5, 7],
        'Previous notice Person\'s notes': [5, 7],
        'Previous notice Voting power': [5, 7],
        'Present notice Person\'s votes': [5, 7],
        'Present notice Voting power': [5, 7],
        'company ACN/ARSN': [5, 7]
    }

    for index, row in df.iterrows():
        if row['key'] in categories:
            allowed_categories = categories[row['key']]
            # Mock function to get the category of a global_id, replace this with actual logic
            category = get_category_of_global_id(row['label'])
            if category not in allowed_categories:
                df.at[index, 'key'] = ' NULL'
                print(f"Set Key: {row['key']} to NULL for Image File: {row['image_file_name']} due to category {category}")

    # Save the corrected data to a new file
    output_file_path = file_path.replace('.txt', '_corrected.txt')
    print(df)
    with open(output_file_path, 'w') as f:
        for index, row in df.iterrows():
            f.write(f"{row['image_file_name']},{row['key']}, {row['label']}\n")
    print(f"Corrected data saved to {output_file_path}")

# Mock function to get the category of a global_id, replace this with actual logic
def get_category_of_global_id(global_id):
    # This function should return the category based on global_id
    # Replace this mock logic with the actual logic to get the category of the global_id
    return 5  # Mock category for demonstration
def find_most_appropriate_key(available_queries, original_key):
    relationships = {
        ' company name': ' substantial holder name',
        ' substantial holder name': ' company name',
        ' holder ACN/ARSN': ' company ACN/ARSN',
        ' company ACN/ARSN': ' holder ACN/ARSN',
        ' There was a change in the interests of the substantial holder on': [
            ' The previous notice was dated', ' The previous notice was given to the company on'
        ],
        ' The previous notice was dated': [
            ' There was a change in the interests of the substantial holder on', ' The previous notice was given to the company on'
        ],
        ' The previous notice was given to the company on': [
            ' There was a change in the interests of the substantial holder on', ' The previous notice was dated'
        ],
        ' Previous notice Voting power': ' Present notice Voting power',
        ' Present notice Voting power': ' Previous notice Voting power',
        ' Previous notice Person\'s notes': ' Present notice Person\'s votes',
        ' Present notice Person\'s votes': ' Previous notice Person\'s notes'
    }

    if original_key in relationships:
        related_keys = relationships[original_key]
        if isinstance(related_keys, list):
            for key in related_keys:
                if key in available_queries:
                    return key
        else:
            if related_keys in available_queries:
                return related_keys

    # If no related key is available, return any available key
    return available_queries.pop()

# Path to your text file
# file_path = '/shared/s3/lab07/jongsong/unilm/layoutlmv3/models_10000_2e-6_large_test_batch3/test_predictions.txt'
check_and_fix_overlapping_keys(os.path.join(file_path, 'test_predictions.txt'))
