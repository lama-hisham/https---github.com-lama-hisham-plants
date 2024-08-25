import zipfile

# Specify the path to the zipped file
zip_file_path = r'C:\Users\Lama\Downloads\Citrus-20240819T101322Z-001.zip'

# Specify the path to extract the files to
extract_dir = r'C:\Users\Lama\Downloads\extractedcitrus'

# Create a ZipFile object
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_dir)