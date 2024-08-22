import zipfile

# Specify the path to the zipped file
zip_file_path = r'C:\Users\Lenovo\Downloads\Citrus-20240820T115133Z-001.zip'

# Specify the path to extract the files to
extract_dir = r'C:\Users\Lenovo\Downloads\extractedcitrus'

# Create a ZipFile object
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all files to the specified directory
    zip_ref.extractall(extract_dir)