"""
This script is a quick and dirty implementation which takes in a file path and cleans up the text by:
1.  Converting all capitals to lower case
2. removing all numerals
3. removing all special characters except ,."'

It will save a file in the same path with the addition of _clean to the file name.

To run the script on an example file type
python clean_text.py --path /path/to/file/ --files file_name_1 file_name_2
"""

import argparse
import re


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reads in txt file and cleans the text')
    parser.add_argument('--path', type=str, help='Path to where all files are located')
    parser.add_argument('--files', metavar='N', type=str, nargs='+', help='list of files that we want to clean')
    args = parser.parse_args()

    for file_name in args.files:
        # Load in Raw Text
        path_to_file = "{path}{file_name}".format(path=args.path, file_name=file_name)
        raw_text = open(path_to_file).read().lower()

        # Remove punctuation and other unwanted characters
        cleaned_text = re.sub("[^A-Za-z\,\.\"\']+", ' ', raw_text)

        # Extract the name from the file name
        name = file_name.split('.')[0]

        # save the file if we feel like it
        temp_file = open('{path}{file_name}_clean.txt'.format(path=args.path, file_name=name), 'w')
        temp_file.write(cleaned_text)
        temp_file.close()