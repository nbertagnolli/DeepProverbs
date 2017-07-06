from __future__ import print_function

import csv
import tweepy

from generate_text import main


if __name__ == "__main__":

    # Parse the credentials for the twitter bot
    with open('../twitter.cred', mode='rU') as infile:
        reader = csv.reader(infile)
        cred_dict = {rows[0]: rows[1] for rows in reader}

    # Set the credentials based on the credentials file
    CONSUMER_KEY = cred_dict['consumer_key']
    CONSUMER_SECRET = cred_dict['consumer_secret']
    ACCESS_KEY = cred_dict['access_key']
    ACCESS_SECRET = cred_dict['access_secret']
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    api = tweepy.API(auth)

    # Parse the spec file and extract model parameters
    with open('../models/proverbs_spec.csv', mode='rU') as infile:
        reader = csv.reader(infile)
        spec_dict = {rows[0]: rows[1] for rows in reader}

    # Randomly generate 1000 character text segment
    generated_text = main('../models/proverbs-39-0.0943.hdf5', spec_dict, 3000)

    # Parse out all "sentences" by splitting on '.'
    split_text = generated_text.split('.')

    # Load in original text
    original_text = open(spec_dict['file_path']).read()

    # Loop and check conditions of a deep proverb
    valid_proverbs = []
    is_good_proverb = False
    for proverb in split_text:
        # make sure proverb isn't in the original text and it is less than 140 characters
        if proverb not in original_text and len(proverb) <= 140:
            valid_proverbs.append(proverb)

    print(valid_proverbs)

    # TODO:: Spell check using pyenchant to both check misspellings and suggest words

    # TODO:: Look for parse consistency

    tweet = valid_proverbs[0]  # TODO:: DO A RANDOM SAMPLE

    # Update twitter status
    api.update_status(tweet)