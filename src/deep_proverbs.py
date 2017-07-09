from __future__ import print_function

import csv

import enchant
import numpy as np
import tweepy

from generate_text import main


def check_spelling(phrases, dictionary='en_US'):
    # type: (List[str], str) -> List[Tuple[str, int]]
    """This method takes in a list of phrases and returns a tuple where the first element is the spell corrected phrase
    and the second element is the number of misspellings in that phrase according to dictionary.

    Args:
        phrases: A list of strings that are the phrases we want to spell check
        dictionary: A string describing one of enchants dictionaries

    Returns:
        checked_phrases: A List of Tuples where each phrase has been spell checked and corrected, and is associated with
            the number of misspellings present in that phrase.
    """

    # Create a holder for our list of phrase, number of misspelling Tuples
    checked_phrases = []

    # Define the spelling dictionary
    d = enchant.Dict(dictionary)

    # Step through all phrases and count the number of misspellings along side correcting them
    for phrase in phrases:
        checked_phrases.append(
            reduce(lambda x, y: (x[0] + ' ' + y[0], x[1] + y[1]),
                   map(lambda x: (x, 0) if not x or d.check(x) else (d.suggest(x)[0].lower(), 1), phrase.split(' ')))
        )

    return checked_phrases


def sample_phrase_on_spelling(checked_phrases, smoothing=2):
    # type: (List[Tuple[str, int]], int) -> str
    """This method takes a spell corrected list and returns a single phrase sampled from that list, where the sampling
    rate has been adjusted by the number of misspellings.  The more misspellings there are the less likely that phrase
    will be chosen

    Args:
        checked_phrases: A list of tuples generated from check_spelling.
        smoothing: An integer the determines how much weight to ascribe to misspellings it must be greater than 0

    Returns:
        A phrase string sampled based on the number of misspellings.
    """

    if smoothing <= 0:
        raise 'ERROR:: Smoothing must be greater than 0'

    # Unzip the phrases and their misspelling counts
    phrases, n_misspellings = zip(*checked_phrases)

    # Calculate the inverse of smoothing + the count of misspellings.  This will be used to assign a probability that
    # the phrase should be sampled.  We want phrases that are good, following the assumption that those phrases which
    # have fewer misspellings are probably better phrases.
    inverse_count = map(lambda x: 1.0 / (x + smoothing), n_misspellings)
    probability_list = inverse_count / np.sum(inverse_count)
    return np.random.choice(phrases, 1, p=probability_list)[0]



if __name__ == "__main__":

    model_spec_path = '../models/proverbs_spec.csv'
    model_path = '../models/proverbs-39-0.0943.hdf5'

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
    with open(model_spec_path, mode='rU') as infile:
        reader = csv.reader(infile)
        spec_dict = {rows[0]: rows[1] for rows in reader}

    # Randomly generate 1000 character text segment
    generated_text = main(model_path, spec_dict, 3000)

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

    # Print the original proverbs and their correctly spelled counter parts
    print(valid_proverbs)
    print(check_spelling(valid_proverbs))

    # TODO:: Look for parse consistency

    tweet = sample_phrase_on_spelling(check_spelling(valid_proverbs))

    print(tweet)
    # Update twitter status
    api.update_status(tweet)