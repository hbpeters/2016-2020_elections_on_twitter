import os
import pandas as pd
import json
import numpy as np

def clean_data(hydrated_twts_path_2016, hydrated_twts_path_2020, output_fp_2016, output_fp_2020, feats_of_interest):
    # Clean 2016 data
    all_days = {}
    for f in os.listdir(hydrated_twts_path_2016): # Loop through files day by day
        onefile_hashtags = get_feats(hydrated_twts_path_2016 + "/" + f, feats_of_interest)
        all_days = {**all_days, **onefile_hashtags} # Merge this day's dict with all_days
    clean = pd.read_json(json.dumps(all_days), orient='index', convert_axes=False)
    #Replace nan with a string containing an empty list
    clean['hashtags'] = clean['hashtags'].replace(np.nan, '[]')

    clean.to_csv(output_fp_2016, index_label="tweet_id")

    # Clean 2020 data
    all_days = {}
    for f in os.listdir(hydrated_twts_path_2020): # Loop through files day by day
        onefile_hashtags = get_feats(hydrated_twts_path_2020 + "/" + f, feats_of_interest)
        all_days = {**all_days, **onefile_hashtags} # Merge this day's dict with all_days
    clean = pd.read_json(json.dumps(all_days), orient='index', convert_axes=False)
    #Replace nan with a string containing an empty list
    clean['hashtags'] = clean['hashtags'].replace(np.nan, '[]')

    clean.to_csv(output_fp_2020, index_label="tweet_id")
    return


def get_tweets(filename):
    print(filename)
    if filename[-9:] != ".DS_Store":
        with open(filename) as fh:
            for tweet in fh:
                yield json.loads(tweet)



def get_feats(filepath, feats_of_interest):
    all_twts_dict = {}
    
    #Calls generator function so that we can read in one tweet at a time
    single_tweets = get_tweets(filepath)
    while(True):
        this_twt_dict = {}
        #Once there are no more tweets in the file, next will return the default ''
        tweet = next(single_tweets, '')
        
        #If there are no more tweets in the file, break out of the while loop
        if tweet == '':
            break
        if tweet['lang'] == "en": # Get only tweets that are in English
            #Gets the hashtags from just the tweet itself NOT from the original tweet if this was a retweet or reply
            for feat in feats_of_interest:
                this_elem = tweet
                feat = feat.split("-") # Get nested keys
                
                if len(feat) > 1: # Must navigate to last nested key to get value
                    for i in range(len(feat)):
                        final_key = feat[i]
                        this_elem = this_elem[final_key]
                        if feat[i] == "hashtags": # Treat "hashtags" list differently
                            this_elem= [dictionary['text'] for dictionary in this_elem if 'text' in dictionary]
                        final_val = this_elem
                else: # Key is not nested
                    final_key = feat[0]
                    final_val = this_elem[final_key]

                this_twt_dict[final_key] = final_val # Add this key, value pair to dictionary for this tweet
            all_twts_dict[tweet['id_str']] = this_twt_dict # Add this tweet's dictionary to dict for all tweets on this day

    return all_twts_dict
    