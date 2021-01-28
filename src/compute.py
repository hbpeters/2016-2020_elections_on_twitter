import pandas as pd
import ast

def analyze(df_16, df_20, l_htags_16, r_htags_16, l_htags_20, r_htags_20):
    

def search_keywords(df, col, keywords):
    "Selects subset of df that contains at least one of the keywords in the specified col"
    pattern = '|'.join(keywords)
    df = df[df[col].str.contains(pattern)]
    return df


def get_twts_for_users(df, user_list):
    return df[df['screen_name'].isin(user_list)]


def filter_by_kwords_and_usrs(df, keywords, users):
    "Get tweets that contain at least one keyword from a list of keywords or are by one of the listed users"
    filtered_kwords = search_keywords(df, 'full_text', keywords)
    filtered_usrs = get_twts_for_users(df, users)
    return filtered_kwords.merge(filtered_usrs, how='outer')


def get_l_and_r(df, l_htags, r_htags):
    left = search_keywords(df, 'full_text', r_htags)
    right = search_keywords(df, 'full_text', l_htags)
    intersect = left.merge(right)
    
    intersect.set_index('tweet_id', inplace=True)
    left.set_index('tweet_id', inplace=True)
    right.set_index('tweet_id', inplace=True)

    # Drop inconclusive cols
    left.drop(intersect.index, inplace=True) 
    right.drop(intersect.index, inplace=True)
    
    # Get all tweets by these users
    left_users = list(left['screen_name'].unique())
    left = get_twts_for_users(df, left_users)
    right_users = list(right['screen_name'].unique())
    right = get_twts_for_users(df, right_users)
    print(left.shape)
    print(right.shape)
    
    # Assign leanings
    left['leaning'] = 'L'
    right['leaning'] = 'R'

    return left, right