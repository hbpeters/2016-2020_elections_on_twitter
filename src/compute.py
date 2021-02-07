import pandas as pd
import ast
import warnings
from textblob import TextBlob

warnings.filterwarnings('ignore')

def analyze(fp_16_data, fp_20_data, l_htags_16, r_htags_16, l_htags_20, r_htags_20, left_users, right_users):
    six = pd.read_csv(fp_16_data)
    twenty = pd.read_csv(fp_20_data)

    #### Call function for Sentiment analysis ####
    six['tweetPolarity'] = six['full_text'].apply(sentiment_polarity)
    twenty['tweetPolarity'] = twenty['full_text'].apply(sentiment_polarity)
    six['tweetSubjectivity'] = six['full_text'].apply(sentiment_subjectivity)
    twenty['tweetSubjectivity'] = twenty['full_text'].apply(sentiment_subjectivity)

    l_16, r_16 = get_l_and_r(six, l_htags_16, r_htags_16, left_users, right_users)
    l_l_dialogue_16, l_r_dialogue_16, mentioned_by_l_16, r_l_dialogue_16, r_r_dialogue_16, mentioned_by_r_16 = get_dialogue(l_16, r_16, left_users, right_users)

    l_20, r_20 = get_l_and_r(twenty, l_htags_20, r_htags_20, left_users, right_users)
    l_l_dialogue_20, l_r_dialogue_20, mentioned_by_l_20, r_l_dialogue_20, r_r_dialogue_20, mentioned_by_r_20 = get_dialogue(l_20, r_20, left_users, right_users)


def get_top_n_hashtags(clean, top_n):
    non_hashtags = clean['hashtags'].loc[clean['hashtags'].str.startswith("[") != True]
    clean_hashtags = clean.drop(non_hashtags.index, axis=0)
    
    clean_hashtags['hashtags'] = clean_hashtags['hashtags'].apply(lambda x: x.lower()) # Convert to lowercase
    clean_hashtags['hashtags'] = clean_hashtags['hashtags'].apply(lambda x: ast.literal_eval(x)) # Turn the string of iterables into a list

    all_hashtags = clean_hashtags['hashtags'].explode().dropna()
    top_n_hashtags = all_hashtags.value_counts().head(n=top_n)
    return top_n_hashtags


def search_keywords(df, col, keywords):
    "Selects subset of df that contains at least one of the keywords in the specified col"
    pattern = '|'.join(keywords)
    df = df[df[col].str.contains(pattern)]
    return df


def get_twts_for_users(df, user_list):
    return df[df['screen_name'].isin(user_list)]

def sentiment_polarity(text):
    blob = TextBlob(text) 
    polar = blob.sentiment.polarity
    return polar

def sentiment_subjectivity(text):
    blob = TextBlob(text) 
    sub = blob.sentiment.subjectivity
    return sub

def filter_by_kwords_and_usrs(df, keywords, users):
    "Get tweets that contain at least one keyword from a list of keywords or are by one of the listed users"
    filtered_kwords = search_keywords(df, 'full_text', keywords)
    filtered_usrs = get_twts_for_users(df, users)
    return filtered_kwords.merge(filtered_usrs, how='outer')


def get_l_and_r(df, l_htags, r_htags, left_users, right_users):
    left = search_keywords(df, 'full_text', l_htags)
    right = search_keywords(df, 'full_text', r_htags)
    intersect = left.merge(right)
    
    intersect.set_index('tweet_id', inplace=True)
    left.set_index('tweet_id', inplace=True)
    right.set_index('tweet_id', inplace=True)

    # Drop all tweets that use a both right and left leaning hashtag
    left.drop(intersect.index, inplace=True)
    right.drop(intersect.index, inplace=True)
    
    # Drop all users who have tweets that contain both left and right leaning hashtags
    r_users = list(right['screen_name'].unique())
    l_users = list(left['screen_name'].unique())
    l_r_users = [usr for usr in l_users if usr in r_users]
    l_users = [usr for usr in l_users if usr not in l_r_users]
    r_users = [usr for usr in r_users if usr not in l_r_users]
    
    # Add l & r leaning news sites and politicians -- guaranteed to be l and r
    l_users = l_users + left_users
    r_users = r_users + right_users
    # Get all tweets by these users
    left = get_twts_for_users(df, l_users)
    right = get_twts_for_users(df, r_users)
    
    # Assign leanings
    left['leaning'] = 'L'
    right['leaning'] = 'R'
    return left, right


def get_dialogue(left, right, left_users, right_users):    
    l_users = list(left['screen_name'].unique())
    r_users = list(right['screen_name'].unique())
    
    # Add l & r leaning news sites and politicians -- guaranteed to be l and r
    l_users += left_users
    r_users += right_users
    
    def classify_dialogue(df):
        users_mentioned = []
        mentions = df[df['user_mentions'] != "[]"] # Get tweets that contain mentions
        mentions['user_mentions'] = mentions['user_mentions'].apply(lambda x: eval(x)) # Convert str to list
        mentions['mentions_leaning'] = ""
        for ind, row in mentions.iterrows():
            mentions_leaning = "" # Start off with inconclusive
            for usr in row['user_mentions']:
                users_mentioned.append(usr)
                if usr in l_users:
                    mentions_leaning += "L"
                elif usr in r_users:
                    mentions_leaning += "R"
                else: # User mentions consist of an inconclusive user
                    mentions_leaning += "M"
            mentions.at[ind, 'mentions_leaning'] = mentions_leaning
            
        users_mentioned = pd.Series(users_mentioned).value_counts()
        polarized = mentions[~mentions["mentions_leaning"].str.contains("M")] # Get rid of rows with inconclusive users
        l_polarized = polarized[~polarized["mentions_leaning"].str.contains("R")]
        r_polarized = polarized[~polarized["mentions_leaning"].str.contains("L")] 
        return l_polarized, r_polarized, users_mentioned
    
    l_l_dialogue, l_r_dialogue, mentioned_by_l = classify_dialogue(left)
    r_l_dialogue, r_r_dialogue, mentioned_by_r = classify_dialogue(right)
    return l_l_dialogue, l_r_dialogue, mentioned_by_l, r_l_dialogue, r_r_dialogue, mentioned_by_r