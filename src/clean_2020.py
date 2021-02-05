import os
import pandas as pd

def search_keywords(df, col, keywords):
    "Selects subset of df that contains at least one of the keywords in the specified col"
    pattern = '|'.join(keywords)
    df = df[df[col].str.contains(pattern)]
    return df

def get_twts_for_users(df, user_list):
    return df[df['screen_name'].isin(user_list)]

def filter_by_kwords_and_users(df, keywords, users):
    "Get tweets that contain at least one keyword from a list of keywords or are by one of the listed users"
    filtered_kwords = search_keywords(df, 'full_text', keywords)
    filtered_usrs = get_twts_for_users(df, users)
    return filtered_kwords.merge(filtered_usrs, how='outer')

keywords_2016 = ["philly convention", "philadelphia convention", "democratic convention", "dnc convention", 
            "#demsinphilly", "#dnc", "#philly", "#demconvention", "#electionday", "#decision2016", "election2016", 
            "election", "clinton", "kaine", "trump", "pence", "#debate", "#debates", "#debatenight", 
            "#debate2016", "#debates2016", "gop cleveland", "cleveland convention", "gop convention", 
            "#gopconvention", "#cleveland", "#rnc", "#gop", "#2016cle", "#rncincle", 
            "#republicationnationalconvention", "vice-presidential debate", "vice presidential debate", 
            "vp debate"]
keywords_2020 = ["biden", "harris", "kamala", "#decision2020", "election2020", "debate2020", "debates2020",
                "#demsinmilwaukee", "#milwaukee"]

users = ["timkaine", "SenSanders", "BernieSanders", "MartinOMalley", "HillaryClinton", "SenateDems", 
        "HouseDemocrats", "TheDemocrats", "GovPenceIN", "mike_pence", "GovChristie", "gov_gilmore", 
        "RealBenCarson", "JebBush", "marcorubio", "realDonaldTrump", "JohnKasich", "tedcruz", "GovMikeHuckabee",
        "ChrisChristie", "RandPaul", "CarlyFiorina", "RickSantorum", "SenateGOP", "HouseGOP", "GOP",
        "debates", "GOPconvention", "KamalaHarris", "SenKamalaHarris"]

all_keywords = keywords_2016 + keywords_2020
output_dir_2020 = "/Users/hannahpeterson/Desktop/DATA_SCIENCE/dsc180/2016-2020_elections_on_twitter/data/temp/2020/clean"
df = pd.DataFrame()
for f in os.listdir(output_dir_2020): 
    this_df = pd.read_csv(output_dir_2020 + "/" + f)
    df = df.append(this_df)
df.dropna(subset=['full_text'], inplace=True)

filtered_2020 = filter_by_kwords_and_users(df, all_keywords, users)
print(filtered_2020.shape)
filtered_2020.to_csv(output_dir_2020 + "/filtered_tweets.csv", index_label="tweet_id")