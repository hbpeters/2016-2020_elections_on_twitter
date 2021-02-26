import pandas as pd
import ast
import warnings
from textblob import TextBlob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

warnings.filterwarnings('ignore')

def analyze(fp_16_data, fp_20_data, l_htags_16, r_htags_16, l_htags_20, r_htags_20, left_users, right_users):
    six = pd.read_csv(fp_16_data)
    twenty = pd.read_csv(fp_20_data)

    # Sentiment analysis
    six['polarity'], six['subjectivity'] = zip(*six['full_text'].map(sentiment_analysis))
    twenty['polarity'], twenty['subjectivity'] = zip(*twenty['full_text'].map(sentiment_analysis))
    # six.to_csv(fp_16_data, index_label="tweet_id")
    # twenty.to_csv(fp_20_data, index_label="tweet_id")
    
    #permutation test
    compare_polarities = permutation_test(six['polarity'],twenty['polarity'], 0.1)
    compare_subjectivities = permutation_test(six['subjectivity'],twenty['subjectivity'], 0.1)

    l_16, r_16 = get_l_and_r(six, l_htags_16, r_htags_16, left_users, right_users)
    l_l_dialogue_16, l_r_dialogue_16, mentioned_by_l_16, r_l_dialogue_16, r_r_dialogue_16, mentioned_by_r_16 = get_dialogue(l_16, r_16, left_users, right_users)
    
    l_16['polarity'], l_16['subjectivity'] = zip(*l_16['full_text'].map(sentiment_analysis))
    l_20['polarity'], l_20['subjectivity'] = zip(*l_20['full_text'].map(sentiment_analysis))
    r_16['polarity'], r_16['subjectivity'] = zip(*r_16['full_text'].map(sentiment_analysis))
    r_20['polarity'], r_20['subjectivity'] = zip(*r_20['full_text'].map(sentiment_analysis))
    
    print('##### 2016 #####')
    print(str(len(l_16)) + " left-leaning tweets")
    print(str(len(r_16)) + " right-leaning tweets")
    print(str(len(l_l_dialogue_16)) +  " instances of L-L dialogue")
    print(str(len(l_r_dialogue_16)) +  " instances of L-R dialogue")
    print(str(len(r_l_dialogue_16)) +  " instances of R-L dialogue")
    print(str(len(r_r_dialogue_16)) +  " instances of R-R dialogue")

    l_20, r_20 = get_l_and_r(twenty, l_htags_20, r_htags_20, left_users, right_users)
    l_l_dialogue_20, l_r_dialogue_20, mentioned_by_l_20, r_l_dialogue_20, r_r_dialogue_20, mentioned_by_r_20 = get_dialogue(l_20, r_20, left_users, right_users)
    print('##### 2020 #####')
    print(str(len(l_20)) + " left-leaning tweets")
    print(str(len(r_20)) + " right-leaning tweets")
    print(str(len(l_l_dialogue_20)) +  " instances of L-L dialogue")
    print(str(len(l_r_dialogue_20)) +  " instances of L-R dialogue")
    print(str(len(r_l_dialogue_20)) +  " instances of R-L dialogue")
    print(str(len(r_r_dialogue_20)) +  " instances of R-R dialogue")
    
    #permutation tests
    left_analysis_polarity = permutation_test(l_16['polarity'],l_20['polarity'], 0.1)
    left_analysis_subjectivity = permutation_test(l_16['subjectivity'],l_20['subjectivity'], 0.1)
    
    right_analysis_polarity = permutation_test(r_16['polarity'],r_20['polarity'], 0.1)
    right_analysis_subjectivity = permutation_test(r_16['subjectivity'],r_20['subjectivity'], 0.1)
    

    # Make directory to save plots to
    plot_dir = fp_16_data[:10] + "plots"
    os.system("mkdir " + plot_dir)

    # Generate histograms for subjectivity and polarity
    plot_for_year("2016", "polarity", six, l_16, r_16, l_l_dialogue_16, l_r_dialogue_16, r_l_dialogue_16, r_r_dialogue_16, plot_dir)
    plot_for_year("2016", "subjectivity", six, l_16, r_16, l_l_dialogue_16, l_r_dialogue_16, r_l_dialogue_16, r_r_dialogue_16, plot_dir)
    plot_for_year("2020", "polarity", twenty, l_20, r_20, l_l_dialogue_20, l_r_dialogue_20, r_l_dialogue_20, r_r_dialogue_20, plot_dir)
    plot_for_year("2020", "subjectivity", twenty, l_20, r_20, l_l_dialogue_20, l_r_dialogue_20, r_l_dialogue_20, r_r_dialogue_20, plot_dir)
    return


def get_top_n_hashtags(clean, top_n):
    non_hashtags = clean['hashtags'].loc[clean['hashtags'].str.startswith("[") != True]
    clean_hashtags = clean.drop(non_hashtags.index, axis=0)
    
    clean_hashtags['hashtags'] = clean_hashtags['hashtags'].apply(lambda x: x.lower()) # Convert to lowercase
    clean_hashtags['hashtags'] = clean_hashtags['hashtags'].apply(lambda x: ast.literal_eval(x)) # Turn the string of iterables into a list

    all_hashtags = clean_hashtags['hashtags'].explode().dropna()
    top_n_hashtags = all_hashtags.value_counts().head(n=top_n)
    return top_n_hashtags


def sentiment_analysis(text):
    blob = TextBlob(text) 
    polar = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    return polar, sub


def permutation_test(col16, col20, sig_level):
    observed_diff = np.abs(col20.mean() - col16.mean())
    all_vals = list(col16) + list(col20)
    len_16 = len(col16)
    pD = []
    p=1000
    for i in range(0,p):
        random.shuffle(all_vals)
        pD.append(np.abs(np.average(all_vals[0:len_16]) - np.average(all_vals[len_16:])))
    n_greater_equal = 0
    for i in range(0,p):
        if pD[i] >= observed_diff:
            n_greater_equal = n_greater_equal + 1
            
    p_val = n_greater_equal/p
    result = p_val < sig_level
    if result == True:
        mesg = "Pvalue is less than the significance level so we must reject the null hypothesis"
    else:
        mesg = "Pvalue is greater than the significance level so we must accept the null hypothesis"
        
    return(mesg, " The pvalue is: {0}".format(p_val))


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


def plot_for_year(year, elem, df, left, right, l_to_l, l_to_r, r_to_l, r_to_r, out_dir):
    figure(num=None, figsize=(20, 20), dpi=150)
    plt.subplot(4, 1, 1)
    plt.hist(df[elem], bins=20, color="#8968CB")
    plt.title(year + " - " + elem)

    plt.subplot(4, 2, 3)
    plt.hist(left[elem], bins=20)
    plt.title("Left")

    plt.subplot(4, 2, 4)
    plt.hist(right[elem], bins=20, color="#F74242")
    plt.title("Right")

    plt.subplot(4, 4, 9)
    plt.hist(l_to_l[elem], bins=20)
    plt.title("Dialogue: L to L")

    plt.subplot(4, 4, 10)
    plt.hist(l_to_r[elem], bins=20)
    plt.title("Dialogue: L to R")

    plt.subplot(4, 4, 11)
    plt.hist(r_to_l[elem], bins=20, color="#F74242")
    plt.title("Dialogue: R to L")

    plt.subplot(4, 4, 12)
    plt.hist(r_to_r[elem], bins=20, color="#F74242")
    plt.title("Dialogue: R to R")

    plt.savefig(out_dir + "/" + year + "_" + elem + '_dists.png')
    return