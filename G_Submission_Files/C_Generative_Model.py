import pandas as pd
import numpy as np
import random
import nltk
import collections
import sklearn as sk
import fire


def prepare_df(csv, exclusion=False):
    """
    This function takes in a csv file and
    returns a dataframe
    :param csv: csv file
    :param exclusion: a word to be excluded from the reviews
    :return: a dataframe
    """
    df = pd.read_csv(csv)
    df["Review"] = df["Review"].astype(str)
    if exclusion != False:

        def clean(raw_data, exclusion):
            useful_words = raw_data.lower().split()
            useful_words = [w.replace(exclusion, "") for w in useful_words]
            return " ".join(useful_words)

        df["Review"] = df["Review"].apply(clean, exclusion=exclusion)
    else:
        pass
    df["Review"] = df["Review"].astype(str)
    df["set_column"] = df["Review"].apply(lambda x: set(x.split()))
    df["review_len"] = df["Review"].apply(lambda x: len(x.split()))
    return df


def balance_df(df, random_state=42):
    """
    This function takes in a dataframe
    and returns a balanced dataframe
    with 5250 recommended and 5250 not recommended reviews
    (randomly sampled from the original dataframe)
    :param df: dataframe
    :param random_state: random state
    :return: balanced dataframe
    """
    temp_recom = df.loc[df["Recommended or Not Recommended"] == True, :]
    temp_recom = temp_recom.sample(5250, random_state=random_state).reset_index(
        drop=True
    )
    temp_not_recom = df.loc[df["Recommended or Not Recommended"] == False, :]
    temp_not_recom = temp_not_recom.sample(5250, random_state=random_state).reset_index(
        drop=True
    )
    temp_df = pd.concat([temp_recom, temp_not_recom], axis=0)
    return temp_df


def split_data_recom(df, test_size=0.33, random_state=42):
    """
    Returns three dataframes (recommended, not recommended, test)
    :param df: dataframe
    :param test_size: test size
    :param random_state: random state
    :return: three dataframes
    """
    train, test = sk.model_selection.train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    df_recom = train.loc[train["Recommended or Not Recommended"] == True, :]
    df_recom = df_recom.reset_index(drop=True)
    df_not_recom = train.loc[train["Recommended or Not Recommended"] == False, :]
    df_not_recom = df_not_recom.reset_index(drop=True)
    return df_recom, df_not_recom, test


def make_bow_dict(df_recom, df_not_recom):
    """
    This function takes in a dataframe
    and returns a dictionary containing:
    - words in the reviews
    - number of times each word appears in the reviews
    :param df_recom: dataframe
    :param df_not_recom: dataframe
    :return: dictionary
    """
    bow_recom_dict, bow_not_recom_dict = dict(
        collections.Counter([y for x in df_recom.Review for y in x.split()])
    ), dict(collections.Counter([y for x in df_not_recom.Review for y in x.split()]))
    bow_recom_set, bot_not_recom_set = dict(
        collections.Counter([y for x in df_recom.set_column for y in x])
    ), dict(collections.Counter([y for x in df_not_recom.set_column for y in x]))
    assert len(bow_recom_dict) > 0
    assert len(bow_not_recom_dict) > 0
    assert len(bow_recom_set) > 0
    assert len(bot_not_recom_set) > 0
    return bow_recom_dict, bow_not_recom_dict, bow_recom_set, bot_not_recom_set


def smoothing_and_probability(dict1, dict2, set1, set2):
    """
    This function takes in two dictionaries and two sets
    to apply smoothing and calculate the probability
    :param dict1: dictionary
    :param dict2: dictionary
    :param set1: set
    :param set2: set
    :return: four dictionaries, dicts are probability, sets are count of reviews as documents
    """
    for key in dict1:
        if key not in dict2:
            dict2[key] = 0
        else:
            pass
    for key in dict2:
        if key not in dict1:
            dict1[key] = 0
        else:
            pass
    for key in dict1:
        dict1[key] += 1
    for key in dict2:
        dict2[key] += 1
    bow_recom_sum_vals = sum(dict1.values())
    bow_not_recom_sum_vals = sum(dict2.values())
    for i in dict1:
        dict1[i] /= bow_recom_sum_vals
    for i in dict2:
        dict2[i] /= bow_not_recom_sum_vals
    assert len(dict1) == len(dict2)
    for key in set1:
        if key not in set2:
            set2[key] = 0
        else:
            pass
    for key in set2:
        if key not in set1:
            set1[key] = 0
        else:
            pass
    for key in set1:
        set1[key] += 1
    for key in set2:
        set2[key] += 1
    assert len(set1) == len(set2)
    return dict1, dict2, set1, set2


def classifier(
    element,
    your_class,
    which_test,
    bow_recom_dict,
    bow_not_recom_dict,
    bow_recom_set,
    bow_not_recom_set,
    df_recom,
    df_not_recom,
    positive_review_probability,
    negative_review_probability,
    full_shape,
    combined_corpus,
):
    """
    This function takes in a string and returns a probability
    :param element: string
    :param your_class: string
    :param which_test: string
    :param bow_recom_dict: dictionary
    :param bow_not_recom_dict: dictionary
    :param bow_recom_set: dictionary
    :param bow_not_recom_set: dictionary
    :param df_recom: dataframe
    :param df_not_recom: dataframe
    :param positive_review_probability: float
    :param negative_review_probability: float
    :param full_shape: int
    :return: float
    """
    df_choices_positive = [
        positive_review_probability,
        bow_recom_set,
        bow_recom_dict,
        df_recom,
    ]
    df_choices_negative = [
        negative_review_probability,
        bow_not_recom_set,
        bow_not_recom_dict,
        df_not_recom,
    ]
    if your_class == "positive":
        df_choices = df_choices_positive
    else:
        df_choices = df_choices_negative
    prob_of_class = df_choices[0] / full_shape
    score = 1 * prob_of_class
    score = float(format(score, ".12f"))
    for i in element.split():
        if i not in df_choices[2].keys():
            pass
        else:
            score /= combined_corpus[i] + 1
            prob_word_given_class = (df_choices[2])[i]
            prob_word_given_class = float(format(prob_word_given_class, ".12f"))

            # if i not in df_choices[1].keys():
            #     print(i,[j for j in df_choices[1].keys() if i in j])

            tf = np.log(prob_word_given_class)
            idf = np.log(df_choices[3].shape[0] / (df_choices[1])[i])
            if which_test == "test_all":
                score *= prob_word_given_class * tf * idf
            elif which_test == "test_only_tf_abs":
                score *= prob_word_given_class * tf
                score = abs(score)
            elif which_test == "test_only_idf_abs":
                score *= prob_word_given_class * idf
                score = abs(score)
            elif which_test == "test_only_prob":
                score *= prob_word_given_class
                score = abs(score)
            elif which_test == "test_all_abs":
                score *= prob_word_given_class * tf * idf
                score = abs(score)
    return score


def calculate_pos_neg(
    element,
    test,
    bow_recom_dict,
    bow_not_recom_dict,
    bow_recom_set,
    bow_not_recom_set,
    df_recom,
    df_not_recom,
    positive_review_probability,
    negative_review_probability,
    full_shape,
    combined_corpus,
):
    """
    This function takes in a string and returns a probability
    works closely with classifier, the main actor in this whole classifying process
    :param element: string
    :param test: string
    :param bow_recom_dict: dictionary
    :param bow_not_recom_dict: dictionary
    :param bow_recom_set: dictionary
    :param bow_not_recom_set: dictionary
    :param df_recom: dataframe
    :param df_not_recom: dataframe
    :param positive_review_probability: float
    :param negative_review_probability: float
    :param full_shape: int
    :return: float
    """
    positive_score = classifier(
        element,
        "positive",
        which_test=test,
        bow_recom_dict=bow_recom_dict,
        bow_not_recom_dict=bow_not_recom_dict,
        bow_recom_set=bow_recom_set,
        bow_not_recom_set=bow_not_recom_set,
        df_recom=df_recom,
        df_not_recom=df_not_recom,
        positive_review_probability=positive_review_probability,
        negative_review_probability=negative_review_probability,
        full_shape=full_shape,
        combined_corpus=combined_corpus,
    )
    negative_score = classifier(
        element,
        "negative",
        which_test=test,
        bow_recom_dict=bow_recom_dict,
        bow_not_recom_dict=bow_not_recom_dict,
        bow_recom_set=bow_recom_set,
        bow_not_recom_set=bow_not_recom_set,
        df_recom=df_recom,
        df_not_recom=df_not_recom,
        positive_review_probability=positive_review_probability,
        negative_review_probability=negative_review_probability,
        full_shape=full_shape,
        combined_corpus=combined_corpus,
    )
    if positive_score > negative_score:
        return True
    elif positive_score == negative_score:
        return False
    else:
        return False


def print_results(df, type_of_dataset, exclude_word=False, origin_data=False):
    """
    This function prints the results of the classification
    :param df: dataframe
    :param type_of_dataset: string
    :param exclude_word: string
    :return: print statement
    """
    if exclude_word != False:
        print(
            f"This is the result without the word {exclude_word.upper()} from a {origin_data.upper()}, and {type_of_dataset.upper()} dataset"
        )
    else:
        print(
            f"This is the result without any excluded words from a {origin_data.upper()}, and {type_of_dataset.upper()} dataset"
        )
    print(
        f"Using class probability, frequency, TF, and IDF: {(sum(df['Recommended or Not Recommended'] == df['score_all'])/df.shape[0])*100:.2f}%"
    )
    print(
        f"Using absolute value of class probability, frequency, and TF: {(sum(df['Recommended or Not Recommended'] == df['score_only_tf_abs'])/df.shape[0])*100:.2f}%"
    )
    print(
        f"Using absolute value of class probability, frequency, and IDF: {(sum(df['Recommended or Not Recommended'] == df['score_only_idf_abs'])/df.shape[0])*100:.2f}%"
    )
    print(
        f"Using absolute value of class probability and frequency: {(sum(df['Recommended or Not Recommended'] == df['score_only_prob'])/df.shape[0])*100:.2f}%"
    )
    print(
        f"Using absolute value of class probability, frequency, TF, and IDF: {(sum(df['Recommended or Not Recommended'] == df['score_all_abs'])/df.shape[0])*100:.2f}%"
    )


def dict_for_synth(df_recom, df_not_recom):
    """
    This function creates a dictionary (foundation) for the synthetic dataset
    :param df: dataframe
    :param df_recom: dataframe
    :param df_not_recom: dataframe
    :return: dictionary
    """
    bow_recom_s = collections.Counter([y for x in df_recom.Review for y in x.split()])
    bow_not_recom_s = collections.Counter(
        [y for x in df_not_recom.Review for y in x.split()]
    )
    bow_recom_dict_s = dict(bow_recom_s)
    bow_not_recom_dict_s = dict(bow_not_recom_s)
    len_count_recom = {}
    len_count_not_recom = {}
    for i in df_recom["review_len"]:
        if i in len_count_recom:
            len_count_recom[i] += 1
        else:
            len_count_recom[i] = 1

    for i in df_not_recom["review_len"]:
        if i in len_count_not_recom:
            len_count_not_recom[i] += 1
        else:
            len_count_not_recom[i] = 1
    return bow_recom_dict_s, bow_not_recom_dict_s, len_count_recom, len_count_not_recom


def make_synthetic_review(bow_corpus, rev_len_count_corpus, sentiment):
    """
    This function creates a synthetic review
    :param bow_corpus: dictionary
    :param rev_len_count_corpus: dictionary
    :param sentiment: string
    :return: dataframe
    """
    if sentiment == "positive":
        value = True
    else:
        value = False

    synthetic_reviews = [["Review", "Recommended or Not Recommended"]]
    counter = 0

    for rev_count in rev_len_count_corpus:
        for i in range(0, rev_len_count_corpus[rev_count]):
            j = 0
            ls_string = []
            while j < rev_count:
                ls_string.append(random.choice(list(bow_corpus.keys())))
                j += 1
            synthetic_string = " ".join(ls_string)
            counter += 1
            print(
                f"The elfs (Eric, Wafi, Zhonglin) are making the {sentiment} reviews!! Progress: {counter/sum(rev_len_count_corpus.values())*100:.2f}%"
            )
            synthetic_reviews.append([synthetic_string, value])

    review_df = pd.DataFrame(synthetic_reviews[1:], columns=synthetic_reviews[0])
    assert sum(rev_len_count_corpus.values()) == review_df.shape[0]
    return review_df


def prompt_syn_data(df):
    """
    This function prompts the user to save or read the synthetic data
    :param df: dataframe
    :return: print statement, dataframe, or csv
    """
    print("Do you want to save or read the synthetic data? (save/read)")
    answer = input()
    if answer == "save":
        df.to_csv(
            f"synthetic_data_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv",
            index=False,
        )
        print(
            f"The synthetic data has been saved as synthetic_data_{pd.to_datetime('today').strftime('%Y-%m-%d')}.csv! Thank you for using our program! Hope we qualify for A+! Please watch Toonami every Saturday at 12:00 AM, also watch Aggretsuko! We love it!"
        )
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(df.head(10))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    else:
        print(df.head(10))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def cyberpunk_sentiment(
    csv, balancing="Unbalanced", make_syn=False, exclude_word=False, origin_data=False
):
    """
    This function runs the Naive Bayes algorithm
    :param csv: csv file
    :param balancing: string
    :param make_syn: boolean
    :param exclude_word: boolean or string
    :return: print statement, dataframe, or csv
    """
    df = prepare_df(csv, exclusion=exclude_word)

    if balancing == "Balanced":
        temp_df = balance_df(df)
    else:
        temp_df = df
    df_recom, df_not_recom, test = split_data_recom(temp_df)
    (
        bow_recom_dict,
        bow_not_recom_dict,
        bow_recom_set,
        bow_not_recom_set,
    ) = make_bow_dict(df_recom, df_not_recom)
    combined_corpus = collections.Counter(bow_recom_dict) + collections.Counter(
        bow_not_recom_dict
    )
    (
        bow_recom_dict,
        bow_not_recom_dict,
        bow_recom_set,
        bow_not_recom_set,
    ) = smoothing_and_probability(
        bow_recom_dict, bow_not_recom_dict, bow_recom_set, bow_not_recom_set
    )

    full_shape = df.shape[0]
    positive_review_probability = len(df_recom) / full_shape
    negative_review_probability = len(df_not_recom) / full_shape

    test["score_all"] = test.Review.apply(
        lambda x: calculate_pos_neg(
            x,
            "test_all",
            bow_recom_dict,
            bow_not_recom_dict,
            bow_recom_set,
            bow_not_recom_set,
            df_recom,
            df_not_recom,
            positive_review_probability,
            negative_review_probability,
            full_shape,
            combined_corpus,
        )
    )
    test["score_only_tf_abs"] = test.Review.apply(
        lambda x: calculate_pos_neg(
            x,
            "test_only_tf_abs",
            bow_recom_dict,
            bow_not_recom_dict,
            bow_recom_set,
            bow_not_recom_set,
            df_recom,
            df_not_recom,
            positive_review_probability,
            negative_review_probability,
            full_shape,
            combined_corpus,
        )
    )
    test["score_only_idf_abs"] = test.Review.apply(
        lambda x: calculate_pos_neg(
            x,
            "test_only_idf_abs",
            bow_recom_dict,
            bow_not_recom_dict,
            bow_recom_set,
            bow_not_recom_set,
            df_recom,
            df_not_recom,
            positive_review_probability,
            negative_review_probability,
            full_shape,
            combined_corpus,
        )
    )
    test["score_only_prob"] = test.Review.apply(
        lambda x: calculate_pos_neg(
            x,
            "test_only_prob",
            bow_recom_dict,
            bow_not_recom_dict,
            bow_recom_set,
            bow_not_recom_set,
            df_recom,
            df_not_recom,
            positive_review_probability,
            negative_review_probability,
            full_shape,
            combined_corpus,
        )
    )
    test["score_all_abs"] = test.Review.apply(
        lambda x: calculate_pos_neg(
            x,
            "test_all_abs",
            bow_recom_dict,
            bow_not_recom_dict,
            bow_recom_set,
            bow_not_recom_set,
            df_recom,
            df_not_recom,
            positive_review_probability,
            negative_review_probability,
            full_shape,
            combined_corpus,
        )
    )
    if make_syn:
        (
            bow_recom_dict_s,
            bow_not_recom_dict_s,
            len_count_recom,
            len_count_not_recom,
        ) = dict_for_synth(df_recom, df_not_recom)
        synth_reviews_positive = make_synthetic_review(
            bow_recom_dict_s, len_count_recom, sentiment="positive"
        )
        synth_reviews_negative = make_synthetic_review(
            bow_not_recom_dict_s, len_count_not_recom, sentiment="negative"
        )
        synth_reviews = pd.concat([synth_reviews_positive, synth_reviews_negative])
        prompt_syn_data(synth_reviews)
    else:
        pass

    print_results(
        test,
        type_of_dataset=balancing,
        exclude_word=exclude_word,
        origin_data=origin_data,
    )


if __name__ == "__main__":
    # comment this out if you prefer to run the script from the command line
    # fire.Fire(cyberpunk_sentiment)

    # This first run is to run the script with the status of:
    # REAL dataset
    # BALANCED dataset,
    # NOT MAKE SYNTHETIC dataset, and
    # EXCLUDE word 'cyberpunk'

    real_review_path = "/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/cleaned_real_reviews.csv"
    cyberpunk_sentiment(
        real_review_path,
        balancing="Balanced",
        make_syn=False,
        exclude_word=False,
        origin_data="real",
    )
    print("+-------------------------------------------------------------------------+")
    real_review_path = "/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/cleaned_real_reviews.csv"
    cyberpunk_sentiment(
        real_review_path,
        balancing="Balanced",
        make_syn=False,
        exclude_word="cyberpunk",
        origin_data="real",
    )
    print("+-------------------------------------------------------------------------+")
    real_review_path = "/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/cleaned_real_reviews.csv"
    cyberpunk_sentiment(
        real_review_path,
        balancing="Unbalanced",
        make_syn=False,
        exclude_word=False,
        origin_data="real",
    )
    print("+-------------------------------------------------------------------------+")
    real_review_path = "/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/cleaned_real_reviews.csv"
    cyberpunk_sentiment(
        real_review_path,
        balancing="Unbalanced",
        make_syn=False,
        exclude_word="cyberpunk",
        origin_data="real",
    )
    print("+-------------------------------------------------------------------------+")

    # The next run is to run the script with the status of:
    # SYNTHETIC dataset
    # BALANCED dataset,
    # NOT MAKE SYNTHETIC dataset, and
    # EXCLUDE word 'cyberpunk'

    synthetic_data_path = "/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/synthetic_reviews_all_trial_1.csv"
    cyberpunk_sentiment(
        synthetic_data_path,
        balancing="Balanced",
        make_syn=False,
        exclude_word=False,
        origin_data="synthetic",
    )
    print("+-------------------------------------------------------------------------+")
    cyberpunk_sentiment(
        synthetic_data_path,
        balancing="Balanced",
        make_syn=False,
        exclude_word="cyberpunk",
        origin_data="synthetic",
    )
    print("+-------------------------------------------------------------------------+")
    cyberpunk_sentiment(
        synthetic_data_path,
        balancing="Unbalanced",
        make_syn=False,
        exclude_word=False,
        origin_data="synthetic",
    )
    print("+-------------------------------------------------------------------------+")
    cyberpunk_sentiment(
        synthetic_data_path,
        balancing="Unbalanced",
        make_syn=False,
        exclude_word="cyberpunk",
        origin_data="synthetic",
    )
