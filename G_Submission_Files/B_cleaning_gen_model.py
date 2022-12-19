import pandas as pd
import re
import string
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)

def read_csv(csv):
    df = pd.read_csv(csv, compression='zip')
    return df

def apply_cleaning(raw_data):
    raw_data = raw_data.translate(str.maketrans('', '', string.punctuation + string.digits)) 
    words = raw_data.lower().split() 
    stops = set(stopwords)
    useful_words = [w for w in words if not w in stops]
    useful_words = [re.sub(r'\b.*cyberpunk.*\b', 'cyberpunk', w) for w in useful_words]
    useful_words = [re.sub(r'c.{6,8}k', 'cyberpunk', w) for w in useful_words]
    useful_words = [w.replace('cyber punk', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunks', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyber punk ', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk2077', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk 2077', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk2077 ', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk 2077 ', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk2077game', 'cyberpunk') for w in useful_words]
    useful_words = [w.replace('cyberpunk2077 game', 'cyberpunk') for w in useful_words]
    useful_words = [re.sub(r'[^\w\s]','',w) for w in useful_words]
    return( " ".join(useful_words))

def clean_csv(csv):
    df = read_csv(csv)
    df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    df["Review"] = df["Review"].astype("str")
    df['Review']=df['Review'].apply(apply_cleaning)
    df["review_len"] = df["Review"].apply(lambda x: len(x.split()))
    df = (df[df['review_len']!=0]).copy()
    df = df.drop(columns=['review_len'])
    # print(df.shape) # Checking the shape of final cleaned dataset
    df.to_csv('/workspaces/NLP-CP2077-Sentiment-Analysis/B_Data_Cleaning/cleaned_real_reviews.csv', index=False)
    print("Cleaned csv file saved to B_Data_Cleaning folder")

if __name__ == '__main__':
    file_location = "/workspaces/NLP-CP2077-Sentiment-Analysis/A_Source_Data/cp2077_reviews.csv.zip"
    clean_csv(file_location)