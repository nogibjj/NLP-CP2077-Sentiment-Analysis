# NLP-CP2077-Sentiment-Analysis
Steps to reproduce the experience:
1. Run the steam scraper (A_steam_scraper.py) (This step was completed, but if run again will generate a new dataset, it won't be the same as ours, since the scraper gets the most recent 16K reviews)
2. Master dataset created (cp2077_reviews.csv.zip)
3. Run the cleaning file for generative model (B_cleaning_gen_model.py)
4. Cleaned REAL dataset for the generative model created (cleaned_real_reviews.csv)
## Real Dataset
5. Run the generative model (C_Generative_Model.py) on the cleaned real review (cleaned_real_reviews.csv), we will get the accuracy result of generative model in print statement on command line and a csv file with all the new columns (test_with_new_columns.csv). If we opted to make the synthetic data, we will get a csv for the synthetic data (synthetic_reviews_all_trial_1.csv), but running this file will create a different result each time, because it is a stochastic generation of text. 
6. Run the Bi-LSTM model (lstm_new.ipynb) on the real data, as it has special cleaning (cp2077_reviews.csv.zip), path changing is maybe needed.
## Synthetic Dataset
7. Run the generative model (C_Generative_Model.py) on the synthetic review file (synthetic_reviews_all_trial_1.csv)
8. Run the Bi-LSTM model (lstm_new_synthetic.ipynb) on the synthetic review file (synthetic_reviews_all_trial_1.csv), path changing is maybe needed.