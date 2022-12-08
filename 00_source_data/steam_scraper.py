import requests 

response = requests.get("http://store.steampowered.com/appreviews/1091500?json=1")


def get_reviews(appid, params={'json':1}):
        url = 'https://store.steampowered.com/appreviews/'
        response = requests.get(url=url+appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        return response.json()

def get_n_reviews(appid, n=100):
    reviews = []
    cursor = '*'
    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'english',
            'day_range' : 9223372036854775807,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)
        n -= 100

        response = get_reviews(appid, params)
        cursor = response['cursor']
        reviews.append(response['reviews'])

        if len(response['reviews']) < 100: break

    return reviews

x = get_n_reviews("1091500",237832) # the number of reviews for CP2077 as reported by Steam

tests = ['Listen up Choom', 'Excellent atmosphere/setting/story',"the most cyberpunk aspect of this game is how a billionaire","tl;dr 5/10, defo rushed, if you must, buy it on sale in a year or two"]


def test_cases(tests_list, review_list):

      tag_dictionary = {}

      for test_case in tests_list:

        for review_dictionary in review_list:

            for review in review_dictionary:

                if review['review'].startswith(test_case):

                    tag_dictionary[(
                        review['review'])] = review['voted_up']
      return tag_dictionary


y = test_cases(tests, x)

boolean_voted_up = []

for key in y:

    boolean_voted_up.append([key, y[key]])

boolean_voted_up[0]
boolean_voted_up[1]
boolean_voted_up[2]
boolean_voted_up[3]

# It's confirmed. voted_up is our truth corpus's label. True is recommended, False is not recommended.
# Also, if you look at how the text was processed by Steam, it is a bit unmalleable. We would have to do some special cleaning. 
# I saw how it handled italics for example, for the second or third test case, not to mention emojis as well. The reviews should be reconstructed
# Also, we need to get rid of the newlines. We are not aiming to make a text generator, so removing the newlines should not be an issue. 
# Looks like the scraper did 16,600 reviews despite all the ns I expected. Still, 10K is amazing. We could subset based on that. That's a good e-mail to ask about.

        






    



                