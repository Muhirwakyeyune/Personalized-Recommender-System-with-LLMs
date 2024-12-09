from faker import Faker
import random
import pandas as pd

fake = Faker()

# Enum TweetType values
tweet_types = ['original', 'replied_to', 'quoted', 'retweeted']

# Possible values for hashtags, mentions, urls, images_urls, videos_urls
hashtags = ['#AI', '#MachineLearning', '#DataScience', '#Python', '#Coding', '#Tech']
mentions = ['@user1', '@user2', '@user3', '@user4', '@user5']
urls = ['http://example.com', 'http://example.org', 'http://example.net']
image_urls = ['http://example.com/image1.jpg', 'http://example.com/image2.jpg', 'http://example.com/image3.jpg']
video_urls = ['http://example.com/video1.mp4', 'http://example.com/video2.mp4', 'http://example.com/video3.mp4']
topics = ['technology', 'science', 'artificial intelligence', 'software development', 'data analysis', 'programming', 'cybersecurity', 'robotics', 'blockchain', 'quantum computing']
categories = ['news', 'updates', 'tutorial', 'opinion', 'review', 'interview', 'event', 'announcement', 'case study', 'research']

# Limit topics and categories to 7-10 items
topics = random.sample(topics, random.randint(7, 10))
categories = random.sample(categories, random.randint(7, 10))

def generate_fake_tweet_user(user_id):
    return {
        'id': f'U{user_id}',
        'username': fake.user_name(),
        'slug': fake.slug(),
        'image': fake.image_url(),
        'profile_url': fake.url(),
        'num_of_followers': fake.random_int(min=0, max=1000000),
        'num_of_following': fake.random_int(min=0, max=5000),
        'bio': fake.text(max_nb_chars=200),
        'num_of_posted_tweets': fake.random_int(min=0, max=10000),
        'created_at': fake.date_time_between(start_date='-5y', end_date='now')
    }

def generate_fake_tweet(tweet_id, user_id):
    tweet_type = random.choice(tweet_types)
    tweet = {
        'id': f'T{tweet_id}',
        'slug': fake.slug(),
        'user': f'U{user_id}',
        'content': fake.text(max_nb_chars=280),
        'published_date': fake.date_time_between(start_date='-5y', end_date='now'),
        'location': None if random.choice([True, False]) else fake.random_number(digits=10, fix_len=True),
        'hashtags': random.sample(hashtags, random.randint(0, 3)),
        'mentions': random.sample(mentions, random.randint(0, 3)),
        'urls': random.sample(urls, random.randint(0, 3)),
        'images_urls': random.sample(image_urls, random.randint(0, 3)),
        'videos_urls': random.sample(video_urls, random.randint(0, 3)),
        'num_of_likes': fake.random_int(min=0, max=10000),
        'num_of_retweets': fake.random_int(min=0, max=5000),
        'num_of_replies': fake.random_int(min=0, max=2000),
        'num_of_quotes': fake.random_int(min=0, max=1000),
        'score': 0,  # This will be calculated later
        'type': tweet_type,
        'parent_tweet_id': None if tweet_type == 'original' else f'T{fake.random_number(digits=10, fix_len=True)}',
        'topics': random.sample(topics, random.randint(0, 3)),
        'categories': random.choice(categories),
        'created_at': fake.date_time_between(start_date='-5y', end_date='now')
    }
    tweet['score'] = tweet['num_of_likes'] + tweet['num_of_retweets'] + tweet['num_of_replies'] + tweet['num_of_quotes']
    return tweet

# Generate fake data
num_users = 100
num_tweets = 1000

tweet_users = [generate_fake_tweet_user(i + 1) for i in range(num_users)]
tweets = []

tweet_id = 1
for user_id, user in enumerate(tweet_users, start=1):
    num_user_tweets = random.randint(1, 20)
    for _ in range(num_user_tweets):
        tweet = generate_fake_tweet(tweet_id, user_id)
        tweets.append(tweet)
        tweet_id += 1

# Convert lists to DataFrames
df_tweet_users = pd.DataFrame(tweet_users)
df_tweets = pd.DataFrame(tweets)

# Save raw data to CSV
df_tweet_users.to_csv('tweet_users.csv', index=False)
df_tweets.to_csv('tweets.csv', index=False)

