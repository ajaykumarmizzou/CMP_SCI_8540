import os
import pandas as pd
#Import hugging face logging in
from huggingface_hub import notebook_login

#Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# To extract hashtags
import neattext.functions as nfx
import re
import warnings
warnings.filterwarnings("ignore")

# import specific functions and classes from NLTK (Natural Language Toolkit library)
from nltk.tokenize import word_tokenize  # used for tokenizing text into individual words
from nltk.corpus import stopwords # provides a list of common words that are often removed from text
from nltk.stem import PorterStemmer # is a stemming algorithm that reduces words to their base or root form

# Initializes stop variable, assigns it the list of English stopwords from the NLTK corpus.
import nltk
# Download stopwords - Stopwords are commonly used words "a", "the" , "an", "is", "are". which are removed since they dont carry significant meaning to the words
nltk.download("stopwords")


# import functions that are needed
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, split,  regexp_replace, lower
from pyspark.sql.functions import pandas_udf 
import pyspark.sql.functions as F 
from pyspark.sql.types import StringType,BooleanType,DateType

from pyspark.sql.functions import length
from pyspark.sql.functions import concat_ws, collect_list

spark = SparkSession.builder.appName("cleaning_data").config('spark.ui.port', '4050').getOrCreate()


notebook_login()
# Disabe W&B
os.environ["WANDB_DISABLED"] = "true"

# Load the dataset and display some values
Link = 'https://raw.githubusercontent.com/Newton23-nk/Covid_Vaccine_Sentiment_Analysis/main/Datasets/Train.csv'
df = pd.read_csv(Link)
df.iteritems = df.items

sdf = spark.createDataFrame(df)  # convert to pyspark dataframe


# A way to eliminate rows containing NaN values
sdf = sdf.dropna(how = "any")
# sdf.show()

# We look at the number of positive, negative and neutral reviews
counts = sdf.groupBy("label").count().orderBy(col("count").desc())
counts.show()

## Step 1: Data Cleaning

# Checking for missing values
miss_v = sdf.select([count(when(isnan(c), c)).alias(c) for c in sdf.columns])
#miss_v.show()

# Checking for duplicated rows
dup_v = sdf.groupBy(sdf.columns).agg(count("*").alias("count")).filter(col("count") > 1)
#dup_v.show()

# get hashtags: extract hashtags and which can also used for analysis like which was the common aside from #Covid #Vaccine
exit()

sdf.withColumn('extract_hashtags', F(nfx.extract_hashtags('safe_text')))
sdf.select('extract_hashtags', 'safe_text').show()

# remove hashtags from the column and save the cleaned text to clean text column
sdf.withColumn('clean_text', F(nfx.remove_hashtags('safe_text')))
# preview
sdf.select('safe_text','clean_text').show()


# remove RT and user handles
@pandas_udf('string') 
def removeRT(text):
    return text.replace("RT" , "")
re_usehandle = F(lambda x: nfx.remove_userhandles(x))

sdf.select(re_usehandle('clean_text'))
sdf.select(removeRT('clean_text'))
#Preview of the safe text and clean text
sdf.select('safe_text','clean_text').show()

# remove multiple white spaces 
@pandas_udf('string')
def stripSpace(text):
    return text.strip()

sdf.select(F(nfx.remove_multiple_spaces('clean_text')))
sdf.select(stripSpace('clean_text'))

# remove all urls To further reduce noise in the data and to remove irrelevant content
sdf.select(F(nfx.remove_urls('clean_text')))
sdf.select('safe_text','clean_text').show()

# remove pucntuations to standardize the data and to ensure consistency in the data
sdf.select(F(nfx.remove_puncts('clean_text')))
sdf.select('safe_text','clean_text').show()

# lets get hashtags into a good string and remove the hashes beside the tag remove punctuation from each hashtag and also remove the '#' symbol. This is to standardize hashtag representations.
@pandas_udf('string')
def clean_hash_tag(text):
    return " ".join([nfx.remove_puncts(x).replace("#", "") for x in text])

sdf.select(clean_hash_tag('extract_hashtags'))

## Step 2:  Dealing with emojis after removing un-related 
# sdf.select(F(nfx.extract_emojis('clean_text'))).take(10)

# removing the emojis
sdf.select(F(nfx.remove_emojis('clean_text')))

# Replace '<user>' with an empty string in the 'clean_text' column
sdf = sdf.withColumn('clean_text',  regexp_replace('clean_text', '<user>', ''))
sdf = sdf.withColumn('clean_text',  regexp_replace('clean_text', '@', ''))
sdf = sdf.withColumn('clean_text',  regexp_replace('clean_text', '<url>', ''))
sdf = sdf.withColumn('clean_text',  regexp_replace('clean_text', 'measles', 'Measles'))
sdf = sdf.withColumn('clean_text',  regexp_replace('clean_text', '“', ''))

# Remove ['vaccine', 'vaccines', 'vaccinate', 'vaccinated', 'vaccinations', 'vaccination'] to ['vaccine'] 
        # define the words to replace
words_to_replace = ['vaccine', 'vaccines', 'vaccinate', 'vaccinated', 'vaccinations', 'vaccination']

# Pattern to match any of the words in the list, using a regular expression
pattern = r'\b(?:{})\b'.format('|'.join(words_to_replace))

# Function to replace the words with 'vaccine'
@pandas_udf('string')
def replace_with_vaccine(text):
    return text.str.replace(pattern, 'vaccine', case=False)

# Apply the function to the 'safe_text' column
sdf.select(replace_with_vaccine('clean_text'))

# Replace ['kids', 'child', 'children'] to ['child']
words_to_replace_2 = ['kids', 'child', 'children']
# Pattern to match any of the words in the list, using a regular expression
pattern_2  = r'\b(?:{})\b'.format('|'.join(words_to_replace_2 ))

# Function to replace the words with 'vaccine'
@pandas_udf('string')
def replace_with_child (text):
    return text.str.replace(pattern_2 , 'child', case=False)

# Apply the function to the 'safe_text' column
sdf.select(replace_with_child('clean_text'))

words_ = nltk.FreqDist(split(sdf.slect(col('clean_text').cast('string').alias("clean_text"))).count())
words = words_.most_common(30)
print(words)

## Step 3:  Removing stop words
# print(",".join(stopwords.words('english')))
stop_words = set(stopwords.words('english'))

# Convert safe_text column to lower so as to apply stop words
sdf=sdf.withColumn("clean_text", lower(sdf["clean_text"]))


# remove stop words
def remove_stop (x):
  return ",".join([word for word in str(x).split() if word not in stop_words])
remove_stop_new = F(lambda x : remove_stop(x))
sdf.select(remove_stop_new('clean_text'))

# To remove punctuations
sdf = sdf.withColumn('clean_text',  regexp_replace('safe_text', r"[&;. ,#@\"!']", " "))

## Step 4: Use Stemmetizaton

# creates an instance of the PorterStemmer class, assigns it to the variable stemmer.
# The stemmer will be used later to perform word stemming, which reduces words to their base or root
ps = PorterStemmer()
final = []
for word in sdf.select('clean_text'):
   final.append(ps.stem(word))
   final.append(" ")


# We Replace ['-', '"', 'u'] with [ ]
words_to_replace_3 = ['-', '"', 'u' ]

# Pattern to match any of the words in the list, using a regular expression
pattern_3  = r'\b(?:{})\b'.format('|'.join(words_to_replace_3 ))

# Function to replace the words with ''
@pandas_udf('string')
def replace_with_vaccine_3 (text):
    return text.str.replace(pattern_3 , '', case=False)

# Apply the function to the 'safe_text' column
sdf.select(replace_with_vaccine_3('clean_text'))
sdf_new = sdf.select('tweet_id','clean_text','label','agreement')

sdf.write.optioins(header='True', delimiter='\t').csv('clean_data.csv')
sdf.write.optioins(header='True', delimiter='\t').text('clean_data.txt')






