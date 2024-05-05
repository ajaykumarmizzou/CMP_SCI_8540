# Importing libraries
import os
import pandas as pd
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, concat_ws, collect_list
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create results and logs directories if they don't exist
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging
logging.basicConfig(filename='logs/logfile.log', level=logging.INFO)

# Creating spark session
spark = SparkSession.builder.appName("DataProcess").config('spark.ui.port', '4050').getOrCreate()
logging.info('Spark session created')

# Disable W&B
os.environ["WANDB_DISABLED"] = "true"
logging.info('W&B disabled')

# Load the dataset
Link = 'data/Train.csv'
df = pd.read_csv(Link)
logging.info('Dataset loaded')

# Convert pandas DataFrame to PySpark DataFrame
sdf = spark.createDataFrame(df)
logging.info('Converted pandas DataFrame to PySpark DataFrame')

# Drop rows containing NaN values
sdf = sdf.dropna()
logging.info('Dropped rows containing NaN values')

# Count the number of each label
counts = sdf.groupBy("label").count().orderBy(col("count").desc())
logging.info('Counted the number of each label')

# Count the number of each agreement
agree_count = sdf.groupBy("agreement").count().orderBy(col("agreement").desc())
logging.info('Counted the number of each agreement')

# Calculate the length of each review
review_length = sdf.select(length('safe_text')).withColumnRenamed("length(safe_text)", "review_length")
logging.info('Calculated the length of each review')

# Find the length of the longest review
max_length = review_length.agg({"review_length": "max"}).withColumnRenamed("max(review_length)", "max_review_length")
logging.info('Found the length of the longest review')

# Find the length of the shortest review
min_length = review_length.agg({"review_length": "min"}).withColumnRenamed("min(review_length)", "min_review_length")
logging.info('Found the length of the shortest review')

# Plot the distribution of text lengths
sns.histplot(review_length.rdd.map(lambda x: x[0]).collect(), kde=True, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.savefig("results/DistributionofTextLengths.png")
logging.info('Saved distribution of text lengths plot')

# Plot the distribution of sentiments
counts_rdd = counts.rdd.map(lambda row: (row['label'], row['count']))
labels, count_values = zip(*counts_rdd.collect())
colors = sns.color_palette('viridis', len(labels))
plt.figure()
plt.bar(labels, count_values, color=colors)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.savefig("results/DistributionofSentiments.png")
logging.info('Saved distribution of sentiments plot')

# Plot the distribution of agreement percentages
agree_counts_rdd = agree_count.rdd.map(lambda row: (row['agreement'], row['count']))
labels, count_values = zip(*agree_counts_rdd.collect())
colors = sns.color_palette('pastel', len(labels))
plt.figure()
plt.bar(labels, count_values, color=colors)
plt.title('Distribution of Agreement Percentages')
plt.xlabel('Agreement Percentage')
plt.ylabel('Frequency')
plt.savefig("results/DistributionofAgreementPercentages.png")
logging.info('Saved distribution of agreement percentages plot')

# Generate a word cloud from the 'safe_text' column
text_df = sdf.agg(concat_ws(' ', collect_list('safe_text')).alias('text'))
text = text_df.first()['text']
cloud_two_cities = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(8, 5))
plt.imshow(cloud_two_cities, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
plt.savefig("results/WordCloud.png")
logging.info('Saved word cloud')
