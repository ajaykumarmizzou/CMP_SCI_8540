import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
#Visualization Libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
# To extract hashtags
import neattext.functions as nfx
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import length
from pyspark.sql.functions import concat_ws, collect_list

spark = SparkSession.builder.appName("DataProcess").config('spark.ui.port', '4050').getOrCreate()
# Disabe W&B
os.environ["WANDB_DISABLED"] = "true"

# Load the dataset and display some values
Link = 'Phase2/Train.csv'

df = pd.read_csv(Link)
#Create a PySpark DataFrame from a pandas DataFrame
import pandas as pd
# Assuming 'df' is your pandas DataFrame
for column_name, dtype in df.dtypes.items():
    try:
        df[column_name].astype(str)
    except Exception as e:
        print(f"Column '{column_name}' has dtype '{dtype}' and may be causing serialization issues: {e}")

sdf = spark.createDataFrame(df)
sdf.show()


exit()
# A way to eliminate rows containing NaN values
sdf = sdf.dropna()
sdf.show()


# We look at the number of positive, negative and neutral reviews
counts = sdf.groupBy("label").count().orderBy(col("count").desc())
counts.show()


# The count of the agrremtns
agree_count = sdf.groupBy("agreement").count().orderBy(col("agreement").desc())
agree_count.show()


# Legnth of the reviews
review_length = sdf.select(length('safe_text')).withColumnRenamed("length(safe_text)", "review_length")
review_length.show()
# Legnth of the longest review
max_length=review_length.agg({"review_length": "max"}).withColumnRenamed("max(review_length)", "max_review_length")
max_length.show()

#Legnth of the shortest review
min_length=review_length.agg({"review_length": "min"}).withColumnRenamed("max(review_length)", "max_review_length")
min_length.show()

# Length of Tweets
#text_length = df['safe_text'].apply(len)
text_length = review_length.rdd.map(lambda x: x[0])
print(text_length)
sns.histplot(text_length.collect(),kde=True, color='skyblue', edgecolor='black', linewidth=1.2)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Count')
#plt.show()
plt.savefig("DistributionofTextLengths.png")

# Distribution of Sentiments
counts_rdd = counts.rdd.map(lambda row: (row['label'], row['count']))
# Extract labels and counts
labels, count_values = zip(*counts_rdd.collect())
# Plot the distribution of counts using Matplotlib
# Define a color palette
colors = sns.color_palette('viridis', len(labels))
plt.figure()
plt.bar(labels, count_values,color=colors)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
#plt.show()
plt.savefig("DistributionofSentiments.png")

# Distribution of Sentiments
agree_counts_rdd = agree_count.rdd.map(lambda row: (row['agreement'], row['count']))
# Extract labels and counts
labels, count_values = zip(*agree_counts_rdd.collect())
# Plot the distribution of counts using Matplotlib
# Define a color palette
colors = sns.color_palette('pastel', len(labels))
plt.figure()
plt.bar(labels, count_values,color=colors)
plt.title('Distribution of Agreement Percentages')
plt.xlabel('Agreement Percentage')
plt.ylabel('Frequency')
#plt.show()
plt.savefig("DistributionofAgreementPercentages.png")


# Concatenate all text from the 'safe_text' column into a single string
text_df = sdf.agg(concat_ws(' ', collect_list('safe_text')).alias('text'))
# Extract the concatenated text
text = text_df.first()['text']
# Generate the word cloud with a white background
cloud_two_cities = WordCloud(width=800, height=400, background_color='white').generate(text)
# Display the word cloud
plt.figure(figsize=(8, 5))
plt.imshow(cloud_two_cities, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=1)
#plt.show()
plt.savefig("WordCloud.png")









