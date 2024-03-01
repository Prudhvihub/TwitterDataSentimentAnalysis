from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, sum, when
from pyspark.sql.types import StructType, StructField, StringType, LongType
from textblob import TextBlob
from pyspark.sql.functions import udf
import pymongo

# Kafka topics
topics = ['NATO', 'Biden', 'Putin', 'Zelensky', 'NoFlyZone']

# Spark session is initialized
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2") \
    .getOrCreate()

# Disable correctness check
spark.conf.set("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")

# Adjusted Schema for the incoming Kafka messages
schema = StructType([
    StructField("id", LongType(), True),
    StructField("text", StringType(), True)
])

# DataFrame representing the stream from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", ",".join(topics)).load()

# Parsing the JSON message
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("event_time"),
    col("topic")
)

# Sentiment Analysis UDF
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return "Positive" if analysis.sentiment.polarity > 0 else "Neutral" if analysis.sentiment.polarity == 0 else "Negative"

sentiment_udf = udf(sentiment_analysis, StringType())

# Applying Sentiment Analysis UDF to the text column
result_df = parsed_df.select(
    col("data.id"),
    col("data.text"),
    sentiment_udf(col("data.text")).alias("sentiment"),
    col("event_time"),
    col("topic")
)

# Define the window specification
window_specification = window("event_time", "10 minutes")

# Group by window, topic, and sentiment, then aggregate count
agg_df = result_df.groupBy(window_specification, "topic", "sentiment").count().na.fill(0)

# Write the data to MongoDB using foreachBatch
def write_to_mongo(df, epoch_id):
    if df.count() == 0:
        print("No data to write")
        return
    
    df_agg = df.groupBy("window", "topic").agg(
        sum(when(col("sentiment") == "Negative", col("count")).otherwise(0)).alias("Negative"),
        sum(when(col("sentiment") == "Neutral", col("count")).otherwise(0)).alias("Neutral"),
        sum(when(col("sentiment") == "Positive", col("count")).otherwise(0)).alias("Positive")
    )
    
    # Display the result on the console
    print("Aggregated Results:")
    df_agg.show(truncate=False)
    
    # Convert to Pandas DataFrame for MongoDB insertion
    pandas_df = df_agg.toPandas()

    # Connect to MongoDB and insert the data
    mongo_uri = "mongodb+srv://aishwaryaashok:P5tH8YCyYOMn8VbW@cluster0.r2jhibn.mongodb.net/?retryWrites=true&w=majority"
    database_name = "TwitterDB"
    collection_name = "TwitterCollection4"
    
    client = pymongo.MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]
    
    for index, row in pandas_df.iterrows():
        data = {
            "window_start": row["window"],
            "topic": row["topic"],
            "Negative": int(row["Negative"]),
            "Neutral": int(row["Neutral"]),
            "Positive": int(row["Positive"])
        }
        collection.insert_one(data)

# Write the data to MongoDB using foreachBatch
query_mongo = agg_df.writeStream.outputMode("update").foreachBatch(write_to_mongo).start()

# Wait for the termination of the query
query_mongo.awaitTermination()
