from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode

spark = SparkSession.builder.appName("WordCount").getOrCreate()


# Read the input file
input_file = r"pythonDemo\pyspark\data\words.txt"
df = spark.read.text(input_file)

# Split each line into words
# 默认text的列是value，withColumn操作旧列，返回一个新列
words = df.withColumn("word", explode(split(df.value, "\s+")))
words.groupBy("word").count().withColumnRenamed("count", "cnt").show()


# Alternatively, you can use SQL to achieve the same result
df.createTempView("df")

spark.sql("""SELECT 
                word, COUNT(1) AS cnt 
          FROM df 
                lateral VIEW explode(split(value, ' ')) tmp aS word 
          GROUP BY word""") \
       .show()



spark.stop()
