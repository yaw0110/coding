from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd

# create spark session
spark = SparkSession.builder.appName("dataframe").getOrCreate()


sc = spark.sparkContext

rdd = sc.textFile(r"pythonDemo\pyspark\data\sql\people.txt") \
        .map(lambda x: x.split(",")) \
        .map(lambda x: (x[0], int(x[1])))

# TODO: rdd转换为DataFrame

spark.createDataFrame(rdd, ["name", "age"]).show()


# TODO: 调用StructType类
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])

df = spark.createDataFrame(rdd, schema)
df.show()

# TODO: toDF()方法
rdd2 = sc.textFile(r"pythonDemo\pyspark\data\sql\people.txt") \
        .map(lambda x: x.split(",")) \
        .map(lambda x: (x[0], int(x[1])))

df2 = rdd2.toDF(["name", "age"])
df2.show()

sc.stop()
spark.stop()