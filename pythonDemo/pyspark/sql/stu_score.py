from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession.builder.appName("stu_score").getOrCreate()
    
    df = spark.read.csv(r"pythonDemo\pyspark\data\stu_score.txt", header=False, inferSchema=True)
    df2 = df.toDF("id", "name", "score")
    df2.printSchema()

    df2.show()
    
    df2.createOrReplaceTempView("stu_score")

    spark.sql("""
              SELECT * 
              FROM stu_score 
              where name='语文' 
              limit 10
              """).show()
    print()
    
    df2.wheres("name='语文'").limit(10).show()

    spark.stop()