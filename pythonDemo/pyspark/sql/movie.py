from pyspark.sql import SparkSession

# create spark session
spark = SparkSession.builder.appName("movie").getOrCreate()


columns = ["user_id", "movie_id", "rating", "timestamp"]


# read movie data from csv file
movies = spark.read.csv(r"pythonDemo\pyspark\data\sql\u.data", header=False, inferSchema=True,sep="\t").toDF(*columns)


# create temporary view for movies table
movies.createOrReplaceTempView("movies")

# TODO: 询用户平均分
spark.sql("SELECT user_id,AVG(rating) avg_rating FROM movies group by user_id order by avg_rating desc").show()


# TODO: 电影平均分
spark.sql("SELECT movie_id, AVG(rating) avg_rating FROM movies group by movie_id order by avg_rating desc").show()

# TODO: 询大于平均分的电影的数量


spark.stop()