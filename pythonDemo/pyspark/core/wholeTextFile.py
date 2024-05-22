from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("wholeTextFile").setMaster("local")
sc = SparkContext(conf=conf)


rdd = sc.wholeTextFiles('pythonDemo/pyspark/data/tiny_files', minPartitions=10)

print(rdd.getNumPartitions())