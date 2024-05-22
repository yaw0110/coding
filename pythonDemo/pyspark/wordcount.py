from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("WordCount").setMaster("local")
sc = SparkContext(conf=conf)

text = sc.textFile("pythonDemo\pyspark\data\word.txt")

text.flatMap(lambda line: line.split(" ")) \
   .map(lambda word: (word, 1)) \
   .reduceByKey(lambda a, b: a + b) \
   .foreach(lambda x: print(x))

sc.stop()