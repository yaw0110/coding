from pyspark import SparkConf, SparkContext
from pyspark.storagelevel import StorageLevel
import re

conf = SparkConf().setAppName("AccAndBoard").setMaster("local")
sc = SparkContext(conf=conf)

data = sc.textFile(r"pythonDemo\pyspark\data\accumulator_broadcast_data.txt")

abnoraml_char = [",",'.','!','$','#','%']
broad = sc.broadcast(abnoraml_char)

acc = sc.accumulator(0)

def filter_data(line):
    if line.strip() == "":
        return []

    newWords = []
    words = re.split(r'\s+', line.strip())

    for word in words:
        if word in broad.value:
            acc.add(1)
        else:
            newWords.append(word)

    return newWords

data.flatMap(filter_data) \
   .map(lambda x: (x, 1)) \
   .reduceByKey(lambda a, b: a + b) \
   .sortBy(lambda x: x[1], ascending=False) \
   .foreach(lambda x: print(x))

print("Total abnormal characters found: ", acc.value)


sc.stop()



