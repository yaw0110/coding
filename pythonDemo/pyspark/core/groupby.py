from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("GroupBy").setMaster("local")
sc = SparkContext(conf=conf)

rdd = sc.parallelize([1,2,3,4,5,6,7,8,9,10])

grouped = rdd.groupBy(lambda num : 'even' if num % 2 == 0 else 'odd')

grouped.map(lambda x: (x[0], sum(list(x[1])))).foreach(print)

print()
for key, value in grouped.collect():
    print(key, list(value))   