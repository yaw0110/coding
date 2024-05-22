from pyspark import SparkContext,SparkConf

conf = SparkConf().setAppName("Fold").setMaster("local")
sc = SparkContext(conf=conf)


rdd = sc.parallelize(range(1,11),3)

rdd.mapPartitionsWithIndex(lambda i,it: (f'inds:{i}',list(it))).foreach(print)
print(rdd.fold(10,lambda x,y:x+y))


sc.stop()