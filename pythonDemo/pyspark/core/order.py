# coding:utf8

from pyspark import SparkConf, SparkContext
import json

conf = SparkConf().setAppName("Order").setMaster("local")
sc = SparkContext(conf=conf)

json_data = sc.textFile("pythonDemo\\pyspark\\data\\order.text") \
    .flatMap(lambda line: line.split("|")) \
    
json_data = json_data.map(lambda line: json.loads(line))
filtered_data = json_data.filter(lambda x: x['areaName'] == '北京')

target_data = filtered_data.map(lambda x: (x['areaName'], x['category'])) \
                            .distinct()

target_data.saveAsTextFile("pythonDemo\\pyspark\\output\\order.text")

