from pyspark import SparkConf, SparkContext
from pyspark.storagelevel import StorageLevel
import jieba

def context_jieba(data):
    """通过jieba分词工具 进行分词操作"""
    seg = jieba.cut_for_search(data)
    l = list()
    for word in seg:
        l.append(word)
    return l


def filter_words(data):
    """过滤不要的 谷 \ 帮 \ 客"""
    return data not in ['谷', '帮', '客']


def append_words(data):
    """修订某些关键词的内容"""
    if data == '传智播': data = '传智播客'
    if data == '院校': data = '院校帮'
    if data == '博学': data = '博学谷'
    return (data, 1)


def extract_user_and_word(data):
    """传入数据是 元组 (1, 我喜欢传智播客)"""
    user_id = data[0]
    content = data[1]
    # 对content进行分词
    words = context_jieba(content)

    return_list = list()
    for word in words:
        # 不要忘记过滤 \谷 \ 帮 \ 客
        if filter_words(word):
            return_list.append((user_id + "_" + append_words(word)[0], 1))

    return return_list

if __name__ == "__main__":
    conf = SparkConf().setAppName("log").setMaster("local")
    sc = SparkContext(conf=conf)

    logData = sc.textFile("pythonDemo\pyspark\data\SogouQ.txt")
    splitData = logData.map(lambda line:line.split("\t"))
    splitData.persist(StorageLevel.MEMORY_AND_DISK)

# TODO: 1. 统计每个用户的搜索词频率 
    userAndWord = splitData.map(lambda line:line[2]) \
                .flatMap(context_jieba) \
                .filter(filter_words) \
                .map(append_words) \
                .reduceByKey(lambda a,b:a+b) \
                .sortBy(lambda x:x[1], ascending=False) \
                .saveAsTextFile(r"pythonDemo\pyspark\output\userAndWord")

# TODO: 2：用户和关键词组合分许

    userAndContent =splitData.map(lambda line:(line[1],line[2])) \
                            .flatMap(extract_user_and_word) \
                            .map(lambda x: (x,1)) \
                            .reduceByKey(lambda a,b:a+b) \
                            .sortBy(ascending=False,keyfunc=lambda x: x[1]) \
                            .saveAsTextFile(r"pythonDemo\pyspark\output\userAndContent")
    
# TODO: 3:热门搜索时间段分析
    timeAnalysis = splitData.map(lambda line:line[0]) \
                            .map(lambda time:(time.split(":")[0],1)) \
                            .reduceByKey(lambda a,b:a+b) \
                            .sortBy(keyfunc=lambda x:x[1], ascending=False) \
                            .saveAsTextFile(r"pythonDemo\pyspark\output\timeAnalysis")

    sc.stop()