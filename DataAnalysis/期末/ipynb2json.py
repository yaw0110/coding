import json

with open(r"DataAnalysis\期末\analysis.ipynb",encoding="utf-8") as fp:
    content = json.load(fp)

with open("1.py","w",encoding="utf-8") as fp:
    for item in content["cells"]:
        fp.writelines([i.rstrip()+"\n" for i in item["source"]])
