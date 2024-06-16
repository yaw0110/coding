import json

with open(r"期末\analysis1.ipynb",encoding="utf-8") as fp:
    content = json.load(fp)

with open("2.py","w",encoding="utf-8") as fp:
    for item in content["cells"]:
        fp.writelines([i.rstrip()+"\n" for i in item["source"]])
