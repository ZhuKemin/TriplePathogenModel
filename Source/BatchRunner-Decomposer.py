import subprocess

# 要处理的数据列表
data_files = [
    ("Yu-2019", "RSV"),
    # ("Zou-2016", "RSV"),
    # ("C702_1", "Flu"),
    # ("E196_11", "Flu")
]

# 遍历数据文件并调用R脚本
for filename, pathogen in data_files:
    print (filename)
    # 调用R脚本并传递参数
    subprocess.run(["Rscript", "Decomposer-AnnualFixed-SingleCycle.R", filename, pathogen], check=True)
    subprocess.run(["Rscript", "Decomposer-AnnualFixed-DualCycle.R", filename, pathogen], check=True)
    subprocess.run(["Rscript", "Decomposer-DualCycle.R", filename, pathogen], check=True)
