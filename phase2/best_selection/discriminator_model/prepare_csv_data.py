import csv

# 路径配置
metrics_csv_path = "" # benchmark csv with 4 metrics
traj_csv_path = "" # traj csv with 1 metric
output_csv_path = "" 

# 读取 traj.csv 到字典：video_name -> similarity
similarity_dict = {}
with open(traj_csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    for row in reader:
        if row and len(row) >= 2:
            similarity_dict[row[0]] = row[1]

# 读取 metrics_csv_path 并合并 similarity
merged_rows = []
with open(metrics_csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    header.append("similarity") 
    merged_rows.append(header)

    for row in reader:
        video_name = row[0]
        similarity = similarity_dict.get(video_name, "") 
        row.append(similarity)
        merged_rows.append(row)

# 写入合并后的 CSV
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(merged_rows)

print(f"合并完成，共写入 {len(merged_rows) - 1} 条数据，结果保存在：{output_csv_path}")
