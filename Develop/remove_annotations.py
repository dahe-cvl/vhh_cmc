import numpy as np

dataset_type = "train"
remove_entries_file = dataset_type + "_remove_annotations.txt"
annotation_entries_file = "/data/share/cmc_eval_dataset/annotation/" + str(dataset_type) + "_shots.flist"
target_file = "/data/share/cmc_eval_dataset/annotation/" + str(dataset_type) + "_shots_reworked.flist"

print("read remove_entries_file ...")
fp = open(remove_entries_file, "r")
lines = fp.readlines()
fp.close()

search_lines = []
for line in lines:
    line = line.replace("\n", "")
    search_lines.append(line)
search_lines_np = np.array(search_lines)
#print(search_lines_np.shape)

print("read annotation entries file ...")
fp = open(annotation_entries_file, "r")
lines_annotations = fp.readlines()
fp.close()

annotation_lines = []
for line in lines_annotations:
    line = line.replace("\n", "")
    annotation_lines.append(line)
annotation_lines_np = np.array(annotation_lines)
#print(annotation_lines_np.shape)

print("start search process ...")
indices = []
for i in range(0, len(search_lines_np)):
    search_line = search_lines_np[i]
    print("--------------------------------------------")
    print(i)
    print("search line: " + str(search_line))

    for j in range(0, len(annotation_lines_np)):
        if (search_line in annotation_lines_np[j]):
            print("find in line: " + str(j))
            print(annotation_lines_np[j])
            indices.append(j)
final_np = np.delete(annotation_lines_np, indices)

print("write result to new file ...")
fp = open(target_file, "w")
for a in range(0, len(final_np)):
    fp.write(final_np[a] + "\n")
fp.close()

print("successfully finished")

