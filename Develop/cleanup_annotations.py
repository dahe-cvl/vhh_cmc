import numpy as np
import os, shutil

dataset_type = "train"
remove_entries_file = dataset_type + "_cleanup_tracks.txt"
annotation_entries_file = "/data/share/cmc_eval_dataset/annotation/" + str(dataset_type) + "_shots_reworked.flist"

target_file = "/data/share/cmc_eval_dataset/annotation/" + str(dataset_type) + "_shots_reworked_tracks.flist"
src_dir = "/data/share/cmc_eval_dataset/"
dst_dir = "/data/share/cmc_eval_dataset/training_data/track/"

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
src_paths = []
for i in range(0, len(search_lines_np)):
    search_line = search_lines_np[i]
    print("--------------------------------------------")
    print(i)
    print("search line: " + str(search_line))

    for j in range(0, len(annotation_lines_np)):
        if (search_line in annotation_lines_np[j]):
            print("find in line: " + str(j))
            print(annotation_lines_np[j])
            print(annotation_lines_np[j].split(" "))
            indices.append(j)
            src_paths.append(src_dir + annotation_lines_np[j].split(" ")[0].replace("\\", "/"))

# move movies to separate folder
for f in src_paths:
    if not os.path.exists(dst_dir + f.split("/")[-1]):
        print("move: " + f)
        shutil.move(f, dst_dir)

# rename
final_l = []
for j in range(0, len(annotation_lines_np)):
    tmp_line = annotation_lines_np[j].replace("\\", "/")
    res = tmp_line
    if (j in indices):
        print("replace in line: " + str(j))
        print(tmp_line.split("/"))
        res = os.path.join(tmp_line.split("/")[0], "track")
        res = os.path.join(res, tmp_line.split("/")[2])
        print(res)
    final_l.append(res)
final_np = np.array(final_l)
print(final_np.shape)

''''''
print("write result to new file ...")
fp = open(target_file, "w")
for a in range(0, len(final_np)):
    fp.write(final_np[a] + "\n")
fp.close()

print("successfully finished")

