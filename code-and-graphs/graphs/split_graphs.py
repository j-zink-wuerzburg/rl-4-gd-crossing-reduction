import tarfile
import os
import random

# Parameters
archive_path = "rome/rome_gexf.tar.gz"
output_dir   = "rome/splits"
seed         = 42
ratios       = (0.80, 0.0, 0.2)  # train, val, test

# Extract
#with tarfile.open(archive_path, "r:gz") as tar:
#    tar.extractall(path=output_dir)

# Gather all GEXF paths
#    <-- change here: graphs live under output_dir/data/
graph_dir = os.path.join(output_dir, "data")
all_graphs = [
    os.path.join(graph_dir, fn)
    for fn in os.listdir(graph_dir)
    if fn.endswith(".gexf")
]

# Shuffle reproducibly
random.seed(seed)
random.shuffle(all_graphs)

# Compute split indices
n_total = len(all_graphs)
n_train = int(n_total * ratios[0])
n_val   = int(n_total * ratios[1])

train_graphs = all_graphs[:n_train]
val_graphs   = all_graphs[n_train : n_train + n_val]
test_graphs  = all_graphs[n_train + n_val :]

# Write out file lists
os.makedirs(output_dir, exist_ok=True)
for split_name, split_list in [("train", train_graphs),
                               ("val",   val_graphs),
                               ("test",  test_graphs)]:
    with open(os.path.join(output_dir, f"{split_name}.txt"), "w") as f:
        for path in split_list:
            f.write(path + "\n")

print(f"Total graphs: {n_total}")
print(f"Train / Val / Test sizes: {len(train_graphs)} / {len(val_graphs)} / {len(test_graphs)}")
