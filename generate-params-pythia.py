DRAFTER_COMBINATIONS = ["0-1-2", "1-2", "0-1", "0-2"]
HIDDEN_DIMS = [32, 128, 256]
NUM_LAYERS = [3, 10, 25]

with open("params-sweeping/params-pythia.txt", "w") as f:
    for d in DRAFTER_COMBINATIONS:
        for h in HIDDEN_DIMS:
            for l in NUM_LAYERS:
                f.write(f"{d} {h} {l}\n")
