import os
import json
import sys

data = sys.argv[1].lower()

if data=="nonnews":
    datainput = "NonNewsOpenL3/_last" 
elif data=="radionews":
    datainput = "RadioNews-BBC/RadioNewsUniform1/x-vectors"
else:
    datainput = "OpenBMAT/BMAT_ATS1/x-vectors"

train = []
validation = []
test = []

test_turn = True
skip_one = True
for index, emb in enumerate(os.listdir(datainput)):
    if index and not (index)%3:
        if test_turn:
            test.append(emb)
            test_turn = False
        else:
            if skip_one:
                train.append(emb)
                skip_one = False
            else:
                validation.append(emb)
                test_turn = True
    else:
        train.append(emb)

split = {"train":train, "test":test, "validation":validation}
for key, value in split.items():
    print(key)
    print(len(value))
    print("\n===================\n")

if data == "nonnews":
    out = "NonNews_split.json" 
elif data=="radionews":
    out = "RadioNews_split.json"
else:
    out = "BMAT_split.json"

with open(out, "w") as f:
    json.dump(split, f)
