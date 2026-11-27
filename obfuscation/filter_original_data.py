import torch, json

with open(".../intersection_apk_ids.json") as f:
    intersection = set(json.load(f))

dataset = torch.load("original_dataset.pt")
filtered = [data for data in dataset if data.apkname in intersection]
torch.save(filtered, "filtered_dataset.pt")