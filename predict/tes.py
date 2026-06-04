import os
path = r"bisindo_kata/npy"
for folder in sorted(os.listdir(path)):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        print(f"{folder}: {count} sample")