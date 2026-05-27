import os

for elem in os.listdir(os.path.join(os.getcwd(), "results")):
    if elem.endswith(".csv"):
        os.remove(elem)
