import os

for elem in os.listdir(os.getcwd()):
    if elem.endswith('.csv'):
        os.remove(elem)