import random 
symbols="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

symbols_weight={}
for i in range(len(symbols)):
    symbols_weight[symbols[i]]=1

symbols_weight['D']=2

def get_material():
    print(symbols, list(symbols_weight.values()))
    print(len(symbols), len(list(symbols_weight.values())))
    random_letter=random.choices(symbols,weights=symbols_weight.values())[0]
    return random_letter


print(get_material())

#generate 1000 random amterials and draw an histogram with the chosen values
#import matplotlib.pyplot as plt
#import numpy as np
#materials = [get_material() for i in range(1000)]
#plt.hist(materials, bins=8)
#plt.show()
