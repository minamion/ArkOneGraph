# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:32:11 2019

@author: sqr_p
"""

Price = dict()
with open('price.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        name, value = line.split()
        Price[name] = int(value)

Credit = dict()
with open('creditPrice.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        name, value = line.split()
        Credit[name] = float(value)