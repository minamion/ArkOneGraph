# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:32:11 2019

@author: sqr_p
"""
import codecs

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

HeYue = dict()
with open('HeYue.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        name, value = line.split()
        HeYue[name] = float(value)

HYO = dict()
with open('HeYueOrd.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        name, value = line.split()
        HYO[name] = float(value)

with codecs.open('materialIO.txt', 'r', 'utf-8') as f:
    material = eval(f.readline())
    required_dct = {x['name']: x['need'] for x in material}
    owned_dct = {x['name']: x['have'] for x in material}

# 当前干员数量
figureCount = {
               '1/2-Star': 7,
               '3-Star': 17,
               '4-Star': 32+1,
               '5-Star': 43+1,
               '6-Star': 16+1
              }
exp_required = {
                '1/2-Star': 9800,
                '3-Star': 115400,
                '4-Star': 484000,
                '5-Star': 734400,
                '6-Star': 1111400
               }
gold_required = {
                 '1/2-Star': 5323,
                 '3-Star': 104040,
                 '4-Star': 482003,
                 '5-Star': 819325,
                 '6-Star': 1334769
                }
required_dct.update({
                        '经验': sum(v*exp_required[k] for k, v in figureCount.items()),
                        '龙门币': sum(v*gold_required[k] for k, v in figureCount.items()),
                        '技巧概要·卷1': 696,
                        '技巧概要·卷2': 1530,
                        '技巧概要·卷3': 4898
                    })
#required_dct.update({
#                        '经验': 0,
#                        '龙门币': 0,
#                        '技巧概要·卷1': 674,
#                        '技巧概要·卷2': 1479,
#                        '技巧概要·卷3': 4709
#                    })
owned_dct.update({'经验': 0, '龙门币': 0, '技巧概要·卷3': 0,
                  '技巧概要·卷2': 0, '技巧概要·卷1': 0})
