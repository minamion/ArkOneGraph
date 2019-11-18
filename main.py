import sys, codecs
from MaterialPlanning import MaterialPlanning
'''
printSetting: 共9位数字,1代表输出,0代表不输出 顺序为:
    总理智消耗, 最优关卡, 合成路线,
    物品价值, 绿票商店, 黄票商店,
    关卡效率, 推荐关卡, 信用商店
    例: '111111111' 表示全部输出, '000000000' 表示全部不输出, '000000010' 表示输出推荐关卡
    '''

if __name__ == '__main__':

    if '-fe' in sys.argv:
        filter_stages = ['GT-'+str(i) for i in range(1,7)]
    else:
        filter_stages = []

    mp = MaterialPlanning(filter_stages=filter_stages, update=False,
                          banned_stages={}, expValue=30, printSetting='000000010')

    with codecs.open('required.txt', 'r', 'utf-8') as f:
        required_dct = {}
        for line in f.readlines():
            required_dct[line.split(' ')[0]] = int(line.split(' ')[1])

    with codecs.open('owned.txt', 'r', 'utf-8') as f:
        owned_dct = {}
        for line in f.readlines():
            owned_dct[line.split(' ')[0]] = int(line.split(' ')[1])

    res, mat1, mat2 = mp.get_plan(required_dct, owned_dct, True, outcome=True,
                                  gold_demand=False, exp_demand=True)
