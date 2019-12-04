import sys, codecs
from MaterialPlanning import MaterialPlanning
#from MaterialPlanningRaw import MaterialPlanning

'''
        Print_functions = [
            self.output_cost,
            self.output_stages,
            self.output_items,
            self.output_values,
            self.output_green,
            self.output_yellow,
            self.output_effect,
            self.output_best_stage,
            self.output_credit,
            self.output_WeiJiHeYue]
'''

if __name__ == '__main__':

    if '-fe' in sys.argv:
        filter_stages = ['GT-'+str(i) for i in range(1,7)]
    else:
        filter_stages = []

    mp = MaterialPlanning(filter_stages=filter_stages, update=False,
                          banned_stages={}, expValue=30, printSetting='1111111110', ConvertionDR=0.18)
#    mp = MaterialPlanning()

    with codecs.open('required.txt', 'r', 'utf-8') as f:
        required_dct = {}
        for line in f.readlines():
            required_dct[line.split(' ')[0]] = int(line.split(' ')[1])

    with codecs.open('owned.txt', 'r', 'utf-8') as f:
        owned_dct = {}
        for line in f.readlines():
            owned_dct[line.split(' ')[0]] = int(line.split(' ')[1])

    res, mat1, mat2 = mp.get_plan(required_dct, owned_dct, True, outcome=True,
                                  gold_demand=True, exp_demand=False)
