from MaterialPlanning import MaterialPlanning
#from MaterialPlanningRaw import MaterialPlanning as MPR
from utils import required_dct, owned_dct

'''
        Print_functions = [
            self.output_cost,           #理智消耗
            self.output_stages,         #关卡次数
            self.output_items,          #合成次数
            self.output_values,         #物品价值
            self.output_green,          #绿票商店
            self.output_yellow,         #黄票商店
            self.output_effect,         #关卡效率
            self.output_best_stage,     #关卡推荐
            self.output_credit,         #信用商店
            self.output_WeiJiHeYue      #危机合约(喧闹法则活动商店)
            ]
'''

if __name__ == '__main__':

    mp = MaterialPlanning(filter_stages=[],
                          filter_freq=1,
                          update=False,
                          banned_stages={},
#                          expValue=30,                 #1224更新后此参数无效, 使用经验需求来调节经验价值
                          printSetting='1111111010',    #参照上面Print_functions的顺序设置, 1输出, 0不输出
                          ConvertionDR=0.18,            #副产物掉落率
                          costLimit=135                 #理智上限
                          )

#    mpr = MPR()

    res, mat1, mat2 = mp.get_plan(required_dct, owned_dct, print_output=True, outcome=True,
                                  gold_demand=True, exp_demand=True)

#    mpr.get_plan(required_dct, owned_dct, print_output=True, outcome=True,
#                                  gold_demand=True, exp_demand=True)
