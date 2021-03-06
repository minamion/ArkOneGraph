import numpy as np
import urllib.request, json, time, os, copy, sys
from scipy.optimize import linprog
from utils import Price, Credit, HeYue, HYO
from collections import defaultdict as ddict
import pandas as pd

global penguin_url
penguin_url = 'https://penguin-stats.io/PenguinStats/api/'

class MaterialPlanning(object):

    def __init__(self,
                 filter_freq=10,
                 filter_stages=[],
                 url_stats='result/matrix?show_stage_details=true&show_item_details=true',
                 url_rules='formula',
                 path_stats='data/matrix.json',
                 path_rules='data/formula.json',
                 update=False,
                 banned_stages={},
#                 expValue=30,
                 ConvertionDR=0.18,
                 printSetting='111111111',
                 costLimit=135,
                 costType='stone'):
        """
        Object initialization.
        Args:
            filter_freq: int or None. The lowest frequence that we consider.
                No filter will be applied if None.
            url_stats: string. url to the dropping rate stats data.
            url_rules: string. url to the composing rules data.
            path_stats: string. local path to the dropping rate stats data.
            path_rules: string. local path to the composing rules data.
        """
        try:
            material_probs, convertion_rules = load_data(path_stats, path_rules)
        except:
            print('exceptRequesting data from web resources (i.e., penguin-stats.io)...', end=' ')
            material_probs, convertion_rules = request_data(penguin_url+url_stats, penguin_url+url_rules, path_stats, path_rules)
            print('done.')
        if update:
            print('Requesting data from web resources (i.e., penguin-stats.io)...', end=' ')
            material_probs, convertion_rules = request_data(penguin_url+url_stats, penguin_url+url_rules, path_stats, path_rules)
            print('done.')


        '''
            添加一些企鹅物流里没有的合成公式
        '''
        convertion_rules.append({'costs': [{'count':3, 'id':'3302', 'name':'技巧概要·卷2', 'rarity':2}],
                                 'extraOutcome': [{'count': 1, 'id': '3303', 'name': '技巧概要·卷3', 'rarity': 3, 'weight': 1}],
                                 'goldCost': 0,
                                 'id': '3303',
                                 'name': '技巧概要·卷3',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count':3, 'id':'3301', 'name':'技巧概要·卷1', 'rarity':1}],
                                 'extraOutcome': [{'count': 1, 'id': '3302', 'name': '技巧概要·卷2', 'rarity': 2, 'weight': 1}],
                                 'goldCost': 0,
                                 'id': '3302',
                                 'name': '技巧概要·卷2',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count':1/200, 'id':'2001', 'name':'基础作战记录', 'rarity':1}],
                                 'extraOutcome': [],
                                 'goldCost': 0,
                                 'id': '00011',
                                 'name': '经验',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count':1/400, 'id':'2001', 'name':'初级作战记录', 'rarity':1}],
                                 'extraOutcome': [],
                                 'goldCost': 0,
                                 'id': '00011',
                                 'name': '经验',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count':1/1000, 'id':'2001', 'name':'中级作战记录', 'rarity':1}],
                                 'extraOutcome': [],
                                 'goldCost': 0,
                                 'id': '00011',
                                 'name': '经验',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count':1/400, 'id':'3003', 'name':'赤金', 'rarity':1}],
                                 'extraOutcome': [],
                                 'goldCost': 0,
                                 'id': '00011',
                                 'name': '经验',
                                 'totalWeight': 1})
        convertion_rules.append({'costs': [{'count': 4, 'id': '30014', 'name': '固源岩组', 'rarity': 2}],
                                    'extraOutcome': [
                                   {'count': 1, 'id': '30013', 'name': '固源岩组', 'rarity': 2, 'weight': 60},
                                   {'count': 1, 'id': '30023', 'name': '糖组', 'rarity': 2, 'weight': 50},
                                   {'count': 1, 'id': '30033', 'name': '聚酸酯组', 'rarity': 2, 'weight': 50},
                                   {'count': 1, 'id': '30043', 'name': '异铁组', 'rarity': 2, 'weight': 40},
                                   {'count': 1, 'id': '31013', 'name': '凝胶', 'rarity': 2, 'weight': 40},
                                   {'count': 1, 'id': '31013', 'name': '凝胶', 'rarity': 2, 'weight': 40},
                                   {'count': 1, 'id': '30053', 'name': '酮凝集组', 'rarity': 2, 'weight': 40},
                                   {'count': 1, 'id': '30063', 'name': '全新装置', 'rarity': 2, 'weight': 30},
                                   {'count': 1, 'id': '30073', 'name': '扭转醇', 'rarity': 2, 'weight': 45},
                                   {'count': 1, 'id': '30083', 'name': '轻锰矿', 'rarity': 2, 'weight': 40},
                                   {'count': 1, 'id': '30093', 'name': '研磨石', 'rarity': 2, 'weight': 36},
                                   {'count': 1, 'id': '30103', 'name': 'RMA70-12', 'rarity': 2, 'weight': 30}],
                                  'goldCost': 300,
                                  'id': '30014',
                                  'name': '提纯源岩',
                                  'totalWeight': 421})

        self.公招出四星的概率 = 0.186
        self.costLimit = costLimit #理智上限
        self.convertion_rules = convertion_rules
        self.material_probs = material_probs
        self.banned_stages = banned_stages
        self.costType = costType

        filtered_probs = []
        needed_stage = []
        for dct in material_probs['matrix']:
            if dct['times']>=filter_freq and dct['stage']['code'] not in filter_stages:
                filtered_probs.append(dct)
            else:
                if dct['stage']['code'] not in needed_stage:
                    print(dct['stage']['code'], dct['times'])
                    needed_stage.append(dct['stage']['code'])
        material_probs['matrix'] = filtered_probs
        self.ConvertionDR = ConvertionDR
        self._set_lp_parameters(*self._pre_processing(material_probs, convertion_rules))
        assert len(printSetting)==10, 'printSetting 长度应为10'
        assert printSetting.count('1') + printSetting.count('0') == 10, 'printSetting 中只能含有0或1'
        self.printSetting = [int(x) for x in printSetting]


    def _pre_processing(self, material_probs, convertion_rules):
        """
        Compute costs, convertion rules and items probabilities from requested dictionaries.
        Args:
            material_probs: List of dictionaries recording the dropping info per stage per item.
                Keys of instances: ["itemID", "times", "itemName", "quantity", "apCost", "stageCode", "stageID"].
            convertion_rules: List of dictionaries recording the rules of composing.
                Keys of instances: ["id", "name", "level", "source", "madeof"].
        """
        # To count items and stages.
        additional_items = {'30135': u'D32钢', '30125': u'双极纳米片',
                            '30115': u'聚合剂', '00011':'经验', '00021':'龙门币',
                            '31014':'聚合凝胶', '31024':'炽合金块', '31013':'凝胶',
                            '31023':'炽合金'}
        item_dct = {}
        stage_dct = {}
        for dct in material_probs['matrix']:
            item_dct[dct['item']['itemId']]=dct['item']['name']
            stage_dct[dct['stage']['code']]=dct['stage']['code']
        item_dct.update(additional_items)

        # To construct mapping from id to item names.
        item_array = []
        item_id_array = []
        for k,v in item_dct.items():
            try:
                float(k)
                item_array.append(v)
                item_id_array.append(k)
            except:
                pass
        self.item_array = np.array(item_array)
        self.item_id_array = np.array(item_id_array)
        self.item_dct_rv = {v:k for k,v in enumerate(item_array)}


        # To construct mapping from stage id to stage names and vice versa.
        stage_array = [x+'-'+y for x in ['LS', 'CE'] for y in '12345']
        for k,v in stage_dct.items():
            if v not in self.banned_stages:
                stage_array.append(v)

        self.stage_array = np.array(stage_array)
        self.stage_dct_rv = {v:k for k,v in enumerate(self.stage_array)}

        # To format dropping records into sparse probability matrix
        probs_matrix = np.zeros([len(stage_array), len(item_array)])
        cost_lst = np.zeros(len(stage_array))

        for dct in material_probs['matrix']:
            try:
                float(dct['item']['itemId'])
                probs_matrix[self.stage_dct_rv[dct['stage']['code']], self.item_dct_rv[dct['item']['name']]] = dct['quantity']/float(dct['times'])

                cost_lst[self.stage_dct_rv[dct['stage']['code']]] = dct['stage']['apCost']
            except:
                pass

        # 添加LS, CE, S4-6, S5-2的掉落
        cost_lst[0:10] = [10,15,20,25,30,10,15,20,25,30]
        for k, stage in enumerate(self.stage_array):
            probs_matrix[k, self.item_dct_rv['龙门币']] = cost_lst[k]*12
        probs_matrix[self.stage_dct_rv['S4-6'], self.item_dct_rv['龙门币']] += 3228
        probs_matrix[self.stage_dct_rv['S5-2'], self.item_dct_rv['龙门币']] += 2484
        probs_matrix[self.stage_dct_rv['CE-1'], self.item_dct_rv['龙门币']] = 1700
        probs_matrix[self.stage_dct_rv['CE-2'], self.item_dct_rv['龙门币']] = 2800
        probs_matrix[self.stage_dct_rv['CE-3'], self.item_dct_rv['龙门币']] = 4100
        probs_matrix[self.stage_dct_rv['CE-4'], self.item_dct_rv['龙门币']] = 5700
        probs_matrix[self.stage_dct_rv['CE-5'], self.item_dct_rv['龙门币']] = 7500
        probs_matrix[self.stage_dct_rv['LS-1'], self.item_dct_rv['经验']] = 1600
        probs_matrix[self.stage_dct_rv['LS-2'], self.item_dct_rv['经验']] = 2800
        probs_matrix[self.stage_dct_rv['LS-3'], self.item_dct_rv['经验']] = 3900
        probs_matrix[self.stage_dct_rv['LS-4'], self.item_dct_rv['经验']] = 5900
        probs_matrix[self.stage_dct_rv['LS-5'], self.item_dct_rv['经验']] = 7400

        # To build equavalence relationship from convert_rule_dct.
        self.convertions_dct = {}
        convertion_matrix = []
        convertion_outc_matrix = []
        convertion_cost_lst = []
        for rule in convertion_rules:
            convertion = np.zeros(len(self.item_array))
            convertion[self.item_dct_rv[rule['name']]] = 1

            comp_dct = {comp['name']:comp['count'] for comp in rule['costs']}
            self.convertions_dct[rule['name']] = comp_dct
            for iname in comp_dct:
                convertion[self.item_dct_rv[iname]] -= comp_dct[iname]
            convertion[self.item_dct_rv['龙门币']] -= rule['goldCost']
            convertion_matrix.append(copy.deepcopy(convertion))

            outc_dct = {outc['name']:outc['count'] for outc in rule['extraOutcome']}
            outc_wgh = {outc['name']:outc['weight'] for outc in rule['extraOutcome']}
            weight_sum = float(sum(outc_wgh.values()))
            for iname in outc_dct:
                convertion[self.item_dct_rv[iname]] += outc_dct[iname]*self.ConvertionDR*outc_wgh[iname]/weight_sum
            convertion_outc_matrix.append(convertion)
            convertion_cost_lst.append(0)

        convertions_group = (np.array(convertion_matrix), np.array(convertion_outc_matrix), convertion_cost_lst)
        farms_group = (probs_matrix, cost_lst)
        return convertions_group, farms_group


    def _set_lp_parameters(self, convertions_group, farms_group):
        """
        Object initialization.
        Args:
            convertion_matrix: matrix of shape [n_rules, n_items].
                Each row represent a rule.
            convertion_cost_lst: list. Cost in equal value to the currency spent in convertion.
            probs_matrix: sparse matrix of shape [n_stages, n_items].
                Items per clear (probabilities) at each stage.
            cost_lst: list. Costs per clear at each stage.
        """
        self.convertion_matrix, self.convertion_outc_matrix, self.convertion_cost_lst = convertions_group
        self.probs_matrix, self.cost_lst = farms_group

        assert len(self.probs_matrix)==len(self.cost_lst)
        assert len(self.convertion_matrix)==len(self.convertion_cost_lst)
        assert self.probs_matrix.shape[1]==self.convertion_matrix.shape[1]


    def update(self,
               filter_freq=20,
               filter_stages=[],
               url_stats='result/matrix?show_stage_details=true&show_item_details=true',
               url_rules='formula',
               path_stats='data/matrix.json',
               path_rules='data/formula.json'):
        """
        To update parameters when probabilities change or new items added.
        Args:
            url_stats: string. url to the dropping rate stats data.
            url_rules: string. url to the composing rules data.
            path_stats: string. local path to the dropping rate stats data.
            path_rules: string. local path to the composing rules data.
        """
        print('Requesting data from web resources (i.e., penguin-stats.io)...', end=' ')
        material_probs, convertion_rules = request_data(penguin_url+url_stats, penguin_url+url_rules, path_stats, path_rules)
        print('done.')

        if filter_freq:
            filtered_probs = []
            for dct in material_probs['matrix']:
                if dct['times']>=filter_freq and dct['stage']['code'] not in filter_stages:
                    filtered_probs.append(dct)
            material_probs['matrix'] = filtered_probs

        self._set_lp_parameters(*self._pre_processing(material_probs, convertion_rules))


    def _get_plan_no_prioties(self, demand_lst, outcome=False, gold_demand=True, exp_demand=True):
        """
        To solve linear programming problem without prioties.
        Args:
            demand_lst: list of materials demand. Should include all items (zero if not required).
        Returns:
            strategy: list of required clear times for each stage.
            fun: estimated total cost.
        """
        A_ub = (np.vstack([self.probs_matrix, self.convertion_outc_matrix])
                if outcome else np.vstack([self.probs_matrix, self.convertion_matrix])).T
        self.farm_cost = (self.cost_lst)
        if self.costType == 'time':
            self.farm_cost = np.array(pd.read_csv('data/time.csv').time)
        cost = (np.hstack([self.farm_cost, self.convertion_cost_lst]))
        assert np.any(self.farm_cost>=0)

        excp_factor = 1.0
        dual_factor = 1.0

        while excp_factor>1e-7:
            solution = linprog(c=cost,
                               A_ub=-A_ub,
                               b_ub=-np.array(demand_lst)*excp_factor,
                               method='interior-point')
            if solution.status != 4:
                break

            excp_factor /= 10.0

        while dual_factor>1e-7:
            dual_solution = linprog(c=-np.array(demand_lst)*excp_factor*dual_factor,
                                    A_ub=A_ub.T,
                                    b_ub=cost,
                                    method='interior-point')
            if solution.status != 4:
                break

            dual_factor /= 10.0


        return solution, dual_solution, excp_factor


    def get_plan(self, requirement_dct, deposited_dct={},
                 print_output=False, outcome=False, gold_demand=True, exp_demand=True):
        """
        User API. Computing the material plan given requirements and owned items.
        Args:
                requirement_dct: dictionary. Contain only required items with their numbers.
                deposit_dct: dictionary. Contain only owned items with their numbers.
        """
        status_dct = {0: 'Optimization terminated successfully. ',
                      1: 'Iteration limit reached. ',
                      2: 'Problem appears to be infeasible. ',
                      3: 'Problem appears to be unbounded. ',
                      4: 'Numerical difficulties encountered.'}

        demand_lst = np.zeros(len(self.item_array))
        for k, v in requirement_dct.items():
            demand_lst[self.item_dct_rv[k]] = v
        for k, v in deposited_dct.items():
            demand_lst[self.item_dct_rv[k]] -= v

        stt = time.time()
        solution, dual_solution, excp_factor = self._get_plan_no_prioties(demand_lst, outcome, gold_demand, exp_demand)
        x, status = solution.x/excp_factor, solution.status
        y, self.slack = dual_solution.x, dual_solution.slack
        self.y = y
        n_looting, n_convertion = x[:len(self.cost_lst)], x[len(self.cost_lst):]

        cost = np.dot(x[:len(self.cost_lst)], self.cost_lst)

        if print_output:
            print(status_dct[status]+(' Computed in %.4f seconds,' %(time.time()-stt)))

        if status != 0:
            raise ValueError(status_dct[status])

        self.stages = []
        self.fullstages = []
        self.effect = dict()
        for i, t in enumerate(n_looting):
#            if t >= 0:
            self.effect[self.stage_array[i]] = sum([probsProb*y[probsidx] for probsidx, probsProb in enumerate(self.probs_matrix[i])])/self.farm_cost[i]
#            if t >= 0.1:
            target_items = np.where(self.probs_matrix[i]>0)[0]
            items = {self.item_array[idx]: float2str(self.probs_matrix[i, idx]*t)
            for idx in target_items if len(self.item_id_array[idx])<=5}
            stage = {
                "stage": self.stage_array[i],
                "count": float2str(t),
                "items": items
            }
            self.stages.append(stage)

        self.syntheses = []
        for i,t in enumerate(n_convertion):
            if t >= 0.1:
                target_item = self.item_array[np.argmax(self.convertion_matrix[i])]
                if target_item in ['经验', '龙门币']:
                    target_item_index = np.argmin(self.convertion_matrix[i])
                    materials = {self.item_array[target_item_index]:\
                        str(np.round(-self.convertion_matrix[i][target_item_index]*int(t+0.9),4))}
                else:
                    materials = {k: str(v*int(t+0.9)) for k,v in self.convertions_dct[target_item].items()}
                synthesis = {
                    "target": target_item,
                    "count": str(int(t+0.9)),
                    "materials": materials
                }
                self.syntheses.append(synthesis)
            elif t >= 0.05:
                target_item = self.item_array[np.argmax(self.convertion_matrix[i])]
                materials = { k: '%.1f'%(v*t) for k,v in self.convertions_dct[target_item].items() }
                synthesis = {
                    "target": target_item,
                    "count": '%.1f'%t,
                    "materials": materials
                }
                self.syntheses.append(synthesis)

        self.values = [{"level":'1', "items":[]},
                  {"level":'2', "items":[]},
                  {"level":'3', "items":[]},
                  {"level":'4', "items":[]},
                  {"level":'5', "items":[]}]
        self.item_value = dict()
        for i,item in enumerate(self.item_array):
            if y[i]>=0:
                if y[i]>0.1:
                    item_value = {
                        "name": item,
                        "value": '%.2f'%y[i]
                    }
                else:
                    item_value = {
                        "name": item,
                        "value": '%.5f'%(y[i])
                    }
                self.item_value[item] = y[i]
                self.values[int(self.item_id_array[i][-1])-1]['items'].append(item_value)
        for group in self.values:
            group["items"] = sorted(group["items"], key=lambda k: float(k['value']), reverse=True)

        self.res = {
            "cost": int(cost),
            "stages": self.stages,
            "syntheses": self.syntheses,
            "values": list(reversed(self.values))
        }

        if print_output:
            self.output()
        return self.res, x, self.effect

    def merge_droprate(self):
        self.droprate = ddict(dict)
        for itemIndex, item in enumerate(self.item_array):
            for stageIndex, stage in enumerate(self.stage_array):
                dr = self.probs_matrix[stageIndex, itemIndex]
                if dr > 0.0001:
                    self.droprate[item][stage] = {
                                'droprate':         dr,
                                'expected_cost':    self.cost_lst[stageIndex]/dr,
                                'effect':           self.effect[stage]
                            }

    def output_best_stage(self):
        self.merge_droprate()
        for item in self.item_array:
            itemLevel = self.item_id_array[self.item_dct_rv[item]][-1]
#            if itemLevel in '45':
#                break
            Max999 = {'Name': '', 'Cost': 1e9, 'dr': 0}
            Max000 = {'Name': '', 'Cost': 1e9, 'dr': 0}
            for stage, values in self.droprate[item].items():
                dr = values['droprate']
                ec = values['expected_cost']
                ef = values['effect']
                if ef > 0.999 and ec < Max999['Cost']:
                    Max999 = {'Name': stage, 'Cost': ec, 'dr': dr, 'ef': ef}
                if ef > 0.000 and ec < Max000['Cost']:
                    Max000 = {'Name': stage, 'Cost': ec, 'dr': dr, 'ef': ef}
            if Max999['Name'] != '':
                print(
                    '\n%s: \n最高效率 %s\t掉率 %.1f 期望理智 %.1f\t效率 %.4f\n最高掉落 %s\t掉率 %.1f 期望理智 %.1f\t效率 %.4f\n'\
                    % (item, Max999['Name'], Max999['dr']*100, Max999['Cost'], Max999['ef'],
                       Max000['Name'], Max000['dr']*100, Max000['Cost'], Max000['ef'])
                        )
                for stage, values in self.droprate[item].items():
                    dr = values['droprate']
                    ec = values['expected_cost']
                    ef = values['effect']
                    if ec/ef < Max999['Cost'] or (ef>0.90 and (dr>Max999['dr'] or ec<Max999['Cost'])):
                        print('%s: \t掉率 %.1f 期望理智 %.1f\t效率 %.4f\t GATE %.1f'% (stage, dr*100, ec, ef, ec/ef))
                        self.output_main_drop(stage)
#            else:
#                print(
#                    '\n%s: \n 最高掉落 %s\t掉率 %.1f 期望理智 %.1f\t效率 %.4f\n'\
#                    % (item,
#                       Max000['Name'], Max000['dr']*100, Max000['Cost'], Max000['ef'])
#                        )


    def output_credit(self):
        self.creditEffect = dict()
        self.creditEffect['技巧概要·卷2'] = self.item_value['技巧概要·卷2'] / (200/3)
        self.creditEffect['技巧概要·卷1'] = self.item_value['技巧概要·卷1'] / (160/5)
        self.creditEffect['龙门币'] = self.item_value['龙门币']*3600/200
        self.creditEffect['作战记录'] = self.item_value['基础作战记录']*9/200
        self.creditEffect['赤金'] = self.item_value['赤金']*6/160
        self.creditEffect['招聘许可'] = ((20*self.公招出四星的概率+10)*0.774+38/258*600/180*self.costLimit*self.公招出四星的概率)/160

        for item, value in Credit.items():
            self.creditEffect[item] = self.item_value[item]/value

        for item, value in sorted(self.creditEffect.items(), key=lambda x:x[1], reverse=True):
            print('%-20s:\t\t%.3f' % (item, value*100))
#            sys.stdout.write('%s>'%item)
        return

    def output_WeiJiHeYue(self):

        print('\n机密圣所(合约商店):')
        self.HeYueDict = {
                '龙门币': 85 * self.item_value['龙门币'] / 1,
                '中级作战记录': self.item_value['中级作战记录'] / 12,
                '技巧概要·卷2(刷CA3)': 20/(3 + 1.18/3) / 15 *(1-self.gold_unit*1*12),
                '技巧概要·卷2(不刷CA3)': self.item_value['技巧概要·卷2'] / 15,
#                '技巧概要·卷2(不刷CA3)': 30/(4 + 3*1.18/3 + 3*1.18*1.18/3/3)*2/3*1.18 / 15*(1-self.gold_unit*1*12),
                '芯片': (18-0.165*0.5*18/3)/(0.5 + 0.5*2/3)/60*(1-self.gold_unit*1*12)
                }
        self.HYODict = {
                '龙门币': 2000 * self.gold_unit / 15,
                '中级作战记录': self.exp_unit*5*2 / 15,
                '零件': 1/1.8,
                '皮肤': 21*self.costLimit/3000
                }
        for item, value in HeYue.items():
            self.HeYueDict[item] = self.item_value[item] / value
        for item, value in HYO.items():
            self.HYODict[item] = self.item_value[item] / value
        for k, v in sorted(self.HeYueDict.items(), key=lambda x:x[1], reverse=True):
            print('%s:\t%.3f'%(k, v))
        print('常规池')
        for k, v in sorted(self.HYODict.items(), key=lambda x:x[1], reverse=True):
            print('%s:\t%.3f'%(k, v))

    def output(self):
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
        for i, function in enumerate(Print_functions):
            if self.printSetting[i]:
                Print_functions[i]()
        return

    def output_cost(self):
        print('消耗理智 %d, 相当于碎石 %d 颗, %d 元'%\
                  (self.res['cost'], np.round(self.res['cost']/self.costLimit),
                   np.round(self.res['cost']/self.costLimit*648/180)))
        if self.costType == 'time':
            print('消耗时间 %d 秒 = %.2f天'%\
                      (self.res['cost'], self.res['cost']/86400))

    def output_stages(self):
        print('Loot at following stages:')
        for stage in self.stages:
            if float(stage['count']) > 1:
                display_lst = [k + '(%s) '%v for k, v in sorted(stage['items'].items(), key=lambda x: float(x[1]), reverse=True)]
                if stage['stage'] not in ['LS-5', 'CE-5']:
                    display_lst = display_lst[1:] + [display_lst[0]]
                print(stage['stage'] + '(%s 次) ===> '%stage['count']
                + ', '.join(display_lst))

    def output_main_drop(self, stage_name):
        stageID = self.stage_dct_rv[stage_name]
        farm_cost = self.farm_cost[stageID]
        itemPercentage = [(self.item_value[self.item_array[k]]*v/farm_cost, self.item_array[k])
                            for k,v in enumerate(self.probs_matrix[stageID])]
        display_lst = [x for x in sorted(itemPercentage, key=lambda x:x[0], reverse=True) if x[0] > 0.1]
        for value, item in display_lst:
            sys.stdout.write('%.3f\t%s\n' % (value, item))

    def output_items(self):
        print('\nSynthesize following items:')
        for synthesis in self.syntheses:
            display_lst = [k + '(%s) '%synthesis['materials'][k] for k in synthesis['materials']]
            print(synthesis['target'] + '(%s) <=== '%synthesis['count']
            + ', '.join(display_lst))

    def output_values(self):
        print('\nItems Values:')
        for i, group in reversed(list(enumerate(self.values))):
            display_lst = ['%s:%s'%(item['name'], item['value']) for item in group['items']]
            print('Level %d items: '%(i+1))
            print(', '.join(display_lst))

    def output_green(self):
        print('\n绿票商店:')
        self.greenTickets = dict()
        for item in self.values[2]['items']:
            try:
                self.greenTickets[item['name']] = {'name': item['name'],
                               'value': item['value'],
                               'efficiency': float(item['value']) / Price[item['name']]}
            except:
                pass
        for k, v in sorted(self.greenTickets.items(), key=lambda x:x[1]['efficiency'], reverse=True):
            print('%s:\t%.3f'%(k, v['efficiency']))

    def output_yellow(self):
        print('\n黄票商店:')
        self.yellowTickets = dict()
        for item in self.values[3]['items']:
            try:
                self.yellowTickets[item['name']] = {'name': item['name'],
                               'value': item['value'],
                               'efficiency': float(item['value']) / Price[item['name']]}
            except:
                pass
        for k, v in sorted(self.yellowTickets.items(), key=lambda x:x[1]['efficiency'], reverse=True):
            print('%s:\t%.3f'%(k, v['efficiency']))

    def output_effect(self):
        print('\n关卡效率:')
        for k, v in sorted(self.effect.items(), key=lambda x:x[1], reverse=True):
            print('%9s:\t%.2f'%(k, v*100))

def Cartesian_sum(arr1, arr2):
    arr_r = []
    for arr in arr1:
        arr_r.append(arr+arr2)
    arr_r = np.vstack(arr_r)
    return arr_r

def float2str(x, offset=0.5):

    if x < 1.0:
        out = '%.1f'%x
    else:
        out = '%d'%(int(x+offset))
    return out

def request_data(url_stats, url_rules, save_path_stats, save_path_rules):
    """
    To request probability and convertion rules from web resources and store at local.
    Args:
        url_stats: string. url to the dropping rate stats data.
        url_rules: string. url to the composing rules data.
        save_path_stats: string. local path for storing the stats data.
        save_path_rules: string. local path for storing the composing rules data.
    Returns:
        material_probs: dictionary. Content of the stats json file.
        convertion_rules: dictionary. Content of the rules json file.
    """
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}
    try:
        os.mkdir(os.path.dirname(save_path_stats))
    except:
        pass
    try:
        os.mkdir(os.path.dirname(save_path_rules))
    except:
        pass
    page_stats = urllib.request.Request(url_stats, headers=headers)
    with urllib.request.urlopen(page_stats) as url:
        material_probs = json.loads(url.read().decode())
        with open(save_path_stats, 'w') as outfile:
            json.dump(material_probs, outfile)

    page_rules = urllib.request.Request(url_rules, headers=headers)
    with urllib.request.urlopen(page_rules) as url:
        convertion_rules = json.loads(url.read().decode())
        with open(save_path_rules, 'w') as outfile:
            json.dump(convertion_rules, outfile)

    return material_probs, convertion_rules

def load_data(path_stats, path_rules):
    """
    To load stats and rules data from local directories.
    Args:
        path_stats: string. local path to the stats data.
        path_rules: string. local path to the composing rules data.
    Returns:
        material_probs: dictionary. Content of the stats json file.
        convertion_rules: dictionary. Content of the rules json file.
    """
    with open(path_stats) as json_file:
        material_probs  = json.load(json_file)
    with open(path_rules) as json_file:
        convertion_rules  = json.load(json_file)

    return material_probs, convertion_rules
