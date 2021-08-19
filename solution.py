import time
import random
from collections import defaultdict
import pandas as pd
import itertools

class Trainer:
    def __init__(self, Env, conf_list):
        self.conf_list = conf_list
        self.Env = Env
        self.checkpoint = None
        self.iter = 0

        # 学習時に壊れるマシンタイプと時刻先読み
        self.break_machine_type = None
        self.break_min_time = None

        for i, conf_i in enumerate(self.conf_list):
            env = self.Env(conf_i)
            print("="*20)
            print("設定：", conf_i)
            print("ジョブ：", env.jobs)
            print("ジョブタイプ：", env.job_types)
            print("マシン：", env.machines)
            print("ptv: ", env.ptv_weight)
            print("breakdown: ", env.breakdowns)
            print("="*20)

            # マシン名からそのマシンタイプを逆引きできるようにする
            machine_list = {}
            for machine_type in env.machines:
                for machine_id in env.machines[machine_type]:
                    machine_list[machine_id] = machine_type
        
            if len(env.breakdowns) != 0:
                for machine in env.breakdowns[0]:
                    # 壊れるマシンのタイプ
                    machine_type = machine_list[machine]
                    # 壊れる時刻
                    breakdown_time = env.breakdowns[0][machine]

                    # 初回のconfigファイルなら
                    if(i == 0):
                        self.break_machine_type = machine_type
                        self.break_min_time = breakdown_time
                    # 他のconfigと壊れるマシンタイプ、時間を比較
                    else:
                        # もしも壊れるマシンタイプが異なっていたら信用できないのでNoneとして処理しないようにする
                        if self.break_machine_type != machine_type:
                            self.break_machine_type = None
                            self.break_min_time = None
                            break
                        self.break_min_time = min(self.break_min_time, breakdown_time)

        # このデータセットのbreakdownするマシンタイプと時間を出力する
        print("break_machine_type: ", self.break_machine_type)
        print("break_min_time: ", self.break_min_time)

    def shift_breakdown(self, env, shift_time):

        # breakdownが設定されていなかったときの例外処理
        if len(env.config['breakdown']) == 0:
            return env

        # マシンが壊れるタイミングを±5秒乱数でずらすようにする
        for key, value in env.config['breakdown'][0].items():
            env.config['breakdown'][0][key] += random.randint(-5, 5)
            
            # 壊れる時間が0以下にならない用に例外処理
            env.config['breakdown'][0][key] = max(env.config['breakdown'][0][key], 0)
        
        return env

    # 学習データに対していろいろな戦略を試してみて安定性の高い手法を採用する
    def train(self, run_time):

        start_time = time.time()

        # 試してみる戦略のリスト
        strategy1_list = ['random', 'qtfirst', 'permutation']
        strategy2_list = ['ptvCalc']

        # スコアを記録する辞書
        score_dic = {}
        PTV_dic= {}
        makespan_dic = {}

        timeout_flag = False
        # breakdownをずらす
        for shift_time in [0, -5, 5]:
            if timeout_flag:
                break
            # 全ての学習条件で実行
            for i, conf_i in enumerate(self.conf_list):
                env = self.Env(conf_i)
                print("設定:", conf_i)
                print("シフトタイム", shift_time)

                # breakdownタイムのシフト
                env = self.shift_breakdown(env, shift_time)

                # 戦略ごとのスコア記録
                score_strategy = {}
                PTV_strategy = {}
                makespan_strategy = {}


                for strategy1 in strategy1_list:
                    for strategy2 in strategy2_list:
                        # 操作優先順を24通り試す場合
                        if strategy1 == 'permutation':
                            jobtype_list = list(env.job_types.keys())
                            jobtype_num = len(env.job_types.keys())
                            for v in itertools.permutations(jobtype_list, jobtype_num):
                                permutation = ""
                                for j in range(jobtype_num):
                                    permutation += v[j]
                                
                                # print("戦略", strategy)
                                # 環境の初期化
                                machine_status, job_status, step_time, job_list = env.reset()
                                done = False # 初期化

                                # 戦略に基づいたエージェント作成
                                agent = Agent(env.job_types, strategy1, strategy2, self.break_machine_type, self.break_min_time, permutation)

                                # 記録するスコアの初期化
                                total_makespan = 0
                                total_PTV = 0
                                total_score = 0

                                while(not done):
                                    # 行動を決定
                                    action = agent.act(machine_status, job_status, step_time, job_list)
                                    # 環境へ行動を渡して状態更新
                                    machine_status, job_status, step_time, reward, job_list, done = env.step(action)

                                    total_makespan += reward['makespan']
                                    total_PTV += reward['PTV']
                                    total_score += reward['makespan'] + reward['PTV']
                                
                                # print(total_makespan, total_PTV, total_score)
                                # この戦略での結果を記録
                                score_strategy[strategy1 + "_" + strategy2 + "_" + permutation] = total_score
                                PTV_strategy[strategy1 + "_" + strategy2 + "_" + permutation] = total_PTV
                                makespan_strategy[strategy1 + "_" + strategy2 + "_" + permutation] = total_makespan
    
                        # 戦略を試す場合
                        else:
                            # 環境の初期化
                            machine_status, job_status, step_time, job_list = env.reset()
                            done = False # 初期化

                            # 戦略に基づいたエージェント作成
                            agent = Agent(env.job_types, strategy1, strategy2, self.break_machine_type, self.break_min_time)

                            # 記録するスコアの初期化
                            total_makespan = 0
                            total_PTV = 0
                            total_score = 0

                            while(not done):
                                # 行動を決定
                                action = agent.act(machine_status, job_status, step_time, job_list)
                                # 環境へ行動を渡して状態更新
                                machine_status, job_status, step_time, reward, job_list, done = env.step(action)

                                total_makespan += reward['makespan']
                                total_PTV += reward['PTV']
                                total_score += reward['makespan'] + reward['PTV']
                            
                            # print(total_makespan, total_PTV, total_score)
                            # この戦略での結果を記録
                            score_strategy[strategy1 + "_" + strategy2] = total_score
                            PTV_strategy[strategy1 + "_" + strategy2] = total_PTV
                            makespan_strategy[strategy1 + "_" + strategy2] = total_makespan

                # このconfでの結果を記録
                score_dic[str(i) + "_shift_" + str(shift_time)] = score_strategy
                PTV_dic[str(i) + "_shift_" + str(shift_time)] = PTV_strategy
                makespan_dic[str(i) + "_shift_" + str(shift_time)] = makespan_strategy

                # もしも学習時間の8割以上使うようなことあれば強制的にループを抜けてタイムアウト回避(たぶん大丈夫)
                now_time = time.time()
                if now_time - start_time > run_time * 0.8:
                    timeout_flag = True
                    break

        # DataFrameで結果を表示
        score_df = pd.DataFrame(score_dic)
        PTV_df = pd.DataFrame(PTV_dic)
        makespan_df = pd.DataFrame(makespan_dic)

        # 平均値を取得
        score_df['mean'] = score_df.mean(axis=1)
        PTV_df['mean'] = PTV_df.mean(axis=1)
        makespan_df['mean'] = makespan_df.mean(axis=1)

        # 学習時の結果の出力
        print("score")
        print(score_df)
        print("")

        print("PTV")
        print(PTV_df)
        print("")

        print("makespan")
        print(makespan_df)
        print("")

        # スコアの平均が最小となった戦略を採用する
        best_strategy = score_df.index[score_df['mean'].argmax(axis=0)]

        best_strategy1 = best_strategy.split('_')[0]
        best_strategy2 = best_strategy.split('_')[1]

        print("best_strategy1:", best_strategy1)
        print("best_strategy2:", best_strategy2)

        if best_strategy1 == 'permutation':
            best_permutation = best_strategy.split('_')[2]
            print("best_permutation:", best_permutation)
            return Agent(env.job_types, best_strategy1, best_strategy2, self.break_machine_type, self.break_min_time, best_permutation)
        else:
            return Agent(env.job_types, best_strategy1, best_strategy2, self.break_machine_type, self.break_min_time)

class Agent:
    def __init__(self, job_types, strategy1, strategy2, break_machine_type=None, break_min_time=None, permutation=None):
        self.job_types = job_types
        # どういう戦略でジョブを詰め込むか
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        
        # train時のbreakdown情報
        self.break_machine_type = break_machine_type
        self.break_min_time = break_min_time

        # permutation戦略の場合
        if permutation is not None:
            self.permutation = {}
            for i, c in enumerate(permutation):
                self.permutation[c] = i

        # 操作ごとの実行時間
        self.op_process_time = {}
        self.job_total_time = {}
        self.op_remain_time = {}
        self.op_max_pend_time = 0
        for job_type in job_types:
            self.job_total_time[job_type] = 0
            for i in job_types[job_type]:
                op_name = i['op_name']
                self.op_process_time[op_name] = i['process_time']
                self.op_max_pend_time = max(self.op_max_pend_time, i['max_pend_time'])
                self.job_total_time[job_type] += self.op_process_time[op_name]
        
        for job_type in job_types:
            temp_time = self.job_total_time[job_type]
            for i in job_types[job_type]:
                op_name = i['op_name']
                self.op_remain_time[op_name] = temp_time
                temp_time -= i['process_time']

    
    def act(self, machine_status, job_status, time, job_list):
        action = {}
        for machine in job_list:
            job = self.adv_wwsqt(machine, machine_status, job_status, time, job_list[machine])
            if job is not None:
                for mm in job_list:
                    try:
                        job_list[mm].remove(a)
                        # job_list[mm].remove(job)
                    except:
                        pass
                    finally:
                        pass
                action[machine] = job
        return action

    def adv_wwsqt(self, machine, machine_status, job_status, time, job_list):

        def get_next_op_info(job):
            if job_status[job]['status'] == 'to_arrive':
                job_type = job_status[job]['type']
                next_op = self.job_types[job_type][0]
                return {'machine':'A', 'next_max_pending_time':next_op['max_pend_time'], \
                    'arrival_time':job_status[job]['arrival'], 'process_time':next_op['process_time'], 'priority':job_status[job]['priority']}
            else:
                job_type = job_status[job]['type']
                now_op = job_status[job]['op']
                for op_idx, op in enumerate(self.job_types[job_type]):
                    if op['op_name'] == now_op:
                        break
                next_op = self.job_types[job_type][op_idx+1] if op_idx < len(self.job_types[job_type]) - 1 else None
                next_op_info = {'machine':next_op['machine_type'], 'next_max_pending_time':next_op['max_pend_time'], \
                    'arrival_time':job_status[job]['remain_process_time'], 'process_time':next_op['process_time'], \
                        'priority':job_status[job]['priority']} if next_op is not None \
                    else {'machine':None, 'next_max_pending_time':None, 'arrival_time':None, 'process_time':None, 'priority':None}
                return next_op_info

        def get_arrive_priority_feature():
            ft = []
            
            min_arrive_time = 3500
            for job in job_status:
                next_op_info = get_next_op_info(job)

                if next_op_info['machine'] == machine_status[machine]['type'] and \
                    job_status[job]['priority'] > 0:
                        if job_status[job]['status'] == 'work':
                            min_arrive_time = min(min_arrive_time, job_status[job]['remain_process_time']+next_op_info['next_max_pending_time'])
                        elif job_status[job]['status'] == 'to_arrive':
                            min_arrive_time = min(min_arrive_time, job_status[job]['arrival']+next_op_info['next_max_pending_time'])

            ft += [min_arrive_time]

            return ft
        
        def get_next_priority_job():            
            future_priority_jobs = []
            for job in job_status:
                next_op_info = get_next_op_info(job)

                if next_op_info['machine'] == machine_status[machine]['type'] and job_status[job]['priority'] > 0:
                    future_priority_jobs.append(next_op_info)
            
            if len(future_priority_jobs) > 0:
                return sorted(future_priority_jobs, key=lambda x: x['arrival_time'])[0]
            else:
                return None

        if len(job_list) == 0:
            return None
        else:
            # 優先度付きジョブのリスト作成
            sorted_list = [a for a in job_list if (job_status[a]['priority']>0)]

            # 優先度付きジョブがないときの処理
            if len(sorted_list) == 0:

                # 既に同タイプのマシンが優先度ジョブを処理中ならばbreakdownに備えて待機しておく
                target_machine_type = machine_status[machine]['type']

                # 同タイプのマシンが処理中の優先度付きジョブの数
                priority_job = 0
                # そのマシンの余裕数
                machine_count = 0
                # 既にbreakdownのマシンがあるか
                is_breakdown = False

                for machine_i in machine_status:
                    # 別のマシンタイプは無視
                    if(machine_status[machine_i]['type'] != target_machine_type):
                        continue

                    # 待機中のマシンの数を数える
                    if(machine_status[machine_i]['status'] == 'idle'):
                        machine_count += 1
                    
                    # breakdonwのマシンの数を数える
                    if(machine_status[machine_i]['status'] == 'down'):
                        is_breakdown = True

                    # 動いていないマシンは無視する
                    if(machine_status[machine_i]['status'] != 'work'):
                        continue
                    
                    # 同じマシンタイプで現在実行中のジョブ
                    job = machine_status[machine_i]['job']
                    
                    # 優先度付きジョブを処理中ならば念のため待機する
                    if(job_status[job]['priority'] > 0):
                        priority_job += 1
                
                # 優先度付きジョブが来るまでの時間
                arrive_priority = get_arrive_priority_feature()[0]

                # たくさん余裕があるときはなるべく処理時間が長いジョブにする前処理
                job_margin = {}
                relax_time_min = 99999
                for job in job_list:
                    op = job_status[job]['op']

                    relax_time = (arrive_priority - self.op_process_time[op])
                    relax_time_min = min(relax_time_min, relax_time)
                    job_margin[job] = abs(relax_time - self.op_process_time[op])

                
                ################################### 
                # 優先度付き以外のジョブを埋める戦略
                ###################################

                # できるジョブの中からランダムに選択する
                if self.strategy1 == 'random':
                    a = random.sample(job_list, len(job_list))

                #  優先度ジョブがないor離れているときは処理時間が長いもの、近づいてくると短いものを優先処理
                elif self.strategy1 == 'margin':
                    a = sorted(job_list, key=lambda x: (job_margin[x], job_status[x]['remain_pending_time']))

                # ジョブが短いものを優先
                elif self.strategy1 == 'qtfirst':
                    a = sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))

                # 処理時間の残りが長いものを優先
                elif self.strategy1 == 'totalfirst':
                    a = sorted(job_list, key=lambda x: (-self.op_remain_time[job_status[x]['op']], job_status[x]['remain_pending_time']))

                # SDT (現在のオペレーションにかかる時間 / すべてのオペレーションにかかる時間 が最小のものを選択)
                elif self.strategy1 == 'SDT':
                    a = sorted(job_list, key=lambda x: (self.op_process_time[job_status[x]['op']] / self.job_total_time[job_status[x]['type']]))
                
                # permutation戦略
                elif self.strategy1 == 'permutation':
                    a = sorted(job_list, key=lambda x: (self.permutation[job_status[x]['type']],job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))

                # 例外処理(戦略は必ず選ばれいているが念の為)
                else:
                    a = sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))

                for job in a:
                    op = job_status[job]['op']

                    # もしも優先度付きジョブを処理中で、
                    # マシンの余裕が1台しかなく、
                    # まだマシンの故障が起こっていなく、
                    # trainで壊れたマシンタイプと同じマシンタイプで、
                    # trainで壊れたマシンの最小時刻までに処理を追えられないならば、
                    # breakdownに備えて処理を保留する
                    if(priority_job > 0 and \
                            machine_count == 1 and \
                                is_breakdown == False and \
                                    target_machine_type == self.break_machine_type and \
                                        self.op_process_time[op] > self.break_min_time - time):
                        continue

                    # 次の優先度付きジョブが来る時間のほうが処理時間よりも大きければ処理可能
                    if(get_arrive_priority_feature()[0] > self.op_process_time[op]):
                        return job
                return None

            ################################### 
            # 優先度のジョブを埋める戦略
            ###################################
            else:
                # デフォルト、残りペンディング時間を優先度で割ったものが小さい順に実行
                if self.strategy2 == "normal":
                    return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],job_status[x]['remain_process_time']))[0]
                # 残りpending時間が短いものから優先的に実行
                elif self.strategy2 == "pending":
                    return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time'], job_status[x]['remain_process_time']))[0]

                # 残りペンディング時間が大きいものは実行しない
                elif self.strategy2 == "stopping":
                    a = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time'], job_status[x]['remain_process_time']))

                    pendtime = self.op_max_pend_time/2 # 最大滞留時間の半分

                    if(job_status[a[0]]['remain_pending_time'] < pendtime):
                        return a[0]
                    else:
                        # qtfirst
                        a = sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))
                        for job in a:
                            op = job_status[job]['op']

                            # 次の優先度付きジョブが来る時間のほうが処理時間よりも大きければ処理可能
                            if(get_arrive_priority_feature()[0] > self.op_process_time[op]):
                                return job
                        return None

                # normalのptvを起こしたときは優先度が高いものを先に処理する修正版
                elif self.strategy2 == "normal2":
                    a = sorted(sorted_list, key=lambda x: (
                        job_status[x]['remain_pending_time']/job_status[x]['priority']
                        if job_status[x]['remain_pending_time'] > 0
                        else job_status[x]['remain_pending_time'] * job_status[x]['priority'],
                        job_status[x]['remain_process_time']))[0]

                    return a

                # ptvによるダメージを事前計算し、処理するジョブを判断する
                elif self.strategy2 == "ptvCalc":
                    a = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],\
                        job_status[x]['remain_process_time']))[0]
                    
                    
                    # 自身と同じマシンタイプのマシンで、idle状態のマシン数（自身含む）を確認
                    idle_machine_num = 0
                    mt = machine_status[machine]["type"]
                    for m in machine_status:
                        if machine_status[m]["type"] == mt and machine_status[m]["status"] == "idle":
                            idle_machine_num += 1
                    
                    # 自身と同じマシンタイプのマシンで、他に手が空いてるマシンがいる場合はシンプルに優先度ジョブをアサイン
                    if idle_machine_num > 1:
                        return a
                    # 手が空いているマシンがいない場合は、直近の未来で到着する優先度ジョブを処理するために、待機するかを判断する
                    else:
                        next_priority_job = get_next_priority_job()
                        if next_priority_job is None:
                            return a
                        else:
                            # 手持ちの優先度ジョブに着手したとき、直近未来の優先度ジョブで発生する滞留違反ペナルティ
                            ptv_1 = max(0, job_status[a]['remain_process_time'] - next_priority_job["arrival_time"] - next_priority_job["next_max_pending_time"])
                            ptv_1 *= next_priority_job["priority"]

                            # 直近未来の優先度ジョブが来るまで待機したとき、手持ちの優先度ジョブで発生する滞留違反ペナルティ
                            ptv_2 = max(0, next_priority_job["arrival_time"] + next_priority_job["process_time"] - job_status[a]["remain_pending_time"])
                            ptv_2 *= job_status[a]["priority"]

                            # 直近未来の優先度ジョブを待つほうが滞留違反ペナルティが小さければ、待機
                            if ptv_1 > ptv_2:
                                return None
                            else:
                                return a

                # ptvによるダメージを事前計算し、処理するジョブを判断するのと、滞留時間がたくさんあるなら待機するの複合
                elif self.strategy2 == "stopptvCalc":
                    a = sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],\
                        job_status[x]['remain_process_time']))
                    
                    pendtime = self.op_max_pend_time/2 # 最大滞留時間の半分

                    if(job_status[a[0]]['remain_pending_time'] < pendtime):
                        # ptv_calcを行う
                        # 自身と同じマシンタイプのマシンで、idle状態のマシン数（自身含む）を確認
                        idle_machine_num = 0
                        mt = machine_status[machine]["type"]
                        for m in machine_status:
                            if machine_status[m]["type"] == mt and machine_status[m]["status"] == "idle":
                                idle_machine_num += 1
                        
                        # 自身と同じマシンタイプのマシンで、他に手が空いてるマシンがいる場合はシンプルに優先度ジョブをアサイン
                        if idle_machine_num > 1:
                            return a[0]
                        # 手が空いているマシンがいない場合は、直近の未来で到着する優先度ジョブを処理するために、待機するかを判断する
                        else:
                            next_priority_job = get_next_priority_job()
                            if next_priority_job is None:
                                return a[0]
                            else:
                                # 手持ちの優先度ジョブに着手したとき、直近未来の優先度ジョブで発生する滞留違反ペナルティ
                                ptv_1 = max(0, job_status[a[0]]['remain_process_time'] - next_priority_job["arrival_time"] - next_priority_job["next_max_pending_time"])
                                ptv_1 *= next_priority_job["priority"]

                                # 直近未来の優先度ジョブが来るまで待機したとき、手持ちの優先度ジョブで発生する滞留違反ペナルティ
                                ptv_2 = max(0, next_priority_job["arrival_time"] + next_priority_job["process_time"] - job_status[a[0]]["remain_pending_time"])
                                ptv_2 *= job_status[a[0]]["priority"]

                                # 直近未来の優先度ジョブを待つほうが滞留違反ペナルティが小さければ、待機
                                if ptv_1 > ptv_2:
                                    return None
                                else:
                                    return a[0]
                    else:
                        # qtfirst
                        a = sorted(job_list, key=lambda x: (job_status[x]['remain_process_time'], job_status[x]['remain_pending_time']))
                        for job in a:
                            op = job_status[job]['op']

                            # 次の優先度付きジョブが来る時間のほうが処理時間よりも大きければ処理可能
                            if(get_arrive_priority_feature()[0] > self.op_process_time[op]):
                                return job
                        return None
                else:
                    # 念のための例外処理(normarlと同じ)
                    return sorted(sorted_list, key=lambda x: (job_status[x]['remain_pending_time']/job_status[x]['priority'],job_status[x]['remain_process_time']))[0]




        









