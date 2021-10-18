import ran_interference
import ran_capacity
import ran_coverage
import multiprocessing as mp
from random import choices
from common import Get_Data, Save_Config_and_Performance, Save_edge
from tqdm import tqdm
import data
import pandas as pd
from NT import num_feature

# mode 1 coverage problem
'''
    label 0 正常
    label 1 弱覆盖
    label 2 越区覆盖
    label 3 重叠覆盖
    label 4 覆盖不均衡
'''
# mode 2 capacity problem
'''
    label 0   正常
    label 5   覆盖质量类(越区/重叠覆盖）
    label 6   切换类
    label 7   基础资源类
'''
# mode 3 interference problem
'''
    label 0   正常
    label 8   杂散干扰
    label 9   邻道干扰
    label 10   阻塞干扰
'''

num_core = 4
num_sample = int(4)


def main(batch: int, mode: str, n: int):
    with open("data" + str(batch) + ".csv", "w", newline="") as datacsv:
        if batch == 1:
            t = tqdm(range(num_sample))
        if mode == 'coverage':  # coverage problem
            for i in range(n):
                if batch == 1:
                    t.update(num_core)
                # label = int((i / n) * 5)  # for test
                label = choices([i for i in range(5)], weights=(1, 3, 3, 3, 3), k=1)[0]
                Get_Data(i, mode, label)
                ran_coverage.Save_Data(datacsv, bool(i))
        # label = int((i / n) * 4)  # for test
        elif mode == 'capacity':  # capacity problem
            for i in range(n):
                if batch == 1:
                    t.update(num_core)
                label = choices([i for i in range(4)], weights=(1, 3, 3, 3), k=1)[0]
                Get_Data(i, mode, label)
                ran_capacity.Save_Data(datacsv, bool(i))
        else:  # interference problem
            for i in range(n):
                if batch == 1:
                    t.update(num_core)
                label = choices([i for i in range(4)], weights=(1, 3, 3, 3), k=1)[-1]
                Get_Data(i, mode, label)
                ran_interference.Save_Data(datacsv, bool(i))
        # # filename1 = "Config_" + str(modes[i])
        # # Save_Config(filename1)
        # # filename2 = "Perform_" + str(modes[i])
        # # Save_Perform(filename2)


if __name__ == '__main__':
    mode = 'coverage'
    # tqdm.write(mode + ' data generating')
    # p1 = mp.Process(target=main, args=(1, mode, num_sample // num_core))
    # p2 = mp.Process(target=main, args=(2, mode, num_sample // num_core))
    # p3 = mp.Process(target=main, args=(3, mode, num_sample // num_core))
    # p4 = mp.Process(target=main, args=(4, mode, num_sample // num_core))
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    Get_Data(0, mode)
    print(data.ran)
    # Save_Config_and_Performance()
    # Save_edge("edge")
    # 合并数据
    # data_list = []
    # for i in range(num_core):
    #     data_temp = pd.read_csv('data' + str(i + 1) + '.csv', header=0, usecols=range(num_feature + 1))
    #     data_list.append(data_temp)
    # data = pd.concat(data_list, ignore_index=True)
    # data.to_csv('data_' + mode + '.csv')
