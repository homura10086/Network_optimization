from tqdm import trange
from data_new import *

# for data generating
'''
mode cover(coverage problem)
    label 0 正常
    label 1 弱覆盖
    label 2 越区覆盖
    label 3 重叠覆盖
    label 4 覆盖不均衡
mode cap(capacity problem)
    label 0   正常
    label 5   覆盖质量类(越区/重叠覆盖）
    label 6   切换类
    label 7   基础资源类
mode inter(interference problem)
    label 0    正常
    label 8    杂散干扰
    label 9    邻道干扰
    label 10   阻塞干扰
'''

ran_num = 1
loops = 1
mode = "cover"
train = "predict_"

print("data generating")
for i in trange(int(loops)):
    label = randint(0, 4) if mode == "cover" else randint(0, 3)
    # label = 2   # for test
    ran = RanSubNetwork(ran_num).get_Ran(mode, label)
    ran.save_Ran()
    # save_Data(i == 0, train)
    save_Relation(ran)
