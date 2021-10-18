from random import randint, sample
from string import ascii_letters, digits
from numpy.random import normal
from data import *
from copy import deepcopy
from tool import getId, getjw, getSupportedType, getDistance, Mean


def get_terminal(i: int, j: int, k: int, l: int, terminalsize: int, x: float, y: float):
    for e in range(terminalsize):
        t = Terminal()
        t.Longitude, t.Latitude = getjw(jws, x, y, 0.001, 0.002, 0)
        if l >= 0:  # CPE接入
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(l) + "/Terminal" + \
                           getId(e)
        else:  # 直接接入
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Terminal" + getId(e)
        # t.TerminalType = getTerminalType(s)
        # t.TerminalBrand = getTerminalBrand(t.TerminalType)
        # t.Storage = getStorage(t.TerminalType)
        # t.Computing = getComputing(t.TerminalType)
        terminals.append(t)


def get_cpe_configuration(i: int, j: int, k: int, x: float, y: float):
    cpe_size = randint(3, 5)
    for e in range(cpe_size):
        cpe = Cpe()
        cpe.CpeId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(e)
        cpe.CpeName = "Cpe-" + ''.join(sample(ascii_letters + digits, 3))
        cpe.MaxDistance = round(normal(100, 10))
        cpe.SupportedType = getSupportedType()
        cpe.Longitude, cpe.Latitude = getjw(jws, x, y, 0.001, 0.003, 0.1)
        get_terminal(i, j, k, e, randint(2, 3), cpe.Longitude, cpe.Latitude)
        cpe.terminals = deepcopy(terminals)
        terminals.clear()
        cpes.append(cpe)


def get_cpe_performance(lamuda_interference: float, label: int, k: int, MaxTxPower: int):
    from ran_interference import dus
    TransRatePeak_mean, RSRP_mean, RSRQ_mean = 0, 0, 0
    for e, cpe in enumerate(cpns[k].cpes):
        cpe_size = len(cpns[k].cpes)
        # 传输速率/接收功率影响因子
        lamuda_coverage = (getDistance(dus[k//3].Longitude, dus[k//3].Latitude, cpe.Longitude, cpe.Latitude) / 0.28) \
            * (240.0 / MaxTxPower)
        cpe.TransRatePeak = round(normal(500, 10) * (1 + 0.01 * (1 - lamuda_coverage)))
        cpe.TransRateMean = round(cpe.TransRatePeak * normal(0.5, 0.03))
        cpe.RSRP = round(normal(-95, 1.5) * (1 + 0.01 * (lamuda_coverage - 1)))
        if label == 3:  # 阻塞干扰
            cpe.RSRQ = round(-18 * (1 - 3 * lamuda_interference))
        else:
            cpe.RSRQ = round(-12 * (1 - 3 * lamuda_interference))
        TransRatePeak_mean = Mean(TransRatePeak_mean, cpe.TransRatePeak, e, cpe_size)
        RSRP_mean = Mean(RSRP_mean, cpe.RSRP, e, cpe_size)
        RSRQ_mean = Mean(RSRQ_mean, cpe.RSRQ, e, cpe_size)
    return TransRatePeak_mean, RSRP_mean, RSRQ_mean


def get_cpn_performance(k: int, lamuda: float, label: int, MaxTxPower: int):
    cpns[k].TransRate_mean, cpns[k].RSRP_mean, cpns[k].RSRQ_mean = get_cpe_performance(lamuda, label, k, MaxTxPower)


def get_cpn_configuration(i: int, j: int, k: int, x: float, y: float, label: int):
    c = CpnSubNetwork()
    c.CpnSNId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k)
    c.CpnSNName = "Cpn-" + ''.join(sample(ascii_letters + digits, 3))
    get_cpe_configuration(i, j, k, x, y)
    c.cpes = deepcopy(cpes)
    cpes.clear()
    cpe_te_size = 0
    for t in c.cpes:
        cpe_te_size += len(t.terminals)
    if label == 1:  # 弱覆盖
        get_terminal(i, j, k, -1, randint(1, 5), x, y)
    elif label == 2 or label == 3:  # 越区覆盖、重叠覆盖
        get_terminal(i, j, k, -1, randint(10, 15), x, y)
    else:
        get_terminal(i, j, k, -1, randint(5, 10), x, y)
    c.terminals = deepcopy(terminals)
    terminals.clear()
    if len(cpns) == 18:
        jws.clear()
    cpns.append(c)
    return cpe_te_size
