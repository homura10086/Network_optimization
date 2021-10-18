from random import randint, sample
from string import ascii_letters, digits
from numpy.random import normal
from data import *
from copy import deepcopy
from tool import getId, getjw, getSupportedType, getDistance, Mean


def get_terminal(i: int, j: int, k: int, l: int, label: int, terminalsize: int, x: float, y: float):
    for e in range(terminalsize):
        t = Terminal()
        if l >= 0:  # CPE接入
            t.Longitude, t.Latitude = getjw(jws, x, y, 0.001, 0.002, 0)
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(l) + "/Terminal" + \
                           getId(e)
        else:  # 直接接入
            if label == 2:  # 切换类
                t.Longitude, t.Latitude = getjw(jws, x, y, 0.0015, 0.002, 0)
            else:
                t.Longitude, t.Latitude = getjw(jws, x, y, 0.001, 0.002, 0)
            t.TerminalId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Terminal" + getId(e)
        # t.TerminalType = getTerminalType(s)
        # t.TerminalBrand = getTerminalBrand(t.TerminalType)
        # t.Storage = getStorage(t.TerminalType)
        # t.Computing = getComputing(t.TerminalType)
        terminals.append(t)


def get_cpe(i: int, j: int, k: int, x: float, y: float, MaxTxPower: int, bw: int, label: int):
    cpe_size = randint(3, 5)
    TransRatePeak_mean, RSRP_mean, RSRQ_mean = 0, 0, 0
    for e in range(cpe_size):
        cpe = Cpe()
        cpe.CpeId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(e)
        cpe.CpeName = "Cpe-" + ''.join(sample(ascii_letters + digits, 3))
        cpe.MaxDistance = round(normal(100, 10))
        cpe.SupportedType = getSupportedType()
        if label == 2:  # 切换类
            cpe.Longitude, cpe.Latitude = getjw(jws, x, y, 0.002, 0.003, 0.1)
        else:
            cpe.Longitude, cpe.Latitude = getjw(jws, x, y, 0.001, 0.003, 0.1)
        lamuda = (getDistance(x, y, cpe.Longitude, cpe.Latitude) / 0.28) * (240.0 / MaxTxPower)
        if label == 1:  # 覆盖质量类
            cpe.RSRP = round(normal(-85, 1.5) * (1 + 0.1 * (lamuda - 1)))
            cpe.RSRQ = round(-10 * (1 + 0.4 * (lamuda - 1)))
        elif label == 2:  # 切换类
            cpe.RSRP = round(normal(-95, 1.5) * (1 + 0.1 * (lamuda - 1)))
            cpe.RSRQ = round(-15 * (1 + 0.4 * (lamuda - 1)))
        else:
            cpe.RSRP = round(normal(-95, 1.5) * (1 + 0.1 * (lamuda - 1)))
            cpe.RSRQ = round(-10 * (1 + 0.4 * (lamuda - 1)))
        cpe.TransRatePeak = round(normal(500, 20) * (1 + 0.2 * (1 - lamuda)) * (bw / 40))
        cpe.TransRateMean = round(cpe.TransRatePeak * normal(0.5, 0.03))
        TransRatePeak_mean = Mean(TransRatePeak_mean, cpe.TransRatePeak, e, cpe_size)
        RSRP_mean = Mean(RSRP_mean, cpe.RSRP, e, cpe_size)
        RSRQ_mean = Mean(RSRQ_mean, cpe.RSRQ, e, cpe_size)
        get_terminal(i, j, k, e, label, randint(2, 3), cpe.Longitude, cpe.Latitude)
        cpe.terminals = deepcopy(terminals)
        terminals.clear()
        cpes.append(cpe)
    return TransRatePeak_mean, RSRP_mean, RSRQ_mean


def get_cpn(i: int, j: int, k: int, x: float, y: float, MaxTxPower: int, bw: int, label: int):
    c = CpnSubNetwork()
    c.CpnSNId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k)
    c.CpnSNName = "Cpn-" + ''.join(sample(ascii_letters + digits, 3))
    c.TransRate_mean, c.RSRP_mean, c.RSRQ_mean = get_cpe(i, j, k, x, y, MaxTxPower, bw, label)
    c.cpes = deepcopy(cpes)
    cpes.clear()
    cpe_te_size = 0
    for t in c.cpes:
        cpe_te_size += len(t.terminals)
    if label == 1:  # 覆盖质量类
        get_terminal(i, j, k, -1, label, randint(10, 20), x, y)
    else:
        get_terminal(i, j, k, -1, label, randint(5, 10), x, y)
    c.terminals = deepcopy(terminals)
    terminals.clear()
    if len(cpns) == 18:
        cpns.clear()
        jws.clear()
    cpns.append(c)
    return cpe_te_size
