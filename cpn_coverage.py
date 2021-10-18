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


def get_cpe(i: int, j: int, k: int, x: float, y: float, MaxTxPower: int, label: int):
    cpe_size = randint(3, 5)
    TransRatePeak_mean, RSRP_mean, RSRQ_mean = 0, 0, 0
    for e in range(cpe_size):
        cpe = Cpe()
        cpe.CpeId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k) + "/Cpe" + getId(e)
        cpe.CpeName = "Cpe-" + ''.join(sample(ascii_letters + digits, 3))
        cpe.MaxDistance = round(normal(100, 10))
        cpe.SupportedType = getSupportedType()
        cpe.Longitude, cpe.Latitude = getjw(jws, x, y, 0.001, 0.003, 0.1)
        lamuda = (getDistance(x, y, cpe.Longitude, cpe.Latitude) / 0.28) * (240.0 / MaxTxPower)
        cpe.TransRatePeak = round(normal(500, 10) * (1 + 0.2 * (1 - lamuda)))
        cpe.TransRateMean = round(cpe.TransRatePeak * normal(0.5, 0.03))
        if label == 0 or label == 4:  # 正常/覆盖不均衡
            cpe.RSRP = round(normal(-95, 1.5) * (1 + 0.1 * (lamuda - 1)))
        elif label == 1:    # 弱覆盖
            cpe.RSRP = round(normal(-105, 1.5) * (1 + 0.1 * (lamuda - 1)))
        else:   # 越区/重叠覆盖
            cpe.RSRP = round(normal(-85, 1.5) * (1 + 0.1 * (lamuda - 1)))
        cpe.RSRQ = round(-11.25 * (1 + 0.4 * (lamuda - 1)))
        TransRatePeak_mean = Mean(TransRatePeak_mean, cpe.TransRatePeak, e, cpe_size)
        RSRP_mean = Mean(RSRP_mean, cpe.RSRP, e, cpe_size)
        RSRQ_mean = Mean(RSRQ_mean, cpe.RSRQ, e, cpe_size)
        get_terminal(i, j, k, e, randint(2, 3), cpe.Longitude, cpe.Latitude)
        cpe.terminals = deepcopy(terminals)
        terminals.clear()
        cpes.append(cpe)
    return TransRatePeak_mean, RSRP_mean, RSRQ_mean


def get_cpn(i: int, j: int, k: int, x: float, y: float, cpe_te_size: int, MaxTxPower: int, label: int):
    c = CpnSubNetwork()
    c.CpnSNId = "GNB" + getId(i) + "/DU" + getId(j) + "/Cpn" + getId(k)
    c.CpnSNName = "Cpn-" + ''.join(sample(ascii_letters + digits, 3))
    c.TransRate_mean, c.RSRP_mean, c.RSRQ_mean = get_cpe(i, j, k, x, y, MaxTxPower, label)
    c.cpes = deepcopy(cpes)
    cpes.clear()
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
        cpns.clear()
        jws.clear()
    cpns.append(c)
    return cpe_te_size
