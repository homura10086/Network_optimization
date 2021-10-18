import csv
from random import choice, uniform
from cpn_interference import get_cpn_performance, get_cpn_configuration
from string import ascii_letters, digits
from random import sample, randint
from numpy.random import normal
from data import *
from tool import getSnaList, shannon, getId


def get_nrcell_configuration(i: int, nrcellsize: int, label_temp: int):
    """Here ! ! !"""
    rand_indexs = sample(range(6), randint(1, 6))
    '''here'''
    for k in range(nrcellsize):
        n = NrCell()
        n.NrCellId = "GNB" + getId(i) + "/NrCell" + getId(k)
        # n.NCGI = n.NrCellId
        # n.CellState = getValue(CellState)
        n.S_NSSAIList = getSnaList(randint(1, 8))
        # n.NrTAC = to_string(getRandom(0, 65535))
        '''Here ! ! !'''
        if label_temp != 0 and k in rand_indexs:
            label = label_temp
        else:
            label = 0
        '''here'''
        labels.append(label)
        if label == 2 and k != 0:  # 邻道干扰，第一个cell无邻道干扰(label=2)
            index = round(ul1.index_main(nrs[k - 1].ArfcnUL) + normal(0, 1))
            if index < 0:
                index = 0
            elif index > (len(ul1) - 1):
                index = len(ul1) - 1
            n.ArfcnUL = ul1[index]
        else:
            n.ArfcnUL = choice(ul1)
        n.ArfcnDL = choice(dl1)
        # //n.ArfcnSUL = getValue(sul)
        n.BsChannelBwDL = choice(bw1)
        n.BsChannelBwUL = choice(bw1)
        # //n.BsChannelBwSUL = getValue(bw_sul)
        # else {
        #     n.ArfcnDL = getValue(dl2)
        #     n.ArfcnUL = getValue(ul2)
        #     n.BsChannelBwDL = getBsChannelBw(bw2)
        #     n.BsChannelBwUL = getBsChannelBw(bw2)
        # }
        # //n.relatedBwp = "Bwp-" + to_string(getRandom(0, 3));
        nrs.append(n)


def get_nrcell_performance():
    k = 0
    for i, x in enumerate(ran.gnbs):
        for j, y in enumerate(x.nrcells):
            te_size = cpe_te_sizes[k] + len(cpns[k].terminals) + len(cpns[k].cpes)
            y.ConnMax = round(te_size * uniform(0.9, 1))
            y.ConnMean = round(y.ConnMax / normal(2, 0.1))
            y.UpOctDL = shannon(gnbs[i].nrcells[j].BsChannelBwDL,
                                rrus[k // 3].antennas[k % 3].MaxTxPower) / 8E3 * normal(0.5, 0.03) * y.ConnMean
            y.AttOutExecInterXn = round(normal(15, 1) * y.ConnMean)
            y.NbrPktDL = round(y.UpOctDL / normal(1, 0.1))
            # 干扰电平影响因子
            if j == 0:  # 每个gnb内的第一个小区
                lamuda_interference = 2e-7 * abs(gnbs[i].nrcells[j].ArfcnUL - gnbs[i].nrcells[j + 1].ArfcnUL)
            elif j == 5:  # 每个gnb内的最后一个小区
                lamuda_interference = 2e-7 * abs(gnbs[i].nrcells[j].ArfcnUL - gnbs[i].nrcells[j - 1].ArfcnUL)
            else:
                lamuda_interference = 2e-7 * (abs(gnbs[i].nrcells[j].ArfcnUL - gnbs[i].nrcells[j - 1].ArfcnUL) +
                                              abs(gnbs[i].nrcells[j].ArfcnUL - gnbs[i].nrcells[j + 1].ArfcnUL)) / 2
            label = labels[k]
            if label == 1:  # 杂散干扰
                y.ULMeanNL = round(normal(-95, 1) * (1 + lamuda_interference))
                y.SuccOutInterXn = round(y.AttOutExecInterXn * normal(0.7, 0.03))
                y.NbrPktLossDL = round(normal(0.3, 0.03) * y.NbrPktDL)
            elif label == 2:  # 邻道干扰
                y.ULMeanNL = round(normal(-90, 1) * (1 + lamuda_interference))
                y.SuccOutInterXn = round(y.AttOutExecInterXn * normal(0.5, 0.03))
                y.NbrPktLossDL = round(normal(0.5, 0.03) * y.NbrPktDL)
            elif label == 3:  # 阻塞干扰
                y.ULMeanNL = round(normal(-85, 1) * (1 + lamuda_interference))
                y.SuccOutInterXn = round(y.AttOutExecInterXn * normal(0.3, 0.03))
                y.NbrPktLossDL = round(normal(0.7, 0.03) * y.NbrPktDL)
            else:
                y.ULMeanNL = round(normal(-105, 1) * (1 + lamuda_interference))
                y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
                y.NbrPktLossDL = round(uniform(0, 0.1) * y.NbrPktDL)
            y.UpOctUL = y.UpOctDL / normal(8, 1)
            y.ULMaxNL = round(y.ULMeanNL / normal(1.2, 0.1))
            y.NbrPktUL = round(y.UpOctUL / normal(1, 0.1))
            y.CellMaxTxPower = rrus[k // 3].antennas[k % 3].MaxTxPower / normal(20, 3)
            y.CellMeanTxPower = y.CellMaxTxPower * normal(0.5, 0.03)
            get_cpn_performance(k, lamuda_interference, label, rrus[k // 3].antennas[k % 3].MaxTxPower)
            k += 1
    cpe_te_sizes.clear()


def get_antenna(i: int, j: int, f: bool, antennasize: int):
    # SupportSeq = ("410MHz-7125MHz", "24250MHz-52600MHz")
    for k in range(antennasize):
        a = Antenna()
        a.AntennaId = "GNB" + getId(i) + "/Rru" + getId(j) + "/Antenna" + getId(k)
        a.AntennaName = "Antenna-" + ''.join(sample(ascii_letters + digits, 3))
        label = labels[i * 6 + j * 3 + k]
        a.MaxTxPower = round(normal(240, 15))
        a.MaxTiltValue = randint(1800, 3600)
        a.MinTiltValue = randint(1, a.MaxTiltValue - 1)
        a.RetTilt = randint(a.MinTiltValue, a.MaxTiltValue)
        # if (f) {
        #     antennas.SupportedSeq = SupportSeq[0]
        #     antennas.ChannelInfo = getStrList(FreqBand1, getRandom(1, 47))
        # }
        # else {
        #     antennas.SupportedSeq = SupportSeq[1]
        #     antennas.ChannelInfo = getStrList(FreqBand2, getRandom(1, 4))
        # }
        # int j = getRandom(1, 64)
        # antennas.beam = get_beam(i, j)
        cpe_te_size = get_cpn_configuration(i, j, k, dus[j].Longitude, dus[j].Latitude, label)
        cpe_te_sizes.append(cpe_te_size)
        antennas.append(a)


def Save_Data(datacsv, f: bool):
    writer = csv.writer(datacsv, dialect="excel")
    if not f:
        writer.writerow(
            ["ULMeanNL", "ConnMean", "AttOutExecInterXn", "SuccOutInterXn", "ArfcnUL", "ArfcnDL", "NbrPktDL",
             "NbrPktLossDL", "UpOctUL", "UpOctDL", "BsChannelBwUL", "BsChannelBwDL", "MaxTxPower", "RetTilt",
             "TransRatePeak", "RSRP", "RSRQ", "Label"])
    temps = Temp()
    for gnb in ran.gnbs:
        for nrcell in gnb.nrcells:
            temps.nrs.append(nrcell)
    for rru in ran.rrus:
        for an in rru.antennas:
            temps.ans.append(an)
    for i in range(len(temps.nrs)):
        writer.writerow(
            [temps.nrs[i].ULMeanNL, temps.nrs[i].ConnMean, temps.nrs[i].AttOutExecInterXn, temps.nrs[i].SuccOutInterXn,
             temps.nrs[i].ArfcnUL, temps.nrs[i].ArfcnDL, temps.nrs[i].NbrPktDL, temps.nrs[i].NbrPktLossDL,
             temps.nrs[i].UpOctUL, temps.nrs[i].UpOctDL, temps.nrs[i].BsChannelBwUL, temps.nrs[i].BsChannelBwDL,
             temps.ans[i].MaxTxPower, temps.ans[i].RetTilt, round(cpns[i].TransRate_mean), round(cpns[i].RSRP_mean),
             round(cpns[i].RSRQ_mean), labels[i] + 7 if labels[i] != 0 else 0]
        )
        # if i % 6 == 5:
        #     writer.writerow('')
    labels.clear()
    cpns.clear()
    writer.writerow('')
