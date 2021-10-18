import csv
from random import choice, uniform
from cpn_coverage import get_cpn
from string import ascii_letters, digits
from data import *
from random import sample, randint
from numpy.random import normal
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
        if label_temp == 0 or (k in rand_indexs and (k != 0 or label_temp != 3)):  # 第一个cell无重叠覆盖（label=3)
            label = label_temp
        else:
            label = 0
        '''here'''
        labels.append(label)
        if label == 3:  # 重叠覆盖
            index = round(dl1.index_main(nrs[k - 1].ArfcnDL) + normal(0, 1))
            if index < 0:
                index = 0
            elif index > (len(dl1) - 1):
                index = len(dl1) - 1
            n.ArfcnDL = dl1[index]
            n.ArfcnUL = choice(ul1)
        elif label == 4:  # 覆盖不均衡
            n.ArfcnDL = choice(dl1)
            n.ArfcnUL = ul1[dl1.index_main(n.ArfcnDL)]
        else:
            n.ArfcnDL = choice(dl1)
            n.ArfcnUL = choice(ul1)
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
            if j == 0:  # 每个gnb内的第一个小区
                lamuda1 = 5e-7 * abs(gnbs[i].nrcells[j].ArfcnDL - gnbs[i].nrcells[j + 1].ArfcnDL)
            elif j == 5:  # 每个gnb内的最后一个小区
                lamuda1 = 5e-7 * abs(gnbs[i].nrcells[j].ArfcnDL - gnbs[i].nrcells[j - 1].ArfcnDL)
            else:
                lamuda1 = 5e-7 * (abs(gnbs[i].nrcells[j].ArfcnDL - gnbs[i].nrcells[j - 1].ArfcnDL) +
                                  abs(gnbs[i].nrcells[j].ArfcnDL - gnbs[i].nrcells[j + 1].ArfcnDL)) / 2
            label = labels[k]
            if label == 1:  # 弱覆盖
                y.ULMeanNL = round(normal(-95, 3) * (1 + lamuda1))
                y.UpOctUL = y.UpOctDL / normal(8, 1)
                y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
                y.NbrPktLossDL = round(uniform(0.1, 0.9) * y.NbrPktDL)
            if label == 3:  # 重叠覆盖
                y.ULMeanNL = round(normal(-95, 3) * (1 + lamuda1))
                y.UpOctUL = y.UpOctDL / normal(8, 1)
                y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
                y.NbrPktLossDL = round(uniform(0.05, 0.1) * y.NbrPktDL)
            elif label == 4:  # 覆盖不均衡
                lamuda2 = 5e-7 * abs(gnbs[i].nrcells[j].ArfcnUL - gnbs[i].nrcells[j].ArfcnDL)
                y.ULMeanNL = round(normal(-95, 3) * (1 + lamuda2))
                y.UpOctUL = y.UpOctDL / normal(14, 1)
                y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.1, 0.9))
                y.NbrPktLossDL = round(uniform(0.05, 0.1) * y.NbrPktDL)
            else:
                y.ULMeanNL = round(-105 * (1 + lamuda1))
                y.UpOctUL = y.UpOctDL / normal(8, 1)
                y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
                y.NbrPktLossDL = round(uniform(0.05, 0.1) * y.NbrPktDL)
            y.ULMaxNL = round(y.ULMeanNL / normal(1.2, 0.1))
            y.NbrPktUL = round(y.UpOctUL / normal(1, 0.1))
            y.CellMaxTxPower = rrus[k // 3].antennas[k % 3].MaxTxPower / normal(20, 3)
            y.CellMeanTxPower = y.CellMaxTxPower * normal(0.5, 0.03)
            k += 1
    cpe_te_sizes.clear()


def get_antenna(i: int, j: int, f: bool, antennasize: int):
    # SupportSeq = ("410MHz-7125MHz", "24250MHz-52600MHz")
    for k in range(antennasize):
        a = Antenna()
        a.AntennaId = "GNB" + getId(i) + "/Rru" + getId(j) + "/Antenna" + getId(k)
        a.AntennaName = "Antenna-" + ''.join(sample(ascii_letters + digits, 3))
        label = labels[i * 6 + j * 3 + k]
        if label == 1:  # 弱覆盖
            a.MaxTxPower = round(normal(150, 15))
            a.MaxTiltValue = randint(1800, 3600)
        elif label == 2:  # 越区覆盖
            a.MaxTxPower = round(normal(330, 15))
            a.MaxTiltValue = randint(100, 1800)
        elif label == 3:  # 重叠覆盖
            a.MaxTxPower = round(normal(330, 15))
            a.MaxTiltValue = randint(100, 1800)
        else:
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
        cpe_te_size = 0
        cpe_te_size = get_cpn(i, j, k, dus[j].Longitude, dus[j].Latitude, cpe_te_size, a.MaxTxPower, label)
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
        for antenna in rru.antennas:
            temps.ans.append(antenna)
    for i in range(len(temps.nrs)):
        writer.writerow(
            [temps.nrs[i].ULMeanNL, temps.nrs[i].ConnMean, temps.nrs[i].AttOutExecInterXn, temps.nrs[i].SuccOutInterXn,
             temps.nrs[i].ArfcnUL, temps.nrs[i].ArfcnDL, temps.nrs[i].NbrPktDL, temps.nrs[i].NbrPktLossDL,
             temps.nrs[i].UpOctUL, temps.nrs[i].UpOctDL, temps.nrs[i].BsChannelBwUL, temps.nrs[i].BsChannelBwDL,
             temps.ans[i].MaxTxPower, temps.ans[i].RetTilt, round(cpns[i].TransRate_mean), round(cpns[i].RSRP_mean),
             round(cpns[i].RSRQ_mean), labels[i]])
        # if i % 6 == 5:
        #     writer.writerow('')
    labels.clear()
    writer.writerow('')
