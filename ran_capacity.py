import csv
from random import choice, uniform
from cpn_capacity import get_cpn
from string import digits, ascii_letters
from random import sample, randint
from numpy.random import normal
from data import *
from tool import getSnaList, shannon, getId


def get_nrcell_configuration(i: int, nrcellsize: int, label_temp: int):
    # label = label_temp  # for test
    """Here ! ! !"""
    if label_temp != 0:
        rand_indexs = sample(range(6), randint(1, 6))
    else:
        rand_indexs = ()
    '''here'''
    for k in range(nrcellsize):
        n = NrCell()
        n.NrCellId = "GNB" + getId(i) + "/NrCell" + getId(k)
        # n.NCGI = n.NrCellId
        # n.CellState = getValue(CellState)
        n.S_NSSAIList = getSnaList(randint(1, 8))
        # n.NrTAC = to_string(getRandom(0, 65535))
        '''Here! ! !'''
        if label_temp != 0:
            if k not in rand_indexs or (i == 0 and label_temp == 2):  # 每个Ran的第一个gnb有切换类问题（label=2)时
                label = 0
            else:
                label = label_temp
        else:
            label = 0
        '''here'''
        labels.append(label)
        n.ArfcnDL = choice(dl1)
        n.ArfcnUL = choice(ul1)
        # //n.ArfcnSUL = getValue(sul)
        if label == 3:  # 基础资源类
            n.BsChannelBwDL = choice(bw1[:len(bw1) // 2])
            n.BsChannelBwUL = choice(bw1[:len(bw1) // 2])
        else:
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
            y.ConnMax = te_size * uniform(0.95, 1)
            y.ConnMean = round(y.ConnMax / normal(2, 0.05))
            label = labels[k]
            if label == 2:  # 切换类
                rand_index = randint(0, 5)
                if labels[(i - 1) * 6 + rand_index] != 2:  # 所选小区非切换类问题小区
                    labels[(i - 1) * 6 + rand_index] = 2
                    ran.gnbs[i - 1].nrcells[rand_index].AttOutExecInterXn = \
                        round(normal(30, 1) * ran.gnbs[i - 1].nrcells[rand_index].ConnMean)
                y.AttOutExecInterXn = round(normal(30, 1) * y.ConnMean)
            else:
                y.AttOutExecInterXn = round(normal(15, 1) * y.ConnMean)
            y.UpOctDL = shannon(gnbs[i].nrcells[j].BsChannelBwDL,
                                rrus[k // 3].antennas[k % 3].MaxTxPower) / (8 * 1024) * normal(0.5, 0.03) * y.ConnMax
            y.SuccOutInterXn = round(y.AttOutExecInterXn * uniform(0.9, 1))
            y.UpOctUL = y.UpOctDL / normal(8, 1)
            y.ULMeanNL = round(normal(-110, 3))
            y.ULMaxNL = round(y.ULMeanNL / normal(1.2, 0.1))
            y.NbrPktDL = round(y.UpOctDL / normal(1, 0.1))
            y.NbrPktUL = round(y.UpOctUL / normal(1, 0.1))
            y.NbrPktLossDL = round(uniform(0, 0.1) * y.NbrPktDL)
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
        if label == 1:  # 覆盖质量类
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
        cpe_te_size = get_cpn(i, j, k, dus[j].Longitude, dus[j].Latitude, a.MaxTxPower,
                              gnbs[i].nrcells[j * 3 + k].BsChannelBwDL, label)
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
             round(cpns[i].RSRQ_mean), labels[i] + 4 if labels[i] != 0 else 0]
        )
        # if i % 6 == 5:
        #     writer.writerow('')
    labels.clear()
    writer.writerow('')
