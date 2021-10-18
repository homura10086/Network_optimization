from data import *
from copy import deepcopy
from string import ascii_letters, digits
from random import sample, randint
from numpy.random import normal
import ran_capacity
import ran_coverage
import ran_interference
import csv
from tool import getId, getjw, getRelatedCellDuList, getTxPower


def get_gnb(i: int, gnbsize: int, label: int, mode: str):
    nrcellsize = 6
    for k in range(gnbsize):
        g = GNBFunction()
        g.GNBId = "Ran" + getId(i) + "/GNB" + getId(k)
        g.GNBName = "GNB-" + ''.join(sample(ascii_letters + digits, 3))
        # gnbs.GNBGId = s + gnbs.GNBId
        g.Longitude, g.Latitude = getjw(jws, lon, lan, 0.005, 0.006, 0.7)
        # gnbs.bwp = get_bwp(j)
        if mode == 'coverage':  # 覆盖类
            ran_coverage.get_nrcell_configuration(k, nrcellsize, label)
        elif mode == 'capacity':  # 容量类
            ran_capacity.get_nrcell_configuration(k, nrcellsize, label)
        else:  # 干扰类
            ran_interference.get_nrcell_configuration(k, nrcellsize, label)
        g.nrcells = deepcopy(nrs)
        nrs.clear()
        gnbs.append(g)


def get_celldu(i: int, j: int, celldusize: int):
    for k in range(celldusize):
        c = CellDu()
        c.CellDuId = "GNB" + getId(i) + "/DU" + getId(j) + "/CellDu" + getId(k)
        # celldus.NCGI = s + "GNB" + getId(i) + "/" + "NrCell" + getId(j)
        # celldus.CellState = getValue(CellState);
        c.S_NSSAIList = gnbs[i].nrcells[j * 3 + k].S_NSSAIList
        if 1:
            c.ArfcnDL = gnbs[i].nrcells[j * 3 + k].ArfcnDL
            c.ArfcnUL = gnbs[i].nrcells[j * 3 + k].ArfcnUL
            # celldus.ArfcnSUL = getValue(sul)
            c.BsChannelBwDL = gnbs[i].nrcells[j * 3 + k].BsChannelBwDL
            c.BsChannelBwUL = gnbs[i].nrcells[j * 3 + k].BsChannelBwUL
            # celldus.BsChannelBwSUL = getValue(bw_sul)
        # else {
        #     c.ArfcnDL = getValue(dl2)
        #     c.ArfcnUL = getValue(ul2)
        #     c.BsChannelBwDL = getValue(bw1)
        #     c.BsChannelBwUL = getValue(bw1)
        # }
        # celldus.relatedBwp = "Bwp-" + to_string(getRandom(0, 3))
        celldus.append(c)


def get_du(dusize: int):
    celldusize = 3
    for j in range(len(gnbs)):
        for k in range(dusize):
            d = DuFunction()
            d.DuId = "GNB" + getId(j) + "/DU" + getId(k)
            d.DuName = "DU-" + ''.join(sample(ascii_letters + digits, 3))
            d.Longitude, d.Latitude = getjw(jws, gnbs[j].Longitude, gnbs[j].Latitude, 0.004, 0.005, 0.5)
            # d.bwp = get_bwp(1)
            get_celldu(j, k, celldusize)
            d.celldus = deepcopy(celldus)
            celldus.clear()
            dus.append(d)


def get_cu(i: int, cusize: int):
    for k in range(cusize):
        c = CuFunction()
        c.CuId = "Ran" + getId(i) + "/CU" + getId(k)
        c.CuName = "CU-" + ''.join(sample(ascii_letters + digits, 3))
        # getPlmnList(cus.PLMNIDList, s)
        c.Longitude = gnbs[k].Longitude
        c.Latitude = gnbs[k].Latitude
        # cus.cucp = get_cucp(i, j, s)
        # cus.cuup = get_cuup(i, getRandom(1, 3))
        cus.append(c)


def get_rru_configuration(rrusize: int, mode: str):
    antennasize = 3
    for j in range(len(gnbs)):
        for k in range(rrusize):
            r = Rru()
            r.RruId = "GNB" + getId(j) + "/Rru" + getId(k)
            r.RruName = "Rru-" + ''.join(sample(ascii_letters + digits, 3))
            # rrus.VendorName = getValue(VendorName)
            # rrus.SerialNumber = "Rru-" + rrus.VendorName + "-" + getId(i)
            # rrus.VersionNumber = to_string(getRandom(1, 3)) + "." + getrandoms(1)
            # rrus.DateOfLastService = getTime()
            f = bool(randint(0, 1))
            # if (f) { rrus.FreqBand1 = getStrList(FreqBand1, getRandom(1, 47)); }
            # else { rrus.FreqBand1 = getStrList(FreqBand2, getRandom(1, 4)); }
            r.relatedCellDuList = getRelatedCellDuList(j * 2 + k)
            if mode == 'coverage':  # 覆盖类
                ran_coverage.get_antenna(j, k, f, antennasize)
            elif mode == 'capacity':  # 容量类
                ran_capacity.get_antenna(j, k, f, antennasize)
            else:  # 干扰类
                ran_interference.get_antenna(j, k, f, antennasize)
            r.antennas = deepcopy(antennas)
            antennas.clear()
            rrus.append(r)


def get_rru_performance(rrusize: int):
    for j in range(len(gnbs)):
        for k in range(rrusize):
            rrus[j * rrusize + k].MeanTxPower = getTxPower(j, k, "mean")
            rrus[j * rrusize + k].MaxTxPower = getTxPower(j, k, "max")
            rrus[j * rrusize + k].MeanPower = rrus[j * rrusize + k].MeanTxPower * normal(10, 0.1)


def Get_Data(i: int, mode: str, label: int = 0):
    dusize = 2
    gnbsize = 3
    rrusize = dusize
    cusize = gnbsize
    ran.RanSNId = "Ran" + getId(i)
    ran.RanSNName = "Ran-" + ''.join(sample(ascii_letters + digits, 3))
    get_gnb(i, gnbsize, label, mode)
    ran.gnbs = deepcopy(gnbs)
    get_du(dusize)
    ran.dus = deepcopy(dus)
    jws.clear()
    get_cu(i, cusize)
    ran.cus = deepcopy(cus)
    get_rru_configuration(rrusize, mode)
    if mode == 'coverage':  # 覆盖类
        ran_coverage.get_nrcell_performance()
    elif mode == 'capacity':  # 容量类
        ran_capacity.get_nrcell_performance()
    else:  # 干扰类
        ran_interference.get_nrcell_performance()
    get_rru_performance(rrusize)
    ran.rrus = deepcopy(rrus)
    gnbs.clear()
    dus.clear()
    cus.clear()
    rrus.clear()


def save_terminal(csvwriter, t: list):
    for x in t:
        csvwriter.writerow([x.TerminalId, x.Longitude, x.Latitude, "Terminal"])


def save_cpn(csvwriter):
    csvwriter.writerow(["Id", "CpnSNName"])
    for x in cpns:
        csvwriter.writerow([x.CpnSNId, x.CpnSNName])
    with open("cpe.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "CpeName", "MaxDistance", "SupportedType", "Longitude", "Latitude", "TransRateMean",
                            "TransRatePeak", "RSRP", "RSRQ"])
        for x in cpns:
            for y in x.cpes:
                csvwriter.writerow([y.CpeId, y.CpeName, y.MaxDistance, y.SupportedType, y.Longitude, y.Latitude,
                                    y.TransRateMean, y.TransRatePeak, y.RSRP, y.RSRQ])
    with open("terminal.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "Longitude", "Latitude", "name"])
        for x in cpns:
            for y in x.cpes:
                save_terminal(csvwriter, y.terminals)
            save_terminal(csvwriter, x.terminals)


def save_cpn_performance(csvwriter):
    csvwriter.writerow(["TransRateMean", "TransRatePeak", "RSRP", "RSRQ"])
    for x in cpns:
        for y in x.cpes:
            csvwriter.writerow([y.TransRateMean, y.TransRatePeak, y.RSRP, y.RSRQ])


def save_nrcell(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.NrCellId, x.S_NSSAIList, x.ArfcnDL, x.ArfcnUL, x.BsChannelBwDL, x.BsChannelBwUL,
                            x.ULMeanNL, x.ULMaxNL, x.UpOctUL, x.UpOctDL, x.NbrPktUL, x.NbrPktDL, x.NbrPktLossDL,
                            x.CellMeanTxPower, x.CellMaxTxPower, x.ConnMean, x.ConnMax, x.AttOutExecInterXn,
                            x.SuccOutInterXn, "NrCell"])


def save_cedu(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.CellDuId, x.S_NSSAIList, x.ArfcnDL, x.ArfcnUL, x.BsChannelBwDL, x.BsChannelBwUL])


def save_du(v: list, csvwriter):
    csvwriter.writerow(["Id", "DuName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.DuId, x.DuName, x.Longitude, x.Latitude])
    with open("cedu.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "SNSSAIList", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL", "name"])
        for x in v:
            save_cedu(x.celldus, csvwriter)
    with open("cpn.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_cpn(csvwriter)
    # p << "BwpContext" << "," << "IsInitalBwp" << "," << "SubCarrierSpacing" << "," <<
    #     "CyclicPrefix" << "," << "StartRB" << "," << "NumOfRBs" << endl;
    # for (auto x : v) {
    #     save_bwp(x.bwp, p);
    # }


def save_antenna(v: list, csvwriter):
    for x in v:
        csvwriter.writerow([x.AntennaId, x.AntennaName, x.RetTilt, x.MaxTiltValue, x.MinTiltValue, x.MaxTxPower])


def save_rru(v: list, csvwriter):
    csvwriter.writerow(["Id", "RruName", "relatedCellDuList", "MeanTxPower", "MaxTxPower", "MeanPower"])
    for x in v:
        csvwriter.writerow([x.RruId, x.RruName, x.relatedCellDuList, x.MeanTxPower, x.MaxTxPower, x.MeanPower])
    with open("antenna.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "AntennaName", "RetTilt", "MaxTilt", "MinTiltValue", "MaxTxPower"])
        for x in v:
            save_antenna(x.antennas, csvwriter)


def save_gnb(v: list, csvwriter):
    csvwriter.writerow(["Id", "GNBName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.GNBId, x.GNBName, x.Longitude, x.Latitude])
    with open("nrcell.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "S-NSSAIList", "ArfcnDL", "ArfcnUL", "BsChannelBwDL", "BsChannelBwUL", "ULMeanNL",
                            "ULMaxNL", "UpOctUL", "UpOctDL", "NbrPktUL", "NbrPktDL", "NbrPktLossDL", "CellMeanTxPower",
                           "CellMaxTxPower", "ConnMean", "ConnMax", "AttOutExecInterXn", "SuccOutInterXn", "name"])
        for x in v:
            save_nrcell(x.nrcells, csvwriter)


def save_cu(v: list, csvwriter):
    csvwriter.writerow(["Id", "CuName", "Longitude", "Latitude"])
    for x in v:
        csvwriter.writerow([x.CuId, x.CuName, x.Longitude, x.Latitude])
        # savelist(x.cus.PLMNIDList, p)


def Save_Config_and_Performance():
    with open("ran.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["Id", "RanSNName"])
        csvwriter.writerow([ran.RanSNId, ran.RanSNName])
    with open("gnb.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_gnb(ran.gnbs, csvwriter)
        # p << "BwpContext" << "," << "IsInitalBwp" << "," << "SubCarrierSpacing" << "," <<
        #     "CyclicPrefix" << "," << "StartRB" << "," << "NumOfRBs"
        # save_bwp(x.gnbs.bwp, p)
    with open("du.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_du(ran.dus, csvwriter)
    with open("cu.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_cu(ran.cus, csvwriter)
        # p << "CuCPId" << "," << "DiscardTimer" << endl
        # p << x.cus.cucp.CuCpId << ","
        # savelist(x.cus.cucp.DiscardTimer, p)
        # save_cuup(x.cus.cuup, p)
        # save_cellcu(x.cus.cucp.cellcu, p)
    with open("rru.csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_rru(ran.rrus, csvwriter)
        # save_beam(x.rru.antenna.beam, p)
        # datacsv.close()


def save_nrcell_performance(csvwriter):
    csvwriter.writerow(["ULMeanNL", "ULMaxNL", "UpOctUL", "UpOctDL", "NbrPktUL", "NbrPktDL", "NbrPktLossDL",
                        "CellMeanTxPower", "CellMaxTxPower", "ConnMean", "ConnMax", "AttOutExecInterXn",
                        "SuccOutInterXn"])
    for x in ran.gnbs:
        for y in x.nrcells:
            csvwriter.writerow([y.ULMeanNL, y.ULMaxNL, y.UpOctUL, y.UpOctDL, y.NbrPktUL, y.NbrPktDL, y.NbrPktLossDL,
                                y.CellMeanTxPower, y.CellMaxTxPower, y.ConnMean, y.ConnMax, y.AttOutExecInterXn,
                                y.SuccOutInterXn])


def save_rru_performance(csvwriter):
    csvwriter.writerow(["MeanTxPower", "MaxTxPower", "MeanPower"])
    for x in ran.rrus:
        csvwriter.writerow([x.MeanTxPower, x.MaxTxPower, x.MeanPower])


def Save_Perform(filename: str):
    with open(filename + ".csv", "a", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        save_cpn_performance(csvwriter)
        save_nrcell_performance(csvwriter)
        save_rru_performance(csvwriter)


def Save_edge(filename):
    with open(filename + ".csv", "w", newline="") as datacsv:
        csvwriter = csv.writer(datacsv, dialect="excel")
        csvwriter.writerow(["node1", "relation", "node2"])
        for gnb in ran.gnbs:
            csvwriter.writerow([ran.RanSNId, "包含", gnb.GNBId])
            for nrcell in gnb.nrcells:
                csvwriter.writerow([gnb.GNBId, "包含", nrcell.NrCellId])
        for index, du in enumerate(ran.dus):
            csvwriter.writerow([ran.RanSNId, "包含", du.DuId])
            for celldu in du.celldus:
                csvwriter.writerow([du.DuId, "包含", celldu.CellDuId])
            csvwriter.writerow([du.DuId, "接入（中传）", ran.cus[index // 2].CuId])
        for cu in ran.cus:
            csvwriter.writerow([ran.RanSNId, "包含", cu.CuId])
        for index, rru in enumerate(ran.rrus):
            csvwriter.writerow([ran.RanSNId, "包含", rru.RruId])
            for i in range(3):
                csvwriter.writerow([rru.RruId, "覆盖", ran.gnbs[index * 3 // 6].nrcells[index * 3 % 6 + i].NrCellId])
            csvwriter.writerow([rru.RruId, "接入（前传）", ran.dus[index].DuId])
            for antenna in rru.antennas:
                csvwriter.writerow([rru.RruId, "包含", antenna.AntennaId])
        for index, cpn in enumerate(cpns):
            csvwriter.writerow([ran.gnbs[index // 6].nrcells[index % 6].NrCellId, "包含", cpn.CpnSNId])
            for index1, terminal in enumerate(cpn.terminals):
                csvwriter.writerow([cpn.CpnSNId, "包含", terminal.TerminalId])
                if (index1 == 1 or index1 == 3) and index1 != len(cpn.terminals) - 1:
                    csvwriter.writerow([terminal.TerminalId, "D2D协作", cpn.terminals[index1 - 1].TerminalId])
                    csvwriter.writerow([terminal.TerminalId, "D2D协作", cpn.terminals[index1 + 1].TerminalId])
            for cpe in cpn.cpes:
                csvwriter.writerow([cpn.CpnSNId, "包含", cpe.CpeId])
                for terminal in cpe.terminals:
                    csvwriter.writerow([cpe.CpeId, "包含", terminal.TerminalId])
