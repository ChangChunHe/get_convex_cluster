#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:20:13 2017
@author: hecc
"""
import convex_structure_class as csc
import numpy as np
import time


def initial(nn):
    n = int(np.floor((2 * nn + 1) ** 2 * 0.7))
    struc = []
    for ii in np.arange(-nn, nn + 1):
        for jj in np.arange(-nn, nn + 1):
            struc.append([ii, jj, ii ** 2 + jj ** 2 + ii * jj])
    struc = np.array(struc)
    q = struc[:, -1].argsort()
    struc = struc[q, :]
    dis = np.unique(struc[:, -1])

    newstr = np.reshape(struc[0, :], (1, 3))
    for k in range(1, len(dis)):
        q = np.where(struc[:, -1] == dis[k])
        q = np.array(q)
        len_q = np.size(q, axis=1)
        theta = np.zeros((len_q, 1))
        for kk in range(0, len_q):
            x, y = struc[q[:, kk], 0] + struc[q[:, kk], 1] / 2, struc[q[:, kk], 1] * np.sqrt(3) / 2
            theta[kk] = angle(x, y)
        pp = np.argsort(theta, axis=0)
        new_q = q[:, pp]
        new_q = np.reshape(new_q, (len_q,))
        # import pdb;pdb.set_trace()
        strnew = struc[new_q, :]
        newstr = np.vstack((newstr, strnew))
    np.savetxt('struc.txt', newstr, fmt="%d")
    struc = newstr.copy()

    # generate neighbouring matrix
    nearest = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]])
    atomlist = np.zeros((n, 6), dtype='int')
    for k in range(0, n):
        atom_neig = nearest + struc[k, 0:2]
        qq = ismember_rows(struc[:, 0:2], atom_neig)
        atomlist[k, 0:length(qq)] = qq
    np.savetxt('neig.txt', atomlist, fmt="%d")
    # geenrate symmetric matrix
    aa = np.loadtxt('A.txt', dtype='int')
    strmap = np.zeros((n, 12), dtype='int')
    for k in range(0, n):
        num = struc[k, :]
        for kk in range(0, 12):
            numnew = np.dot(np.reshape(num[0:2], (1, 2)), aa[2 * kk:2 * kk + 2])
            qq = ismember_rows(struc[:, 0:2], numnew)
            strmap[k, kk] = qq
    np.savetxt('strmap.txt', strmap, fmt="%d")


def all_convex_below(max_atom):
    # expat_layer = csc.get_below_atom.get_expa_layer(max_atom)
    expa_layer = csc.GetBelowAtom(max_atom).get_expa_layer()
    max_line = csc.GetBelowAtom(max_atom).get_max_line()
    convex_set = []
    a = np.ceil(max_atom / 2).astype('int')
    for N in range(2, a + 1):
        time_start = time.time()
        # conv_stru = csc.generate_convex_structure.get_convex_cluster(N, max_line[N-2], expa_layer[N-2])
        conv_stru = csc.GenerateConvexStructure(N, max_line[N - 2], expa_layer[N - 2]).get_convex_cluster
        convex_set.append(conv_stru)
        time_end = time.time()
        elapse = time_end - time_start
        print('number of the basic line is ', N, ', the structures has ', len(conv_stru), ","
                                                                                          "and time has pasted", elapse)

    all_set = []
    for ii in convex_set:
        for jj in ii:
            all_set.append(jj)

    less_set = []
    for ii in all_set:
        if np.size(ii, axis=0) <= max_atom:
            less_set.append(ii)
    # check repeatation
    unique_less_set = np.zeros((len(less_set), max_atom), dtype='int')
    k = 0
    for ii in less_set:
        leng = np.size(ii, axis=0)
        unique_less_set[k, 0:leng] = ii
        k = k + 1
    unique_less_set = np.unique(unique_less_set, axis=0)

    n = np.size(unique_less_set, axis=0)
    atom_number = np.zeros((n, 2), dtype='int')
    atom_number[:, 0] = range(3, 3 + n)
    for jj in range(3, 3 + n):
        ij = 1
        for ii in unique_less_set:
            ii = ii[ii > 0]
            ii = np.hstack((np.array([0]), ii))
            if len(ii) == jj:
                ij = ij + 1
        atom_number[jj - 3, 1] = ij

    lists = [[] for i in range(max_atom)]
    for ii in range(2, max_atom + 1):
        for jj in range(0, n):
            a = unique_less_set[jj]
            a = a[a > 0]
            leng = len(a)
            a = np.hstack((np.array([0]), a))
            if leng == ii:
                a = np.reshape(a, (1, ii + 1))
                if np.shape(lists[ii]) == (0,):
                    lists[ii] = a
                else:
                    lists[ii] = np.vstack((lists[ii], a))
    return lists


def draw_fig(lists, vaca_para, n_end, n_start=0):
    # n = len(lists)
    for jj in range(n_start, n_end):
        ij = 1
        temp = lists[jj]
        for ii in temp:
            string = str(jj + 1) + f'_{ij:d}' + '.png'
            # suf.dra(ii,1,string)
            csc.advanced_process(ii, vaca_para=vaca_para, dra_para=1, string=string).dra()
            ij = ij + 1


def write_poscar(lists, vaca_para, n_end, n_start=0):
    nn = 1
    for ii in range(n_start, n_end):
        numb = lists[ii]
        for jj in range(0, np.size(numb, axis=0)):
            # suf.write_txt(n,numb[jj])
            # __init__(self, index, vaca_para, dra_para=0, string=[], n=1):
            csc.advanced_process(numb[jj], vaca_para=vaca_para, n=nn).write_poscar()
            nn = nn + 1


def vacan(lists, max_vaca, n_end, n_start=0):
    atomlist = np.loadtxt('neig.txt', dtype='int')
    all_cavern = []
    for ii in range(n_start, n_end):
        start = time.time()
        temp = lists[ii]
        number = np.size(temp, axis=0)
        for jj in range(0, number):
            struc = temp[jj]
            # center = suf.get_center(struc,atomlist)
            center = csc.GetVacancy(origin_struc=struc, atomlist=atomlist, center=0, cavancy_n=0).get_center()
            # check repeat and cannot remove two who are neighbour
            opt_number = np.shape(center)[0]
            if opt_number != 0:
                # add_cavern = []
                if opt_number > max_vaca:
                    for ki in range(0, max_vaca):
                        # opt_choice = suf.get_nonear_cav(struc, center,ki+1)
                        opt_choice = csc.GetVacancy(struc, atomlist, center, ki + 1).get_nonear_cav()
                        all_cavern.append(opt_choice)

                else:
                    for ki in range(0, opt_number):
                        opt_choice = csc.GetVacancy(struc, atomlist, center, ki + 1).get_nonear_cav()
                        all_cavern.append(opt_choice)

        end = time.time()
        print(ii + 1, 'has elpased', end - start, 's')
    return all_cavern


def ismember_rows(aa, pat):
    rows = np.size(pat, axis=0)
    index = np.zeros((1, rows), dtype='int')
    for k in range(0, rows):
        sub_pat = pat[k]
        sub_pat = np.reshape(sub_pat, (1, length(sub_pat)))
        a = np.where(np.all(aa == sub_pat, axis=1))
        a = np.array(a)
        if np.shape(a) == (1, 0):
            index[0, k] = 0.1
        else:
            index[0, k] = a[0, 0]

    return index


def check_rep(symlist, atom, sequence):
    # import numpy as np
    all_poslist = symlist[atom, :]
    all_poslist = reshape(all_poslist)
    sequence = np.unique(sequence, axis=1)
    nn = length(sequence)
    seq_list = np.zeros([nn, 2]).astype('int')
    seq_list[0, :] = np.array([0, sequence[:, 0] + 1])
    for ii in np.arange(1, nn):
        seq_list[ii] = np.array([seq_list[ii - 1, 1], sequence[:, ii] + 1])

    for jj in np.arange(0, nn):
        each_element = np.arange(seq_list[jj, 0], seq_list[jj, 1], dtype='int')
        all_poslist[each_element] = np.sort(all_poslist[each_element], axis=0)
    all_poslist = all_poslist.transpose()
    all_poslist = np.unique(all_poslist, axis=0)
    min_atom = all_poslist[0]
    min_atom = np.reshape(min_atom, (1, length(min_atom)))
    return min_atom


def gene_list(m, n):
    elelist = np.arange(0, m)
    elelist = np.reshape(elelist, (m, 1))
    for k in range(0, n - 1):
        midlist1 = np.kron(np.ones((m, 1)), elelist)
        midlist2 = np.kron(np.ones((1, m)), elelist[:, 0])

        midlist2 = np.sort(midlist2).transpose()
        elelist = np.hstack((midlist2, midlist1))
    return elelist


def angle(x, y):
    import math
    if x == 0 and y < 0:
        theta = 3 * math.pi / 2
    elif x == 0 and y > 0:
        theta = math.pi / 2
    elif x > 0 and y == 0:
        theta = 0
    elif x < 0 and y == 0:
        theta = math.pi
    else:
        theta = math.atan(y / x)
        if y < 0 < x:
            theta = theta + 2 * math.pi
        elif x < 0 < y:
            theta = theta + math.pi
        elif x < 0 and y < 0:
            theta = theta + math.pi
    if x == 0 and y == 0:
        theta = 0
    return theta


def length(x):
    x = np.array(x)
    try:
        a = np.size(x, axis=0)
    except IndexError:
        a = 0
    try:
        b = np.size(x, axis=1)
    except IndexError:
        b = 0
    if a > b:
        return a
    else:
        return b


def reshape(array):
    shape = np.shape(array)
    if 0 in shape:
        array = np.reshape(array, (1, 0))
    elif 1 in shape:
        shape = list(shape)
        shape.remove(1)
        array = np.reshape(array, shape)
    return array
