#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:33:11 2017
@author: hecc
"""
import numpy as np
import main_fun as mf


class GenerateConvexStructure:

    def __init__(self, length, max_line, expand_layer, vacancy=0):

        self.length = length
        self.max_line = max_line
        self.expand_layer = expand_layer
        self.vacancy = vacancy

    @property
    def get_convex_cluster(self):
        nn = self.length
        max_line = self.max_line
        expa_layer = self.expand_layer
        struc = np.loadtxt('struc.txt', dtype='int')
        strmap = np.loadtxt('strmap.txt', dtype='int')
        basis_line = np.zeros((nn, 2), dtype='int')
        n = int(nn / 2)
        mod_nn = np.mod(nn, 2)
        ii = 0
        if mod_nn == 0:
            for k in range(-n + 1, n + 1):
                basis_line[ii] = [k, 0]
                ii = ii + 1
        else:
            for k in range(-n, n + 1):
                basis_line[ii] = [k, 0]
                ii = ii + 1
        arange_list = self.gene_list(max_line)
        all_full_str = []
        all_full_number = []
        arange_list_rows = np.size(arange_list, axis=0)
        for k in range(0, arange_list_rows):
            lower = arange_list[k, 0]
            upper = arange_list[k, 1]
            basis = basis_line.copy()
            old_str = basis_line.copy()
            if upper != 0:
                for ii in range(0, upper):
                    new_str = self.get_new_str(old_str, 1)
                    basis = np.vstack((basis, new_str))
                    old_str = new_str.copy()
            old_str = basis_line.copy()
            if lower != 0:
                for ii in range(0, lower):
                    new_str = self.get_new_str(old_str, 0)
                    basis = np.vstack((basis, new_str))
                    old_str = new_str.copy()
            basis = np.unique(basis, axis=0).astype('int')
            all_full_str.append(basis)

        # equavalent length expansion
        for ii in range(0, expa_layer):
            for jj in range(0, arange_list_rows):
                a = all_full_str[jj].copy()
                q_above = np.where(a[:, 1] > 0)
                q_below = np.where(a[:, 1] < 0)
                # expand upper
                add_aa = np.array([]).reshape((0, 2))
                for kk in range(0, ii + 1):
                    add_aa = np.vstack((add_aa, basis_line + np.array([[0, 1]]) * (kk + 1)))
                # add_A = np.array(add_A)
                expa_stru_up = np.vstack((basis_line,
                                          add_aa,
                                          a[q_above] + np.array([[0, 1]]) * (ii + 1),
                                          a[q_below])).astype('int')
                all_full_str.append(expa_stru_up)
                # expand below
                add_aa = np.array([]).reshape((0, 2))
                for kk in range(0, ii + 1):
                    add_aa = np.vstack((add_aa, basis_line + np.array([[0, -1]]) * (kk + 1)))
                # add_A = np.array(add_A)[0]
                expa_stru_below = np.vstack((basis_line,
                                             add_aa,
                                             a[q_above],
                                             a[q_below] + np.array([[0, -1]]) * (ii + 1))).astype('int')
                all_full_str.append(expa_stru_below)
        # expand filament
        if expa_layer > 0:
            for ii in range(0, expa_layer):
                add_aa = np.array([]).reshape((0, 2))
                for kk in range(0, ii + 1):
                    add_aa = np.vstack((add_aa, basis_line + np.array([[0, 1]]) * (kk + 1)))
                # add_A= np.array(add_A)[0]
                add_aa = np.vstack((basis_line, add_aa)).astype('int')
                all_full_str.append(add_aa)

        all_full_rows = np.size(all_full_str, axis=0)
        # all_full_center = []
        for ii in range(0, all_full_rows):
            index = mf.ismember_rows(struc[:, 0:2], all_full_str[ii])
            index = np.reshape(index, (mf.length(index),))
            index = namestr(index, strmap, struc)
            all_full_number.append(index)
            # all_full_center.append(get_center(index,atomlist))

        rows = len(all_full_number)
        max_size = 0
        for k in range(0, rows):
            a = mf.length(all_full_number[k])
            if max_size < a:
                max_size = a

        unique_full_str = np.zeros((rows, max_size))
        for k in range(0, rows):
            a = all_full_number[k]
            a = np.reshape(a, (mf.length(a),))
            unique_full_str[k, 0:mf.length(a)] = a
        unique_full_str = np.unique(unique_full_str, axis=0)
        unique_str = []
        for k in range(0, len(unique_full_str)):
            a = unique_full_str[k]
            a = a[a > 0]
            a = np.hstack((np.array([0]), a)).astype('int')
            unique_str.append(a)
        return unique_str

    @staticmethod
    def gene_list(n):
        list_1 = []
        for ii in range(0, n):
            for jj in range(ii, n):
                list_1.append([ii, jj])
        list_1.pop(0)
        return np.array(list_1, dtype='int')

    @staticmethod
    def get_new_str(basis_str, para):
        neig_matrix = np.array([[0, 1], [-1, 1], [1, -1], [0, -1]])
        center = np.mean(basis_str[:, 0])
        new_str = np.zeros(np.shape(basis_str))
        rows = np.size(basis_str, axis=0)

        if para == 1:
            for k in range(0, rows):
                if basis_str[k, 0] <= center:
                    new_str[k] = basis_str[k] + neig_matrix[0]
                else:
                    new_str[k] = basis_str[k] + neig_matrix[1]
        elif para == 0:
            for k in range(0, rows):
                if basis_str[k, 0] <= center:
                    new_str[k] = basis_str[k] + neig_matrix[2]
                else:
                    new_str[k] = basis_str[k] + neig_matrix[3]
        new_str = np.unique(new_str, axis=0)
        return new_str


class GetBelowAtom:

    def __init__(self, max_atom):
        self.max_atom = max_atom

    def get_expa_layer(self):
        max_atom = self.max_atom
        max_n = np.ceil(max_atom / 2).astype('int')
        length = max_n - 1
        expa_layer = np.zeros((length,))
        for ii in range(0, length):
            expa_layer[ii] = np.ceil(max_atom / (ii + 2))
        expa_layer = expa_layer.astype('int')
        for ii in range(0, length):
            if expa_layer[ii] > ii:
                expa_layer[ii] = ii
        return expa_layer

    def get_max_line(self):
        max_atom = self.max_atom
        max_n = np.ceil(max_atom / 2).astype('int')
        length = max_n - 1
        max_line = np.zeros((length,))

        for ii in range(0, length):
            k = 1
            a = ii + 2
            old_a = a
            add_a = old_a - 1
            while add_a > 0:
                a = a + add_a
                add_a = add_a - 1
                k = k + 1
                if a > max_atom:
                    break
            max_line[ii] = k
        max_line = max_line.astype('int')
        return max_line


class GetVacancy:

    def __init__(self, origin_struc, atomlist, center, cavancy_n):
        self.origin_struc = origin_struc
        self.atomlist = atomlist
        self.center = center
        self.n = cavancy_n

    def get_nonear_cav(self):
        origin_struc = self.origin_struc
        center = self.center
        n = self.n
        from itertools import combinations
        neig = np.loadtxt('neig.txt', dtype='int')
        combins = [c for c in combinations(center, n)]
        opt_choice = []
        for ii in range(0, len(combins)):
            choice = combins[ii]
            all_neigh = neig[choice, :].flatten()
            section = np.intersect1d(all_neigh, choice)
            if np.shape(section) == (0,):
                # choice = namestr(choice,strmap,struc)
                if np.shape(opt_choice) == (0,):
                    opt_choice = choice
                else:
                    opt_choice = np.vstack((opt_choice, choice))
        if len(opt_choice) != 0:
            shape = np.shape(opt_choice)
            if len(shape) == 1:
                opt_choice = np.reshape(opt_choice, (1, mf.length(opt_choice)))
            opt_choice = np.unique(opt_choice, axis=0)
            strmap = np.loadtxt('strmap.txt', dtype='int')
            struc = np.loadtxt('struc.txt', dtype='int')
            rows = np.size(opt_choice, axis=0)
            cols = np.size(origin_struc, axis=0)
            new_struc = np.zeros((rows, cols - n), dtype='int')
            for ii in range(0, rows):
                new_struc[ii] = np.setdiff1d(origin_struc, opt_choice[ii])
                new_struc[ii] = namestr(new_struc[ii], strmap, struc)
            new_struc = np.unique(new_struc, axis=0)
        else:
            new_struc = []
        return new_struc

    def get_center(self):

        atomlist = self.atomlist
        struc = self.origin_struc
        nn = np.size(struc, axis=0)
        neig_num = np.zeros((nn, 2))
        neig_num[:, 0] = range(0, nn)
        for ii in range(0, nn):
            q = np.intersect1d(atomlist[struc[ii]], struc)
            neig_num[ii, 1] = mf.length(q)
        index = np.where(neig_num[:, 1] == 6)
        center = struc[index]
        return center


class advanced_process:

    def __init__(self, index, vaca_para, dra_para=0, string='a.png', n=1):
        self.index = index
        self.dra_para = dra_para
        self.string = string
        self.n = n
        self.vaca_para = vaca_para

    def dra(self):

        index = self.index
        para = self.dra_para
        string = self.string
        vaca_para = self.vaca_para
        import matplotlib.pyplot as plt
        # index = [1,2,3,4,5,6,7,8,9,10]
        struc = np.loadtxt('struc.txt', dtype='int')
        stru = struc[index, 0:2]
        new_struc = np.zeros((np.shape(stru)))
        new_struc[:, 0] = stru[:, 0] + stru[:, 1] / 2
        new_struc[:, 1] = stru[:, 1] * np.sqrt(3) / 2
        # scatter
        plt.figure()
        rows = mf.length(stru)
        d = np.zeros((rows, rows))
        for ii in range(0, rows):
            for jj in range(ii + 1, rows):
                d[ii, jj] = ((stru[ii, 0] - stru[jj, 0]) ** 2 + (stru[ii, 1] - stru[jj, 1]) ** 2
                             + (stru[ii, 0] - stru[jj, 0]) * (stru[ii, 1] - stru[jj, 1]))
                if (d[ii, jj] - 1) < 0.1:
                    x1 = new_struc[ii, 0]
                    y1 = new_struc[ii, 1]
                    x2 = new_struc[jj, 0]
                    y2 = new_struc[jj, 1]
                    plt.plot([x1, x2], [y1, y2], linewidth=1, color='blue', zorder=0)
        plt.scatter(new_struc[:, 0], new_struc[:, 1], color='red', s=70, marker='o', alpha=1, zorder=1)
        # T:散点的颜色
        # s：散点的大小
        # alpha:是透明程度
        plt.xticks(())
        plt.yticks(())
        plt.axis('equal')
        if para == 1:
            # plt.savefig(string)
            fig = plt.gcf()
            # fig.set_size_inches(18.5, 10.5)
            fig.savefig(vaca_para + string, dpi=450)
            plt.close()

    def write_poscar(self):
        index = self.index
        n = self.n
        vaca_para = self.vaca_para
        struc = np.loadtxt('struc.txt', dtype='int')
        cluster = struc[index, 0:2]
        x = (cluster[:, 0] + cluster[:, 1] / 2) * 1.7
        y = (cluster[:, 1] * np.sqrt(3) / 2) * 1.7
        a = (max(x) - min(x)) + 12
        b = (max(y) - min(y)) + 12
        c = 12
        basis = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        xx = np.reshape(x + a / 2, (len(index), 1))
        yy = np.reshape(y + b / 2, (len(index), 1))
        zz = np.ones((len(index), 1)) * c / 2
        pp = np.hstack((xx, yy, zz))
        filename = 'POSCAR' + vaca_para + str(n) + '.vasp'
        with open(filename, 'a') as file_object:
            file_object.write('Bsheet\n')
            file_object.write('1.00000\n')
            for ii in range(0, 3):
                # write basis
                line = str(basis[ii])
                newline = ''
                for jj in line:
                    if jj != '[' and jj != ']':
                        newline = newline + jj
                file_object.write(newline + '\n')
            file_object.write('B\n')
            file_object.write(str(len(index)) + '\n')
            file_object.write('Cart\n')
            for ii in range(0, len(index)):
                line = str(pp[ii])
                newline = ''
                for jj in line:
                    if jj != '[' and jj != ']':
                        newline = newline + jj
                file_object.write(newline + '\n')


def namestr(atom, strmap, struc):

    atom_coor = struc[atom, 0:2]
    num = mf.length(atom)
    # distance matrix
    temp_d = np.zeros((num, num), dtype='int')

    for k in range(0, num):
        tem_coor = atom_coor - atom_coor[k]
        index = mf.ismember_rows(struc[:, 0:2], tem_coor)
        temp_d[k] = np.sort(struc[index, 2])

    new_temp_d = np.unique(temp_d, axis=0)

    min_temp_d = new_temp_d[0]
    index = []
    for ii in range(0, num):
        if np.all(temp_d[ii] == min_temp_d):
            index.append(ii)
    index = np.array(index)
    # index = np.reshape(index, (mf.length(index)))
    center = atom_coor[index]
    num_center = np.size(center, axis=0)
    min_name = []
    for k in range(0, num_center):
        cen = center[k, 0:2]
        str1 = atom_coor - cen
        qq = mf.ismember_rows(struc[:, 0:2], str1)
        all_poss = mf.reshape(strmap[qq, :])
        all_poss = np.sort(all_poss, axis=0)
        all_poss = np.unique(all_poss, axis=1).T
        min_name.append(all_poss[0])
    min_name = np.unique(min_name, axis=0)
    min_atom = min_name[0]
    return min_atom
