import math
import time
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class SimulateThread(object):

    def __init__(self, population, individual, pool, ratio, effective_point, save_path):
        super(SimulateThread, self).__init__()
        self.population = population
        self.individuals = individual
        self.pool = pool
        self.ratio = ratio
        self.effective_point = effective_point
        self.save_path = save_path

    def recombine(self, seq1, seq2, initial_point):
        temporary_list = seq1[initial_point:].copy()
        seq1[initial_point:] = seq2[initial_point:].copy()
        seq2[initial_point:] = temporary_list.copy()
        del temporary_list
        return seq1, seq2

    def generator_per_time(self, chrom, random_numbers):
        chrom = np.array(chrom).transpose()
        ref = chrom[2].copy()
        mut = chrom[3].copy()
        for initial_point in random_numbers:
            ref, mut = self.recombine(ref, mut, initial_point)
        return ref, mut

    def generate_effective_points(self, directory_path, ep_numbers):
        integration = np.load(os.path.join(os.getcwd(), "Simulate_data", "integration_data.npy"))
        effective_points = []
        save_path = os.path.join(directory_path, "effective_points")  # r"D:\2.5k-10-10%\effective_points"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        choices = np.random.choice([i for i in range(len(integration))], size=ep_numbers, replace=False)
        for choice in choices:
            effective_points.append((choice, integration[choice]))  # (index, [chrom, position, ref, mut])
        np.save(os.path.join(save_path, str(ep_numbers) + "_effective_points.npy"), effective_points)
    
    def finetune_ef(self, directory_path, ep_numbers):
        start = time.time()
        integration = np.load(os.path.join(os.getcwd(), "Simulate_data", "reduce_integration_data.npy"))
        save_path = os.path.join(directory_path, "effective_points")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        min_dist = 6e6
        re = True
        start_tmp = start
        while re:
            tmp = time.time()
            if tmp - start_tmp > 60:
                min_dist *= 0.8
                start_tmp = tmp
                print("reduce min distance")
            ef = []
            distances = []
            choices = np.random.choice([i for i in range(len(integration))], size=ep_numbers)
            for choice in choices:
                ef.append(np.array([choice, integration[choice]]))  # (index, [chrom, position,  ref, mut])
                distances.append(integration[choice][1])
            ef = sorted(ef, key=lambda x: x[1][0])
            distances = sorted(distances)
            near = []
            re = False
            for i, j in zip(range(len(distances) - 1), range(1, len(distances))):
                if distances[j] - distances[i] < min_dist:
                    re = True
                    near.append(distances[i])
                    near.append(distances[j])
                    break
        np.save(os.path.join(save_path, str(ep_numbers) + "_effective_points.npy"), ef)

    def generate_muti_pairs_data(self, directory_path, max_individual):
        dir_path = os.path.join(os.getcwd(), "Simulate_data", "reduce_filtered_data")
        dir_list = os.listdir(dir_path)
        save_path = os.path.join(directory_path, "saved_pairs_data")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        generator_number = math.floor(max_individual * 1.2)
        for count in range(generator_number):
            per_time_result = []
            ref_result = []
            mut_result = []
            for file_name in dir_list:
                data = np.load(os.path.join(dir_path, file_name))
                # times = round(np.random.normal(3, 0.5, 1)[0])
                times = round(np.random.normal(5, 0.5, 1)[0])
                random_numbers = np.random.randint(len(data), size=times)
                ref, mut = self.generator_per_time(data, random_numbers)
                if ref_result != []:
                    ref_result = np.concatenate((ref_result, ref))
                    mut_result = np.concatenate((mut_result, mut))
                else:
                    ref_result = ref.copy()
                    mut_result = mut.copy()
                    del ref, mut
            per_time_result.append(ref_result.copy())
            per_time_result.append(mut_result.copy())
            del ref_result, mut_result
            np.save(os.path.join(save_path, str(count + 1) + ".npy"), per_time_result)

    def recombination(self, directory_path, max_individual):
        dir_path = os.path.join(directory_path, "saved_pairs_data")
        dir_list = os.listdir(dir_path)
        save_dir_path = os.path.join(directory_path, "recombination_data")
        if not os.path.isdir(save_dir_path):
            os.mkdir(save_dir_path)
        choosable_group = []
        recombination_result = []
        for i, c in zip(dir_list, range(len(dir_list))):
            for j in [0, 1]:
                choosable_group.append((i, j))
        for i in range(max_individual):
            group_choose = np.random.choice(len(choosable_group), size=2, replace=False)
            group1 = choosable_group[group_choose[0]]
            group2 = choosable_group[group_choose[1]]
            recombination_result.append([group1, group2])
            choosable_group.remove(group1)
            choosable_group.remove(group2)
        np.save(os.path.join(save_dir_path, "recombination.npy"), recombination_result)

    def cal_and_sort(self, directory_path, sub_dir, pairs, ep_numbers):

        effective_points = np.load(os.path.join(directory_path, r"effective_points/"
                                                + str(ep_numbers) + "_effective_points.npy"), allow_pickle=True)
        plants_dir = np.load(os.path.join(directory_path, "recombination_data",
                                          "recombination.npy"), allow_pickle=True)
        plants_indexes = np.random.choice([i for i in range(len(plants_dir))], pairs, replace=False)
        calculated_phenotype = []
        save_path = os.path.join(directory_path, sub_dir, "cal_and_sort")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for single_single_plant, c in zip(plants_indexes, range(len(plants_indexes))):
            count = 0
            single_plant = plants_dir[single_single_plant]
            data1 = np.load(os.path.join(directory_path, "saved_pairs_data",
                                         single_plant[0][0]))[int(single_plant[0][1])]
            data2 = np.load(os.path.join(directory_path, "saved_pairs_data",
                                         single_plant[1][0]))[int(single_plant[1][1])]
            data = np.array([data1, data2]).transpose()
            for effective_point in effective_points:  # 因效应点改动
                index = effective_point[0]
                effective_ref = effective_point[1][2]
                effective_mut = effective_point[1][3]
                # -----------------------------------------------------------------------
                # if data[index][0] == effective_ref and data[index][1] == effective_ref:
                #     count += value * 2
                # elif data[index][0] == effective_mut and data[index][1] == effective_mut:
                #     pass
                # else:
                #     count += value
                # -----------------------------------------------------------------------
                if data[index][0] == effective_ref and data[index][1] == effective_ref:
                    # count += 0.05   # 二十个点，Aa2.5%，AA5%
                    # count += 0.1  # 十个点，Aa5%，AA10%
                    # count += 0.2  # 五个点，Aa10%，AA20%
                    count += 1/ep_numbers
                elif data[index][0] == effective_mut and data[index][1] == effective_mut:
                    pass
                else:
                    # count += 0.025
                    # count += 0.05
                    # count += 0.1
                    count += (1/ep_numbers)/2
                # -----------------------------------------------------------------------
            count += np.random.normal(0, 0.06, 1)[0]
            if count < 0:
                count = 0
            elif count > 1:
                count = 1
            calculated_phenotype.append((single_plant, count))  # (file_name, frequency)
            del data1, data2
        np.save(os.path.join(save_path, "calculated_phenotype.npy"),
                calculated_phenotype)
        # sort
        calculated_phenotype = np.load(os.path.join(save_path, "calculated_phenotype.npy"), allow_pickle=True)
        sorted_data = sorted(calculated_phenotype, key=lambda x: float(x[1]))
        np.save(os.path.join(save_path, "sorted_phenotype.npy"),
                sorted_data)  # r"D:\2.5k-10-10%\cal_and_sort\sorted_phenotype.npy"

    def divide_pools(self, sp, pairs, directory_path, pool_num):
        sorted_path = os.path.join(directory_path, "cal_and_sort", "sorted_phenotype.npy")
        plants_path = os.path.join(directory_path, "saved_pairs_data")
        save_path = os.path.join(sp, "divide_pools")
        sorted_data = np.load(sorted_path, allow_pickle=True)
        indexes = np.random.choice([i for i in range(len(sorted_data))], size=pairs, replace=False)
        sorted_data = sorted(sorted_data[indexes], key=lambda x: float(x[1]))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        summary_data = []
        # for index, pool in enumerate(range(0, pairs, int(pairs / pool_num))):
            # pool_name = sorted_data[pool: pool + int(pairs / pool_num)]
        for index in range(pool_num):
            pool = int(pairs / pool_num) * index
            if index == 0:
                pool_name = sorted_data[pool: pool + int(pairs * self.ratio)]
            elif index == pool_num - 1:
                pool_name = sorted_data[-1 * int(pairs * self.ratio) :]
            else:
                s = int((int(pairs / pool_num) - int(pairs * self.ratio)) / 2) + pool
                pool_name = sorted_data[s: s + int(pairs * self.ratio)]
            pool_data = []
            for name in pool_name:
                a = [np.load(os.path.join(plants_path, name[0][0][0]))[int(name[0][0][1])],
                     np.load(os.path.join(plants_path, name[0][1][0]))[int(name[0][1][1])]]
                if pool_data == []:
                    pool_data = a
                else:
                    pool_data = [pool_data[0] + a[0], pool_data[1] + a[1]]
            pool_data = pool_data[0] + pool_data[1]
            summary_data.append(pool_data / (int(pairs / pool_num) * 2))
        np.save(os.path.join(save_path, "summary_data.npy"), summary_data)

        chr_pos = np.load(os.path.join(os.getcwd(), "Simulate_data", "reduce_integration_data.npy"), allow_pickle=True)
        # txt
        if not os.path.exists(os.path.join(save_path, "simulate_data.txt")):
            f = open(os.path.join(save_path, "simulate_data.txt"), "x")
        else:
            f = open(os.path.join(save_path, "simulate_data.txt"), "w")
        with f:
            for i, j in zip(chr_pos, np.array(summary_data).T):
                line = str(i[0]) + "\t" + str(i[1]) + "\t" + "I" + "\t" + "I"
                for k in j:
                    line += "\t"
                    line += "%.3f,%.3f" % (k, 1 - k)
                f.writelines(line + "\n")
        # csv
        data = []
        for i, j in zip(chr_pos, np.array(summary_data).T):
            row = []
            row.append(i[0])
            row.append(i[1])
            row.append("I")
            row.append("I")
            for k in j:
                row.append(k)
                row.append(1 - k)
            data.append(row)
        df = pd.DataFrame(data)
        import time
        df.to_csv(os.path.join(save_path, "simulate_data_{}.csv".format(hash(time.time()))), header=False, index=False)

    def run(self):
        # self.generate_effective_points(self.save_path, self.effective_point)
        print("--Genetate Effective Points")
        self.finetune_ef(self.save_path, self.effective_point)
        print("--Generate Pairs Data")
        self.generate_muti_pairs_data(self.save_path, max(self.individuals))
        print("Recombination")
        self.recombination(self.save_path, max(self.individuals))
        print("Calculate and Sort")
        self.cal_and_sort(self.save_path, "", max(self.individuals), self.effective_point)
        # individual-ep-ev
        directory = []
        for individual in self.individuals:
            directory.append(str(individual) + "-" + str(self.effective_point) +
                             "-" + str(self.ratio))
        pairs = self.individuals
        iter_num = self.population
        print("Generate Final Data")
        for d, p, c in zip(directory, pairs, range(len(pairs))):
            dir_name = d
            save_path = os.path.join(self.save_path, dir_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for n in tqdm(range(iter_num)):
                dir_name_ = d + "--" + str(n + 1)
                save_path_ = os.path.join(save_path, dir_name_)
                if not os.path.exists(save_path_):
                    os.mkdir(save_path_)
                self.divide_pools(save_path_, p, self.save_path, self.pool)

def main(args):
    simulate = SimulateThread(population=1, individual=[args.i], pool=args.p, ratio=args.r, effective_point=args.e, save_path=args.s)
    simulate.run()


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", default=None, required=True, type=int, help="individual")
    parser.add_argument("--p", default=None, required=True, type=int, help="pools")
    parser.add_argument("--r", default=None, required=True, type=float, help="ratio")
    parser.add_argument("--e", default=None, required=True, type=int, help="effective points")
    parser.add_argument("--s", default=None, required=True, type=str, help="save path")

    args = parser.parse_args()

    main(args)