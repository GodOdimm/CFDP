import numpy as np
import os

def ReadFile(file_name):
    l = []
    with open(file_name) as f:
        for line in f:
            str_num = line.strip().split()
            num = [eval(s) for s in str_num]
            l.append(num)
    print('{0} points have been read...'.format(l.__len__()))
    return np.array(l)


def CountDistance(points):
    p_num, p_dim = points.shape
    dis = np.zeros((p_num, p_num))
    for i in range(p_num):
        for j in range(i, p_num):
            rel_vec = points[i] - points[j]
            dis_ij = np.linalg.norm(rel_vec,ord=2)
            dis[i][j] = dis[j][i] = dis_ij
    print('counted distance...')
    return dis


def CountDensity(dis, dc):  # the density of every points
    assert dc > 0
    p_num, _ = dis.shape
    density = []
    for vec in dis:
        small_than_dc_pos = [vec < dc]
        density.append(np.sum(small_than_dc_pos))
    return np.array(density).astype(float)


def Count_delta(dis, density):
    delta = []
    neighbor = []  # the nearest neighbor that has large density than it
    p_num, _ = dis.shape
    assert len(density) == p_num
    max_density = np.max(density)
    for i in range(p_num):
        if density[i] == max_density:
            delta.append(np.max(dis[i]))
            neighbor.append(-1)
        else:
            min_j = 0
            tmp_delta = np.max(dis[i])
            for j, val in enumerate(dis[i]):
                if density[j] > density[i] and tmp_delta > val:
                    tmp_delta = val
                    min_j = j
            delta.append(tmp_delta)
            neighbor.append(min_j)

            # pos = [density > density[i]]  # for Pi ,find all positions Pj larger than Pi
            # all_dij = dis[i]
            # candi_dij = all_dij[pos]
            # dij = np.min(candi_dij)
            # delta.append(dij)
            # for ind, val in enumerate(dis[i]):
            #     if val == dij:
            #         neighbor.append(ind)
            #         break

    assert len(neighbor) == p_num
    return np.array(delta), np.array(neighbor)


def SearchDc(dis, tau=0.02):
    sides = []
    p_num, _ = dis.shape
    for i in range(p_num - 1):
        for j in range(i + 1, p_num):
            sides.append(dis[i][j])
    assert sides.__len__() == p_num * (p_num - 1) / 2
    sides.sort()
    pos = round(len(sides) * tau)
    return sides[pos - 1]


def CDFofNormalDistribution(x):
    p0 = 220.2068679123761
    p1 = 221.2135961699311
    p2 = 112.0792914978709
    p3 = 33.91286607838300
    p4 = 6.373962203531650
    p5 = .7003830644436881
    p6 = .03326249659989109
    q0 = 440.4137358247552
    q1 = 793.8265125199484
    q2 = 637.3336333788311
    q3 = 296.5642487796737
    q4 = 86.78073220294608
    q5 = 16.06417757920695
    q6 = 1.755667163182642
    q7 = 0.08838834764831844
    cutoff = 7.071
    root2pi = 2.506628274631001
    xabs = np.abs(x)
    res = 0
    if x > 37.0:
        res = 1.0
    elif x < -37.0:
        res = 0.0
    else:
        expntl = np.exp(-.5 * xabs * xabs)
        pdf = expntl / root2pi
        if (xabs < cutoff):
            res = expntl * ((((((p6 * xabs + p5) * xabs + p4) * xabs + p3) * xabs + p2) * xabs + p1) * xabs + p0) / ((((
                                                                                                                               (
                                                                                                                                       (
                                                                                                                                               (
                                                                                                                                                       q7 * xabs + q6) * xabs + q5) * xabs + q4) * xabs + q3) * xabs + q2) * xabs + q1) * xabs + q0)
        else:
            res = pdf / (xabs + 1.0 / (xabs + 2.0 / (xabs + 3.0 / (xabs + 4.0 / (xabs + 0.65)))))

    if x > 0:
        res = 1 - res
    return res


def numberOfClusters(gamma, threshold):
    sort_gamma = np.sort(gamma)
    mean = np.mean(sort_gamma)
    var = np.var(sort_gamma)
    std = np.sqrt(var)
    nClusters=0
    for i in range(len(gamma) - 1, 0, -1):
        tmp = (sort_gamma[i] - mean) / std
        prob = CDFofNormalDistribution(tmp)
        if i > len(gamma) - 100:
            pass
            # print(prob)
        if prob < threshold or 1 - prob < threshold:
            continue
        nClusters = len(gamma) - 1 - i
        break
    return nClusters


def getnCluster(density, delta, threshold):
    #normalized

    def zScoreNormailzation(x):
        x=(x-np.mean(x))/np.std(x)
        return x

    def Normailzation(x):
        x=(x-np.min(x))/(np.max(x)-np.min(x))
        return x

    den,dell=density.copy(),delta.copy()
    den=Normailzation(den)
    dell=Normailzation(dell)
    gamma=den*dell

    # print(gamma[83],gamma[229])

    #
    # den_min, den_max = np.min(density), np.max(density)
    # delta_min, delta_max = np.min(delta), np.max(delta)
    # den_range = den_max - den_min
    # delta_range = delta_max - delta_min
    # gamma = []
    # for i in range(len(density)):
    #     tmp = (density[i] - den_min) * (delta[i] - delta_min) / (den_range * delta_range)
    #     gamma.append(tmp)
    # gamma = np.array(gamma)
    ncluster = numberOfClusters(gamma=gamma, threshold=threshold)

    vec = []
    # find the K th max numbers

    # for test
    # ncluster=15
    print(ncluster)
    THE_MIN=np.min(gamma)-1
    for _ in range(ncluster):
        max_ind = np.argmax(gamma)
        gamma[max_ind] = THE_MIN
        vec.append(max_ind)

    return ncluster, vec


def assignCluster(neighbor, nvec_index, ord,density):
    p_num = len(neighbor)
    point_class = np.zeros(p_num)
    class_num = 1
    for index in nvec_index:  # give class to core
        point_class[index] = class_num
        class_num += 1
    # assign class to other point
    for index in ord:
        if point_class[index] == 0:  # haven't assign class
            point_class[index] = point_class[neighbor[index]]
            if 0== point_class[neighbor[index]]:
                print(density[index],density[neighbor[index]])
                break

    tmp=np.sort(point_class)
    return point_class.astype(int)


def sortByDensity(density):
    p_num = len(density)
    density = density.tolist()
    l = [i for i in range(p_num)]
    tmp_tuple = [(a, b) for a, b in zip(density, l)]
    sorted_tuple = sorted(tmp_tuple, key=lambda x: -x[0])
    ord = [j for i, j in sorted_tuple]
    return ord


if __name__ == '__main__':
    file_name = input("Please enter file name: ").strip() + '.txt'
    points = ReadFile(os.path.join('data', file_name))
    dis = CountDistance(points[:, ])
    tau=0.015
    dc = SearchDc(dis, tau=tau)
    density = CountDensity(dis, dc)
    ord = sortByDensity(density)
    delta, neighbor = Count_delta(dis, density)

    _, nvec_index = getnCluster(density, delta, threshold=0.05)

    #plus 1% to the max gamma point,and repeat once!
    # max_pos=nvec_index[0]
    # density[max_pos]=density[max_pos]*1.001
    for i in nvec_index:
        pre=density[i]
        density[i]=density[i]*1.001



    ord = sortByDensity(density)
    delta, neighbor = Count_delta(dis, density)
    ncluster, nvec_index = getnCluster(density, delta, threshold=0.05)

    point_class = assignCluster(neighbor=neighbor, nvec_index=nvec_index, ord=ord,density=density)

    # print(ncluster)

    from showResult import DrawDistribute, drawPoint,showAcc

    showAcc(points,point_class)

    # drawPoint(points, point_class=point_class,filename=file_name,dc=tau)
    # DrawDistribute(rho=density, delta=delta)
