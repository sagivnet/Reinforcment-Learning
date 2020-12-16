import statistics
import numpy as np
import json

def calc(list_of_scores_list): # get [ [<num_of_scores> Scores], [<num_of_scores> Scores], .. ]

    num_of_runs = len(list_of_scores_list)
    num_of_scores = len(list_of_scores_list[0])

    res = [[] for x in xrange(num_of_scores)]

    for run in range(num_of_runs):
        for ep in range(num_of_scores):
            res[ep].append(list_of_scores_list[run][ep])

    for ep in range(num_of_scores):
        res[ep] = [ep+1,np.mean(res[ep]),statistics.stdev(res[ep])]

    return res

def compress(l):
    # len(l) = 1000
    temp = [0, 0, 0]
    res = []
    for i in range(len(l)):
        temp[1] += l[i][1]
        temp[2] += l[i][2]
        if (i % 100 == 0 and i != 0) or i == len(l)-1:
            temp[1] /= 100
            temp[2] /= 100
            temp[0] = l[i][0] - 1
            if i == len(l) - 1:
                temp[0] += 1
            res.append(temp)
            temp = [0, 0, 0]
    return res


inp = raw_input("List Of Lists file: ")
with open(inp, "r") as f:
    output = calc(json.loads(f.read()))
    # print(output)
    print(compress(output))