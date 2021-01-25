#!/usr/bin/env python3

import sys
import math

def readFile(filepath):
    itemsets = []
    items = []
    l_string = []
    with open(filepath) as f:
        for line in f:
            l = line.strip()
            l =  ''.join([i for i in l if not i.isdigit()])
            l = l.strip()
            if l:
                l_string.append(l)
                items.append(l)
            else: 
                if l_string:
                    itemsets.append(l_string)
                    l_string = []
    return itemsets, list(set(items))

# create the vertical view of the itemsets  (dictionary with the itemset as key)
def vertic(itemsets, items): 
    vertic = {k: [] for k in items}

    for idx1, t in enumerate(itemsets):
        for idx2, c in enumerate(t):
            if not vertic[c]:
                vertic[c].append([idx1, [idx2]])
            else:
                exist = False
                for idx3, i in enumerate(vertic[c]):
                    if i[0] == idx1:
                        vertic[c][idx3][1].append(idx2)
                        exist = True
                if not exist:
                    vertic[c].append([idx1, [idx2]])
    return vertic

def generate_key(listt):
    key = ''
    
    for i in listt:
        key = key + i + ', '
    key = key[:-2]
    return key

def print_result(ll):
    string = ''
    for e in ll:
        string += e + ', '

    return string[:-2]

# compute information gain
def Information_Gain(P, N, p, n):
    if p + n == 0:
        return 0
    if p + n == P + N:
        return 0
    entropy1 = Entropy(P / (P + N))
    entropy2 = (-(p + n) / (P + N)) * Entropy(p / (p + n))
    entropy3 = -(P + N - p - n) / (P + N) * (Entropy((P - p) / (P + N - p - n)))
    gain = entropy1 + entropy2 + entropy3
    return round(gain, 5)

def Entropy(x):
    if (x == 0) or (x == 1):
        return 0
    entro = (- x * math.log(x, 2) - (1 - x) * math.log(1 - x, 2))
    return entro

#generate candidates from q that passes the threshold mini
def generate_quandidate_info(q, items, mini, vertic_pos, vertic_neg, P, N):
    new_candi = []
    
    for i in items:
        candi = [q + [i], 0, 0, 0]
#         pos
        verti_pos = []
        if generate_key(q) in vertic_pos and i in vertic_pos:
            for r in vertic_pos[generate_key(q)]:
                for s in vertic_pos[i]:
                    l_t = []
                    if r[0] == s[0]:
                        for t in s[1]:
                            if t > r[1][0]:
                                l_t.append(t)
                    if l_t:
                        verti_pos.append([r[0], l_t])
            if verti_pos:
                candi[1] = len(verti_pos)
        
        #neg
        verti_neg = []
        if generate_key(q) in vertic_neg and i in vertic_neg:
            for r in vertic_neg[generate_key(q)]:
                for s in vertic_neg[i]:
                    l_t = []
                    if r[0] == s[0]:
                        for t in s[1]:
                            if t > r[1][0]:
                                l_t.append(t)
                    if l_t:
                        verti_neg.append([r[0], l_t])
            if verti_neg:
                candi[2] = len(verti_neg) 
            
        candi[3] = Information_Gain(P, N, candi[1], candi[2])
#         check threshold on positive or negative
        if candi[1] >= mini or candi[2] >= mini:
            new_candi.append(candi)
            vertic_pos[str(generate_key(q)+ ', '+ i)] = verti_pos
            vertic_neg[str(generate_key(q)+ ', '+ i)] = verti_neg
            
    return new_candi, vertic_pos, vertic_neg

def Spade_info(pos_filepath, neg_filepath, k):

    itemsets_pos, items_pos = readFile(pos_filepath)
    vertic_pos = vertic(itemsets_pos, items_pos)

    itemsets_neg, items_neg = readFile(neg_filepath)
    vertic_neg = vertic(itemsets_neg, items_neg)

    P = len(itemsets_pos)
    N = len(itemsets_neg)
    
#     set threshold to 1
    mini = 1
    finish = False

    queue = list(set(items_pos + items_neg))
    QUEUE = []
    ITEMS = []
    FREQ_items = []


    # create the big queue [itemset, pos_support, neg_support, Information_Gain]
    for q in queue:
        pos = 0
        neg = 0
        if q in vertic_pos:
            pos = len(vertic_pos[q])
        if q in vertic_neg:
            neg = len(vertic_neg[q])
        QUEUE.append([[q], pos, neg, Information_Gain(P, N, pos, neg)])
        ITEMS.append(q)
        FREQ_items.append(Information_Gain(P, N, pos, neg))

    QUEUE = sorted(QUEUE, key=lambda x: x[3], reverse=True)
    ITEMS = [x for _,x in sorted(zip(FREQ_items, ITEMS), reverse=True)]

    result = []
    curr_score = []
    while QUEUE:
        new_candi = []
        q = QUEUE.pop(0)
#         threshold is the absolute diff of the positive supp and negative
        thresh = abs(q[1] - q[2])
        
#         check threshold
        if  q[1] >= mini or q[2] >= mini:
            result.append(q)

                #  update threshold

            if thresh not in curr_score:
                curr_score.append(thresh)

            if len(curr_score) > k:
                curr_score.remove(min(curr_score))
                mini = (min(curr_score))

                
            new_candi, vertic_pos, vertic_neg = generate_quandidate_info(q[0], ITEMS, mini, vertic_pos, vertic_neg, P, N)

            for c in reversed(new_candi):
                QUEUE.insert(0, c)

                # sort the queue to speed up the search on the Information_Gain
            QUEUE = sorted(QUEUE, key=lambda x: x[3], reverse=True)
            
        
#     remove the non closed itemsets by keeping track of the supersets in a dictionary with a key 
#     the positive and negative support
    supersets = {}
    final_result = []
    
#     first sort the result by the length of the itemset (start by longer ones)
    closed_result = sorted(result, key=lambda x: len(x[0]), reverse=True)
    for r in closed_result:
        if (str(r[1]) + str(r[2])) not in supersets:
            supersets[str(r[1]) + str(r[2])] = []
        check = True
#         check if there exist a superset in the dict
        for t in supersets[str(r[1]) + str(r[2])]:
            pointer = 0
            for u in t:
                if len(r[0]) > pointer:
                    if u == r[0][pointer]:
                        pointer += 1
            if pointer == len(r[0]):
                check = False
        if check:
            final_result.append(r)
        for s in range(0, len(r[0])):
            if r[0][:s+1] not in supersets[str(r[1]) + str(r[2])]:
                supersets[str(r[1]) + str(r[2])].append(r[0][:s+1])

    score = []
    final_result = sorted(final_result, key=lambda x: x[3], reverse=True)
    for r in final_result:
        if r[3] not in score:
            score.append(r[3])
        if len(score) > k:
            break
        print('[' + str(print_result(r[0])) + ']', r[1], r[2], r[3])
        


def main():
    pos_filepath = sys.argv[1]  # filepath to positive class file
    neg_filepath = sys.argv[2]  # filepath to negative class file
    k = int(sys.argv[3])

    # TODO: read the dataset files and call your miner to print the top k itemsets
    Spade_info(pos_filepath, neg_filepath, k)


if __name__ == "__main__":
    main()