import timeit
# start = timeit.default_timer()

temps = []

class Dataset:
    """Utility class to manage a dataset stored in a external file."""

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            lines = [line for line in lines if line]  # Skipping blank lines
            for line in lines:
                transaction = list(map(int, line.split(" ")))
                self.transactions.append(transaction)
                for item in transaction:
                    self.items.add(item)
        except IOError as e:
            print("Unable to read dataset file!\n" + e)

    def trans_num(self):
        """Returns the number of transactions in the dataset"""
        return len(self.transactions)

    def items_num(self):
        """Returns the number of different items in the dataset"""
        return len(self.items)

    def get_transaction(self, i):
        """Returns the transaction at index i as an int array"""
        return self.transactions[i]
    def get_list_items(self):
        return self.items


# apriori
def apriori(filepath, minFrequency):

    data = Dataset(filepath)
    transac = data.transactions
    queue = []

    for d in data.items:
        queue.append([d])
    result = []
    res = []
    size_queue = []

    def generate_items(queue):
        new_queue = []
        new_items = []
        for idx1, i in enumerate(queue):
            for j in  range(idx1+1, len(queue)):
                new_item = list(set(i + queue[j]))
                new_item = sorted(new_item)
                if (len(new_item) > len(queue[0])) and not new_item in new_items and not new_item in res:
                    new_queue.append(new_item)
                    new_items.append(new_item)
        return new_queue

    while queue:
        new_queue = []
        for q in queue:
            count = 0
            for t in transac:
                if all(elem in t for elem in q):
                    count +=1
#                     check support threshold
            if count/len(data.transactions) >= minFrequency:
                result.append([q, count/len(data.transactions)])
                res.append(q)
                new_queue.append(q)
        queue = generate_items(new_queue)
        size_queue.append(len(queue))
#         print(queue)


#     print('queue: ', queue)

#     print('RESULT : ', result)

    for r in result:
        print(r[0], '(',r[1],')')

#     print(max(size_queue))




def alternative_miner(filepath, minFrequency):

    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

#     remove items with a lower support than the threshold
    def prune(nb_trans, vertic, item):
        new_item = []
        for i in item:
#             print(len(vertic[i]))
            if len(vertic[i])/nb_trans < minFrequency:
                del vertic[i]
            else:
                new_item.append(i)
        return vertic, new_item

    def generate_items(item, vertic, res, nb_trans):
        cur = item.pop()
        new_item = item.copy()

        cur_vertic = dico_vertic[cur]
        del vertic[cur]


        for i in range(len(item) - 1, -1, -1) :
            if item[i] != cur:
                add = str(item[i])

                new_i = []
                new_i = list(set().union(add.split (","), cur.split (",") ))
                new_i.sort(reverse = False)
                temp = ''
                for k in new_i:
                    temp += (k + ',')
                temp = temp[:-1]
#                 print('temp:',temp)
                if not(temp in new_item) and temp != cur and not(temp in res):
                    intersec = list(set(cur_vertic).intersection(vertic[add]))
#                     intersec = intersection(cur_vertic, vertic[add]) SLOW
                    if len(intersec)/nb_trans >= minFrequency:
                        new_item.append(temp)

                        vertic[temp] = (intersec)
        return new_item, vertic


    itemss = Dataset(filepath)
    itemsets = []
    vertical = []

    result = []
    queue = []
    RES = []


# #     read the data without using the class dataset
#     itemsets = []
#     items = []
#     max_items = []
#     with open(filepath) as f:
#         for line in f:
#             it = list(map(int, line.split()))
#             itemsets.append(it)
#             max_items.append(it[-1])
#     # items = list(set(items))
#     max_items = max(max_items)
#     items = list(range(1, max_items + 1))
#     items = list(map(str, items))



#   read the data  using the class dataset
    items = list(itemss.get_list_items())

    for idx, i in enumerate(items):
        items[idx] = str(i)

    for i in range(0, len(itemss.transactions)):
        itemsets.append(itemss.get_transaction(i))

#     stop = timeit.default_timer()
#     print('Time: ', stop - start, '\'s')

# create the vertical representation old very slow
#     for i in items:
#         trans_list = []
#         for idx, j in enumerate(itemsets):
#             if int(i) in j:
#                 trans_list.append(idx)
#         vertical.append(trans_list)

#     dico_vertic = dict(zip(items, vertical))

    dico_vertic = {k: [] for k in items}

#     print(dico_vertic)

#     build vertical representation
    for idx,j in enumerate(itemsets):
        for r in j:
            dico_vertic[str(r)].append(idx)

#     stop = timeit.default_timer()
#     print('Time: ', stop - start, '\'s')

    queue = items.copy()
    dico_vertic, queue = prune(len(itemsets), dico_vertic, queue)

#     stop = timeit.default_timer()
#     print('Time: ', stop - start, '\'s')

    count = 0
    while queue:
        result.append([queue[-1], len(dico_vertic[queue[-1]]) / len(itemsets)])
        RES.append(queue[-1])
        queue, dico_vertic = generate_items(queue, dico_vertic, RES, len(itemsets))


#     print(result)
    for i in result:
        sets = i[0].replace(",", ", ")
        print("["+str(sets)+"]","(" + str(i[1]) +")")

#     get number for datasets
#     print('number of items :', len(itemss.items))
#     print('number of transactions : ', len(itemss.transactions))
#     avg = []
#     for t in itemss.transactions:
# #         print(len(t))
#         avg.append(len(t))
# #     print(itemss.transactions)

#     print('average number of items in transactions : ', sum(avg)/len(avg))


#apriori("C:\\Users\\ficus\\Desktop\\UCL 2nd Sem\\Mining patterns in Data\\Proj 1\\Datasets\\Datasets\\toy.dat", 0.125)
#alternative_miner('Datasets/pumsb_star.dat', 0.99)

# import timeit
# start = timeit.default_timer()

# stop = timeit.default_timer()
# print('Time: ', stop - start, '\'s')
