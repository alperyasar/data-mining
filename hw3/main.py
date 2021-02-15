from fp_growth import *

def main(filename,min_sup):
    item,D = read_data(filename)

    itemset = find_frequent_itemset(item,min_sup,D)
    itemset = sort(itemset)
    header_table = create_HeaderTable(itemset)
    
    # Create the root node
    root = Node('root',1,None,None)
    print("  Itemset   |   Sup count")
    frequent_list = frequent_item(itemset)
    # Run the tree generation algorithm
    root = create_FPTree(D,frequent_list,header_table,root)
    final = Run_FPGrowth(root,min_sup,header_table)

    print("Item | Frequent Patterns Generated")
    for item in final:
        if len(item[0]) == 1:
            print("\n", item[0][0] , end="    ")
        else: 
            print("{",end="")
            for i in range(len(item[0]) - 1):
                print(item[0][i], end=", ")
            print(item[0][len(item[0])-1],":",item[1], end="}, ")

def read_data(filename):
    data = open(filename, "r")
    itemset = []
    transaction = {}
    id = 1
    for line in data:
        fLine = line.strip().split(',')
        for item in fLine:
            if item not in itemset:
                itemset.append(item)
        transaction[id] = fLine
        id += 1
    data.close()
    print(len(itemset))
    return itemset,transaction

main("data.txt",4)
#main("data2.txt",5)
#main("data3.txt",250)