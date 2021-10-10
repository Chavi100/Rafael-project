
def CountPosNeg(values):
    pair_dict={}
    for num in values:
        if num in pair_dict:
                pair_dict[num]+=1
        else:
            pair_dict[num]=1
    pairs=[]
    for key in pair_dict:
        if pair_dict[key]!=0 and key!=0:
         add=min(pair_dict[key],pair_dict[key*-1])
         for i in range(add):
             pairs.append(abs(key))
         pair_dict[key]=0
         pair_dict[key*-1]=0
    pairs=sorted(pairs)
    print(pairs)
    return pairs
CountPosNeg([-1,1,0,1,-1,1])


def findLargestDiff(a, b):
    find_min_1 = a[0]
    find_min_2 = a[0]
    find_max_1 = b[0]
    find_max_2 = b[0]
    for value in a:
        if value < find_min_1:
            find_min_1 = value
        if value > find_max_1:
            find_max_1 = value
    for value in b:
        if value < find_min_2:
            find_min_2 = value
        if value > find_max_2:
            find_max_2 = value
    if find_max_2 - find_min_1 > find_max_1 - find_min_2:
        print([find_min_1,find_max_2])
        return [find_min_1,find_max_2]
    else:
        print([find_max_1, find_min_2])
        return [find_max_1, find_min_2]
findLargestDiff([1,2,3],[4,5,6])


def findGivenDifference(A, B, D):
    dict_b = {}
    for value in B:
        dict_b[value] = None
    with_dif = []
    for value in A:
        if value - D in dict_b or value + D in dict_b:
            with_dif.append(value)
    with_dif = sorted(set(with_dif))
    print(with_dif)
    return with_dif



def sortAnagrams(strings):
    ascii_dict={}
    anagram_dict={}
    for string in strings:
        an=''.join(sorted(string))
        ascii_sum=0
        for char in string:
            ascii_sum+=ord(char)
        print(ascii_sum)
        if an in anagram_dict:
            anagram_dict[an].append(string)
        else:
            anagram_dict[an]=[string]
        if ascii_sum in ascii_dict:
            ascii_dict[ascii_sum].append(an)
        else:
            ascii_dict[ascii_sum]=[an]
    sorted_array=[]
    for anagram in anagram_dict:
        anagram_dict[anagram] = sorted(anagram_dict[anagram])
        if anagram_dict[anagram][0] != anagram:
            ascii_sum = 0
            for char in anagram_dict[anagram][0]:
                ascii_sum += ord(char)
            ascii_dict[ascii_sum].remove(anagram)
            ascii_dict[ascii_sum].append(anagram_dict[anagram][0])
    sort_by_ascii= sorted(ascii_dict)
    for value in sort_by_ascii:
        ascii_dict[value]=sorted(ascii_dict[value])
        for anagram in ascii_dict[value]:
            an = ''.join(sorted(anagram))
            for str in anagram_dict[an]:
                sorted_array.append(str)
    print(sorted_array)






sortAnagrams(["eat","tea","tan","ate","nat","bat"])







