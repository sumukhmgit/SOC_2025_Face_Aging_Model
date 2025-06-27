#Use a loop to pop elements from the end or build a new list in reverse.
def reverse_list(lst):
    # Write your logic here
    for i in range(len(lst)):
      lst.append(lst.pop(len(lst)-1-i))
    return lst