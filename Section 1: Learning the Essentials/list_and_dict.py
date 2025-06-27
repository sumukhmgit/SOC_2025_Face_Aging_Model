#Use str.lower().
#Use str.split() to break the sentence into words.
#Use a dictionary to count frequencies.
def count_words(sentence):
    # Write your logic here
    low=sentence.lower()
    words=low.split()
    word_count={}
    for word in words:
      if word in word_count:
        word_count[word]+=1
      else:
        word_count[word]
    return word_count