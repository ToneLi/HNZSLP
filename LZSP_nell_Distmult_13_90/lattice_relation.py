

def generate_N_grams(word,ngram):
  # words=[word for word in text.split(" ") if word not in set(stopwords.words('english'))]
  words=[x for x in word]
  # print("Sentence after removing stopwords:",words)

  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[''.join(ngram) for ngram in temp]
  return ans



def  get_bytes_relations(word,ngram):

  N=[]
  for i in range(ngram):
    i=i+1
    n_gram=generate_N_grams(word,i)
    # print(n_gram)
    After_p = []
    if len(n_gram)!=1:
      for j in range(len(n_gram)):

        if j==0:
          n_gram_j="0|"+n_gram[j]+"|"+n_gram[j+1]
          After_p.append(n_gram_j)
        elif j==len(n_gram)-1:
          n_gram_j = n_gram[j-1] + "|" + n_gram[j]+"|0"
          After_p.append(n_gram_j)
        else:
          # print(j)
          n_gram_j = n_gram[j - 1] + "|" +n_gram[j] +"|"+n_gram[j+1]
          After_p.append(n_gram_j)
    else:
      n_gram_j =  "0|" + n_gram[0] + "|0"
      After_p.append(n_gram_j)

    N.append(After_p)



  Full_unit_connection=[]
  for i in range(len(N)):
    if i>0:
      level_i=N[i]
      j=-1
      L=[]
      for u in level_i:
        j=j+1
        # print("N[i-1]",N[i-1])
        last_level_connet=[x.split("|")[1] for x in N[i-1][j:j+2]]
        L_level_i=u+"-"+last_level_connet[0]+"|"+last_level_connet[1]
        L.append(L_level_i)
      Full_unit_connection.append(L)
    else:
        Full_unit_connection.append(N[0])


  if len(word)>ngram:
    # print("Full_unit_connection",len(Full_unit_connection))
    # print("n--",ngram)
    last_2_level=Full_unit_connection[ngram-2]
    last_level_c=""
    # print("last_2_level",last_2_level)
    for u in last_2_level:
      last2=u.split("-")[0].split("|")[1]
      last_level_c=last_level_c+last2+"|"
    last_level_c=last_level_c.strip("|")

    last_level="0|"+word+"|0"+"-"+last_level_c

    Full_unit_connection[-1]=[last_level]

  return Full_unit_connection




# results=get_bytes_relations(word,ngram)
if __name__=="__main__":
  W=['agricultural', 'product', 'cut', 'into', 'geometric', 'shape']
  W1 = ['agent00', 'created', '0000000', '0000000']
  ngram=7
  for word in W1:
    results = get_bytes_relations(word, ngram)
    print(results)
    # print(len(results))


"""
[['0|h|a', 'h|a|s', 'a|s|0'], ['0|ha|as-h|a', 'ha|as|0-a|s'], ['0|has|0-ha|as']]
ngram-1: ['0|h|a', 'h|a|s', 'a|s|0']     left neighbor|unit|right neighbor
ngram-2: ['0|ha|as-h|a', 'ha|as|0-a|s'] left neighbor|unit|right neighbor--the composition of, unit
ngram-3: ['0|has|0-ha|as']
"""