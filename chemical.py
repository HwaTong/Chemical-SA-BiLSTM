# code from https://github.com/ali-i-abbas/protein_function_prediction

def getAminoAcidCharge(x):
    if x in "KR":
        return 1.0
    if x == "H":
        return 0.1
    if x in "DE":
        return -1.0
    return 0.0

def getAminoAcidHydrophobicity(x):
    AminoAcids = "ACDEFGHIKLMNPQRSTVWY"
    _hydro = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2,-0.9, -1.3]
    return _hydro[AminoAcids.find(x)]

def isAminoAcidPolar(x):
    return x in "DEHKNQRSTY"

def isAminoAcidAromatic(x):
    return x in "FWY"

def hasAminoAcidHydroxyl(x):
    return x in "ST"

def hasAminoAcidSulfur(x):
    return x in "CM"

def createdata(inputs, targets,vocab, gram_len):
    index = np.arange(len(inputs))
    batch = [inputs[k] for k in index]
    labels = np.asarray([targets[k] for k in index])
    ngrams = list()
    np_prots = list()
    for seq in batch:
        grams = np.zeros((len(seq) - gram_len + 1,), dtype='int32')
        np_prot = list()
        for i in range(len(seq) - gram_len + 1):
            a = seq[i]
            descArray = [float(x) for x in [getAminoAcidCharge(a), getAminoAcidHydrophobicity(a), isAminoAcidPolar(a),isAminoAcidAromatic(a), hasAminoAcidHydroxyl(a), hasAminoAcidSulfur(a)]]
            np_prot.append(descArray)
            grams[i] = vocab[seq[i: (i + gram_len)]]
        np_prots.append(np_prot)
        ngrams.append(grams)
    np_prots = sequence.pad_sequences(np_prots, maxlen=MAXLEN)
    ngrams = sequence.pad_sequences(ngrams, maxlen=MAXLEN)
    res_inputs = to_categorical(ngrams, num_classes=len(vocab) + 1)
    res_inputs = np.concatenate((res_inputs, np_prots), 2)
    return res_inputs,labels
