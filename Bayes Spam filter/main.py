import string
import math
from collections import Counter

hamPath = "D:\\Egyetem\\4.felev\AI\lab3\data\ham\\"
spamPath = "D:\\Egyetem\\4.felev\AI\lab3\data\spam\\"

spamHists = []
hamHists = []

PS = 0
PH = 0
V = 0
cardWS = 0
cardWH = 0

DicS = {}
DicH = {}


# STEP1) Preprocess -> tokens , hists

def GetTokensOfEmail(infile):
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\stopwords.txt", "r")
    stopwords1 = f.read().splitlines()
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\stopwords2.txt", "r")
    stopwords2 = f.read().splitlines()
    tokens = infile.read().lower().split()
    for token in tokens:
        if token in string.punctuation or token == "subject:" or token in stopwords1 or token in stopwords2:
            for i in range(tokens.count(token)):
                tokens.remove(token)
    return tokens


def DataPreProcess():
    print("\n Data preprocessing (getting tokens and histograms) is running.. \n\n")
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\stopwords.txt", "r")
    stopwords1 = f.read().splitlines()
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\stopwords2.txt", "r")
    stopwords2 = f.read().splitlines()
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\\train.txt", "r")
    train = f.read().splitlines()
    for file in train:
        if "ham" in file:
            infile = open("D:\Egyetem\\4.felev\AI\lab3\data\ham\\" + file, encoding="latin-1")
            tokens = infile.read().lower().split()
            for token in tokens:
                if token in string.punctuation or token == "subject:" or token in stopwords1 or token in stopwords2:
                    for i in range(tokens.count(token)):
                        tokens.remove(token)
            hamHists.append((Counter(tokens)))
        else:
            infile = open("D:\Egyetem\\4.felev\AI\lab3\data\spam\\" + file, encoding="latin-1")
            tokens = infile.read().lower().split()
            for token in tokens:
                if token in string.punctuation or token == "subject:" or token in stopwords1 or token in stopwords2:
                    for i in range(tokens.count(token)):
                        tokens.remove(token)
            spamHists.append(Counter(tokens))


# STEP2) Bayes Model Calculations:

def BayesModelParameters(alpha):
    global PS
    global PH
    global V
    global cardWS
    global cardWH

    PS = len(spamHists) / (len(spamHists) + len(hamHists))
    PH = len(hamHists) / (len(spamHists) + len(hamHists))

    CS = spamHists[0]
    CH = hamHists[0]
    for i in range(1, len(spamHists)):
        CS += spamHists[i]
    for i in range(1, len(hamHists)):
        CH += hamHists[i]

    cardWS = sum(Counter.values(CS))
    cardWH = sum(Counter.values(CH))

    # Spam dictionary
    V = len(CS + CH)
    # print(len(CS), " ", len(CH), " ", V)
    for elem in Counter.items(CS):
        DicS[elem[0]] = (elem[1] + alpha) / (alpha * V + cardWS)

    # Ham dictionary
    for elem in Counter.items(CH):
        DicH[elem[0]] = (elem[1] + alpha) / (alpha * V + cardWH)


def TestEmail(file, alpha):
    default = []
    if alpha == 0:
        default.append(0.00000001)
        default.append(0.00000001)
    else:
        default.append(alpha / (alpha * V + cardWS))
        default.append(alpha / (alpha * V + cardWH))

    file = open(file, "r", encoding="latin-1")
    tokens = GetTokensOfEmail(file)
    counterd = Counter(tokens)
    lnR = math.log(PS) - math.log(PH)
    s = 0
    for token in counterd.items():
        s += token[1] * (math.log(DicS.get(token[0], default[0])) - math.log(DicH.get(token[0], default[1])))
    lnR += s
    if lnR > 1:
        return "spam"
    else:
        return "ham"


def TrainingError(alpha):
    wrong = 0
    processed = 0
    falseNegative = 0
    falsePositive = 0

    f = open("D:\Egyetem\\4.felev\AI\lab3\data\\train.txt", "r", encoding="latin-1")
    files = f.read().splitlines()
    for file in files:
        processed += 1
        if "spam" in file:
            if TestEmail("D:\Egyetem\\4.felev\AI\lab3\data\spam\\" + file, alpha) != "spam":
                wrong += 1
                falsePositive += 1
        elif "ham" in file:
            if TestEmail("D:\Egyetem\\4.felev\AI\lab3\data\ham\\" + file, alpha) != "ham":
                wrong += 1
                falseNegative += 1

    print("Wrong/processed = ", wrong, "/", processed)
    print("Training error = ", wrong / processed * 100, "%")
    print("False Positive/False Negative = ", falsePositive / falseNegative)


def TestError(alpha):
    wrong = 0
    processed = 0
    falseNegative = 0
    falsePositive = 0
    f = open("D:\Egyetem\\4.felev\AI\lab3\data\\test.txt", "r", encoding="latin-1")
    files = f.read().splitlines()
    for file in files:
        processed += 1
        if "spam" in file:
            if TestEmail("D:\Egyetem\\4.felev\AI\lab3\data\spam\\" + file, alpha) != "spam":
                wrong += 1
                falsePositive += 1
        elif "ham" in file:
            if TestEmail("D:\Egyetem\\4.felev\AI\lab3\data\ham\\" + file, alpha) != "ham":
                wrong += 1
                falseNegative += 1

    print("Wrong/processed = ", wrong, "/", processed)
    print("Test error = ", wrong / processed * 100, "%")
    print("False Positive/False Negative = ", falsePositive / falseNegative)


if __name__ == '__main__':
    DataPreProcess()
    # for elem in Counter.items(spamHists[0]):
    #    print(elem[0])

    print("\n----------------------------------------------------\n")
    print("Spam filter with alpha = 0:\n")
    alpha = 0
    BayesModelParameters(alpha)
    TrainingError(alpha)
    print()
    TestError(alpha)
    print("\n\n----------------------------------------------------\n")

    alphas = [1, 0.1, 0.01, 0.0000001]
    for alpha in alphas:
        print("Spam filter with alpha = ", alpha, "\n")
        BayesModelParameters(alpha)
        TrainingError(alpha)
        print()
        TestError(alpha)
        print("\n----------------------------------------------------\n")
