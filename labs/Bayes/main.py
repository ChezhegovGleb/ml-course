import os
import matplotlib.pyplot as plt

countClasses = 2
fines = [4e3, 1]
alpha = 1e-1

gramm = 1
k = 10

parts = [[] for i in range(k)]

cnt = -1

for d, dirs, files in os.walk('./messages'):
    for f in files:
        clazz = 0
        if (f.count("spmsg")):
            clazz = 1

        fileName = d + '/' + f
        with open(fileName) as fileName:
            subject = list(map(str, next(fileName).split()))[1:]
            subject = [int(item) for item in subject]
            text = list(map(int, next(fileName).split()))
            message = [clazz] + subject + text
            parts[cnt].append(message)
    cnt += 1

trueAnswers = 0
trueLegitAnswers = 0
legitAnswers = 0
spamAnswers = 0
answers = 0

for i in range(k):

    allWords = set()
    countMessagesInClass = [0] * countClasses
    classToTextCountByWord = [dict() for d in range(countClasses)]

    n = 0

    for j in range(k):
        if (j == i):
            continue

        for message in parts[j]:
            n += 1
            currentDict = set()
            clazz = message[0]
            countMessagesInClass[clazz] += 1

            for ind in (1, len(message) - gramm):
                word = ' '.join(map(str, message[ind:ind + gramm]))
                allWords.add(word)
                currentDict.add(word)

            for word in currentDict:
                if (classToTextCountByWord[clazz].get(word, 0)):
                    classToTextCountByWord[clazz][word] += 1
                else:
                    classToTextCountByWord[clazz][word] = 1

    for message in parts[i]:
        results = [0] * countClasses
        sum = 0
        dictionaryWords = set()
        realClazz = message[0]
        for ind in (1, len(message) - gramm):
            word = ' '.join(map(str, message[ind:ind + gramm]))
            dictionaryWords.add(word)

        for clazz in range(countClasses):
            ans = 1.0
            for word in allWords:
                numerator = alpha
                denominator = 2 * alpha
                if (classToTextCountByWord[clazz].get(word, 0)):
                    numerator += classToTextCountByWord[clazz][word]

                denominator += countMessagesInClass[clazz]
                pwc = float(numerator) / denominator

                flag = int(word in dictionaryWords)
                ans *= (flag * pwc + (1 - flag) * (1 - pwc))

            pc = float(countMessagesInClass[clazz]) / n
            ans *= pc * fines[clazz]
            results[clazz] = ans

        sum = results[0] + results[1]
        results[0] /= sum
        results[1] /= sum

        if ((results[1] < 0.8) and (realClazz == 0)):
            trueLegitAnswers += 1
        if (results[1] < 0.8):
            trueAnswers += 1
        if (realClazz == 0):
            legitAnswers += 1
        if (realClazz == 1):
            spamAnswers += 1
        answers += 1

print("accuracy = ", end = "")
print(trueAnswers / answers)
print("findLegit / realLegit = ", end = "")
print(trueLegitAnswers / legitAnswers)

# draw accuracy from lambda_legit

lmbda = []
accuracy = []

for lmb in range(1, int(4e3) + 1, 100):
    fines = [lmb, 1]
    for alpha in [1e-1]:
        for gramm in range(1, 2):
            trueAnswers = 0
            trueLegitAnswers = 0
            legitAnswers = 0
            spamAnswers = 0
            answers = 0

            for i in range(k):

                allWords = set()
                countMessagesInClass = [0] * countClasses
                classToTextCountByWord = [dict() for d in range(countClasses)]

                n = 0

                for j in range(k):
                    if (j == i):
                        continue

                    for message in parts[j]:
                        n += 1
                        currentDict = set()
                        clazz = message[0]
                        countMessagesInClass[clazz] += 1

                        for ind in (1, len(message) - gramm):
                            word = ' '.join(map(str, message[ind:ind + gramm]))
                            allWords.add(word)
                            currentDict.add(word)

                        for word in currentDict:
                            if (classToTextCountByWord[clazz].get(word, 0)):
                                classToTextCountByWord[clazz][word] += 1
                            else:
                                classToTextCountByWord[clazz][word] = 1

                for message in parts[i]:
                    results = [0] * countClasses
                    sum = 0
                    dictionaryWords = set()
                    realClazz = message[0]
                    for ind in (1, len(message) - gramm):
                        word = ' '.join(map(str, message[ind:ind + gramm]))
                        dictionaryWords.add(word)

                    for clazz in range(countClasses):
                        ans = 1.0
                        for word in allWords:
                            numerator = alpha
                            denominator = 2 * alpha
                            if (classToTextCountByWord[clazz].get(word, 0)):
                                numerator += classToTextCountByWord[clazz][word]

                            denominator += countMessagesInClass[clazz]
                            pwc = float(numerator) / denominator

                            flag = int(word in dictionaryWords)
                            ans *= (flag * pwc + (1 - flag) * (1 - pwc))

                        pc = float(countMessagesInClass[clazz]) / n
                        ans *= pc * fines[clazz]
                        results[clazz] = ans

                    sum = results[0] + results[1]
                    results[0] /= sum
                    results[1] /= sum

                    if ((results[1] < 0.8) and (realClazz == 0)):
                        trueLegitAnswers += 1

                    if (results[1] < 0.8):
                        trueAnswers += 1

                    if (realClazz == 0):
                        legitAnswers += 1
                    if (realClazz == 1):
                        spamAnswers += 1
                    answers += 1

        print(trueAnswers / answers, trueLegitAnswers / legitAnswers, gramm, fines, alpha)
        print(trueAnswers, trueLegitAnswers, legitAnswers, answers)
    lmbda.append(lmb)
    accuracy.append(trueAnswers / answers)

plt.plot(lmbda, accuracy)
plt.xlabel("lambda_legit")
plt.ylabel("accuracy")
plt.title("Graphic 2")
plt.show()

# build ROC

rocX = []
rocY = []

mu = 0.0
while (mu <= 1):
    trueAnswers = 0
    trueLegitAnswers = 0
    falseLegitAnswers = 0
    legitAnswers = 0
    spamAnswers = 0
    answers = 0
    for i in range(k):

        allWords = set()
        countMessagesInClass = [0] * countClasses
        classToTextCountByWord = [dict() for d in range(countClasses)]

        n = 0

        for j in range(k):
            if (j == i):
                continue

            for message in parts[j]:
                n += 1
                currentDict = set()
                clazz = message[0]
                countMessagesInClass[clazz] += 1

                for ind in (1, len(message) - gramm):
                    word = ' '.join(map(str, message[ind:ind + gramm]))
                    allWords.add(word)
                    currentDict.add(word)

                for word in currentDict:
                    if (classToTextCountByWord[clazz].get(word, 0)):
                        classToTextCountByWord[clazz][word] += 1
                    else:
                        classToTextCountByWord[clazz][word] = 1

        for message in parts[i]:
            results = [0] * countClasses
            sum = 0
            dictionaryWords = set()
            realClazz = message[0]
            for ind in (1, len(message) - gramm):
                word = ' '.join(map(str, message[ind:ind + gramm]))
                dictionaryWords.add(word)

            for clazz in range(countClasses):
                ans = 1.0
                for word in allWords:
                    numerator = alpha
                    denominator = 2 * alpha
                    if (classToTextCountByWord[clazz].get(word, 0)):
                        numerator += classToTextCountByWord[clazz][word]

                    denominator += countMessagesInClass[clazz]
                    pwc = float(numerator) / denominator

                    flag = int(word in dictionaryWords)
                    ans *= (flag * pwc + (1 - flag) * (1 - pwc))

                pc = float(countMessagesInClass[clazz]) / n
                ans *= pc * fines[clazz]
                results[clazz] = ans

            sum = results[0] + results[1]
            results[0] /= sum
            if ((results[0] >= mu) and (realClazz == 0)):
                trueLegitAnswers += 1
            if ((results[0] >= mu) and (realClazz == 1)):
                falseLegitAnswers += 1
            if (realClazz == 0):
                legitAnswers += 1
            if (realClazz == 1):
                spamAnswers += 1
            answers += 1

    print(falseLegitAnswers, trueLegitAnswers, legitAnswers)
    rocX.append(falseLegitAnswers / spamAnswers)
    rocY.append(trueLegitAnswers / legitAnswers)
    mu += 0.025

plt.plot(rocX, rocY)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.show()