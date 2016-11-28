import numpy as np

vb = []
arp = []

for idx in xrange(10):
    with open("results"+str(idx+1)+".txt") as f:
        vb.append([])
        arp.append([])
        for line in f.readlines():
            if "Variational Bayes Logistic Regression:" in line:
                vb[-1].append(float(line.strip().replace("Variational Bayes Logistic Regression: ","")))
            elif "Adv. Reference Prior Logistic Regression: " in line:
                arp[-1].append(float(line.strip().replace("Adv. Reference Prior Logistic Regression: ","")))

vb = np.array(vb)
arp = np.array(arp)

print vb
print arp
print
print vb.mean(0)
print arp.mean(0)
print
print vb.std(0)
print arp.std(0)
