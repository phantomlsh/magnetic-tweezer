from math import sqrt, floor

r = 40
X = []
Y = []
W = []
for i in range(0, r, 1):
	X.append([])
	Y.append([])
	W.append([])

for i in range(-r, r, 1):
	for j in range(-r, r, 1):
		d = sqrt(i*i + j*j)
		if d + 1 >= r or d <= 1:
			continue
		f = floor(d)
		if (1+f-d > 0.0001):
			X[f].append(i)
			Y[f].append(j)
			W[f].append((1+f-d)/f)
		if (d-f > 0.0001):
			X[f+1].append(i)
			Y[f+1].append(j)
			W[f+1].append((d-f)/f)