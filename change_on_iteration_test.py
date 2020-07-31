x= {i : i for i in range(10)}

for k,v in x.items():
	if k == 5:
		x[k] = "found"

print(x)
