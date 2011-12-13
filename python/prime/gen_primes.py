#!/usr/bin/python
import prime
import pickle

x = [2, 3]
for n in range(5, 1000000, 2):
    if prime.isprime(n):
        x.append(n)

f = open('primes', 'wb')
pickle.dump(x, f)
f.close()

f = open('primes', 'rb')
y = pickle.load(f)
f.close()

print(len(x))
print(len(y))
