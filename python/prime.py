import pickle
import search
from math import sqrt, ceil

def get_list(fname='primes'):
    try :
        f = open(fname,'rb')
        l = pickle.load(f)
        f.close()
    except :
        l = []
    if (len(l) == 0): l = [2, 3]
    return l

def set_list(l, fname='primes'):
    f = open(fname,'wb')
    pickle.dump(l, f)
    f.close()

def isprime(num, l=[]):

    # Basic Test
    if (num % 2 == 0 or
        num % 3 == 0):
        return False

    # Load prime number list
    if len(l) == 0: l = get_list()
    
    e = l[-1]

    # Check if number in list
    if num <= e:
        if search.binary(l, num) >= 0:
            return True
        else: return False

    # Primality test, Step 1
    for n in l:
        if (n * n > num):  return True
        if (num % n == 0): return False

    # Primality test, Step 2
    # You are only here if num > e * e
    n = int(e/6) * 6
    m = n - 1
    k = n + 1
    while (m * m <= num):
        if (num % m == 0 or
            num % k == 0):
            return False
        m += 6
        k += 6
    return True

def primes(num, l=[]):
    
    if len(l) == 0: l = get_list()

    # Check to see if a sublist can be created
    e = l[-1]
    if (num < e):
        res = search.binary_low(l, num)
        return l[:res[0]+1]

    e = 6*(ceil(e/6))

    lim = num + 1
    # Extend the current list
    for n in range(e, lim, 6):
        m = n - 1
        if isprime(m, l): l.append(m)
        m = n + 1
        if isprime(m, l): l.append(m)

    # Save to pickle
    set_list(l)

    return l

def pfactors(num, l=[]):
    if (num == 1): return [1]

    if len(l) == 0: l = get_list()
    loc, found = search.binary_low(l, num)
    e = l[-1]


    # If number is in list
    # It means the number is prime
    # Hence it is the only prime factor
    if (found): return [num]

    # Get all primes below num
    ll = primes(num, l)
    return [n for n in ll if(num % n == 0)]

def factors(num):
    return [n for n in range(2, num) if (num % n == 0)]
    
def all_factors(num):
    if (num == 1): res = [1]
    else: res = [1, num]
    res.extend(factors(num))
    return res

def totient(n, l=[]):
    # Totient function
    # T(n) = n * (p1 - 1)*(p2 - 1)*(p3 -1)...(pn -1)/(p1 * p2 * p3 .. pn)
    # Where p1, p2, p3 ... pn are prime factors of n
    P = pfactors(n, l)
    num = n
    den = 1
    for p in P:
        num *= (p-1)
        den *= p
    return int(num / den)
