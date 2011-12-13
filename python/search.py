def binary_low(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval < x:
            lo = mid+1
        elif midval > x: 
            hi = mid
        else:
            return mid, True
    return (hi -1), False

def binary(a, x, lo=0, hi=None):
    loc, found = binary_low(a, x, lo, hi)
    if (found): 
        return loc
    else :
        return -1
