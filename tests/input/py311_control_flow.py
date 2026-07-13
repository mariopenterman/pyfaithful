def classify(n, items):
    result = []
    for x in items:
        if x < 0:
            continue
        elif x == 0 or x == n:
            result.append('edge')
        else:
            result.append('mid')
    else:
        result.append('done')
    while n > 0:
        n -= 1
        if n == 3:
            break
    return result and n >= 0
