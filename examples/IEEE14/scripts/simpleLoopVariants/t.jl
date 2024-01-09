
st = time()
    j = 0
    for i in 1:10
        j = 10*i^2 + i^3
    end
et = time()

print(et - st)