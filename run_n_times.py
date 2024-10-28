import w2v

n = 1000  # Number of times to run
with open('./final_results.txt', 'w') as w:
    for i in range(n):
        w2v.main(plot_results=False)
