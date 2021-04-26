import os

def gen_neg_description_file():
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negImg'):
            f.write('negImg\\' + filename + '\n')