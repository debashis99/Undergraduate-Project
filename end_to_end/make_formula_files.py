from utils import greyscale, get_vocab

_dir_data = 'data/sample/'

with open(_dir_data + 'formulas.norm.lst') as my_file:
    l_no = 1
    for line in my_file:
        f = _dir_data + "formula_processed/" + str(l_no) + ".txt"
        f= open(f,"w+")
        f.write(line)
        f.close()
        l_no = l_no + 1