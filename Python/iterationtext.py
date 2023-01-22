def wrap(string, max_width):
    s=""
    for i in range(0,len(string),max_width):
        s += string[i:i+max_width] + "\n"
    return s
