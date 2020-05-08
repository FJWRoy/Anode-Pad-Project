import sys
from tabulate import tabulate
import pandas as pd
dictInput = {}
with open("input.txt", newline=None) as f:
	for line in f:
		if line.strip() != '' and line[0] != '#':
		    key= line.split(":")[0]
		    val= line.split(":")[1].replace("\n","")
		    dictInput[key] = val

def display():
    df = pd.DataFrame(dictInput.items(), columns=['Input', 'Value'])
    print(tabulate(df,["input", "value"], tablefmt="grid"))

def input_check():
    pass
    """
    g = input("Are these listed parameters correct? y/n : ")
    if (g != 'n' and g!='y' and g!='no' and g!='yes'):
        raise Exception('Error: Please enter either y or n')
    elif g == 'n' or g=='no':
        print("Please edit input.txt")
        sys.exit(0)
        """

if __name__ == "__main__":
    print("error: run parameter module as main")