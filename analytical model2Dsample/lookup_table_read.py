import pickle
if __name__ == "__main__":
    g = input("Type file name to read:")
    with open(g, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        print("Read Complete!")