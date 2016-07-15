import pickle

def main():
    dataset = "boston"
    filepath = "hypers_tensorflow_" + dataset + ".pickle"

    print filepath
    with open(filepath, 'rb') as f:
            best_params = pickle.load(f)
    
    print best_params

if __name__ == "__main__":
    main()
