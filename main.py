#from asl_detector.data import load_data, preprocess_data

class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task"""
    def __init__(self) -> None:
        return

    def run(self):
        data = load_data()
        print(data)
        #preprocess_data()

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()
    