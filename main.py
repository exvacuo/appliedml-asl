from asl_detector.data.dataloader import load_data
from asl_detector.data.preprocess_data import preprocess
import cv2
class Pipeline:
    """Class that encapsulates the pipeline for the ASL classification task"""
    def __init__(self) -> None:
        self.data = None
        return

    def run(self):
        data = load_data()

    def run_baseline(self):
        self.data = load_data()
        for i in range(len(self.data[0])):
            self.data[0][i] = preprocess(self.data[0][i], 64)
        return self.data
        
        

if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run_baseline()
    print(pipeline.data[0][0])

    cv2.imshow('Preprocessed Image', pipeline.data[0][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()