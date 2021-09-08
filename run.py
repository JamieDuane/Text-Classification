import os.path
from train_model.text_classification import textClassification
from pathlib import Path

def excute():
    path = os.path.curdir
    path = Path(path+'/data')
    tc = textClassification(path)
    y_test, test_data, model = tc.train_model(method = 'cnn')
    tc.test_model(y_test, test_data, model)

if __name__=='__main__':
    excute()