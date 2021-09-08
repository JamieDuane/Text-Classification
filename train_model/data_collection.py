import numpy as np

class dataExtraction(object):
    def __init__(self, path):
        self.path = path
        self.tmp_path = self.path / 'tmp'

    def collect_data(self, name=None):
        data = open(self.path/f'{name}.json')
        label, sentence = [], []
        for i in data.readlines():
            data_line = eval(i)
            label.append(data_line['label'])
            sentence.append(data_line['sentence'])
        return np.array(label), sentence

    def load_data(self):
        data = {name: {'label':[], 'sentence':[]} for name in ['train', 'test', 'dev']}
        for key in list(data.keys()):
            data[key]['label'], data[key]['sentence'] = self.collect_data(name=key)
        return data