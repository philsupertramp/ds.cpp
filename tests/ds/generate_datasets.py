#!/bin/env/python
import math
import random


class DataSet:
    def __init__(self, input_count, output_count, target_method, step_width):
        self.Training = dict()
        self.Validation = dict()
        self.Test = dict()
        self.target_method = target_method
        self.input_count = input_count
        self.output_count = output_count
        self.step_width = step_width

    def generate(self, train_count, validation_count, test_count):
        self.Training['input'] = list()
        self.Training['output'] = list()
        self.Validation['input'] = list()
        self.Validation['output'] = list()
        self.Test['input'] = list()
        self.Test['output'] = list()
        index = 0
        stepwidth = self.step_width or 2./train_count
        i = -(train_count*stepwidth/2.)
        while index < train_count:
            self.Training['input'].append(i)
            self.Training['output'].append(self.target_method(self.Training['input'][index]))
            i += stepwidth
            index += 1

        index = 0
        stepwidth = self.step_width or 2./validation_count
        i = -(validation_count*stepwidth/2.)
        while index < validation_count:
            self.Validation['input'].append(i + random.random())
            self.Validation['output'].append(self.target_method(self.Validation['input'][index]))
            i += stepwidth
            index += 1

        index = 0
        stepwidth = self.step_width or 2./test_count
        i = -(test_count*stepwidth/2.)
        while index < test_count:
            self.Test['input'].append(i + random.random())
            self.Test['output'].append(self.target_method(self.Test['input'][index]))
            i += stepwidth
            index += 1

    def write(self, path):
        with open(f'{path}/training.dat', 'w') as f:
            for i in range(len(self.Training['input'])):
                f.write(f'{self.Training["input"][i]}\t{self.Training["output"][i]}\n')

        with open(f'{path}/validation.dat', 'w') as f:
            for i in range(len(self.Validation['input'])):
                f.write(f'{self.Validation["input"][i]}\t{self.Validation["output"][i]}\n')

        with open(f'{path}/test.dat', 'w') as f:
            for i in range(len(self.Test['input'])):
                f.write(f'{self.Test["input"][i]}\t{self.Test["output"][i]}\n')


if __name__ == '__main__':
    ds = DataSet(1, 1, lambda x: x * x, 0.1)
    ds.generate(360, 150, 140)
    ds.write('.')
