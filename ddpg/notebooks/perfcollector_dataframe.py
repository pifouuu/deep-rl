import matplotlib.pyplot as plt
import numbers
import pandas as pd

def is_wrong(values):
    if (len(values) <2):
        return True
    valpair = values[len(values)-1]
    return isinstance(valpair,numbers.Number)

def is_conv(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val >= 80.0)

def is_stuck(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val <= -19.0)

def is_noConv(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val > -19.0 and val <80.0)

class PerfCollectorData():
    def __init__(self, image_folder, **kwargs):
        self.data = {}
        self.image_folder = image_folder
        d1 = pd.DataFrame()
        d2 = pd.DataFrame()
        d1['type'] = ['conv', 'stuck', 'noconv']
        d2['type'] = ['conv', 'stuck', 'noconv', 'total']
        self.data = d1.set_index('type')
        self.data_num = d2.set_index('type')

    def init(self, delta):
        self.data[delta] = [[],[],[]]
        self.data_num[delta] = [0, 0, 0, 0]

    def add(self, delta, values):
        if is_wrong(values):
            print("ignored: ",values)
        else:
            self.data_num.loc['total', delta] += 1
            if is_conv(values):
                self.data.loc['conv', delta].append(values)
                self.data_num.loc['conv', delta] += 1
            elif is_stuck(values):
                self.data.loc['stuck', delta].append(values)
                self.data_num.loc['stuck', delta] += 1
            elif is_noConv(values):
                self.data.loc['noconv', delta].append(values)
                self.data_num.loc['noconv', delta] += 1
            else:
                print("PerfCollector::add: WTF, this should not happen!!!")

    def stats(self):
        print (self.data_num)

    def plot(self, delta):
        plt.figure(1, figsize=(20,13))
        plt.xlabel("time steps")
        plt.ylabel("performance")
        plt.title("Performance for delta = {}".format(delta))

        for values in self.data.loc['stuck', delta]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="stuck", c='r')

        for values in self.data.loc['noconv', delta]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="noconv", c='g')

        for values in self.data.loc['conv', delta]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="conv", c='b')
        #plt.legend()
        #plt.show()
        plt.savefig(self.image_folder + 'perf_' + delta + '.png', bbox_inches='tight')

    def plot_all(self):
        plt.figure(1, figsize=(20,13))
        plt.xlabel("time steps")
        plt.ylabel("performance")
        plt.title("Performance")

        for values in self.data.loc['stuck',:]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="stuck", c='r')

        for values in self.data.loc['noconv',:]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="noconv", c='g')

        for values in self.data.loc['conv',:]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="conv", c='b')
        #plt.legend()
        #plt.show()
        plt.savefig(self.image_folder + 'perf.png', bbox_inches='tight')