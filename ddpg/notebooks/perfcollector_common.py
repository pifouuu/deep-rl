import numpy as np
import matplotlib.pyplot as plt
import numbers

def isWrong(values):
    if (len(values) <2):
        return True
    valpair = values[len(values)-1]
    return isinstance(valpair,numbers.Number)

def isConv(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val >= 80.0)

def isStuck(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val <= -19.0)

def isNoConv(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val > -19.0 and val <80.0)

class PerfCollectorCommon(image_folder):
    def __init__(self,**kwargs):
        self.collec = {}
        self.image_folder = image_folder

    def init(self):
        sort_dict = {}
        sort_dict["conv"] = []
        sort_dict["stuck"] = []
        sort_dict["noconv"] = []
        self.collec = sort_dict

    def add(self, values):
        if isWrong(values):
            print("ignored: ",values)
        elif isConv(values):
            self.collec["conv"].append(values)
        elif isStuck(values):
            self.collec["stuck"].append(values)
        elif isNoConv(values):
            self.collec["noconv"].append(values)
        else:
            print("PerfCollector::add: WTF, this should not happen!!!")

    def stats(self):
        conv = len(self.collec["conv"])
        stuck = len(self.collec["stuck"])
        noconv = len(self.collec["noconv"])
        print ("    nb conv : ", conv)
        print ("    nb stuck : ", stuck)
        print ("    nb noconv : ", noconv)
        print ("    total : ", conv+stuck+noconv)

    def plot(self):
        plt.figure(1, figsize=(20,13))
        plt.xlabel("time steps")
        plt.ylabel("performance")
        plt.title("Performance")

        for values in self.collec["stuck"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="stuck", c='r')

        for values in self.collec["noconv"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="noconv", c='g')

        for values in self.collec["conv"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="conv", c='b')
        #plt.legend()
        plt.show()  
        plt.savefig(self.imageFolder + 'perf.svg', bbox_inches='tight')