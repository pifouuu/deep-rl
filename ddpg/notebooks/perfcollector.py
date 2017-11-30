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

def is_stuck(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val <= -19.0)

def isNoConv(values):
    valpair = values[len(values)-1]
    val = valpair[1]
    return (val > -19.0 and val <80.0)

class PerfCollector():
    def __init__(self, image_folder, **kwargs):
        self.collec = {}
        self.image_folder = image_folder

    def init(self, delta):
        sort_dict = {}
        sort_dict["conv"] = []
        sort_dict["stuck"] = []
        sort_dict["noconv"] = []
        self.collec[delta] = sort_dict

    def add(self, delta, values):
        if isWrong(values):
            print("ignored: ",values)
        elif isConv(values):
            self.collec[delta]["conv"].append(values)
        elif is_stuck(values):
            self.collec[delta]["stuck"].append(values)
        elif isNoConv(values):
            self.collec[delta]["noconv"].append(values)
        else:
            print("PerfCollector::add: WTF, this should not happen!!!")

    def stats(self):
        for i in self.collec.keys():
            print (i,":")
            conv = len(self.collec[i]["conv"])
            stuck = len(self.collec[i]["stuck"])
            noconv = len(self.collec[i]["noconv"])
            print ("    nb conv : ", conv)
            print ("    nb stuck : ", stuck)
            print ("    nb noconv : ", noconv)
            print ("    total : ", conv+stuck+noconv)

    def plot(self, delta):
        plt.figure(1, figsize=(20,13))
        plt.xlabel("time steps")
        plt.ylabel("performance")
        plt.title("Performance for delta = {}".format(delta))

        for values in self.collec[delta]["stuck"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="stuck", c='r')

        for values in self.collec[delta]["noconv"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="noconv", c='g')

        for values in self.collec[delta]["conv"]:
            if len(values)>1:
                x = []
                y = []
                for i in range(len(values)):
                    x.append(values[i][0])
                    y.append(values[i][1])
                plt.plot(x,y, label="conv", c='b')
        #plt.legend()
        #plt.show()
        plt.savefig(self.image_folder + 'perf_' + delta + '.svg', bbox_inches='tight')
