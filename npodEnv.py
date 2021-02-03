from rpy2.robjects.packages import importr
from rpy2.robjects.packages import STAP
from rpy2.robjects import r
from rpy2.robjects import FloatVector
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
importr("NPOD")

#https://stackoverflow.com/questions/24152160/converting-an-rpy2-listvector-to-a-python-dictionary
def r_list_to_py_dict(r_list):
    converted = {}
    for name in r_list.names:
        val = r_list.rx(name)[0]
        if isinstance(val, ro.vectors.DataFrame):
            converted[name] = pandas2ri.ri2py_dataframe(val)
        elif isinstance(val, ro.vectors.ListVector):
            converted[name] = r_list_to_py_dict(val)
        elif isinstance(val, ro.vectors.FloatVector) or isinstance(val, ro.vectors.StrVector):
            if len(val) == 1:
                converted[name] = val[0]
            else:
                converted[name] = list(val)
        else: # single value
            converted[name] = val
    return converted

class NpodEnv:
    def __init__(self):
        self.pkdata_file = "data/data_1comp_neely.csv"
        self.sim_file = ""
        self.a=[0.001, 50]
        self.b=[2, 250]
        model_str = """
        model <- function(theta,t){
            #t<c(0.5,1,2,3,4,5,...)
            #this equation assumes a dose of 500mg, an infusion of 0.5h and a time vector like c(0.5,...)
            x05<-(500/(0.5*theta[1]*theta[2]))*(1-exp(-theta[1]*t[1]))
            val <- (x05)*exp(-theta[1]*(t[-1]-0.5))
            return(c(x05,val))
        }
        """
        my_fn = STAP(model_str, "my_fn")
        self.model = my_fn.model
        self.individuals = r['seq'](1,51)
        self.c0 = 0
        self.c1 = 0.1
        self.cache_folder_name = "1comp_neely"
    
    def run(self):
        result_dic = r_list_to_py_dict(r['NPOD'](
            self.sim_file,
            self.pkdata_file,
            list(map(FloatVector, [self.a,self.b])),
            self.individuals,
            model = self.model,
            c0 = self.c0,
            c1 = self.c1,
            size_theta0 = 10000,
            cache_folder_name = self.cache_folder_name))
        del result_dic['PSI']
        return({**result_dic, 'a': self.a, 'b': self.b, 'c0': self.c0, 'c1': self.c1})
    def ver(self):
        return(0.2)