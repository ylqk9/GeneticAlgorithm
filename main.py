import numpy
import pandas
from imp import reload
import func
reload(func)
from func import *

gam = GeneticAlgorithm(ngene = 50, ngeneration=150, mutation=0.001)
gam.Iterate()
