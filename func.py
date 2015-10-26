import pandas
import numpy
from sklearn.linear_model import LinearRegression

def Load(filename):
	data = pandas.read_csv(filename, sep = '\t')
	y = data['target'].as_matrix()
	data_string = data.select_dtypes(include = ["object"])
	data_num = data.drop(['target'], axis = 1).select_dtypes(exclude = ["object"])
	x = pandas.get_dummies(data_string.icol(0)).as_matrix()[:,:-1]
	for i in range(1,data_string.shape[1]):
		x1 = pandas.get_dummies(data_string.icol(i)).as_matrix()[:,:-1]
		x = numpy.hstack((x,x1))
	#fill data since data.dropna() kills too many
	x = numpy.hstack((x,data_num.fillna(data_num.mean()).as_matrix()))
	return x, y

def GeneMake(nsamples, nluci):
	gene = numpy.ndarray((nsamples, nluci))
	gene[:,0:nluci/2] = 0
	gene[:,nluci/2:] = 1
	for i in range(0, nsamples):
		numpy.random.shuffle(gene[i,:])
	return gene

class GeneEval(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.n = y.shape[0]
		self.s = x.shape[1]
		self.lr = LinearRegression()
	def Score(self, gene):
		self.lr.fit(self.x[:,gene == 1], self.y)
		AIC = self.n*numpy.log(self.lr.residues_/self.n) + 2*(sum(gene) + 2)
		return AIC

def GeneCrossover(g1, g2):
	#numpy arrays are passed by ref
	nl = g1.shape[0]
	#crosspt = nl/2
	crosspt = numpy.random.randint(nl)
	temp = numpy.copy(g2[0:crosspt])
	numpy.copyto(g2[0:crosspt], g1[0:crosspt])
	numpy.copyto(g1[0:crosspt], temp)
	return

def GeneNextGen(gene, AIC):
	p = len(AIC)
	rank = p - AIC.argsort().argsort()
	fitness = 2.*rank/(p*(p+1.))
	genenext = numpy.copy(gene)
	for i in range(0, int(p/2)):
		pa = numpy.random.choice(p, p=fitness)
		pb = numpy.random.randint(p)
		genea = numpy.copy(gene[pa,:])
		geneb = numpy.copy(gene[pb,:])
		GeneCrossover(genea, geneb)
		genenext[2*i,:] = genea
		genenext[2*i+1,:] = geneb
	return genenext

def GeneMutation(gene, rate):
	mask = numpy.random.choice(2, size = len(gene), p = (1-rate, rate)).astype(bool)
	gene[mask] = numpy.abs(gene[mask] - 1)

class GeneticAlgorithm(object):
	def __init__(self, ngene = 500, ngeneration = 20, mutation = 0.02):
		self.ncreature = ngene
		self.niter = ngeneration
		self.mutrate = mutation
		x, y = Load("train.txt")
		self.gene = GeneMake(self.ncreature, x.shape[1])
		self.model = GeneEval(x, y)
		self.allscr = []
		self.bestscr = []
		self.bestresidual = []
		self.bestgene = []
	def Iterate(self):
		for iter in range(0, self.niter):
			scr = []
			for i in range(0, self.ncreature):
				GeneMutation(self.gene[i,:], self.mutrate)
				scr.append(self.model.Score(self.gene[i,:]))
			self.allscr.append(scr)
			self.gene = GeneNextGen(self.gene, numpy.array(scr))
			self.bestscr.append(min(scr))
			bestind = scr.index(min(scr))
			self.bestresidual.append(self.model.Score(self.gene[bestind,:]))
			self.bestgene.append(self.gene[bestind,:])
			print(iter, self.bestgene[-1].sum(), self.bestscr[-1], self.bestresidual[-1], 5000*numpy.log(self.bestresidual[-1]/5000))
			#print(numpy.sum(self.gene, axis = 1))
