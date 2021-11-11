import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
mpl.style.use('tableau-colorblind10')
# the function, which is y = x^2 here

mpl.rcParams['text.usetex'] = True

def plot_taufunc():

	x = np.linspace(0,10,100)
	y = (np.e**x - 1)/(np.e**x+1)
	
	plt.plot(x,y)
	plt.rc('text', usetex=True)
	plt.xlabel('x')

	plt.ylabel(r'\frac{e^{\tau}-1}{e^{\tau}+1}')

	plt.savefig('./plots/plot_tau_func.png')
	plt.show()
	plt.clf()

plot_taufunc()