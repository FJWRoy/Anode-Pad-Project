import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def plot_scales():

	data = [6e6, 9e7, 5e8]
	xs = [4.5e1, 3e3, 1e4]
	fig, ax = plt.subplots(figsize=(12, 10))
	ax.scatter(xs, data, color='black', s=80)

	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_ylim([1e6, 1e9])
	ax.set_xlim([1, 3e4])
	ax.set_xlabel(r"# of channels / m$^{2}$", fontsize=20)
	ax.set_ylabel("max rate @ 1% double occupancy", fontsize=20)

	ax.grid(True, axis='both', which='minor', color='pink', alpha=0.6, linewidth=1)
	ax.grid(True, axis='both', which='major', linewidth=1)
	
	ax.get_xaxis().set_tick_params(labelsize=17, length=14, width=2, which='major')
	ax.get_xaxis().set_tick_params(labelsize=17,length=7, width=2, which='minor')
	ax.get_yaxis().set_tick_params(labelsize=17,length=14, width=2, which='major')
	ax.get_yaxis().set_tick_params(labelsize=17,length=7, width=2, which='minor')
	ax.tick_params(bottom=True, top=True, left=True, right=True, which='both')
	#plt.show()
	plt.savefig("anode_occupancy.jpg", bbox_inches='tight')



#sample a gaussian distribution. 
def sample_normal():
	mu, sigma = 0, 0.01 

	#the brute force way, sometimes
	#needed if mu and sigma vary 
	many_samples = []
	nsamples = 10000
	for n in range(nsamples):
		many_samples.append(np.random.normal(mu, sigma))


	#or all at once
	mu, sigma = 0.03, 0.01
	all_at_once = np.random.normal(mu, sigma, nsamples)


	#quick plot for demonstration
	fig, ax = plt.subplots()
	#how i like to do bins
	binwidth = 0.001 
	bins = np.arange(min(many_samples), max(many_samples), binwidth)
	ax.hist(many_samples, bins, histtype='step', fill = False, color='black', linewidth=2, alpha=0.8)
	bins = np.arange(min(all_at_once), max(all_at_once), binwidth)
	ax.hist(all_at_once, bins, histtype='step', fill = False, color='blue', linewidth=2, alpha=0.8)

	#then very good to have the binwidth in the y-axis label
	ax.set_ylabel("events per " + str(binwidth) + " bin width")
	plt.show()


if __name__ == "__main__":
	#plot_scales()
	sample_normal()