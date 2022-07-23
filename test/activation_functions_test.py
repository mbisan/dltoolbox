import dltoolbox.activationFunctions as af

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 150

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')

fig, axs = plt.subplots(1,2)

linspace = np.arange(-5,5,0.01)

sigmoid = af.sigmoid.f(linspace)
dsigmoid = af.sigmoid.df(linspace)

relu = af.relu.f(linspace)
drelu = af.relu.df(linspace)

axs[0].set_title(r"Sigmoidea")
axs[0].set(ylabel = r"$f(x) = \frac{1}{1+e^{-x}}$, $f'(x)=f(x)(1-f(x))$")
axs[0].set(xlabel = r"$x$")
axs[0].plot(linspace, sigmoid, linspace, dsigmoid)
axs[1].set_title(r"Zuzentzailea")
axs[1].set(ylabel = r"$f(x) = x\mathbf{1}_{\{x>0\}}(x)$, $f'(x)=\mathbf{1}_{\{x>0\}}(x)$" )
axs[1].set(xlabel = r"$x$")
axs[1].plot(linspace, relu, linspace, drelu)

plt.show()