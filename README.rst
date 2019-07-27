Bayesian parameter estimation of RHIC yields
============================================

Attribution: This repository is based on Jonah Bernhard's Bayesian parameter estimation package for relativistic heavy-ion collisions: `qcd.phy.duke.edu/hic-param-est <http://qcd.phy.duke.edu/hic-param-est>`__.

Quick start
-----------

Follow the installation instructions for hic-param-est: `qcd.phy.duke.edu/hic-param-est <http://qcd.phy.duke.edu/hic-param-est/#installation>`__.

This project also requires the latest version of trento: `qcd.phy.duke.edu/trento <http://qcd.phy.duke.edu/trento/installation.html>`__.

Once both packages are installed, the first step is to generate the computer experiment design.
Take a look at `src.design` to change the number of design points, parameters, and parameter ranges.
Generate the design points using ::

   python3 -m src.design design

The next step is to evaluate the trento initial condition model at each design point. ::

   python3 -m src.trento

Running the events will take some time, depending on the number of events and systems specified.
On my relatively old quad-core machine, running the events took about 24 hours.

After the events are created, following the remaining steps explained in Jonah's hic-param-est documentation.
Specifically,

Build the experimental observables dictionary ::

   python3 -m src.expt

Pre-process the trento event data to calculate the model observables ::

   python3 -m src.model

Train the emulator on the trento observables ::

   python3 -m src.emulator

Sample the Bayesian posterior distribution using MCMC ::

   python3 -m src.mcmc --nwalkers 1000 --nburnsteps 1000 10000

And finally visualize the results of the analysis ::

   python3 -m src.plots
