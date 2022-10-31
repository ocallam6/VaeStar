# VAE Star

This directory aims to create an algorithm to explore stellar parameters using data science methods and variational autoencoders.

1. data_prep -  This file takes a cluster and turns the data into a form usable by other files. The form of the input data is in blocks which are designed to line up with the isochrones along their final axis.

2. vaestar - this is a variational autoencoder which aims to make posterior estimates on the stellar parameters. It is not done yet and needs a lot of work

3. dist_fun_age - this is a notebook which aims to compute the distance between an isochrone and a point in two different ways. Both will assume the value of feh as supplied by Gaia. Furthermore the first method will calculate the perpendicular distance to the isochrone as a function of age. Main issues with this method is due to ages making isochrones vastly different and points showing up in simply incorrect places. To combat this we will look at a second method. Which will essentially do the same, except use the value of logg as derived by Gaia and find the distance of this to the isochrone.