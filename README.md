# Allsky-Identifier
### To Identify Stars in an Allsky-image and determine their FWHM

Needs to be adapted depending on the orientation of one camera (images).<br/>
In parameters.py change location and other parameters according to location/requirements.<br/>

Locates the Stars in an Image:<br/>
<img src="images/star_ident_example.png" width = 600>

Then one can plot the row/column - wise distributions of measured counts of the pixels.<br/>
Gaussians will be fitted over this to determine the FWHM and size of the star/object.<br/>
<img src="images/mars_hists.png" width = 400>
<img src="images/mars_ident.png" width = 400>
<br/>
<br/>
Optionally can also produce image like this for orientation of night sky.<br/>
<img src="images/orientedsky.png" width = 400>
