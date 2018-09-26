from geohashmaps import * 

# Input file should be at text file with two columns (no headers or indexes) separated by tabs
# For example, you could turn a pandas df into this using df.to_csv(output.txt, header=None, index=None, sep="\t")
input_file = 'earnings_example.txt'
output_file = 'earnings_example.kml'

# Initialize the KML maker using the input file
kml = KmlMaker(input_file)
# Load all the locations (loads them into a dict)
kml.loadLocations()

# There are a couple output options. First is continuous. Continuous isn't truly continuous, but does 100 colors. Can do "normed", which normalizes the values as (value - min)/(max-min) without imposing any sort of distribution of colors. This will get weird if there is one super high/low value (because most colors will be in the center of the distrib)
# 1) 
kml.continuous_kml_output(output_filename=output_file, normed=True)

# The second contionuos optoin is to do "normed=False", in which the values are not normalized but instead uses 0-100th ptiles. Most green colors will be the top 1%.  
kml.continuous_kml_output(output_filename=output_file, normed=False)

# Finally, can just do 4 colors for each quartile. Have to explicitly choose the color divides for the quartiles (lazy code)
kml.quartiles_kml_output(output_filename=output_file, color_divides=[22.99,23.73,25.16])
