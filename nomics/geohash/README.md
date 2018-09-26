# Geohash maps

This module will help create maps of geohashes, with shading for values, using KML files. For an example, see [this](https://drive.google.com/a/uber.com/open?id=1bIF3aWP5aQoN_26IPozl_i7mSkfQBfNW&usp=sharing) map of average hourly earnings in Chicago for the top 2000 (by trip count) geohash6s.

## Requirements

Make sure you have pandas, numpy, geohash-python, matplotlib, seaborn, and scipy installed. 

## Input file

Your input file should be a tab-separated .txt file with two columns, one for the geohash identifier and one for the value you want to map. You can turn a pandas df into this format by running: 

```
df.to_csv('output.txt', header=None, index=None, sep="\t")
```

## Creating the KML file

See the file `examples.py` for information on how to turn your input file into a kml and the various options for how to color a map

There's no real way to get a legend, but the code will spit out the min and max values to give you a sense of the range... you'll have to create your own legend. You can do this, sort of, by running the following: 

```
cmap = sns.diverging_palette(20, 130, n=100, as_cmap=True)
sns.palplot(cmap)
```

That will give you a nice picture of the range of colors. You'll have to manually label parts of it to communicate what the scale means.

## Mapping KML files

KML files can be used in Google Maps editor. Once you have a KML file ready, go to [My Google Maps](https://www.google.com/maps/d/u/0/) and hit "Create a new map." Then click "add layer" in the box on the left and select your KML file to upload. There is a limit of 2000 shapes, so you'll have to only do the top 2k geohashes (if you have more than 2k) 
