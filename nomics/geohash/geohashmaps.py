#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import geohash
from kml_template import header, footer, box_template, red_template, orange_template, yellow_template, green_template
from kml_template_continuous import header_cont, footer_cont, box_template_cont, raw_style_cont
import pandas as pd 
import numpy as np
from scipy import stats
from matplotlib.colors import to_hex
import seaborn as sns


class KmlMaker(object):
    def __init__(self,filename):
        self.filename = filename
        self.locations = {}

    def makeGoogleEarthBox(self,geo):
        bbox = geohash.bbox(geo)
        lowerleft = "%s,%s,1"%(bbox['w'],bbox['s'])
        upperleft = "%s,%s,1"%(bbox['w'],bbox['n'])
        lowerright = "%s,%s,1"%(bbox['e'],bbox['s'])
        upperright = "%s,%s,1"%(bbox['e'],bbox['n'])
        polygon = "%s %s %s %s %s"%(lowerleft,upperleft,upperright,lowerright,lowerleft)
        return polygon

    def loadLocations(self):
        counts = {}
        for line in open(self.filename,"rU"):
            (geohashcode, count) = line.strip().split("\t")
            self.locations[geohashcode] = count
        print 'Done loading geohashcode counts.'

    def get_template(self,input_value,color_ramp=[4,7,15], reverse_order=True):
        low = float(color_ramp[0])
        medium = float(color_ramp[1])
        high = float(color_ramp[2])
        input_value = float(input_value)
        template = box_template
        if reverse_order: 
            if input_value < low:
                template = red_template 
            elif input_value < medium:
                template = orange_template 
            elif input_value < high:
                template = yellow_template 
            else:
                template = green_template 
        else:
            if input_value < low:
                template = green_template
            elif input_value < medium:
                template = yellow_template
            elif input_value < high:
                template = orange_template
            else:
                template = red_template
        return template

    def quartiles_kml_output(self,title='Location Indicators',output_filename ='output_advanced.kml',color_divides=[1,500,1000],polygon_height=1000):
        f = open(output_filename,"w")
        header2 = header.replace('__title__',title)
        f.write(header2)
        for key,value in self.locations.items():
            #value = int(value)
            #if value < 1: continue
            t = self.get_template(value,color_ramp=color_divides)
            poly = self.makeGoogleEarthBox(key)
            poly = poly.replace("elevation",1)
            t = t.replace("__name__",key)
            t = t.replace("__coordinates__",poly)
            t = t.replace('__title__',title)
            f.write(t+"\n")
        f.write(footer)

    def create_all_ptile_templates(self, lower_bound=-98, upper_bound=130, n=101, opacity="BF"):
        # Create color map
        cmap = list(sns.diverging_palette(lower_bound, upper_bound, n=n))
        cmap = map(to_hex, cmap)
        
        # Fill the raw template
        rv = ''
        for i in np.arange(0,n):
            name = 'contstyle%d_' % i
            color = '#'+str(opacity)+cmap[i][1:]
            rv = rv + raw_style_cont.format(**{'name':name, 'color':color}) + '\n'
        return rv


    def continuous_kml_output(self, title="Location Indicators", output_filename='output.kml', normed=False):
        # Load locations into a dataframe
        locations = pd.DataFrame(self.locations.items())
        locations.columns = ['geohash', 'value']
        locations['value'] = locations['value'].astype(float)
        print "Lowest value:  %0.2f" % (locations.value.min())
        print "Highest value: %0.2f" % (locations.value.max())
        
        # Either do the ptile (by default) or the normed value. Needs to be an int 0-100 (can't do true continuous...)
        if normed:
            locations['color_val'] = locations['value'].apply(lambda x: int(100*(x-locations.value.min())/(locations.value.max()-locations.value.min())))
        else:
            # Find the percentile score of values (where they fall in distribution of all values)
            locations['color_val'] = locations['value'].apply(lambda x: int(stats.percentileofscore(locations['value'], x)))

        # Get the cont colors template 
        colors_template = self.create_all_ptile_templates()
        
        # Write the header and the colors template to the output file
        f = open(output_filename, "w")
        header2 = header_cont.format(**{'title':title})
        f.write(header2)
        f.write(colors_template)

        # For each location, write the placemark of the corresponding ptile color to the output
        for index, row in locations.iterrows():
            poly = self.makeGoogleEarthBox(row['geohash'])
            style = 'contstyle%d_' % (row['color_val'])
            template = box_template_cont.format(**{'name':row['geohash'], 'style':style, 'coordinates':poly})
            f.write(template+"\n")
        
        # Write the footer, then close
        f.write(footer)
        f.close()

