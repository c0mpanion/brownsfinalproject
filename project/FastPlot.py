import gmplot
import numpy
import pandas
import multiprocessing
import time
from collections import deque
from itertools import chain
#import __main__



class FastPlotter(gmplot.GoogleMapPlotter):
    
    def dictionary_append(sendrlist1, sendrlist2, setDict):
        setDict.append((sendrlist1, sendrlist2))

    def threadedHeatMap(self, lats, lngs, threshold=10, radius=10, gradient=None, opacity=0.6, maxIntensity=1, dissipating=True):
        """
        :param lats: list of latitudes
        :param lngs: list of longitudes
        :param maxIntensity:(int) max frequency to use when plotting. Default (None) uses max value on map domain.
        :param threshold:
        :param radius: The hardest param. Example (string):
        :return:
        """
        settings = {}
        # Try to give anyone using threshold a heads up.
        if threshold != 10:
            warnings.warn("The 'threshold' kwarg is deprecated, replaced in favor of maxIntensity.")
        settings['threshold'] = threshold
        settings['radius'] = radius
        settings['gradient'] = gradient
        settings['opacity'] = opacity
        settings['maxIntensity'] = maxIntensity
        settings['dissipating'] = dissipating
        settings = self._process_heatmap_kwargs(settings)

        heatmap_points = []

        #Shared variables for each thread to communicate
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        # List of jobs
        jobs = []

        # Iterate through all zip codes and create a process
        for i in range(len(lats)):
            p = multiprocessing.Process(target=dictionary_append, args=(lats[i], lngs[i], heatmap_points))
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish
        for proc in jobs:
	    proc.join()

        self.heatmap_points.append((heatmap_points, settings))
     
	 	 
    def _process_heatmap_kwargs(self, settings_dict):
        settings_string = ''
        settings_string += "heatmap.set('threshold', %d);\n" % settings_dict['threshold']
        settings_string += "heatmap.set('radius', %d);\n" % settings_dict['radius']
        settings_string += "heatmap.set('maxIntensity', %d);\n" % settings_dict['maxIntensity']
        settings_string += "heatmap.set('opacity', %f);\n" % settings_dict['opacity']

        dissipation_string = 'true' if settings_dict['dissipating'] else 'false'
        settings_string += "heatmap.set('dissipating', %s);\n" % (dissipation_string)

        gradient = settings_dict['gradient']
        if gradient:
            gradient_string = "var gradient = [\n"
            for r, g, b, a in gradient:
                gradient_string += "\t" + "'rgba(%d, %d, %d, %d)',\n" % (r, g, b, a)
            gradient_string += '];' + '\n'
            gradient_string += "heatmap.set('gradient', gradient);\n"

            settings_string += gradient_string

        return settings_string	
if __name__ == '__main__':
       gmap = FastPlotter(37.766956, -122.438481, 13)
       path4 = [(37.433302 , 37.431257 , 37.427644 , 37.430303), (-122.14488, -122.133121, -122.137799, -122.148743)] 	
       gmap.threadedHeatMap(path4[0], path4[1], radius=40)
       gmap.draw("gmplot_map.html")




