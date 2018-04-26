from gmplot import gmplot
import numpy
import pandas
import multiprocessing
import time
from collections import deque

class FastPlotter(gmplot.GoogleMapPlotter):
    
    def threadedHeatMap(self, lats, lngs, threshold=10, radius=10, gradient=None, opacity=0.6, maxIntensity=1, dissipating=True):
        """
        Modified version of the heatmap function from the parent class GoogleMapPlotter. Raw Lists are passed in for lats and lngs.
        :param lats: list of latitudes
        :param lngs: list of longitudes
        :param maxIntensity:(int) max frequency to use when plotting. Default (None) uses max value on map domain.
        :param threshold:
        :param radius: The hardest param. Example (string):
        :return:
        """
        settings = {}
        # Trying to give anybody using threshold a heads up.
        # if threshold != 10:
        #     warnings.warn("The 'threshold' kwarg is deprecated, replaced in favor of maxIntensity.")
        settings['threshold'] = threshold
        settings['radius'] = radius
        settings['gradient'] = gradient
        settings['opacity'] = opacity
        settings['maxIntensity'] = maxIntensity
        settings['dissipating'] = dissipating
        settings = self._process_heatmap_kwargs(settings)

        # Shared variables for each thread to communicate
        manager = multiprocessing.Manager()
        sharedDict = manager.dict()
        # heatmap_points = []

        # List of jobs
        jobs = []

        # Iterate through the indexes of one of the lists and create a process

        for i in range(len(lats)):
            p = multiprocessing.Process(target=self.dictionary_append, args=(i, lats[i], lngs[i], sharedDict))
            jobs.append(p)
            p.start()

        # Wait for all jobs to finish
        for proc in jobs:
            proc.join()

        # print(sharedDict)

        # Convert ProxyDict to regular Dict and
        nonSharedDict = dict(sharedDict)
        heatmap_points = nonSharedDict.values()

        # print(heatmap_points)
        self.heatmap_points.append((heatmap_points, settings))

    def dictionary_append(self, process_num, sendrlist1, sendrlist2, sharedDict):
        tempList = deque()
        tempList.append((sendrlist1, sendrlist2))
        # Create paired lists of latitude and longitude
        sharedDict[process_num] = numpy.array(tempList).flatten().tolist()

        return sharedDict

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

# if __name__ == '__main__':
#        gmap = FastPlotter(37.766956, -122.438481, 13)
#        path4 = [(37.433302 , 37.431257 , 37.427644 , 37.430303), (-122.14488, -122.133121, -122.137799, -122.148743)]
#        gmap.threadedHeatMap(path4[0], path4[1], radius=40)
#        gmap.draw("gmplot_map.html")


