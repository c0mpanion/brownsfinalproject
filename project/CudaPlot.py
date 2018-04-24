import gmplot
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

class CudaPlot(gmplot.GoogleMapPlotter):
     
      def heatmap(self, lats, lngs, threshold=10, radius=10, gradient=None, opacity=0.6, maxIntensity=1, dissipating=True):
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

         #heatmap_points = []
         #for lat, lng in zip(lats, lngs):
         #   heatmap_points.append((lat, lng))

	 heatmapGPU = SourceModule (
	 """
	    __global__ void heatmapGPU(float *lats, float *lngs, float *heatmap_points)
	    {
		int idx = 0;
		int idy = 0;
		heatmap_points[0][idy] = lngs[idy];
		heatmpa_points[idx][1] = lats[idx];	
	    }	
	 """		
	 )
         func = mod.get_function("heatmapGPU")
	 func(lats_args, lngs_args, heatmap_points, block = (400,1,1))
	 heatmap_points = numpty.empty_like(heatmaps_points)
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

#if __name__ == '__main__':
