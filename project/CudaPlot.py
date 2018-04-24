import gmplot
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

class CudaPlotter(gmplot.GoogleMapPlotter):
     
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
	 

	 #Sequential CPU Implementation  
         #heatmap_points = []
         #for lat, lng in zip(lats, lngs):
         #   heatmap_points.append((lat, lng))

	  
	 #GPU Implementation
	 #Change lists to Numpy Arrays 
         heatmap_points = heatmap_points.astype(numpy.float32)
	 tempLats = lats.astype(numpy.float32)
	 tempLngs = lngs.astype(numpy.float32)
	
	 #Allocate memory in GPU  
	 lats_gpu = cuda.mem_alloc(tempLats.nbytes)
	 lngs_gpu = cuda.mem_alloc(tempLngs.nbytes)
	 points_gpu = cuda.mem_alloc(heatmap_points.nbytes)

	 #Transfer data from CPU(Host) to GPU(Device) 
	 cuda.memcpy_htod(lats_gpu, tempLats)
	 cuda.memcpy_htod(lngs_gpu, tempLngs)
	 cuda.memcpy_htod(points_gpu, heatmap_points)

	 #C Implementation of commented out code above
	 heatmapGPU = SourceModule (
	 """
	    __global__ void heatmapGPU(float *lats, float *lngs, float *heatmap_points)
	    {
		int idx = 0;
		int idy = 0;
		heatmap_points[idx][0] = lats[idx]; #Column 0 is for lats values 
		heatmpa_points[idy][1] = lngs[idy]; #Column 1 is for lngs values

	    }	
	 """		
	 )
	
 	 #Execute kernel function using GPU versions of the "Lists" as args
         func = mod.get_function("heatmapGPU")
	 func(lats_gpu, lngs_gpu, points_gpu, block = (400,1,1))

	 #Bring back data to the CPU from temp var and into constructor variable heatmap_points
	 tempPoints = numpty.empty_like(heatmaps_points)
	 cuda.memcpy_dtoh(tempPoints, points_gpu)

	 #Convert back to list for following line
	 heatmaps_points = tempPoints.tolist()

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

     #gmap = CudaPlot.gmplot.GoogleMapPlotter.from_geocode("San Francisco")
     # Place map
     gmap = CudaPlotter(37.766956, -122.438481, 13)

     

     # Draw
     gmap.draw("my_map.html")
     #gmap.draw("my_map.html")
