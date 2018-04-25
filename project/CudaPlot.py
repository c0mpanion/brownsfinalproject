import gmplot
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import pandas



#import __main__



class CudaPlotter(gmplot.GoogleMapPlotter):

     # def __init__(self, center_lat, center_lng, zoom, data_frame, apikey=''):
     # 	 gmplot.GoogleMapPlotter.__init__(self, center_lat, center_lng, zoom, apikey='')
     #	 self.df = data_frame	
     
      def heatmap_GPU(self, lats, lngs, threshold=10, radius=10, gradient=None, opacity=0.6, maxIntensity=1, dissipating=True):
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
 
	 heatmap_points = numpy.array(heatmap_points)
	 lats = numpy.array(lats)
	 lngs = numpy.array(lngs)
	 #GPU Implementation
	 #Change lists to Numpy Arrays 
         heatmap_points = heatmap_points.astype(numpy.float32)
	 tempLats = lats.astype(numpy.float32)
	 tempLngs = lngs.astype(numpy.float32)
         
	 print("works here")

	 		
	 #Allocate memory in GPU  
	 lats_gpu = cuda.mem_alloc(tempLats.nbytes)
	 lngs_gpu = cuda.mem_alloc(tempLngs.nbytes)
	 points_gpu = cuda.mem_alloc(tempLats.nbytes*2)

	 print ("works here 2")
	 #Transfer data from CPU(Host) to GPU(Device) 
	 cuda.memcpy_htod(lats_gpu, tempLats)
	 cuda.memcpy_htod(lngs_gpu, tempLngs)
	 #cuda.memcpy_htod(points_gpu, heatmap_points)

	 N = tempLats.nbytes
	 print("works here 3")
	 #C Implementation of commented out code above
	 heatmapGPU = SourceModule("""
	    __global__ void heatmapGPU(float *lats, float *lngs, float **heatmap_points)
	    {
		printf("works here in module 1");
		int N = (sizeof(lats) - sizeof(int))- 1; 
		int idx = blockIdx.x;
		int idy = blockIdx.y;
	        if (idx < N)
			heatmap_points[idx][0] = lats[idx]; 
		if (idy < N) 
			heatmap_points[idy][1] = lngs[idy];
		printf("works here in module 2");
	    }	
	    """)
	
  	 print("works here 4")
 	 #Execute kernel function using GPU versions of the "Lists" as args
         func = heatmapGPU.get_function("heatmapGPU")
	 func(lats_gpu, lngs_gpu, points_gpu, block = (400,1,1))

	 print("works here 5")
	 #Bring back data to the CPU from temp var and into constructor variable heatmap_points
	 tempPoints = numpty.empty_like(heatmaps_points)
	 cuda.memcpy_dtoh(tempPoints, points_gpu)

	 print("works here 6")
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

     # Test code for checking to see if results are the same or different from the parent class
     
     gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13)
     cudamap = CudaPlotter(37.766956, -122.438481, 13)	

     path4 = [(37.433302 , 37.431257 , 37.427644 , 37.430303), (-122.14488, -122.133121, -122.137799, -122.148743)]
     
     #Both Heatmap function require Python List type in the args	
     gmap.heatmap(path4[0], path4[1], threshold=10, radius=40, dissipating=False)
     cudamap.heatmap_GPU(path4[0], path4[1], threshold=10, radius=40, dissipating=False)	
     # Draw
     gmap.draw("gmplot_map.html")
     cudamap.draw("cuda_plot.html")	    
