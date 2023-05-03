/*-
 * #%L
 * ST-AnD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * ST-AnD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License 
 * along with ST-AnD.  If not, see <https://www.gnu.org/licenses/>.
 * #L%
 */

package qupath.ext.sptx2;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.locationtech.jts.geom.Geometry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.opencsv.CSVReader;

import ij.gui.Roi;
import ij.plugin.filter.ThresholdToSelection;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import javafx.geometry.Point2D;
import qupath.imagej.tools.IJTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.PathRootObject;
import qupath.lib.objects.TMACoreObject;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.AbstractDetectionPlugin;
import qupath.lib.plugins.DetectionPluginTools;
import qupath.lib.plugins.ObjectDetector;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.ImagePlane;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.interfaces.ROI;

/**
 * Plugin for loading 10x Visium Annotation 
 * 
 * @author Chao Hui Huang
 *
 */
public class VisiumAnnotation extends AbstractDetectionPlugin<BufferedImage> {
	
	final private static Logger logger = LoggerFactory.getLogger(VisiumAnnotation.class);
	
	private ParameterList params;

	private final List<String> operationList = new ArrayList<>(List.of("Spots Only", "Connected Clusters", "Surrounding Regions of Spots"));
	
	private String lastResults = null;
	
	/**
	 * Constructor.
	 */
	public VisiumAnnotation() {
		params = new ParameterList()
			// .addTitleParameter(lastResults)
			.addStringParameter("spatialFile", "Spatial File", "", "Spatial File")
			.addStringParameter("clusterFile", "Cluster File", "", "Cluster File")
			.addDoubleParameter("spotDiameter", "Spot Diameter", 55, GeneralTools.micrometerSymbol(), "Spot Diameter")
			.addDoubleParameter("minSpotDist", "Minimal Spot Distance", 100, GeneralTools.micrometerSymbol(), "Minimal Spot Distance")
			.addChoiceParameter("operation", "Operation", operationList.get(0), operationList, "Operation")
			.addBooleanParameter("rotatedImage", "Is the image rotated?", false, "Is the image rotated?")
			.addBooleanParameter("tissueRegionsOnly", "Tissue regions only?", true, "Tissue regions only?")
			.addBooleanParameter("estPxlSize", "Using estimated pixel size? (default: false)", false, "Using estimated pixel size?")			
			.addDoubleParameter("10xMPP", "microns per pixel on original 10x images (0 == ignored)", 0, GeneralTools.micrometerSymbol(), "Sampled microns per pixel on original images (0 == ignored)")	
			.addDoubleParameter("xShift", "X-Shift", 0, GeneralTools.micrometerSymbol(), "X-Shift")
			.addDoubleParameter("yShift", "Y-Shift", 0, GeneralTools.micrometerSymbol(), "X-Shift");
	}
	
	class AnnotationLoader implements ObjectDetector<BufferedImage> {
		
		@Override
		public Collection<PathObject> runDetection(final ImageData<BufferedImage> imageData, final ParameterList params, final ROI pathROI) throws IOException {
			
			final List<PathObject> pathObjects = new ArrayList<PathObject>();
			
			
			try {
				final ImageServer<BufferedImage> server = imageData.getServer();
				
		        final FileReader spatialFileReader = new FileReader(params.getStringParameterValue("spatialFile"));
		        final CSVReader spatialReader = new CSVReader(spatialFileReader);
		        final HashMap<String, List<Integer>> spatialHMap = new HashMap<String, List<Integer>>();
		     
		        String[] spatgialNextRecord;
		        List<Point2D> posList = new ArrayList<Point2D>();
		        
		        while ((spatgialNextRecord = spatialReader.readNext()) != null) {
		        	List<Integer> list = new ArrayList<Integer>();
		        	list.add(Integer.parseInt(spatgialNextRecord[1]));
		        	list.add(Integer.parseInt(spatgialNextRecord[2]));
		        	list.add(Integer.parseInt(spatgialNextRecord[3]));
		        	list.add(Integer.parseInt(spatgialNextRecord[4]));
		        	list.add(Integer.parseInt(spatgialNextRecord[5]));
		        	
		        	posList.add(new Point2D(Double.parseDouble(spatgialNextRecord[4]), Double.parseDouble(spatgialNextRecord[5])));
		        	
		        	spatialHMap.put(spatgialNextRecord[0], list);
		        }
		        
		        spatialReader.close();
		        
	        	final double spotDiameter = params.getDoubleParameterValue("spotDiameter");
	        	final double minSpotDist = params.getDoubleParameterValue("minSpotDist");		        
		        
	        	double minDistPxl = -1;
		        
		        for(int i = 0; i < posList.size(); i ++) {
		        	for(int j = 0; j < posList.size(); j ++) {
		        		final double distPxl = posList.get(i).distance(posList.get(j));
		        		minDistPxl = (i != j && (minDistPxl < 0 || distPxl < minDistPxl))? distPxl: minDistPxl;
		        	}
		        }
		        
	        	final double imagePixelSizeMicrons = params.getBooleanParameterValue("estPxlSize")? minSpotDist / minDistPxl: server.getPixelCalibration().getAveragedPixelSizeMicrons();
	        	
	        	ImageServerMetadata metadataNew = new ImageServerMetadata.Builder(server.getMetadata())
	        			.pixelSizeMicrons(imagePixelSizeMicrons, imagePixelSizeMicrons)
	    				.build();
	    		
	        	if (!server.getMetadata().equals(metadataNew)) imageData.updateServerMetadata(metadataNew);
	        	
	        	final FileReader clusterFileReader = new FileReader(params.getStringParameterValue("clusterFile"));
		        final CSVReader clusterReader = new CSVReader(clusterFileReader);
		        final HashMap<String, Integer> analysisHMap = new HashMap<String, Integer>();

		        String[] clusterNextRecord;
		        int clsNum = 0;
		        while ((clusterNextRecord = clusterReader.readNext()) != null) {
		            try {
		                final Integer cls = Integer.parseInt(clusterNextRecord[1]);
		                clsNum = cls > clsNum? cls: clsNum;
		                analysisHMap.put(clusterNextRecord[0], cls);
		            } catch (NumberFormatException nfe) {}
		        }
		        clusterReader.close();

		        final Color[] palette = new Color[clsNum+1];
	    	    for(int i = 0; i < clsNum+1; i++) palette[i] = Color.getHSBColor((float) i / (float) clsNum+1, 0.85f, 1.0f);
		    			        
		        Set<String> barcodeSet = spatialHMap.keySet();
	        	final double sampledPixelSizeMicrons = params.getDoubleParameterValue("10xMPP");	  
		        final double pixelSizeRatio = sampledPixelSizeMicrons > 0.0? sampledPixelSizeMicrons / imagePixelSizeMicrons: 1.0;
		        
		        if(params.getChoiceParameterValue("operation").equals(operationList.get(0))) {
		            final double xShiftMicrons = params.getDoubleParameterValue("xShift");
		            final double yShiftMicrons = params.getDoubleParameterValue("yShift");
					final int rad = (int)Math.round(0.5*spotDiameter/imagePixelSizeMicrons);
					final int dia = (int)Math.round(spotDiameter/imagePixelSizeMicrons);
					final int xshift = (int)Math.round(xShiftMicrons/imagePixelSizeMicrons);
					final int yshift = (int)Math.round(yShiftMicrons/imagePixelSizeMicrons);
					
			        for(String barcode: barcodeSet) {
			        	List<Integer> list = spatialHMap.get(barcode);
			        	
			        	final int in_tissue = list.get(0);
			        	final int pxl_row_in_fullres = (int)Math.round(list.get(3) * pixelSizeRatio);
			        	final int pxl_col_in_fullres = (int)Math.round(list.get(4) * pixelSizeRatio);
			        	
			        	if(params.getBooleanParameterValue("tissueRegionsOnly") && (in_tissue == 1) || !params.getBooleanParameterValue("tissueRegionsOnly")) {
			        		final int cluster = analysisHMap.containsKey(barcode)? analysisHMap.get(barcode): 0;
							final String pathObjName = barcode;
							final String pathClsName = barcode;
							
							ROI pathRoi = ROIs.createEllipseROI(pxl_col_in_fullres-rad+xshift, pxl_row_in_fullres-rad+yshift, dia, dia, null);
							
					    	final PathClass pathCls = PathClass.fromString(pathClsName);
							
					    	final PathAnnotationObject pathObj = (PathAnnotationObject) PathObjects.createAnnotationObject(pathRoi, pathCls);
					    	
					    	pathObj.setName(pathObjName);
					    	pathObj.setColor(palette[cluster].getRGB());
					    	
							final MeasurementList pathObjMeasList = pathObj.getMeasurementList();
							
							pathObjMeasList.close();
							pathObjects.add(pathObj);  
							
			        	}
			        }
		        }
		        else if(params.getChoiceParameterValue("operation").equals(operationList.get(1))) {
		            final double xShiftMicrons = params.getDoubleParameterValue("xShift");
		            final double yShiftMicrons = params.getDoubleParameterValue("yShift");
		            
			        for(String barcode: barcodeSet) {
			        	List<Integer> list = spatialHMap.get(barcode);
			        	
			        	final int in_tissue = list.get(0);
			        	final int pxl_row_in_fullres = (int)Math.round(list.get(3) * pixelSizeRatio);
			        	final int pxl_col_in_fullres = (int)Math.round(list.get(4) * pixelSizeRatio);
			        	
			        	if(params.getBooleanParameterValue("tissueRegionsOnly") && (in_tissue == 1) || !params.getBooleanParameterValue("tissueRegionsOnly")) {
			        		final int cluster = analysisHMap.containsKey(barcode)? analysisHMap.get(barcode): 0;
			        		
							final String pathObjName = barcode;
							final String pathClsName = barcode;
									
							final int rad = (int)Math.round(0.5*spotDiameter/imagePixelSizeMicrons);
							final int dia = (int)Math.round(spotDiameter/imagePixelSizeMicrons);
							
							final int rad2 = (int)Math.round(Math.sqrt(2.0  * Math.pow(0.5*spotDiameter/imagePixelSizeMicrons, 2.0)));
							final int dia2 = (int)Math.round(2.0 * Math.sqrt(2.0  * Math.pow(0.5*spotDiameter/imagePixelSizeMicrons, 2.0)));
							
							final int xshift = (int)Math.round(xShiftMicrons/imagePixelSizeMicrons);
							final int yshift = (int)Math.round(yShiftMicrons/imagePixelSizeMicrons);
							
							final ROI spotPathRoi = ROIs.createEllipseROI(pxl_col_in_fullres-rad+xshift, pxl_row_in_fullres-rad+yshift, dia, dia, null);		
							final ROI surroundingPathRoi = ROIs.createEllipseROI(pxl_col_in_fullres-rad2+xshift, pxl_row_in_fullres-rad2+yshift, dia2, dia2, null);	
							
							final Geometry spotPathGeom = spotPathRoi.getGeometry();
							final Geometry expandedPathGeom = surroundingPathRoi.getGeometry();
							
							final Geometry surroundingPathGeom = expandedPathGeom.difference(spotPathGeom);
							final ROI surroundingPathROI = GeometryTools.geometryToROI(surroundingPathGeom, ImagePlane.getDefaultPlane());
							
					    	final PathClass surroundingPathCls = PathClass.fromString(pathClsName);
							
					    	final PathAnnotationObject surroundingPathObj = (PathAnnotationObject) PathObjects.createAnnotationObject(surroundingPathROI, surroundingPathCls);

					    	surroundingPathObj.setName(pathObjName);
					    	surroundingPathObj.setColor(palette[cluster-1].getRGB());

							pathObjects.add(surroundingPathObj);
			        	}
			        }
		        }
		        else { 
			        final ImageServer<BufferedImage> imageServer = imageData.getServer();
					final int imageWidth = imageServer.getWidth();
					final int imageHeight = imageServer.getHeight();
		            final double xShiftMicrons = params.getDoubleParameterValue("xShift");
		            final double yShiftMicrons = params.getDoubleParameterValue("yShift");
					final int xshift = (int)Math.round(xShiftMicrons/imagePixelSizeMicrons);
					final int yshift = (int)Math.round(yShiftMicrons/imagePixelSizeMicrons);
					
			        for(int c = 1; c < clsNum; c ++) {
			        	final BufferedImage image = new BufferedImage(imageWidth/4, imageHeight/4, BufferedImage.TYPE_BYTE_GRAY);
			        	final Graphics2D graphic = image.createGraphics();
			            
			        	for(String barcode: barcodeSet) {
				        	List<Integer> list = spatialHMap.get(barcode);
				        	
				        	final int in_tissue = list.get(0);
				        	final int pxl_row_in_fullres = (int)Math.round(list.get(3) * pixelSizeRatio);
				        	final int pxl_col_in_fullres = (int)Math.round(list.get(4) * pixelSizeRatio);

				        	if(in_tissue == 1) {
				        		final Integer cluster = analysisHMap.get(barcode);
				        		final int r = (int)Math.round(Math.ceil(0.5*minSpotDist/Math.sqrt(3.0)*2.0/imagePixelSizeMicrons)/4.0);
						    	
				        		if (cluster == c) {
									final Polygon p = new Polygon();
									final int x = (int)Math.round((double)(pxl_col_in_fullres+xshift)/4.0);
									final int y = (int)Math.round((double)(pxl_row_in_fullres+yshift)/4.0);

								    for (int i = 0; i < 6; i++)								    	
									    if(params.getBooleanParameterValue("rotatedImage")) {
									    	
									    	p.addPoint((int) (x + r * Math.cos((i * 2 * Math.PI / 6)+(2 * Math.PI / 12))),
									    			(int) (y + r * Math.sin((i * 2 * Math.PI / 6)+(2 * Math.PI / 12))));
									    }
									    else {
									    	p.addPoint((int) (x + r * Math.cos((i * 2 * Math.PI / 6))),
									    			(int) (y + r * Math.sin((i * 2 * Math.PI / 6))));
									    }
								    graphic.setColor(Color.WHITE);
								    graphic.fillPolygon(p);		        			
				        		}
				        	}
			        	}
			        	
			        	final ByteProcessor bp = new ByteProcessor(image);
			        	bp.setThreshold(127.5, 255, ImageProcessor.NO_LUT_UPDATE);
			        	final Roi roiIJ = new ThresholdToSelection().convert(bp);
			        	
			        	 if (roiIJ != null) {
					    	final ROI roi = IJTools.convertToROI(roiIJ, 0, 0, 4, ImagePlane.getDefaultPlane());
					    	final PathClass pathCls = PathClass.fromString("cluster-"+String.valueOf(c));
							
					    	final PathObject p = PathObjects.createAnnotationObject(roi, pathCls);
					    	pathObjects.add(p);
					    }
			        }
		        }
			}
			catch(Exception e) {			
			}
			
			if (Thread.currentThread().isInterrupted())
				return null;
			
			if (pathObjects == null || pathObjects.isEmpty())
				lastResults =  "No regions detected!";
			else if (pathObjects.size() == 1)
				lastResults =  "1 region detected";
			else
				lastResults =  pathObjects.size() + " regions detected";
			
			logger.info(lastResults);
			
			return pathObjects;
		}
		
		
		@Override
		public String getLastResultsDescription() {
			return lastResults;
		}
		
		
	}
	
	
		@Override
	public ParameterList getDefaultParameterList(final ImageData<BufferedImage> imageData) {
		// boolean micronsKnown = imageData.getServer().getPixelCalibration().hasPixelSizeMicrons();
		// params.setHiddenParameters(!micronsKnown, "requestedPixelSizeMicrons", "minAreaMicrons", "maxHoleAreaMicrons");
		// params.setHiddenParameters(micronsKnown, "requestedDownsample", "minAreaPixels", "maxHoleAreaPixels");
		return params;
	}

	@Override
	public String getName() {
		return "Simple tissue detection";
	}

	@Override
	public String getLastResultsDescription() {
		return lastResults;
	}


	@Override
	public String getDescription() {
		return "Detect one or more regions of interest by applying a global threshold";
	}


	@Override
	protected void addRunnableTasks(ImageData<BufferedImage> imageData, PathObject parentObject, List<Runnable> tasks) {
		tasks.add(DetectionPluginTools.createRunnableTask(new AnnotationLoader(), getParameterList(imageData), imageData, parentObject));
	}


	@Override
	protected Collection<? extends PathObject> getParentObjects(final PluginRunner<BufferedImage> runner) {
		
		PathObjectHierarchy hierarchy = getHierarchy(runner);
		if (hierarchy.getTMAGrid() == null)
			return Collections.singleton(hierarchy.getRootObject());
		
		return hierarchy.getSelectionModel().getSelectedObjects().stream().filter(p -> p.isTMACore()).collect(Collectors.toList());

	}

	@Override
	public Collection<Class<? extends PathObject>> getSupportedParentObjectClasses() {
		// TODO: Re-allow taking an object as input in order to limit bounds
		// Temporarily disabled so as to avoid asking annoying questions when run repeatedly
		List<Class<? extends PathObject>> list = new ArrayList<>();
		list.add(TMACoreObject.class);

		list.add(PathRootObject.class);
		return list;
	}

}
