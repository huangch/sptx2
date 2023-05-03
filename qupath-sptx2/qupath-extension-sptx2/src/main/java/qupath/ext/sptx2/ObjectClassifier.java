/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2014 - 2016 The Queen's University of Belfast, Northern Ireland
 * Contact: IP Management (ipmanagement@qub.ac.uk)
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

package qupath.ext.sptx2;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.plugins.AbstractTileableDetectionPlugin;
import qupath.lib.plugins.ObjectDetector;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;

/**
 * Default command for cell detection within QuPath, assuming either a nuclear or cytoplasmic staining.
 * <p>
 * To automatically classify cells as positive or negative along with detection, see {@link PositiveCellDetection}.
 * <p>
 * To quantify membranous staining see {@link WatershedCellMembraneDetection}.
 * 
 * @author Pete Bankhead
 *
 */
public class ObjectClassifier extends AbstractTileableDetectionPlugin<BufferedImage> {
	
	protected boolean parametersInitialized = false;
	 private static Semaphore semaphore;
	// CELLPOSE PARAMETERS

    protected static SpTx2Setup sptx2Setup = SpTx2Setup.getInstance();

	private final static Logger logger = LoggerFactory.getLogger(ObjectClassifier.class);
	protected static SpTx2Setup sptxSetup = SpTx2Setup.getInstance();
	
	private static int m_samplingFeatureSize;
	private static double m_modelPreferredPixelSizeMicrons;
	private static String m_modelName;
	private static List<PathObject> m_availabelObjList;
	private static int m_batchSize;
    protected static String model;

    ParameterList params;
	
	static class ObjClassifier implements ObjectDetector<BufferedImage> {
		private List<PathObject> pathObjects = null;
			
		@Override
		public Collection<PathObject> runDetection(final ImageData<BufferedImage> imageData, ParameterList params, ROI pathROI) throws IOException {
			// final AtomicBoolean success = new AtomicBoolean(false);

			try {
				if (pathROI == null) throw new IOException("Object classification requires a ROI!");
				
				final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();
				final String serverPath = server.getPath();			
				final RegionRequest tileRegion = RegionRequest.createInstance(server.getPath(), 1.0, pathROI);
				
		    	pathObjects = Collections.synchronizedList(new ArrayList<PathObject>());
				
				m_availabelObjList.parallelStream().forEach( objObject -> {
					final ROI objRoi = objObject.getROI();
					final int x = (int)(0.5+objRoi.getCentroidX());
					final int y = (int)(0.5+objRoi.getCentroidY());
					
					if(tileRegion.contains(x, y, 0, 0)) {
						synchronized(pathObjects) {
							pathObjects.add(objObject);
						}
					}
				});			
				
				if(pathObjects.size() > 0) {
					// Create a temporary directory for imageset
					final String timeStamp = Long.toString(System.nanoTime());
					
					final Path imageSetPath = Files.createTempDirectory("sptx2-classification_imageset-" + timeStamp + "-");
					final String imageSetPathString = imageSetPath.toAbsolutePath().toString();
                    imageSetPath.toFile().deleteOnExit();
                    
        			final String modelLocationStr = sptxSetup.getObjclsModelLocationPath();
        			final String modelPathStr = Paths.get(modelLocationStr, m_modelName+".pt").toString();
        			
                    final Path resultPath = Files.createTempFile("sptx2-classification_result-" + timeStamp + "-", ".json");
                    final String resultPathString = resultPath.toAbsolutePath().toString();
                    resultPath.toFile().deleteOnExit();
                    
                    IntStream.range(0, pathObjects.size()).parallel().forEachOrdered(i -> { 
                    // for(int i = 0; i < pathObjects.size(); i ++) {
						final PathObject objObject = pathObjects.get(i);
						final ROI objRoi = objObject.getROI();
					    final int x0 = (int) (0.5 + objRoi.getCentroidX() - ((double)m_samplingFeatureSize / 2.0));
					    final int y0 = (int) (0.5 + objRoi.getCentroidY() - ((double)m_samplingFeatureSize / 2.0));
					    final RegionRequest objRegion = RegionRequest.createInstance(serverPath, 1.0, x0, y0, m_samplingFeatureSize, m_samplingFeatureSize);
						
						try {
							// Read image patches from server
							final BufferedImage img = (BufferedImage)server.readRegion(objRegion);
						    
							//  Assign a file name by sequence
							final String imageFileName = Integer.toString(i)+".png";
							
							// Obtain the absolute path of the given image file name (with the predefined temporary imageset path)
							final Path imageFilePath = Paths.get(imageSetPathString, imageFileName);
							
							// Make the image file
							File imageFile = new File(imageFilePath.toString());
							ImageIO.write(img, "png", imageFile);
						} 
						catch (IOException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					});
                    // }
					
                    if(semaphore != null) semaphore.acquire();
					// Create command to run
			        VirtualEnvironmentRunner veRunner;
			        veRunner = new VirtualEnvironmentRunner(sptx2Setup.getEnvironmentNameOrPath(), sptx2Setup.getEnvironmentType(), ObjectClassifier.class.getSimpleName());
				
			        // This is the list of commands after the 'python' call
			        final String script_path = Paths.get(sptx2Setup.getScriptLocationPath(), "object_classification.py").toString();
			        List<String> sptx2Arguments = new ArrayList<>(Arrays.asList("-W", "ignore", script_path, "eval", resultPathString));
			        
			        sptx2Arguments.add("--model_file");
			        sptx2Arguments.add("" + modelPathStr);
			        veRunner.setArguments(sptx2Arguments);
			        
			        sptx2Arguments.add("--image_path");
			        sptx2Arguments.add("" + imageSetPathString);
			        veRunner.setArguments(sptx2Arguments);
			        
			        sptx2Arguments.add("--batch_size");
			        sptx2Arguments.add("" + m_batchSize);
			        veRunner.setArguments(sptx2Arguments);
			        
			        // Finally, we can run Cellpose
			        final String[] logs = veRunner.runCommand();
			        for (String log : logs) logger.info(log);
			        // veRunner.runCommand(); 
			        // logger.info("Object classification command finished running");
					
					if(semaphore != null) semaphore.release();
					
					final FileReader resultFileReader = new FileReader(new File(resultPathString));
					final BufferedReader bufferedReader = new BufferedReader(resultFileReader);
					final Gson gson = new Gson();
					final JsonObject jsonObject = gson.fromJson(bufferedReader, JsonObject.class);
					
					final Boolean ve_success = gson.fromJson(jsonObject.get("success"), new TypeToken<Boolean>(){}.getType());
					assert ve_success: "object_classification.py returned failed";
					
					final List<Double> ve_predicted = gson.fromJson(jsonObject.get("predicted"), new TypeToken<List<Double>>(){}.getType());
					assert ve_predicted != null: "object_classification.py returned null";
					assert ve_predicted.size() == pathObjects.size(): "object_classification.py returned wrong size";
					
					for(int i = 0; i < ve_predicted.size(); i ++) {
						final PathClass pc = PathClass.fromString(m_modelName+":prediction:"+Integer.toString(1+ve_predicted.get(i).intValue()));
						pathObjects.get(i).setPathClass(pc);
					}
					
//					// IntStream.range(0, predicted.size()).parallel().forEach(i -> {
//					for(int i = 0; i < predicted.size(); i ++) {
//						final PathObject objObject = pathObjects.get(i);
//						final int r = predicted.get(i).intValue();
//						final MeasurementList measList = objObject.getMeasurementList();
//							
//						for(int j = 0; j < m_labelList.length; j ++) {
//							measList.put(m_taskLabel + ":" + m_labelList[j], r+m_additionalLabelList.length == j? 1: 0);
//						}
//						
//						measList.put("classification:celltype", r);
//						measList.close();
//					// });
//					}
					
					// success.set(true);
				}
		    }
			catch (Exception e) {				    	
				e.printStackTrace();
				
			}
		    finally {
			    System.gc();
			    
		    }

			return pathObjects;
		}
		
		@Override
		public String getLastResultsDescription() {
			if (pathObjects == null) return null;
			
			final int nDetections = pathObjects.size();
			
			if (nDetections == 1) return "1 nucleus classified";
			else return String.format("%d nuclei classified", nDetections);
		}
	}
	
	@Override
	protected void preprocess(final PluginRunner<BufferedImage> pluginRunner) {
		try {
			final ImageData<BufferedImage> imageData = pluginRunner.getImageData();
			
			m_modelName = (String)getParameterList(imageData).getChoiceParameterValue("modelName");
			final String modelLocationStr = sptxSetup.getObjclsModelLocationPath();
			final String modelPathStr = Paths.get(modelLocationStr, m_modelName+".pt").toString();

			final String timeStamp = Long.toString(System.nanoTime());
			final Path resultPath = Files.createTempFile("sptx2-classification_result-" + timeStamp + "-", ".json");
            final String resultPathString = resultPath.toAbsolutePath().toString();
            resultPath.toFile().deleteOnExit();
            
			final String s = ProgramDirectoryUtilities.getProgramDirectory();
			logger.debug(s);
			
			// Create command to run
	        VirtualEnvironmentRunner veRunner;
			
			veRunner = new VirtualEnvironmentRunner(sptx2Setup.getEnvironmentNameOrPath(), sptx2Setup.getEnvironmentType(), ObjectClassifier.class.getSimpleName());
		
	        // This is the list of commands after the 'python' call
	        // List<String> sptx2Arguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", "/workspace/sptx2/qupath-sptx2/qupath-extension-sptx2/scripts/object_classifier"));
			final String script_path = Paths.get(sptx2Setup.getScriptLocationPath(), "object_classification.py").toString();
			
			// List<String> sptx2Arguments = new ArrayList<>(Arrays.asList("-W", "ignore", "/workspace/sptx2/qupath-sptx2/qupath-extension-sptx2/scripts/object_classification.py", "param", resultPathString));
			List<String> sptx2Arguments = new ArrayList<>(Arrays.asList("-W", "ignore", script_path, "param", resultPathString));
			
			
	        sptx2Arguments.add("--model_file");
	        sptx2Arguments.add("" + modelPathStr);
	        veRunner.setArguments(sptx2Arguments);

	        // Finally, we can run Cellpose
	        final String[] logs = veRunner.runCommand();
	        // veRunner.runCommand();
	        
	        for (String log : logs) logger.info(log);
	        // logger.info("Object classification command finished running");
			
	        final FileReader resultFileReader = new FileReader(new File(resultPathString));
			final BufferedReader bufferedReader = new BufferedReader(resultFileReader);
			final Gson gson = new Gson();
			final JsonObject jsonObject = gson.fromJson(bufferedReader, JsonObject.class);
			
//			List<Double> image_std = gson.fromJson(jsonObject.get("image_std"), new TypeToken<List<Double>>(){}.getType());
			
			m_modelPreferredPixelSizeMicrons = jsonObject.get("pixel_size").getAsDouble();
			m_samplingFeatureSize = jsonObject.get("image_size").getAsInt();
			
			final PathObjectHierarchy hierarchy = imageData.getHierarchy();
			final Collection<PathObject> selectedObjects = hierarchy.getSelectionModel().getSelectedObjects();
			final Predicate<PathObject> pred = p -> selectedObjects.contains(p.getParent());
			
			m_availabelObjList = Collections.synchronizedList(QPEx.getObjects(hierarchy, pred));
			
			m_batchSize = getParameterList(imageData).getIntParameterValue("batchSize");
			final int maxThread = getParameterList(imageData).getIntParameterValue("maxThread");
			
			semaphore = maxThread <= 0? null: new Semaphore(maxThread);
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
		    System.gc();
		}
	}	

	@Override
	protected void postprocess(final PluginRunner<BufferedImage> pluginRunner) {
		m_availabelObjList.clear();
		System.gc();			
	}
	
	private ParameterList buildParameterList(final ImageData<BufferedImage> imageData) {
		ParameterList params = null;
		
		params = new ParameterList();
		params.addTitleParameter("Setup parameters");
	
		try {			
			if(!imageData.getServer().getPixelCalibration().hasPixelSizeMicrons()) {
				Dialogs.showErrorMessage("Error", "Please check the image properties in left panel. Most likely the pixel size is unknown.");
				throw new Exception("No pixel size information");
			}
	        
			final List<String> classificationModeNamelList = Files.list(Paths.get(sptxSetup.getObjclsModelLocationPath()))
					.filter(Files::isRegularFile)
            	    .map(p -> p.getFileName().toString())
            	    .filter(s -> s.endsWith(".pt"))
            	    .map(s -> s.replaceAll(".pt", ""))
            	    .collect(Collectors.toList());

			if(classificationModeNamelList.size() == 0) throw new Exception("No model exist in the model directory.");
			
			
			params.addChoiceParameter("modelName", "Model", classificationModeNamelList.get(0), classificationModeNamelList, 
					"Choose the model that should be used for object classification");

			params.addEmptyParameter("");
			params.addEmptyParameter("Adjust below parameters if GPU resources are limited.");
			// params.addIntParameter("maxThread", "Max thread number (exprimental, due to GPU capacity limitation)", 1, null, "Max thread number (due to GPU constaint)");		
			params.addIntParameter("batchSize", "Batch Size", 128, null, "Batch siize");		
			
			params.addIntParameter("maxThread", "Max number of parallel threads (0: using qupath setup)", 0, null, "Max number of parallel threads (0: using qupath setup)");		
			
		} catch (Exception e) {
			params = null;
			
			// TODO Auto-generated catch block
			e.printStackTrace();
			Dialogs.showErrorMessage("Error", e.getMessage());
		} finally {
		    System.gc();
		}
		
				
		return params;
	}
	
	@Override
	protected boolean parseArgument(ImageData<BufferedImage> imageData, String arg) {		
		return super.parseArgument(imageData, arg);
	}

	@Override
	public ParameterList getDefaultParameterList(final ImageData<BufferedImage> imageData) {
		
		if (!parametersInitialized) {
			params = buildParameterList(imageData);
		}
		
		return params;
	}

	@Override
	public String getName() {
		return "Object Classification";
	}

	
	@Override
	public String getLastResultsDescription() {
		return "";
	}

	@Override
	public String getDescription() {
		return "Object classification based on deep learning";
	}


	@Override
	protected double getPreferredPixelSizeMicrons(ImageData<BufferedImage> imageData, ParameterList params) {
		return m_modelPreferredPixelSizeMicrons;
	}


	@Override
	protected ObjectDetector<BufferedImage> createDetector(ImageData<BufferedImage> imageData, ParameterList params) {
		return new ObjClassifier();
	}


	@Override
	protected int getTileOverlap(ImageData<BufferedImage> imageData, ParameterList params) {
		return 0;
	}
		
}