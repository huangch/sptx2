/*-
 * #%L
 * This file is part of QuPath.
 * %%
 * Copyright (C) 2014 - 2016 The Queen's University of Belfast, Northern Ireland
 * Contact: IP Management (ipmanagement@qub.ac.uk)
 * Copyright (C) 2018 - 2020 QuPath developers, The University of Edinburgh
 * %%
 * QuPath is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * QuPath is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License 
 * along with QuPath.  If not, see <https://www.gnu.org/licenses/>.
 * #L%
 */

package qupath.ext.sptx2;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.AbstractInteractivePlugin;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.ext.stardist.StarDist2D;
import qupath.ext.stardist.StarDist2D.Builder;

/**
 * Plugin to support Star-Dist nucleus detection UI.
 * 
 * @author Chao Hui Huang
 *
 * @param <T>
 */
public class StarDistCellNucleusDetection<T> extends AbstractInteractivePlugin<T> {
	protected static SpTx2Setup sptxSetup = SpTx2Setup.getInstance();
	
	private static Semaphore semaphore;
	
	private static Logger logger = LoggerFactory.getLogger(StarDistCellNucleusDetection.class);
	
	private String resultString = null;

	@Override
	protected void preprocess(final PluginRunner<T> pluginRunner) {
	};
	
	@Override
	public Collection<Class<? extends PathObject>> getSupportedParentObjectClasses() {
		return Collections.singleton(PathAnnotationObject.class);
	}

	@Override
	public String getName() {
		return "Stardict-based Cell Nucleus Detection";
	}

	@Override
	public String getDescription() {
		return "Stardict-based Cell Nucleus Detection";
	}

	@Override
	public String getLastResultsDescription() {
		return resultString;
	}

	
	@SuppressWarnings("unchecked")
	@Override
	public ParameterList getDefaultParameterList(ImageData<T> imageData) {
		
		try {
			if(!imageData.getServer().getPixelCalibration().hasPixelSizeMicrons()) {
				Dialogs.showErrorMessage("Error", "Please check the image properties in left panel. Most likely the pixel size is unknown.");
				throw new Exception("No pixel size information");
			}
			
			final List<String> stardistModeNamelList = Files.list(Paths.get(sptxSetup.getStardistModelLocationPath()))
            	    .filter(Files::isRegularFile)
            	    .map(p -> p.getFileName().toString())
            	    .collect(Collectors.toList());
            
    		final ImageServer<BufferedImage> server = (ImageServer<BufferedImage>) imageData.getServer();		
    		final double imagePixelSizeMicrons = server.getPixelCalibration().getAveragedPixelSizeMicrons();
    		
			final ParameterList params = new ParameterList()
					.addTitleParameter("General Parameters")			
					.addDoubleParameter("threshold", "Probability (detection) threshold", 0.1, null, "Probability (detection) threshold")
					.addDoubleParameter("normalizePercentilesLow", "Percentile normalization (lower bound)", 1, null, "Percentile normalization (lower bound)")
					.addDoubleParameter("normalizePercentilesHigh", "Percentile normalization (higher bound)", 99, null, "Percentile normalization (lower bound)")
					.addTitleParameter("Measurements")
					.addBooleanParameter("includeProbability", "Add probability as a measurement (enables later filtering). Default: false", false, "Add probability as a measurement (enables later filtering)")
					.addBooleanParameter("measureShape", "Add shape measurements. Default: false", false, "Add shape measurements")
					.addBooleanParameter("measureIntensity", "Add intensity measurements. Default: false", false, "Add shape measurements")
					.addTitleParameter("Additional Parameters")
					.addChoiceParameter("starDistModel", "Specify the model .pb file", stardistModeNamelList .get(0), stardistModeNamelList, "Choose the model that should be used for object classification")
					.addDoubleParameter("pixelSize", "Resolution for detection. Default: value provided by QuPath.", imagePixelSizeMicrons, null, "Resolution for detection")
					.addStringParameter("channel", "Select detection channel (e.g., DAPI. Default: [empty] = N/A)", "")
					.addDoubleParameter("cellExpansion", "Approximate cells based upon nucleus expansion (e.g., 5.0. Default: -1 = N/A)", -1, null, "Approximate cells based upon nucleus expansion")		
					.addDoubleParameter("cellConstrainScale", "Constrain cell expansion using nucleus size (e.g., 1.5. Default: -1 = N/A)", -1, null, "Constrain cell expansion using nucleus size")
					.addIntParameter("maxThread", "Max thread number (exprimental, due to GPU capacity limitation)", 1, null, "Max thread number (due to GPU constaint)")			
					.addEmptyParameter("(Try to stay with 1. CUDA for JavaCPP in multi-threaded Java is not stable.)")			
					;
			
			return params;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			logger.error(e.getMessage());
			e.printStackTrace();
		}
		
		return null;
	}

	@Override
	protected Collection<? extends PathObject> getParentObjects(PluginRunner<T> runner) {
		return getHierarchy(runner).getSelectionModel().getSelectedObjects().stream().filter(p -> p.isAnnotation()).collect(Collectors.toList());
	}

	@Override
	protected void addRunnableTasks(ImageData<T> imageData, PathObject parentObject, List<Runnable> tasks) {}
	
	@SuppressWarnings("unchecked")
	@Override
	protected Collection<Runnable> getTasks(final PluginRunner<T> runner) {
		final Collection<? extends PathObject> parentObjects = getParentObjects(runner);
		if (parentObjects == null || parentObjects.isEmpty())
			return Collections.emptyList();
		
		// Add a single task, to avoid multithreading - which may complicate setting parents
		final List<Runnable> tasks = new ArrayList<>(parentObjects.size());
		
		final String modelFilePath = (String)params.getChoiceParameterValue("starDistModel");
		final double threshold = params.getDoubleParameterValue("threshold");
		final String channels = params.getStringParameterValue("channel");
		final double normalizePercentilesLow = params.getDoubleParameterValue("normalizePercentilesLow");
		final double normalizePercentilesHigh = params.getDoubleParameterValue("normalizePercentilesHigh");
		final double pixelSize = params.getDoubleParameterValue("pixelSize");
		final double cellExpansion = params.getDoubleParameterValue("cellExpansion");
		final double cellConstrainScale = params.getDoubleParameterValue("cellConstrainScale");
		final boolean measureShape = params.getBooleanParameterValue("measureShape");
		final boolean measureIntensity = params.getBooleanParameterValue("measureIntensity");
		final boolean includeProbability = params.getBooleanParameterValue("includeProbability");
		final int maxThread = params.getIntParameterValue("maxThread");
		
		semaphore = new Semaphore(maxThread);
		
		parentObjects.forEach(p -> {
			tasks.add(() -> {
				runDetection(
						(ImageData<BufferedImage>) runner.getImageData(), 
						p, 
						modelFilePath, 
						threshold, 
						channels, 
						normalizePercentilesLow, 
						normalizePercentilesHigh, 
						pixelSize, 
						cellExpansion,
						cellConstrainScale, 
						measureShape, 
						measureIntensity, 
						includeProbability);
			});
		});
		return tasks;
	}
	
	/**
	 * Create and add a new annotation by expanding the ROI of the specified PathObject.
	 * 
	 * 
	 * @param bounds
	 * @param hierarchy
	 * @param pathObject
	 * @param radiusPixels
	 * @param constrainToParent
	 * @param removeInterior
	 */
	private static void runDetection(
			ImageData<BufferedImage> imageData, PathObject parentObject, String modelPath,
			double threshold, String channels, double normalizePercentilesLow, double normalizePercentilesHigh,
			double pixelSize, double cellExpansion, double cellConstrainScale, boolean measureShape,
			boolean measureIntensity, boolean includeProbability
			) {
		
		try {
		
			semaphore.acquire();
			
			
			final List<PathObject> parentObjects = new ArrayList<PathObject>();
			parentObjects.add(parentObject);

			final Path stardistModelPath = Paths.get(sptxSetup.getStardistModelLocationPath(), modelPath);
			
			final Builder stardistBuilder = StarDist2D.builder(stardistModelPath.toString())
			        .threshold(threshold)
			        .normalizePercentiles(normalizePercentilesLow, normalizePercentilesHigh)
			        .pixelSize(pixelSize);
	
	        if(!channels.isBlank()) stardistBuilder.channels(channels);
	        if(cellExpansion > 0) stardistBuilder.cellExpansion(cellExpansion);
	        if(cellConstrainScale > 0) stardistBuilder.cellConstrainScale(cellConstrainScale);
			if(measureShape) stardistBuilder.measureShape();
			if(measureIntensity) stardistBuilder.measureIntensity();
			if(includeProbability) stardistBuilder.includeProbability(true);
			
			final StarDist2D stardist = stardistBuilder.build();
			
			
			
			stardist.detectObjects((ImageData<BufferedImage>) imageData, parentObjects);
			
			
			
			semaphore.release();
				
			 
			
			

		} catch (InterruptedException e) {
			 // TODO Auto-generated catch block
			 e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
