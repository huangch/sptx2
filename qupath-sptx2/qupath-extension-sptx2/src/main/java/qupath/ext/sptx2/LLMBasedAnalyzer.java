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
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.plugins.AbstractInteractivePlugin;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;

/**
 * Plugin to support Star-Dist nucleus detection UI.
 * 
 * @author Chao Hui Huang
 *
 * @param <T>
 */
public class LLMBasedAnalyzer<T> extends AbstractInteractivePlugin<T> {
	protected static SpTx2Setup sptxSetup = SpTx2Setup.getInstance();
	
	private static Semaphore semaphore;
	
	private static Logger logger = LoggerFactory.getLogger(LLMBasedAnalyzer.class);
	
	// CELLPOSE PARAMETERS

    protected static SpTx2Setup sptx2Setup = SpTx2Setup.getInstance();

    // Parameters and parameter values that will be passed to the sptx2 command
    // protected static LinkedHashMap<String, String> parameters;

    // No defaults. All should be handled by the builder
    protected static String model;
    protected Integer overlap;

    protected File modelDirectory;
    protected File trainDirectory;
    protected File valDirectory ;

    private static File sptx2TempFolder;


    // Results table from the training

    
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
	
	@Override
	public ParameterList getDefaultParameterList(ImageData<T> imageData) {
		
		try {
			if(!imageData.getServer().getPixelCalibration().hasPixelSizeMicrons()) {

				
				Dialogs.showErrorMessage("Error", "Please check the image properties in left panel. Most likely the pixel size is unknown.");
				
				throw new Exception("No pixel size information");
			}
			
			
	
    		
			final ParameterList params = new ParameterList()
					.addTitleParameter("General Parameters")			
					.addIntParameter("maxThread", "Max thread number (exprimental, due to GPU capacity limitation)", 1, null, "Max thread number (due to GPU constaint)")			
					.addEmptyParameter("(Try to stay with 1. CUDA for JavaCPP in multi-threaded Java is not stable.)")			
					;
			
			return params;
		} catch (Exception e) {
			// TODO Auto-generated catch block
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

		final int maxThread = params.getIntParameterValue("maxThread");

		semaphore = new Semaphore(maxThread);
		
		parentObjects.forEach(p -> {
			tasks.add(() -> {
				runDetection(
						(ImageData<BufferedImage>) runner.getImageData(), p);
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
			ImageData<BufferedImage> imageData, PathObject parentObject
			) {
		
		try {
			
			
		
			semaphore.acquire();
			final String s = ProgramDirectoryUtilities.getProgramDirectory();
			logger.debug(s);
			System.out.print("DEBUG: DEBUG: DEBUG: DEBUG: DEBUG: DEBUG: DEBUG: DEBUG: "+s);
			
			// Create command to run
	        VirtualEnvironmentRunner veRunner;
			
			veRunner = new VirtualEnvironmentRunner(sptx2Setup.getEnvironmentNameOrPath(), sptx2Setup.getEnvironmentType(), LLMBasedAnalyzer.class.getSimpleName());
		
	        // This is the list of commands after the 'python' call
	        List<String> sptx2Arguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", "lls_based_analyzer"));

	        sptx2Arguments.add("--dir");
	        sptx2Arguments.add("" + sptx2TempFolder);

	        sptx2Arguments.add("--pretrained_model");
	        sptx2Arguments.add("" + model);

//	        parameters.forEach((parameter, value) -> {
//	            sptx2Arguments.add("--"+parameter);
//	            if( value != null) {
//	                sptx2Arguments.add(value);
//	            }
//	        });

	        // These all work for sptx2 v2

	        sptx2Arguments.add("--save_tif");

	        sptx2Arguments.add("--no_npy");

	        sptx2Arguments.add("--use_gpu");

	        sptx2Arguments.add("--verbose");

	        veRunner.setArguments(sptx2Arguments);

	        // Finally, we can run python
	        veRunner.runCommand();
	        logger.info("Cellpose command finished running");
			
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
