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

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import javafx.beans.property.StringProperty;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.measure.ObservableMeasurementTableData;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.images.ImageData;
import qupath.lib.measurements.MeasurementList;
import qupath.lib.objects.PathAnnotationObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectConnections;
import qupath.lib.objects.TMACoreObject;
import qupath.lib.objects.PathDetectionObject;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.plugins.AbstractDetectionPlugin;
import qupath.lib.plugins.DetectionPluginTools;
import qupath.lib.plugins.ObjectDetector;
import qupath.lib.plugins.PluginRunner;
import qupath.lib.plugins.parameters.ParameterList;
import qupath.lib.roi.interfaces.ROI;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Plugin for loading 10x Visium Annotation 
 * 
 * @author Chao Hui Huang
 *
 */
public class CellCellInteractionAnalysis extends AbstractDetectionPlugin<BufferedImage> {
	final private static Logger logger = LoggerFactory.getLogger(CellCellInteractionAnalysis.class);
	final private StringProperty CCIAnalLRPFileProp = PathPrefs.createPersistentPreference("CCIAnalLRPFile", ""); 
	final private StringProperty CCIAnalVendorProp = PathPrefs.createPersistentPreference("CCIAnalVendor", "xenium"); 
	final private StringProperty CCIAnalLigandReceptorProp = PathPrefs.createPersistentPreference("CCIAnalLigandReceptor", "ligand"); 
	
	private ParameterList params;

	final private List<String> vendorlList = Arrays.asList("xenium", "cosmx");
	final private List<String> ligandreceptorList = Arrays.asList("ligand", "receptor");
	
	private String lastResults = null;
	
	/**
	 * Constructor.
	 */
	public CellCellInteractionAnalysis() {
		params = new ParameterList()
			.addTitleParameter("Cell-Cell Interaction Analysis")
			.addStringParameter("lrpfile", "Ligand-Receptor Pair list file", CCIAnalLRPFileProp.get(), "Ligand-Receptor Pair list file")
			.addChoiceParameter("vendor", "Vendor", CCIAnalVendorProp.get(), vendorlList, "Choose the vendor that should be used for object classification")
			.addChoiceParameter("ligandorreceptor", "Ligand-based or Receptor-based", CCIAnalLigandReceptorProp.get(), ligandreceptorList, "Summary")
			;
	}
	
	class AnnotationLoader implements ObjectDetector<BufferedImage> {
		
		
		
		@Override
		public Collection<PathObject> runDetection(final ImageData<BufferedImage> imageData, final ParameterList params, final ROI pathROI) throws IOException {
			CCIAnalLRPFileProp.set(params.getStringParameterValue("lrpfile"));
			CCIAnalVendorProp.set((String)params.getChoiceParameterValue("vendor"));
			CCIAnalLigandReceptorProp.set((String)params.getChoiceParameterValue("ligandorreceptor"));
			
			final PathObjectHierarchy hierarchy = imageData.getHierarchy();
			final ObservableMeasurementTableData model = new ObservableMeasurementTableData();
			model.setImageData(imageData, imageData == null ? Collections.emptyList() : hierarchy.getObjects(null, PathDetectionObject.class));
			final PathObjectConnections connections = (PathObjectConnections) imageData.getProperty("OBJECT_CONNECTIONS");
			
			try {
				/*
	             * Generate cell masks with their labels
	             */
				
				final List<PathObject> selectedAnnotationPathObjectList = new ArrayList<>();
				
				for (PathObject pathObject : hierarchy.getSelectionModel().getSelectedObjects()) 
					if (pathObject.isAnnotation() && pathObject.hasChildObjects())
						selectedAnnotationPathObjectList.add(pathObject);
				
				if(selectedAnnotationPathObjectList.isEmpty()) throw new Exception("Missed selected annotations");
				
				final List<String> availGeneList = model.getAllNames().stream().filter(c -> c.startsWith((String)params.getChoiceParameterValue("vendor")+":cell_transcript:")).collect(Collectors.toList());
				
				final List<List<String>> lrpList = new ArrayList<>();
				
				final String lprFilePath = params.getStringParameterValue("lrpfile");
				final FileReader lrpFileReader = new FileReader(new File(lprFilePath));
				final BufferedReader lrpReader = new BufferedReader(lrpFileReader);
				lrpReader.readLine();
				String lrpNextRecord;
				
				while ((lrpNextRecord = lrpReader.readLine()) != null) {
		        	final String[] lrpNextRecordArray = lrpNextRecord.split(",");
		        	final String ligand = lrpNextRecordArray[1].replaceAll("\"", "");
		        	final String receptor = lrpNextRecordArray[2].replaceAll("\"", "");
		        		
		        	if(availGeneList.contains((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+ligand) && availGeneList.contains((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+receptor))
		        		lrpList.add(Arrays.asList(ligand, receptor));
				}
				
				lrpReader.close();
				
//				for(PathObject p: selectedAnnotationPathObjectList) {
				selectedAnnotationPathObjectList.parallelStream().forEach(p -> {
//					for(PathObject c: p.getChildObjects()) {
					p.getChildObjects().parallelStream().forEach(c -> { 
						final List<PathObject> connectedObj = connections.getConnections(c);
						final MeasurementList cMeasList = c.getMeasurementList();
						final List<String> cgList = cMeasList.getMeasurementNames().stream().filter(g -> g.startsWith((String)params.getChoiceParameterValue("vendor")+":cell_transcript:")).collect(Collectors.toList());
						
						if(lrpList.stream().map(g -> cMeasList.get((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+g.get(0))).anyMatch(g -> g.isNaN())) return;
						if(cgList.stream().map(g -> cMeasList.get(g)).anyMatch(g -> g.isNaN())) return;
						
						final double cgSum = cgList.stream().map(g -> cMeasList.get(g)).mapToDouble(Double::doubleValue).sum();
						final Map<String, Double> cgMap = cgList.stream().collect(Collectors.toMap(g -> g, g -> cMeasList.get(g)/cgSum));
						
						synchronized (cMeasList) {
							lrpList.stream().forEach(g -> {
								cMeasList.put((String)params.getChoiceParameterValue("vendor")+":cell_cci:"+g.get(0)+"_"+g.get(1), 0);
							});
						}
						
						Map<List<String>,Double> sumBuf = Collections.synchronizedMap(new HashMap<List<String>,Double>());
					    Map<List<String>,AtomicBoolean> flagBuf = Collections.synchronizedMap(new HashMap<List<String>,AtomicBoolean>());
					      
						lrpList.stream().forEach(g -> {
							sumBuf.put(g, Double.valueOf(0.0));
							flagBuf.put(g, new AtomicBoolean(false));
						});
						
//						for(PathObject d: connectedObj) {
						connectedObj.parallelStream().forEach(d -> {
							final MeasurementList dMeasList = d.getMeasurementList();
							final List<String> dgList = dMeasList.getMeasurementNames().stream().filter(g -> g.startsWith((String)params.getChoiceParameterValue("vendor")+":cell_transcript:")).collect(Collectors.toList());

							if(lrpList.stream().map(g -> dMeasList.get((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+g.get(1))).anyMatch(g -> g.isNaN())) return;
							if(dgList.stream().map(g -> dMeasList.get(g)).anyMatch(g -> g.isNaN())) return;
							
							final double dgSum = dgList.stream().map(g -> dMeasList.get(g)).mapToDouble(Double::doubleValue).sum();
							final Map<String, Double> dgMap = dgList.stream().collect(Collectors.toMap(g -> g, g -> dMeasList.get(g)/dgSum));
							
//							for(List<String> lrp: lrpList) {
							lrpList.parallelStream().forEach(lrp -> {
								final Double cv = cgMap.get((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+lrp.get(0));
								final Double dv = dgMap.get((String)params.getChoiceParameterValue("vendor")+":cell_transcript:"+lrp.get(1));
								if(cv.isNaN() || dv.isNaN()) return;
								
								final Double prob = cv*dv;
								flagBuf.get(lrp).set(true);
								
								synchronized (sumBuf) {
									sumBuf.put(lrp, sumBuf.get(lrp)+prob);
								}
//							}
							});
							synchronized (dMeasList) {
								dMeasList.close();
							}
//						 }
						});
						
//						for(List<String> g: lrpList) {
						lrpList.parallelStream().forEach(g -> {
							synchronized (cMeasList) {
								final double resultValue = flagBuf.get(g).get()? sumBuf.get(g): 0.0;
								cMeasList.put((String)params.getChoiceParameterValue("vendor")+":cell_cci:"+g.get(0)+"_"+g.get(1), resultValue);
							}
//						}
						});
						
						synchronized (cMeasList) {
							cMeasList.close();
						}
//					 }
					});
//				 }
				});
	            
		        hierarchy.getSelectionModel().setSelectedObject(null);
				
			}
			catch(Exception e) {	

				Dialogs.showErrorMessage("Error", e.getMessage());
				lastResults = e.getMessage();
				logger.error(lastResults);
				
				// return new ArrayList<PathObject>(hierarchy.getRootObject().getChildObjects());
			}				
			
			if (Thread.currentThread().isInterrupted()) {

				Dialogs.showErrorMessage("Warning", "Interrupted!");
				lastResults =  "Interrupted!";
				logger.warn(lastResults);
				// return new ArrayList<PathObject>(hierarchy.getRootObject().getChildObjects());
			}
			
			return new ArrayList<PathObject>(hierarchy.getRootObject().getChildObjects());
		}
		
		
		@Override
		public String getLastResultsDescription() {
			return lastResults;
		}
		
		
	}

	@Override
	public ParameterList getDefaultParameterList(final ImageData<BufferedImage> imageData) {
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
//		List<Class<? extends PathObject>> list = new ArrayList<>();
//		list.add(TMACoreObject.class);
//		list.add(PathRootObject.class);
//		return list;
		
		return Arrays.asList(
				PathAnnotationObject.class,
				TMACoreObject.class
				);		
	}

}
