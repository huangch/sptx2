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

import qupath.ext.sptx2.VirtualEnvironmentRunner.EnvType;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.scene.control.Menu;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.panes.PreferencePane;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.tools.MenuTools;
/**
 * Install SpTx as an extension.
 * 
 * @author Chao Hui Huang
 */
public class SpTx2Extension implements QuPathExtension, GitHubProject {
	
//	@SuppressWarnings("unchecked")
	@Override
	public void installExtension(QuPathGUI qupath) {
		final SpTx2Setup sptxOptions = SpTx2Setup.getInstance();
		
		// Create stardistModel Property Instance
        final StringProperty stardistModelLocationPathProp = PathPrefs.createPersistentPreference("stardistModelLocationPath", "");
        sptxOptions.setStardistModelLocationPath(stardistModelLocationPathProp.get());
        stardistModelLocationPathProp.addListener((v,o,n) -> sptxOptions.setStardistModelLocationPath(n));
        
        // Add stardistModel Property to Preference Page
        final PreferencePane stardistPrefs = QuPathGUI.getInstance().getPreferencePane();
        stardistPrefs.addPropertyPreference(stardistModelLocationPathProp, String.class, "Stardist model directory", "SpTx2",
                "Enter the directory where the stardist models are located.");
        
		// Create Property Instance
        final StringProperty objclsModelLocationPathProp = PathPrefs.createPersistentPreference("objclsModelLocationPath", "");
        sptxOptions.setObjclsModelLocationPath(objclsModelLocationPathProp.get());
        objclsModelLocationPathProp.addListener((v,o,n) -> sptxOptions.setObjclsModelLocationPath(n));
        
        // Add Property to Preference Page
        final PreferencePane objclsPrefs = QuPathGUI.getInstance().getPreferencePane();
        objclsPrefs.addPropertyPreference(objclsModelLocationPathProp, String.class, "Object Classification model directory", "SpTx2",
                "Enter the directory where the object classification models are located.");        
        
        
        
        // Create the options we need
        ObjectProperty<EnvType> envType = PathPrefs.createPersistentPreference("sptx2EnvType", EnvType.CONDA, EnvType.class);
        StringProperty envPath = PathPrefs.createPersistentPreference("sptx2EnvPath", "");

        //Set options to current values
        sptxOptions.setEnvironmentType(envType.get());
        sptxOptions.setEnvironmentNameOrPath(envPath.get());

        // Listen for property changes
        envType.addListener((v,o,n) -> sptxOptions.setEnvironmentType(n));
        envPath.addListener((v,o,n) -> sptxOptions.setEnvironmentNameOrPath(n));

        // Add Permanent Preferences and Populate Preferences
        PreferencePane prefs = QuPathGUI.getInstance().getPreferencePane();

        prefs.addPropertyPreference(envPath, String.class, "SpTx2 LLM Environment name or directory", "SpTx2",
                "Enter either the directory where your chosen Cellpose virtual environment (conda or venv) is located. Or the name of the conda environment you created.");
        prefs.addChoicePropertyPreference(envType,
                FXCollections.observableArrayList(VirtualEnvironmentRunner.EnvType.values()),
                VirtualEnvironmentRunner.EnvType.class,"SpTx2 LLM Environment Type", "SpTx2",
                "This changes how the environment is started.");
        
        
        
        
        
        
		Menu menu = qupath.getMenu("Extensions>SpTx Analysis Toolbox", true);

		Menu importMenu = MenuTools.addMenuItems(menu, "Import...");
		
		MenuTools.addMenuItems(
				importMenu,
				qupath.createPluginAction("ST Annotation", STAnnotation.class, null)
				);
		
		MenuTools.addMenuItems(
				importMenu,
				qupath.createPluginAction("10x Visium Annotation", VisiumAnnotation.class, null)
				);
		
		MenuTools.addMenuItems(
				importMenu,
				qupath.createPluginAction("10x Xenium Annotation", XeniumAnnotation.class, null)
				);
		
		MenuTools.addMenuItems(
				importMenu,
				qupath.createPluginAction("NanoString CosMX Annotation", CosmxAnnotation.class, null)
				);
		
		MenuTools.addMenuItems(
				importMenu,
				qupath.createPluginAction("Pixel Size Calibration by Xenium Affine Matrix", XeniumPixelSizeCalibration.class, null)
				);
		
		MenuTools.addMenuItems(
				menu,
				null,
				qupath.createPluginAction("StarDist-based Nucleus Detection", StarDistCellNucleusDetection.class, null)
				);		
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Pseudo Spot Generation", PseudoVisiumSpotGeneration.class, null)
				);
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Layerwise Spatial Profiling", LayerwiseSpatialProfiling.class, null)
				);

		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Cell-Cell Interaction Analysis", CellCellInteractionAnalysis.class, null)
				);
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Export Detection Object Images", PathDetectionObjectImageAcquisition.class, null)
				);
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("Object Classification", ObjectClassifier.class, null)
				);
		
		MenuTools.addMenuItems(
				menu,
				qupath.createPluginAction("LLM-based Analysis", LLMBasedAnalyzer.class, null)
				);
	}

	@Override
	public String getName() {
		return "SpTx Extension";
	}

	@Override
	public String getDescription() {
		return "Run SpTx Extension.\n"
				+ "See the extension repository for citation information.";
	}
	
	@Override
	public Version getQuPathVersion() {
		return Version.parse("0.3.0-rc2");
	}

	@Override
	public GitHubRepo getRepository() {
		return GitHubRepo.create(getName(), "qupath", "qupath-extension-sptx");
	}

}
