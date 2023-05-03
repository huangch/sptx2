package qupath.ext.sptx2;

import java.nio.file.Paths;

import qupath.ext.sptx2.VirtualEnvironmentRunner.EnvType;

public class SpTx2Setup {
	private static final boolean DEPLYMENT = false;
	
	public String getScriptLocationPath() {
    	final String scriptLocationPath = DEPLYMENT? 
    			Paths.get(ProgramDirectoryUtilities.getProgramDirectory().toString()).toString():
    			// Paths.get(ProgramDirectoryUtilities.getProgramDirectory().toString(), "build", "install", "QuPath-0.4.3", "lib").toString();
    			"/workspace/sptx2/qupath-sptx2/qupath-extension-sptx2/scripts";
    	
    	return scriptLocationPath;
    }

	private EnvType envType;
    private String environmentNameOrPath;
    private String stardistModelLocationPath;
    private String objclsModelLocationPath;
    
    private static SpTx2Setup instance = new SpTx2Setup();

    
    public EnvType getEnvironmentType() {
        return envType;
    }

    public void setEnvironmentType(EnvType envType) {
        this.envType = envType;
    }

    public String getEnvironmentNameOrPath() {
        return environmentNameOrPath;
    }

    public void setEnvironmentNameOrPath(String environmentNameOrPath) {
        this.environmentNameOrPath = environmentNameOrPath;
    }


    
    
    public static SpTx2Setup getInstance() {
        return instance;
    }
    
    public String getStardistModelLocationPath() {
        return stardistModelLocationPath;
    }

    public void setStardistModelLocationPath(String stardistModelLocationPath) {
        this.stardistModelLocationPath = stardistModelLocationPath;
    }
    
    public String getObjclsModelLocationPath() {
        return objclsModelLocationPath;
    }

    public void setObjclsModelLocationPath(String objclsModelLocationPath) {
        this.objclsModelLocationPath = objclsModelLocationPath;
    }
    
    
}