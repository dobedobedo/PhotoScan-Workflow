# PhotoScan-Workflow
Python Script for automation in Agisoft PhotoScan
  
# Usage  
This script will process all chunks with the following behaviour:  
- **If Photo Alignment is not done yet, align the photos**  
- **If there is tie point, do the rest workflow**  
  
**Note:**  
The GCP must be marked manually, following camera optimisation  
You can change some variables in the script for specific purposes  
Run the script in Tools > Run script
  
**Extra Correction**  
This branch add an experiment feature to correct BRDF effect using Walthal model.  
To turn on this feature, set the _BRDF_ user variable to _True_. Otherwise, _False_ to turn it off.  
It will generated BRDF corrected image and use them to create orthomosaic.  
For a 300 4-band-image (Parrot Sequoia images) project with 9 cm resolution, it costs me about one more day to process.  
  
**Dependencies**  
- numpy  
- pysolar  
- scikit-learn  
  
For information of how to install extra Python modules in PhotoScan, check the below link:  
https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-photoscan-professional-pacakge  
