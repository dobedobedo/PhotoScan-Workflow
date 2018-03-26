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
  
**Enable NIR tie points only**  
Some studies show enabling only NIR tie points can improve photogrammetric point clouds quality.  
To turn on this feature, set the _**NIR_only**_ user variable to _**True**_. Otherwise, _**False**_ to turn it off.
  
**Extra Correction**  
This branch add an experiment feature to correct BRDF effect using Walthal model.  
To turn on this feature, set the _**BRDF**_ user variable to _**True**_. Otherwise, _**False**_ to turn it off.  
It will generated BRDF corrected image and use them to create orthomosaic.  
  
_Remember to check whether image date time is recorded as UTC, and set the **UTC** user variable **True** if it is_  
  
For a 300 4-band-image (Parrot Sequoia images) project with 9 cm resolution, it costs me about two more days to process.  
  
**Dependencies**  
- numpy  
- pysolar  
- scikit-learn  
  
For information of how to install extra Python modules in PhotoScan, check the below link:  
https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-photoscan-professional-pacakge  
