# PastureAI-Image2Biomass-Prediction
In livestock farming systems, knowing the amount of available pasture biomass is extremely important. Farmers rely on this information to make decisions about grazing management, livestock feeding, and pasture sustainability. If pasture biomass is underestimated, livestock may be underfed, while overestimation can lead to overgrazing and degradation of the pasture ecosystem.
The traditional method of measuring biomass includes cutting, drying and followed by weighing. This method is desstructive, slow, labour intensive and has limitations of spatial coverage

PastureAI is a deep learning system that estimates pasture biomass directly from top-view smartphone photos (or any other photo), requiring zero expert knowledge from the farmer. Given an image of a paddock, the system predicts green biomass which represents the actively growing grass and which is highly nutricious, green dry matter (GDM) refelcting the digestable portion of teh biomass, and total biomass in grams - and translates those numbers into an immediate grazing decision. 
A brief introduction of the dataset, it is a multimodel dataset which has visual featutre of around 350 images, structural feature as the canopy height and spectral feature as NDVI index. 
The system offers two models. Model 1 requires only a photo. Model 2 additionally accepts **NDVI** and **canopy height** readings from cheap handheld sensors (*falling plate*), improving accuracy when available. 

## Model 1 — Image Only
ResNet34 backbone pretrained on ImageNet, fine-tuned for pasture biomass regression. The classifier head is replaced with a custom regression head (256 → 128 → 3 outputs) with ReLU activations and dropout. Trained on 357 pasture images split 80/20 by image to prevent leakage. Heavy augmentation applied due to small dataset size. Predicts green biomass, GDM, and total biomass from a single top-view photo with no additional inputs required.
```
── Final Val R² Scores ──
  Dry_Green_g         : 0.8011
  GDM_g               : 0.7453
  Dry_Total_g         : 0.6745
  ```
  <img width="1200" height="400" alt="predictions" src="https://github.com/user-attachments/assets/428da781-2c55-42ac-bfb1-8522d1289856" />


## Model 2 (MultiModal) — Image + Sensors
Same ResNet34 backbone as Model 1, with an additional MLP branch (2 → 256 → 256) that processes two tabular inputs: NDVI from a handheld GreenSeeker sensor and compressed canopy height from a falling plate meter. The two streams are concatenated and passed through the shared regression head to predict green biomass, GDM, and total biomass.
NDVI (Normalized Difference Vegetation Index) directly measures the ratio of live green vegetation to bare ground — information that is partially visible in an image but captured much more precisely by a dedicated sensor. Canopy height from the plate meter adds a density dimension that the camera simply cannot see: two paddocks can look identical in a photo but have very different biomass if one is tall and dense versus short and sparse. Together these two sensors give the model information that exists beyond the RGB image, which is why Model 2 consistently outperforms Model 1.
The tabular branch was intentionally kept separate from the backbone rather than encoding sensor values into the image — this way the model can learn image features and sensor features independently before fusing them, which is more effective than trying to force all information through a single pathway.
```
── Final Val R² Scores (Model 2) ──
  Dry_Green_g         : 0.7647
  GDM_g               : 0.7003
  Dry_Total_g         : 0.7202
```
<img width="1189" height="413" alt="model2_predictions" src="https://github.com/user-attachments/assets/a2f452b7-9f2d-4b23-a5a4-c59bf0d101c1" />
