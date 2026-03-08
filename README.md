# PastureAI-Image2Biomass-Prediction
In livestock farming systems, knowing the amount of available pasture biomass is extremely important. Farmers rely on this information to make informed decisions about grazing management, livestock feeding, and long-term pasture sustainability. If pasture biomass is underestimated, livestock may be underfed, which reduces productivity. On the other hand, overestimating biomass can lead to overgrazing, which damages pasture ecosystems, reduces regrowth capacity, and ultimately affects farm profitability.
Traditionally, biomass estimation is performed through destructive sampling methods. In this process, vegetation is physically clipped from a small area of the field, dried in a laboratory to remove moisture, and then weighed to measure the dry matter content. While this method provides accurate measurements, it is slow, labor-intensive, destructive to the pasture, and limited in spatial coverage, meaning it cannot easily represent the variability across an entire field.
To address these limitations, we introduce PastureAI, a deep learning–based system designed to estimate pasture biomass directly from top-view smartphone photos or similar field images, requiring no specialized equipment or expert knowledge from farmers. Given an image of a pasture paddock, the system predicts several key biomass components. These include green biomass, which represents actively growing and highly nutritious grass; green dry matter (GDM), which reflects the digestible portion of the biomass important for livestock nutrition; and total dry biomass, which indicates the overall dry matter content. These predictions are reported in grams of biomass per quadrat, corresponding to the small sampled area represented by the image, and can be translated into practical grazing decisions for farmers.
The system is trained using the CSIRO Image2Biomass dataset from Kaggle, which is a multimodal dataset combining several types of information. First, it contains visual features in the form of approximately 350 pasture images captured from top-view perspectives. Second, it includes structural features such as canopy or pasture height measurements that provide information about vegetation density and growth. Third, it incorporates spectral features including the NDVI vegetation index, which measures plant greenness and health based on reflectance properties.
The images in this dataset were collected from multiple regions across Australia and across different seasons, capturing a wide variety of pasture conditions, species compositions, and environmental conditions. This diversity helps models learn robust patterns linking visual appearance and environmental signals to biomass production.
PastureAI offers two models. Model 1 requires only a photo as input, making it simple and accessible for farmers using a smartphone in the field. Model 2 additionally incorporates NDVI and canopy height measurements obtained from inexpensive handheld tools such as a falling plate meter. By combining image information with these structural and spectral signals, the second model can improve prediction accuracy when these measurements are available.
Overall, by combining computer vision with structural and spectral indicators, PastureAI provides a fast, scalable, and non-destructive method for estimating pasture biomass. This approach can help farmers monitor pasture availability more efficiently, prioritize field scouting, and make more informed grazing management decisions


## Model 1 — Image Only
ResNet34 backbone pretrained on ImageNet, fine-tuned for pasture biomass regression. The classifier head is replaced with a custom regression head (256 → 128 → 3 outputs) with ReLU activations and dropout. Trained on 357 pasture images split 80/20 by image to prevent leakage. Heavy augmentation applied due to small dataset size. Predicts green biomass, GDM, and total biomass from a single top-view photo with no additional inputs required.
<img width="812" height="716" alt="image" src="https://github.com/user-attachments/assets/2546163c-0244-4fb6-aeec-69dddb04383f" />
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

<img width="773" height="745" alt="image" src="https://github.com/user-attachments/assets/2766707d-4daf-4ab4-935d-aab1c5f146e4" />
```
── Final Val R² Scores (Model 2) ──
  Dry_Green_g         : 0.7647
  GDM_g               : 0.7003
  Dry_Total_g         : 0.7202
```
<img width="1189" height="413" alt="model2_predictions" src="https://github.com/user-attachments/assets/a2f452b7-9f2d-4b23-a5a4-c59bf0d101c1" />

## Streamlit DashBoard
<img width="2524" height="1252" alt="image" src="https://github.com/user-attachments/assets/09536abf-73f3-4c0c-9597-af648c15b0f2" />

