# PastureAI-Image2Biomass-Prediction
In livestock farming systems, knowing the amount of available pasture biomass is extremely important. Farmers rely on this information to make decisions about grazing management, livestock feeding, and pasture sustainability. If pasture biomass is underestimated, livestock may be underfed, while overestimation can lead to overgrazing and degradation of the pasture ecosystem.
The traditional method of measuring biomass includes cutting, drying and followed by weighing. This method is desstructive, slow, labour intensive and 

PastureAI is a deep learning system that estimates pasture biomass directly from top-view smartphone photos (or any other photo), requiring zero expert knowledge from the farmer. Given an image of a paddock, the system predicts green biomass, green dry matter (GDM), and total biomass in grams - and translates those numbers into an immediate grazing decision.
The system offers two models. Model 1 requires only a photo. Model 2 additionally accepts **NDVI** and **canopy height** readings from cheap handheld sensors (*falling plate*), improving accuracy when available. 
