# Automatic-Extractor-of-Fusion-Geometry-Features-for-Cartography-Objects-in-Wemap
You can use AutomaticExtractor1 to obtain your map that you want to extract.
You can calculate geometry features and fuise them for your data of shapefile format by code named Factors_calculation_Fusion, 
which will obtain a new format that could used directly to extract in code named AotomaticExtractor1.

All of data is in the directiornary named ExperimentalData, and the area_experiments_proj2.shp is orinal data, and the Including_salienceAttribite.json and Weights_Entropy_Effectivevalue.json are data after running code named Factors_calculation_Fusion.

Automatic_extractor.pkl is the tranined model, you can use it and do not re-train it.

Chinese(中文):

直接使用AutomaticExtractor1提取想要的地图层级数据
提取地图层级数据前，需要计算几何特征并融合，使用Factors_calculation_Fusion中的代码预处理shapefile格式的文件
将完成预处理的数据在AutomaticExtractor1运行，并提取最终你想要的分层数据

原始数据：area_experiments_proj2.shp
计算和融合几何特征后的数据：Including_salienceAttribite.json 和 Weights_Entropy_Effectivevalue.json

已经训练好的模型：Automatic_extractor.pkl，直接在AutomaticExtractor1中使用，不需再次训练
