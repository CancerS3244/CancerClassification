import os, json
from PIL import Image
import pandas as pd

filenames = os.listdir("Descriptions")

df_dict = {"y": []}
size = 50, 35
for x in range(size[0]):
    for y in range(size[1]):
        df_dict["x" + str(y + x * size[1])] = []

for file in filenames:
    # Get the y value
    with open("Descriptions/" + file, "r") as metaFile:
        load = json.load(metaFile)
    try:
        benign_malignant = load['meta']['clinical']['benign_malignant']
    except:
        benign_malignant = "benign"
    df_dict["y"].append(benign_malignant)
    
    # Get the pixels
    im = Image.open("Images/"+file+".jpg").convert("L").resize(size, Image.LANCZOS)
    pix = im.load()
    for x in range(size[0]):
        for y in range(size[1]):
            df_dict["x" + str(y + x * size[1])].append(pix[x, y])

df = pd.DataFrame.from_dict(df_dict)


# Let's try running h2o
import h2o
h2o.init(ip='localhost', port='54321')

hf = h2o.H2OFrame(df)
hf['y'] = hf['y'].asfactor()
response = "y"
predictors = ["x" + str(x) for x in range(size[0] * size[1])]
train, valid = hf.split_frame(ratios = [0.8])

# Try GBM
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm = H2OGradientBoostingEstimator(balance_classes = True)
gbm.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
#gbm.model_performance()._metric_json
print(gbm.logloss())

from h2o.estimators.random_forest import H2ORandomForestEstimator
drf = H2ORandomForestEstimator(balance_classes = True, )
drf.train(x = predictors, y = response, training_frame = train, validation_frame = valid)
#drf.r2(train= True)
#drf.r2(valid = True)
print(drf.logloss())


from h2o.estimators.deeplearning import H2ODeepLearningEstimator
hls = [[10, 10], [10,10,10], [5,5], [5,5,5], [20, 20]]
for hl in hls:
    dl = H2ODeepLearningEstimator(hidden = hl, epochs = 20, balance_classes = True)
    dl.train(x = predictors, y = response, training_frame=train, validation_frame=valid)
    print(hl, dl.logloss())
