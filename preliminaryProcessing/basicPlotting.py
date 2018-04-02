import pandas as pd
#import os, json

# The code below can be used to generate the CSV
#filenames = os.listdir("Descriptions")
#
#metas = {}
#counter = 0
#for name in filenames:
#    with open("Descriptions/" + name, "r") as file:
#        load = json.load(file)
#    try:
#        img_type = load['meta']['acquisition']['image_type']
#        age = load['meta']['clinical']['age_approx']
#        benign_malignant = load['meta']['clinical']['benign_malignant']
#        diagnosis = load['meta']['clinical']['diagnosis']
#        diagnosis_confirm_type = load['meta']['clinical']['diagnosis_confirm_type']
#        melanocytic = load['meta']['clinical']['melanocytic']
#        sex = load['meta']['clinical']['sex']
#        #diagnosis2 = load['meta']['unstructured']['diagnosis']
#        #localization = load['meta']['unstructured']['localization']
#        #site = load['meta']['unstructured']['site']
#    except:
#        # Skip files which don't have 
#        continue
#    metas[name] = {"img_type": img_type, "age": age, "benign_malignant": benign_malignant, 
#     "diagnosis": diagnosis, "diagnosis_confirm_type": diagnosis_confirm_type, 
#     "melanocytic": melanocytic, "sex": sex}
#    counter += 1
#    if counter % 200 == 0:
#        print(round(counter/len(filenames)*100, 2), "% done")
#
#df = pd.DataFrame.from_dict(metas, orient="index")

# # Save dataframe into csv, for easier importing next time
# df.to_csv("metadata.csv")


# # We can also load it from the csv if we want to 
df = pd.DataFrame.from_csv("metadata.csv")

import matplotlib.pyplot as plt

# Let's do some plotting
df.img_type.value_counts().plot.bar().set_title("Image Types")

plt.clf()
df.age.plot.hist().set_title("Age Distribution")

plt.clf()
df.sex.value_counts().plot.bar().set_title("Gender Distribution")

plt.clf()
df.benign_malignant.value_counts().plot.bar().set_title("Benign/Malignant")

plt.clf()
df.melanocytic.value_counts().plot.bar().set_title("Melanocytic?")

plt.clf()
df.diagnosis.value_counts().plot.bar().set_title("Diagnoses")

plt.clf()
df.diagnosis_confirm_type.value_counts().plot.bar().set_title("Diagnosis Confirmation Types")