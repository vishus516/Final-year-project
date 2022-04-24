import os
import pandas as pd
import numpy
import numpy as np
import pickle
import sklearn.ensemble as ek
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
#from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.ensemble as ske
from sklearn import tree, linear_model

from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import log_loss
from matplotlib import pyplot
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

import pefile
import os
import array
import math
import pickle
import sys
import argparse
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve

data = pd.read_csv('data.csv',sep='|', low_memory=False)

data.head(5)

X = data.drop(['Name','md5','legitimate'],axis=1).values
y = data['legitimate'].values


#Feature Selection
print("============================Feature Selection===============================")
randomf = RandomForestClassifier(n_estimators=50).fit(X,y)
model = SelectFromModel(randomf, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]



X_train, X_test, y_train, y_test =train_test_split(X_new, y ,test_size=0.2)
features = []
print('%i features identified as important:' % nb_features)


#important features sored
indices = np.argsort(randomf.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], randomf.feature_importances_[indices[f]]))


for f in sorted(np.argsort(randomf.feature_importances_)[::-1][:nb_features]): 
    features.append(data.columns[2+f])


def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

algorithms = {

        "LR": LogisticRegression(),

        "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),

        "RandomForest": RandomForestClassifier(n_estimators=50),

        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),

        "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),

        "GNB": GaussianNB()

    }

results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results[algo] = score
    #print("%s : %f %%" % (algo, score*100))
    y_pred = clf.predict(X_test)
    print(algo + ":Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print (algo + ":Accuracy : ", accuracy_score(y_test,y_pred)*100)
    #confusion Matrix
    matrix =confusion_matrix(y_test, y_pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    fig.canvas.set_window_title(algo)
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    #ROC_AUC curve
    probs = clf.predict_proba(X_test) 
    probs = probs[:, 1]  
    auc = roc_auc_score(y_test, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(y_test)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    plot_roc_curve(fpr, tpr)
    #Classification Report
    target_names = ['Yes', 'No']
    prediction=clf.predict(X_test)
    print(classification_report(y_test, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(clf, classes=classes, support=True)
    visualizer.fit(X_train, y_train)  
    visualizer.score(X_test, y_test)  
    g = visualizer.poof()

winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))

# Save the algorithm and the feature list for later predictions
print('Saving algorithm and feature list in classifier directory...')
#joblib.dump(algorithms[winner], 'classifier/classifier.pkl')
open('classifier/features.pkl', 'wb').write(pickle.dumps(features))
open('classifier/classifier.pkl', 'wb').write(pickle.dumps(algorithms[winner]))

print('Saved')


clf = algorithms[winner]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)

# print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
# print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))




def get_entropy(data):
    if len(data) == 0:
        return 0.0
    occurences = array.array('L', [0]*256)
    for x in data:
        occurences[x if isinstance(x, int) else ord(x)] += 1

    entropy = 0
    for x in occurences:
	    if x:
	        p_x = float(x) / len(data)
	        entropy -= p_x*math.log(p_x, 2)

    return entropy


def get_resources(pe):
    """Extract resources :
    [entropy, size]"""
    resources = []
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        try:
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                data = pe.get_data(resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
                                size = resource_lang.data.struct.Size
                                entropy = get_entropy(data)

                                resources.append([entropy, size])
        except Exception as e:
            return resources
    return resources

def get_version_info(pe):
    """Return version infos"""
    res = {}
    for fileinfo in pe.FileInfo:
        if fileinfo.Key == 'StringFileInfo':
            for st in fileinfo.StringTable:
                for entry in st.entries.items():
                    res[entry[0]] = entry[1]
        if fileinfo.Key == 'VarFileInfo':
            for var in fileinfo.Var:
                res[var.entry.items()[0][0]] = var.entry.items()[0][1]
    if hasattr(pe, 'VS_FIXEDFILEINFO'):
          res['flags'] = pe.VS_FIXEDFILEINFO.FileFlags
          res['os'] = pe.VS_FIXEDFILEINFO.FileOS
          res['type'] = pe.VS_FIXEDFILEINFO.FileType
          res['file_version'] = pe.VS_FIXEDFILEINFO.FileVersionLS
          res['product_version'] = pe.VS_FIXEDFILEINFO.ProductVersionLS
          res['signature'] = pe.VS_FIXEDFILEINFO.Signature
          res['struct_version'] = pe.VS_FIXEDFILEINFO.StrucVersion
    return res

#extract the info for a given file
def extract_infos(fpath):
    res = {}
    pe = pefile.PE(fpath)
    res['Machine'] = pe.FILE_HEADER.Machine
    res['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
    res['Characteristics'] = pe.FILE_HEADER.Characteristics
    res['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
    res['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
    res['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
    res['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
    res['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
    res['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    res['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
    try:
        res['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
    except AttributeError:
        res['BaseOfData'] = 0
    res['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    res['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
    res['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
    res['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    res['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
    res['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
    res['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
    res['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    res['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
    res['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    res['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    res['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
    res['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    res['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    res['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
    res['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
    res['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
    res['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
    res['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
    res['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes

    # Sections
    res['SectionsNb'] = len(list(pe.sections))

    entropy = list(map(lambda x:x.get_entropy(), pe.sections))

    if float(len(list(entropy))) > 0.0:
        print("len of entropy=" +str(float(len(list(entropy)))))
        res['SectionsMeanEntropy'] = sum(entropy)/float(len(list(entropy)))
        res['SectionsMinEntropy'] = min(entropy)
        res['SectionsMaxEntropy'] = max(entropy)
    else:
        res['SectionsMeanEntropy'] = 0
        res['SectionsMinEntropy'] = 0
        res['SectionsMaxEntropy'] = 0

    raw_sizes = list(map(lambda x:x.SizeOfRawData, pe.sections))
    if len(list(raw_sizes)) > 0:
        print("len of raw_sizes=" +str(len(list(raw_sizes))))
        res['SectionsMeanRawsize'] = sum(raw_sizes)/len(list(raw_sizes))
        res['SectionsMinRawsize'] = min(raw_sizes)
        res['SectionsMaxRawsize'] = max(raw_sizes)
    else:
        res['SectionsMeanRawsize'] = 0
        res['SectionsMinRawsize'] = 0
        res['SectionsMaxRawsize'] = 0


    virtual_sizes = list(map(lambda x:x.Misc_VirtualSize, pe.sections))
    if len(list(virtual_sizes)) > 0:
        print("len of virtual_sizes=" +str(len(list(virtual_sizes))))
        res['SectionsMeanVirtualsize'] = sum(virtual_sizes)/float(len(list(virtual_sizes)))
        res['SectionsMinVirtualsize'] = min(virtual_sizes)
        res['SectionMaxVirtualsize'] = max(virtual_sizes)
    else:
        res['SectionsMeanVirtualsize'] = 0
        res['SectionsMinVirtualsize'] = 0
        res['SectionMaxVirtualsize'] =  0

    #Imports
    try:
        res['ImportsNbDLL'] = len(list(pe.DIRECTORY_ENTRY_IMPORT))
        imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
        res['ImportsNb'] = len(imports)
        res['ImportsNbOrdinal'] = len(list(filter(lambda x:x.name is None, imports)))
    except AttributeError:
        res['ImportsNbDLL'] = 0
        res['ImportsNb'] = 0
        res['ImportsNbOrdinal'] = 0

    #Exports
    try:
        res['ExportNb'] = len(list(pe.DIRECTORY_ENTRY_EXPORT.symbols))
    except AttributeError:
        # No export
        res['ExportNb'] = 0
    #Resources
    resources= get_resources(pe)
    res['ResourcesNb'] = len(list(resources))
    if len(resources)> 0:
        entropy = list(map(lambda x:x[0], resources))
        res['ResourcesMeanEntropy'] = sum(entropy)/float(len(list(entropy)))
        res['ResourcesMinEntropy'] = min(entropy)
        res['ResourcesMaxEntropy'] = max(entropy)
        sizes = list(map(lambda x:x[1], resources))
        res['ResourcesMeanSize'] = sum(sizes)/float(len(list(sizes)))
        res['ResourcesMinSize'] = min(sizes)
        res['ResourcesMaxSize'] = max(sizes)
    else:
        res['ResourcesNb'] = 0
        res['ResourcesMeanEntropy'] = 0
        res['ResourcesMinEntropy'] = 0
        res['ResourcesMaxEntropy'] = 0
        res['ResourcesMeanSize'] = 0
        res['ResourcesMinSize'] = 0
        res['ResourcesMaxSize'] = 0

    # Load configuration size
    try:
        res['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
    except AttributeError:
        res['LoadConfigurationSize'] = 0


    # Version configuration size
    try:
        version_infos = get_version_info(pe)
        res['VersionInformationSize'] = len(list(version_infos.keys()))
    except AttributeError:
        res['VersionInformationSize'] = 0
    return res


# if __name__ == '__main__':
	
#     clf = joblib.load('classifier/classifier.pkl')
#     features = pickle.loads(open(os.path.join('classifier/features.pkl'),'rb').read())
#     data = extract_infos('file/TestDummy2.exe')
#     pe_features = list(map(lambda x:data[x], features))

#     res= clf.predict([pe_features])[0]    
#     print ('The file is %s' % (['legitimate', 'malicious'][res]))

#####os.system("temp.py F:/Project_2019-2020/Malware_Detection_ML/MicrosoftEdgeSetup.exe")


from PIL import ImageTk,Image
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image



def Browse():
	textbox1.delete('1.0',"end-1c")
	filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =((("Executable file",".exe"),("Executable file",".exe")) ))
	textbox.delete('1.0', "end-1c")
	textbox.insert("end-1c", filename)
	#clf = joblib.load('classifier/classifier.pkl')
	clf = pickle.loads(open(os.path.join('classifier/classifier.pkl'),'rb').read())
	features = pickle.loads(open(os.path.join('classifier/features.pkl'),'rb').read())
	data = extract_infos(filename)
	pe_features = list(map(lambda x:data[x], features))

	res= clf.predict([pe_features])[4]
	print ('The file is %s' % (['legitimate', 'malicious'][res]))
	textbox1.insert("end-1c",str(['Legitimate', 'Malicious'][res]))
	if(['Legitimate', 'Malicious'][res] == 'Legitimate'):
			load = Image.open('tick.jpg')
			load = load.resize((630, 400), Image.ANTIALIAS)
			render = ImageTk.PhotoImage(load)
			img = Label(image=render)
			img.image = render
			img.place(x=75, y=235)
	else:
			load = Image.open('cross.jpg')
			load = load.resize((630, 400), Image.ANTIALIAS)
			render = ImageTk.PhotoImage(load)
			img = Label(image=render)
			img.image = render
			img.place(x=75, y=235)





app = tk.Tk()

HEIGHT = 700
WIDTH = 700

app.resizable(0,0)
canvas = Canvas(width=1300, height=700)
canvas.pack()
filename=('malware.jpg')
load = Image.open(filename)
load = load.resize((1300, 700), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
load = Image.open(filename)
img.place(x=1, y=1)


frame = tk.Frame(app,  bg='#3e3e32', bd=5)
frame.place(relx=0.3, rely=0.1, relwidth=0.5, relheight=0.25, anchor='n')
#frame_window = C.create_window(100, 40, window=frame)

textbox = tk.Text(frame, font=20,width="30",height=2)
textbox.grid(row=2, column=1)

submit = tk.Button(frame,font=40, text='BROWSE',height=1,width="13",command=lambda: Browse())
submit.grid(row=2, column=2,padx=20,pady=20)

textbox1 = tk.Text(frame, font=10,width="30",height=2)
textbox1.grid(row=3, column=1)

lower_frame = tk.Frame(app, bg='#3e3e32', bd=10)
lower_frame.place(relx=0.3, rely=0.32, relwidth=0.5, relheight=0.6, anchor='n')


bg_color = 'white'
results = tk.Text(lower_frame)
results.config(font=40, bg=bg_color)
results.place(relwidth=1, relheight=1)


app.mainloop()





