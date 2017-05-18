from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plot
clf = svm.SVC(gamma=0.001, C=100.)
digits = datasets.load_digits()

data1=[ 0,0,7,8,13,16,15,1,
0 , 0 , 7 , 7 , 12 , 11 , 12 , 0 ,
0 , 0 , 0 , 0 , 0 , 0 , 12 , 0 ,
0 , 0 , 0 , 0 , 0 , 15 , 0 , 0 ,
0 , 0 , 0 , 0 , 15 , 0 , 0 , 0 ,
0 , 0 , 0 , 16 , 0 , 0 , 0 , 0 ,
0 , 0 , 9 , 0 , 0 , 0 , 0 , 0 ,
0 , 12 , 0 , 5 , 0 , 0 , 0 , 0
]

clf.fit(digits.data[:-1], digits.target[:-1])
result=clf.predict(data1)
print result

data1_plot=[ [0,0,7,8,13,16,15,1],
[0 , 0 , 7 , 7 , 12 , 11 , 12 , 0 ],
[0 , 0 , 0 , 0 , 0 , 0 , 12 , 0] ,
0 , 0 , 0 , 0 , 0 , 15 , 0 , 0 ,
0 , 0 , 0 , 0 , 15 , 0 , 0 , 0 ,
0 , 0 , 0 , 16 , 0 , 0 , 0 , 0 ,
0 , 0 , 9 , 0 , 0 , 0 , 0 , 0 ,
0 , 12 , 0 , 5 , 0 , 0 , 0 , 0
]


plot.figure(1,figsize=(3,3))
plot.imshow(data1_plot,cmap=plot.cm.gray_r,interpolation='nearest')
plot.show()



