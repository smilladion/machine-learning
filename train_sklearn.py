from trainers import SKLearnTrainer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logistic = SKLearnTrainer(LogisticRegression())
logistic.train()
logistic.save()

svm_linear = SKLearnTrainer(LinearSVC())
svm_linear.train()
svm_linear.save()

svm_poly = SKLearnTrainer(SVC(kernel='poly'))
svm_poly.train()
svm_poly.save()

nearest = SKLearnTrainer(KNeighborsClassifier())
nearest.train()
nearest.save()
