from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import joblib
#print('gaussian',cross_val_score(GaussianNB(), patches_hog, y_train))

def train_SVM(projections,labels):
    grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
    grid.fit(projections, labels)
    print('svm',grid.best_score_)
    # bensave 2el model 3ashan mane3melsh train kol shwaya
    filename = 'Recognition_model.sav'
    joblib.dump(grid, filename)
    