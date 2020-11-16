from src.model import * 

gao = pd.read_csv('data/G1.csv')
wang = pd.read_csv('data/W1.csv')
li = pd.read_csv('data/L1.csv')
yan = pd.read_csv('data/Y1.csv')

data = pd.concat([wang, yan, gao, li],axis=0)
data = shuffle(data)

X = np.array(data[['p','x','y','z']])
y = np.array(data[['label']])
sacler = StandardScaler()
sacler = sacler.fit(X)
X = sacler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=2020,
                                                    shuffle=True)

SVM_clf(X_train, X_test, y_train, y_test, X, y)
PCA_KNN(X_train, X_test, y_train, y_test, X, y)