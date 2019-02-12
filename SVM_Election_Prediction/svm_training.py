from svmutil import *
from utils import *
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import warnings
#warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
from numpy import random
warnings.filterwarnings(action='once')

def train_svm(x_train,y_train,x_test,y_test, param):

    prob = svm_problem(y_train, x_train)
    m = svm_train(prob, param)
    svm_save_model('part1a.model', m)
    m = svm_load_model('part1a.model')
    p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    ACC, MSE, SCC = evaluations(y_test, p_label)
    print("Accuracy =  ", ACC)
    print("MSE =  ",MSE)
    return ACC,MSE

def find_best_svm_2(landmarks,annotations ):

    x = Normalize(landmarks)
    # x = x.tolist()
    y = annotations[:, 0]

    X_train_folds = []
    Y_train_folds = []
    k = 5
    step_size = int(x.shape[0] / k)
    k1 = 0
    for i in range(0, k):
        X_train_folds.append(x[k1:k1 + step_size, :])
        Y_train_folds.append(y[k1:k1 + step_size, :])
        k1 = k1 + step_size

    g = [1, 0.5, 2]
    c = [1]
    results = np.zeros(shape=(len(g), len(c)))

    for g_index, g_val in enumerate(g):
        for c_index, c_val in enumerate(c):
            errors_of_this_k_fold = np.zeros(k)
            for j in range(0, 5):
                X_test = np.reshape(X_train_folds[j], (X_train_folds[j].shape[0], -1))
                y_train_4folds = []
                y_test = np.reshape(Y_train_folds[j], (Y_train_folds[j].shape[0], -1))
                X_train_4folds = []

                for l in range(0, 5):
                    if (l != j):
                        # np.stack()
                        X_train_4folds.append(X_train_folds[l])
                        y_train_4folds.append(Y_train_folds[l])

                X_train_4folds = np.array(X_train_4folds)
                a = (X_train_4folds[0, :])
                b = (X_train_4folds[1, :])
                a = np.concatenate((a, b), axis=0)
                c = (X_train_4folds[2, :])
                a = np.concatenate((a, c), axis=0)
                d = (X_train_4folds[3, :])
                a = np.concatenate((a, d), axis=0)
                # print("a",(a.shape))
                X_train = a

                Y_train_4folds = np.array(y_train_4folds)
                a = (Y_train_4folds[0, :])
                b = (Y_train_4folds[1, :])
                a = np.concatenate((a, b), axis=0)
                c = (Y_train_4folds[2, :])
                a = np.concatenate((a, c), axis=0)
                d = (Y_train_4folds[3, :])
                a = np.concatenate((a, d), axis=0)

                y_train = a
                # print("y_train",(y_train.shape))
                accuracy, mse = train_svm(X_train.tolist(), y_train.tolist())
                errors_of_this_k_fold[j] = mse

        results[g_index, c_index] = np.average(errors_of_this_k_fold)
        print("error for current g:" + g_val + ", current c:" + c_val + "is = ", results[k])

    # ks_min = ks[np.argsort(results)[0]]
    results_min = min(results)

    # print('Set k = {0} and get minimum error as {1}'.format(ks_min,results_min))

def find_best_svm_(X_train, y_train,X_test,y_test,param_grid,save_dir_clf= None,computeagain = False):
    #X_train  = X
    #y_train = Y
    print("X_train.shape",X_train.shape)
    #print("X_train", X_train)

    print("X_test.shape", X_test.shape)
    #print("X_test", X_test)

    print("y_train.shape", y_train.shape)
    #print("y_train", y_train)

    print("y_test.shape", y_test.shape)
    #print("y_test", y_test)

    threshold = np.mean(y_train)

    if save_dir_clf is not None and os.path.exists(save_dir_clf) and computeagain==False:
        print('[Find cached clf, %s loading...]' % save_dir_clf)
        clf = np.load(save_dir_clf)
    else:

        print("Fitting the classifier to the training set")
        t0 = time()


        clf = GridSearchCV(SVR(kernel='rbf'),
                          param_grid, cv=5,scoring = ['neg_mean_squared_error'],refit='neg_mean_squared_error', n_jobs=-1)

        #clf = GridSearchCV(SVR(kernel='rbf'),
                           #param_grid, cv=5, n_jobs=-1)

        clf = clf.fit(X_train, y_train)
        if save_dir_clf is not None:
            pickle.dump(clf, open(save_dir_clf, 'wb'))
        print("done in %0.3fs" % (time() - t0))

    print("best_score_  found by grid search:")
    print(clf.best_score_ )


    print("Predicting on the training set")
    t0 = time()
    y_pred_train = clf.predict(X_train)
    print("done in %0.3fs" % (time() - t0))
    y_pred_train_labels = get_discrete(y_pred_train,threshold)
    y_train_labels = get_discrete(y_train,threshold)
    train_accuracy = accuracy_score(y_train_labels, y_pred_train_labels)
    train_precision = average_precision_score(y_train_labels, y_pred_train_labels)
    train_mse  = mean_squared_error(y_train, y_pred_train)

    print("Predicting on the test set")
    t0 = time()
    y_pred_test = clf.predict(X_test)
    y_pred_test_labels = get_discrete(y_pred_test,threshold)
    y_test_labels = get_discrete(y_test,np.mean(y_test))
    print("done in %0.3fs" % (time() - t0))
    test_accuracy = accuracy_score(y_test_labels, y_pred_test_labels)
    test_precision = average_precision_score(y_test_labels, y_pred_test_labels)

    test_mse = mean_squared_error(y_test, y_pred_test)

    best_params = clf.best_params_
    result = {
        "train_accuracy" :train_accuracy,
        "train_precision" :train_precision,
        "train_mse" :train_mse,
        "test_accuracy":test_accuracy,
        "test_precision":test_precision,
        "test_mse" : test_mse,
        "best_params":best_params,
    }
    return result

def find_best_svc_part21(X_train, y_train,X_test, y_test,save_dir_clf= None):
    #X_train  = X
    #y_train = Y
    threshold = np.mean(y_train)

    if save_dir_clf is not None and os.path.exists(save_dir_clf):
        print('[Find cached clf, %s loading...]' % save_dir_clf)
        clf = np.load(save_dir_clf)
    else:

        print("Fitting the classifier to the training set")
        t0 = time()

        param_grid = {'C': [1e3, 5e3, 10, 1, 1000, ]}
        clf = GridSearchCV(LinearSVC(fit_intercept=False),
                          param_grid, cv=5, n_jobs=-1)
        #clf = GridSearchCV(SVR(kernel='rbf'),
                           #param_grid, cv=5, n_jobs=-1)
        clf = clf.fit(X_train, y_train)
        if save_dir_clf is not None:
            pickle.dump(clf, open(save_dir_clf, 'wb'))
        print("done in %0.3fs" % (time() - t0))

    print("best_score_  found by grid search:")
    print(clf.best_score_ )
    print("best projection param w found by grid search:")
    print(clf.best_estimator_.coef_)

    print("Predicting on the training set")
    t0 = time()
    y_pred_train = clf.predict(X_train)
    print("done in %0.3fs" % (time() - t0))
    train_accuracy = accuracy_score(y_train, y_pred_train)

    print("Predicting on the test set")
    t0 = time()
    y_pred_test = clf.predict(X_test)
    print("done in %0.3fs" % (time() - t0))
    test_accuracy = accuracy_score(y_test, y_pred_test)

    mean_train_accuracy = clf.cv_results_['mean_train_score'][clf.best_index_]
    mean_test_accuracy = clf.cv_results_['mean_test_score'][clf.best_index_]

    best_params = clf.best_params_
    result = {
        "mean_train_accuracy_kfold" : mean_train_accuracy,
        "mean_test_accuracy_kfold" : mean_test_accuracy,
        "train_accuracy" :train_accuracy,
        "test_accuracy":test_accuracy,
        "best_params":best_params,
    }

    #return [mean_train_accuracy,mean_test_accuracy,mean_train_precision,mean_test_precision,train_accuracy,train_precision,test_accuracy,test_precision],best_params
    return result

def find_best_svc_part21_nosplit(X,Y,save_dir_clf= None):
    #X_train  = X
    #y_train = Y
    threshold = np.mean(Y)

    if save_dir_clf is not None and os.path.exists(save_dir_clf):
        print('[Find cached clf, %s loading...]' % save_dir_clf)
        clf = np.load(save_dir_clf)
    else:

        print("Fitting the classifier to the training set")
        t0 = time()

        param_grid = {'C': [ 2e-9, 2e-7, 2 ** 15, 10 ** 10]}
        clf = GridSearchCV(LinearSVC(fit_intercept=False),
                          param_grid, cv=5, n_jobs=-1)
        #clf = GridSearchCV(SVR(kernel='rbf'),
                           #param_grid, cv=5, n_jobs=-1)
        clf = clf.fit(X, Y)
        if save_dir_clf is not None:
            pickle.dump(clf, open(save_dir_clf, 'wb'))
        print("done in %0.3fs" % (time() - t0))

    print("best_score_  found by grid search:")
    print(clf.best_score_ )

    mean_train_accuracy = clf.cv_results_['mean_train_score'][clf.best_index_]
    mean_test_accuracy = clf.cv_results_['mean_test_score'][clf.best_index_]

    best_params = clf.best_params_
    result = {
        "mean_train_accuracy_kfold" : mean_train_accuracy,
        "mean_test_accuracy_kfold" : mean_test_accuracy,
        "best_params":best_params,
    }

    #return [mean_train_accuracy,mean_test_accuracy,mean_train_precision,mean_test_precision,train_accuracy,train_precision,test_accuracy,test_precision],best_params
    return result


def train_and_plot_svm3(X, Y, param_grid, type, model_no, seed=None,computeagain=False):
    mean_train_accuracies_kfold = []
    mean_test_accuracies_kfold = []
    train_accuracies_finalmodel = []
    test_accuracies_finalmodel = []

    mean_train_mse_kfold = []
    mean_test_mse_kfold = []
    train_mse_finalmodel = []
    test_mse_finalmodel = []

    mean_train_precision_kfold = []
    mean_test_precision_kfold = []
    train_precision_finalmodel = []
    test_precision_finalmodel = []

    no_of_models = 1

    if (seed == None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.10, shuffle=True, random_state=seed)

    results = find_best_svm_(X_train, y_train[:, model_no], X_test,
                             y_test[:, model_no], param_grid, save_dir_clf='clf_' + str(model_no) + type + '.pkl',computeagain = computeagain)


    print('model no: ', model_no)
    print("results = ", results)
    train_accuracies_finalmodel.append(results['train_accuracy'])
    test_accuracies_finalmodel.append(results['test_accuracy'])
    train_mse_finalmodel.append(results['train_mse'])
    test_mse_finalmodel.append(results['test_mse'])
    train_precision_finalmodel.append(results['train_precision'])
    test_precision_finalmodel.append(results['test_precision'])


    return train_accuracies_finalmodel, test_accuracies_finalmodel, train_mse_finalmodel, test_mse_finalmodel, train_precision_finalmodel, test_precision_finalmodel



def train_and_plot_svm(X,Y,param_grid, type,seed=None, ylim  = None):
    mean_train_accuracies_kfold = []
    mean_test_accuracies_kfold = []
    train_accuracies_finalmodel = []
    test_accuracies_finalmodel = []

    mean_train_mse_kfold = []
    mean_test_mse_kfold = []
    train_mse_finalmodel = []
    test_mse_finalmodel = []

    mean_train_precision_kfold = []
    mean_test_precision_kfold = []
    train_precision_finalmodel = []
    test_precision_finalmodel = []

    no_of_models = 14

    if(seed == None):
        X_train, X_test, y_train, y_test = train_test_split(
         X, Y, test_size=0.1, shuffle = False )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.10, shuffle=True,random_state=seed)

    for i in range(0,no_of_models):

        results = find_best_svm_(X_train,y_train[:,i],X_test,
                                y_test[:,i],param_grid,save_dir_clf='clf_'+str(i) + type +'.pkl')
        print('model no: ',i)
        print("results = ", results)
        train_accuracies_finalmodel.append(results['train_accuracy'])
        test_accuracies_finalmodel.append(results['test_accuracy'])
        train_mse_finalmodel.append(results['train_mse'])
        test_mse_finalmodel.append(results['test_mse'])
        train_precision_finalmodel.append(results['train_precision'])
        test_precision_finalmodel.append(results['test_precision'])

    plot_graphs(no_of_models,train_accuracies_finalmodel,test_accuracies_finalmodel,'mean train-','mean test-','mean accuracies-','mean accuracies-',type,ylim)
    plot_graphs(no_of_models,train_mse_finalmodel,test_mse_finalmodel,'mean train-','mean test-','mean mse-','mean mse-',type)
    plot_graphs(no_of_models,train_precision_finalmodel,test_precision_finalmodel,'mean train-','mean test-','mean precision-','mean precision-',type)
    return train_accuracies_finalmodel,test_accuracies_finalmodel,train_mse_finalmodel,test_mse_finalmodel,train_precision_finalmodel, test_precision_finalmodel

def train_and_plot_part21(X,Y,type,type_of_politician,seed):
    mean_train_accuracies_kfold = []
    mean_test_accuracies_kfold = []
    train_accuracies_finalmodel = []
    test_accuracies_finalmodel = []

    length_x = X.shape[0]
    no_OF_Train = int(0.9 * length_x)

    Y_new = Y
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, shuffle=True,random_state=seed) #33(governor), #53(sen)

    results = find_best_svc_part21(X_train, y_train, X_test,
                            y_test, save_dir_clf='clf_' + type_of_politician + type + '.pkl')
    #results = find_best_svc_part21_nosplit(X,Y, save_dir_clf='clf_' + type_of_politician + type + '.pkl')
    print("results = ", results)
    train_accuracies_finalmodel.append(results['train_accuracy'])
    test_accuracies_finalmodel.append(results['test_accuracy'])
    mean_train_accuracies_kfold.append(results['mean_train_accuracy_kfold'])
    mean_test_accuracies_kfold.append(results['mean_test_accuracy_kfold'])
    #plot_graphs(no_of_models, train_accuracies_finalmodel, test_accuracies_finalmodel, 'mean train-bestmodel',
                #'mean test-bestmodel', 'mean accuracies-bestmodel', 'Part2.1 mean accuracies-bestmodel', type)
    #plot_graphs(no_of_models,mean_train_accuracies_kfold,mean_test_accuracies_kfold,'mean train-kfold','mean test-kfold','mean accuracies-kfold','Part2.1 mean accuracies-kfold')
    return train_accuracies_finalmodel, test_accuracies_finalmodel

def plot_graphs(no_of_models, values_A,values_B,label_A,label_B,Y_label,Title,type, ylim = None):
    plt.figure()
    no_of_models = list(range(no_of_models))
    plt.plot(no_of_models, values_A, 'o-', color="r",
             label=label_A+type)
    plt.plot(no_of_models, values_B, 'o-', color="g",
             label=label_B+type)
    plt.xlabel("Model No")
    plt.ylabel(Y_label+type)
    if(ylim != None):
        plt.ylim(ylim, 1.0)
    plt.title(Title +type )
    plt.legend(loc="best")
    plt.savefig(Title +type + '.png')
    #plt.show()

def get_discrete(y_pred, threshold):
    result = []
    mean = np.mean(y_pred)

    for j in range(0, y_pred.shape[0]):
        if (y_pred[j] >= threshold):
            label = 1
        else:
            label = -1
        result.append(label)

    return np.array(result)

def mean_sqr_error(y_true,y_pred):
    mse = np.square(y_true - y_pred)
    mse = np.mean(mse)
    return mse

def predict_using_svm(save_dir_clf,X):
    if os.path.exists(save_dir_clf):
        print('[Find cached clf, %s loading...]' % save_dir_clf)
        clf = np.load(save_dir_clf)
    else:
        print('trained svm not found!')
        return

    print("Predicting on X")
    y_pred = clf.predict(X)
    return y_pred