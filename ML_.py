from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import Colours
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import os

data = pd.read_excel("HDH.xlsx")
X = data[["SR", "Tamb", "Uair", "Twci", "Twce", "Twhi", "Twhe", "Taci", "Tace", "Mair"]]
Y = data[["Output"]]


def svc_cv(C, epsilon, loss_n, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    loss_n = int(round(loss_n))
    loss_ = ['epsilon_insensitive', 'squared_epsilon_insensitive']
    t_size = 0.3
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = LinearSVR(epsilon=epsilon, C=C, loss=loss_[loss_n], max_iter=1000000, random_state=r_number, tol=0.01)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_svr(data, targets):
    """Apply Bayesian Optimization to SVC parameters."""

    def svc_crossval(C, epsilon, loss_n):
        return svc_cv(C=C, epsilon=epsilon, loss_n=loss_n, data=data, targets=targets)

    optimizer = BayesianOptimization(
        f=svc_crossval,
        pbounds={"C": (0.0001, 1000), "epsilon": (0, 0.2), 'loss_n': (0, 1)},
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./linsvrlogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./linsvrlogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def rfc_cv(n_estimators, min_samples_split, max_features, max_depth, loss_n, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    loss_n = int(round(loss_n))
    loss_ = ['mae', 'mse']
    t_size = 0.2
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_features=max_features,
            max_depth=max_depth,
            criterion=loss_[loss_n],
            random_state=r_number
        )
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_rf(data, targets):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def rfc_crossval(n_estimators, min_samples_split, max_features, max_depth, loss_n):
        return rfc_cv(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(round(max_features)),
            max_depth=int(max_depth),
            loss_n=loss_n,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds={
            "n_estimators": (10, 5000),
            "min_samples_split": (2, 25),
            "max_features": (1, 8),
            "max_depth": (2, 10),
            'loss_n': (0, 1)
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./randomlogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./randomlogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def mlp(h1, h2, h3, ac, so, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    ac = int(round(ac))
    so = int(round(so))
    ac_ = ['identity', 'logistic', 'tanh', 'relu']
    so_ = ['lbfgs', 'sgd', 'adam', 'sgd']
    t_size = 0.2
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = MLPRegressor(hidden_layer_sizes=(h1, h2, h3), activation=ac_[ac], solver=so_[so], random_state=r_number,
                          max_iter=100000)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_mlp(data, targets):
    """Apply Bayesian Optimization to mlp parameters."""

    def mlp_crossval(h1, h2, h3, ac, so):
        return mlp(
            h1=int(h1),
            h2=int(h2),
            h3=int(h3),
            ac=ac,
            so=so,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=mlp_crossval,
        pbounds={
            "h1": (1, 300),
            "h2": (1, 300),
            "h3": (1, 300),
            "ac": (0, 3),
            'so': (0, 3)
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./MLPreglogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./MLPreglogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


def svr(C, epsilon, ker, data, targets):
    scaler_zscore = preprocessing.StandardScaler()
    r21 = []
    ker = int(round(ker))
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    t_size = 0.2
    for r_number in range(10):
        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
        sc_ale = scaler_zscore.fit(X_train)
        X_train = scaler_zscore.transform(X_train)
        X_test = scaler_zscore.transform(X_test)
        sr = SVR(epsilon=epsilon, C=C, kernel=kernel[ker], max_iter=100000)
        sr.fit(X_train, y_train)
        y_pred_test = sr.predict(X_test)
        r21.append(r2_score(y_test, y_pred_test))
    r21 = np.array(r21)
    return r21.mean()


def optimize_ssvr(data, targets):
    """Apply Bayesian Optimization to ssvr parameters."""

    def ssvr_crossval(C, epsilon, ker):
        return svr(
            C=C,
            epsilon=epsilon,
            ker=ker,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=ssvr_crossval,
        pbounds={
            "C": (0.0001, 1000),
            "epsilon": (0, 0.2),
            "ker": (0, 3),
        },
        random_state=1234,
        verbose=2
    )
    logger = JSONLogger(path="./SVRmaclogs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=300)
    with open('./SVRmaclogs.json', 'a', encoding='utf-8', errors='replace')as f:
        f.write(str(optimizer.max))


# 误差和迭代次数的关系
def plot_optimization(filename):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        values = []
        for id, line in enumerate(lines):
            line = eval(line)
            if id == 0:
                values.append(line["target"])
            else:
                max_ = max(values)
                now = line["target"]
                if now > max_:
                    values.append(now)
                else:
                    values.append(max_)
        x = range(len(values))
        fig = plt.figure()
        fig.suptitle("optimization result of %s" % filename)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(x, values)
        ax1.set_xlabel("Generations")
        ax1.set_ylabel("r2 value of test data")
        if not os.path.exists('picture'):
            os.mkdir('picture')
        fig.savefig('picture/%s.png' % filename[7:13])
        plt.show()


def plot_r2_rf(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        n_estimators = int(dic['n_estimators'])
        min_samples_split = int(dic['min_samples_split'])
        max_features = int(round(dic['max_features']))
        max_depth = int(dic['max_depth'])
        loss_n = int(round(dic['loss_n']))
        loss_ = ['mae', 'mse']
        t_size = 0.2
        mse0 = []
        mae0 = []
        r20 = []
        mse1 = []
        mae1 = []
        r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(5):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)

            sr = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                max_features=max_features,
                max_depth=max_depth,
                criterion=loss_[loss_n],
                random_state=r_number
            )
            sr.fit(X_train, y_train.values.ravel())
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)

            # You need to adjust the shape of the data

            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:       # y_train Training label
                train.append(i)
            for i in y_pred_train:  # y_train predicting label
                train_pred.append(i)
            for i in y_test:
                test.append(i)
            for i in y_pred_test:
                test_pred.append(i)
            mse0.append(mean_squared_error(y_train, y_pred_train))  # MSE
            mae0.append(mean_absolute_error(y_train, y_pred_train))  # MAE
            r20.append(r2_score(y_train, y_pred_train))  # The model R2 accuracy, the best is 1, may be negative
            mse1.append(mean_squared_error(y_test, y_pred_test))
            mae1.append(mean_absolute_error(y_test, y_pred_test))
            r21.append(r2_score(y_test, y_pred_test))

        mse0 = np.array(mse0)
        mae0 = np.array(mae0)
        r20 = np.array(r20)
        mse1 = np.array(mse1)
        mae1 = np.array(mae1)
        r21 = np.array(r21)
        fig = plt.figure(figsize=(10, 4))  # Set canvas
        fig.suptitle("bayesian optimization of %s forest" % filename)  # name
        ax1 = fig.add_subplot(1, 2, 1)  # Divide the canvas into a row and two columns, a total of 2 subgraphs, and locate on the first subgraph
        ax2 = fig.add_subplot(1, 2, 2)  # locate on the second subgraph
        x = [0, 7]
        y = [0, 7]
        ax1.plot(x, y, color='grey', linewidth=0.5)
        ax1.grid(True, color='grey', linewidth=0.5)
        ax1.scatter(train, train_pred, color='lightpink', marker='.')  # Scatter plot
        ax1.text(0.06, 6.50, s='train_mae=%f' % mae0.mean())
        ax1.text(0.06, 6.20, s='train_mse=%f' % mse0.mean())
        ax1.text(0.06, 5.90, s='train_r2 score=%f' % r20.mean())
        ax1.set_xlabel("Actual value")  # Set horizontal axis title
        ax1.set_ylabel("Predict value")  # Set vertical axis title
        ax2.plot(x, y, color='grey', linewidth=0.5)
        ax2.grid(True, color='grey', linewidth=0.5)
        ax2.scatter(test, test_pred, color='cornflowerblue', marker='.')
        ax2.set_xlabel("Actual value")
        ax2.set_ylabel("Predict value")
        ax2.text(0.06, 6.50, s='test_mae=%f' % mae1.mean())
        ax2.text(0.06, 6.20, s='test_mse=%f' % mse1.mean())
        ax2.text(0.06, 5.90, s='test_r2 score=%f' % r21.mean())
        ax2.set_xlim(0.00, 7.00)
        ax2.set_ylim(0.00, 7.00)
        ax1.set_xlim(0.00, 7.00)
        ax1.set_ylim(0.00, 7.00)
        if not os.path.exists('picture'):
            os.mkdir('picture')
        fig.savefig('picture/%svalue.png' % filename[0:4])
        plt.grid(color='grey', linewidth=0.5)
        plt.show()


def plot_r2_mlp(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        h1 = int(dic['h1'])
        h2 = int(dic['h2'])
        h3 = int(dic['h3'])
        ac = int(dic['ac'])
        so = int(dic['so'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        mse0 = []
        mae0 = []
        r20 = []
        mse1 = []
        mae1 = []
        r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(10):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # Data standardization
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            # Different algorithms use different regression methods
            sr = MLPRegressor(
                # h1=h1,
                # h2=h2,
                # h3=h3,
                # activation=ac,
                # solver=so,
                # criterion=loss_[loss_n],
                hidden_layer_sizes=(h1, h2, h3),
                activation="relu",
                solver='adam',
                max_iter=100000,
                random_state=r_number
            )
            sr.fit(X_train, y_train)
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)


            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:
                train.append(i)
            for i in y_pred_train:
                train_pred.append(i)
            for i in y_test:
                test.append(i)
            for i in y_pred_test:
                test_pred.append(i)
            mse0.append(mean_squared_error(y_train, y_pred_train))
            mae0.append(mean_absolute_error(y_train, y_pred_train))
            r20.append(r2_score(y_train, y_pred_train))
            mse1.append(mean_squared_error(y_test, y_pred_test))
            mae1.append(mean_absolute_error(y_test, y_pred_test))
            r21.append(r2_score(y_test, y_pred_test))

        mse0 = np.array(mse0)
        mae0 = np.array(mae0)
        r20 = np.array(r20)
        mse1 = np.array(mse1)
        mae1 = np.array(mae1)
        r21 = np.array(r21)
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle("bayesian optimization of %s" % filename[0:15])
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        x = [0, 7]
        y = [0, 7]
        ax1.plot(x, y, color='grey', linewidth=0.5)
        ax1.scatter(train, train_pred, color='lightpink')
        ax1.text(0.06, 6.50, s='train_mae=%f' % mae0.mean())
        ax1.text(0.06, 6.00, s='train_mse=%f' % mse0.mean())
        ax1.text(0.06, 5.50, s='train_r2 score=%f' % r20.mean())
        ax1.set_xlabel("Actual value")
        ax1.set_ylabel("Predict value")
        ax2.plot(x, y, color='grey', linewidth=0.5)
        ax2.scatter(test, test_pred, color='cornflowerblue')
        ax2.set_xlabel("Actual value")
        ax2.set_ylabel("Predict value")
        ax2.text(0.06, 6.50, s='test_mae=%f' % mae1.mean())
        ax2.text(0.06, 6.00, s='test_mse=%f' % mse1.mean())
        ax2.text(0.06, 5.50, s='test_r2 score=%f' % r21.mean())
        ax2.set_xlim(0.00, 7.00)
        ax2.set_ylim(0.00, 7.00)
        ax1.set_xlim(0.00, 7.00)
        ax1.set_ylim(0.00, 7.00)
        if not os.path.exists('picture'):
            os.mkdir('picture')
        # fig.savefig('picture/%svalue.png' % filename[7:13])
        plt.show()

        # Using support vector machine algorithm


def plot_r2_svr(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        # loss_n = int(round(dic['loss_n']))
        epsilon = int(dic['epsilon'])
        C = int(dic['C'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        mse0 = []
        mae0 = []
        r20 = []
        mse1 = []
        mae1 = []
        r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(10):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()
            # 数据标准化 z-score
            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            sr = LinearSVR(
                # loss_n=loss_n,
                epsilon=epsilon,
                C=C,
                # ac=ac,
                # so=so,
                # criterion=loss_[loss_n],
                # random_state=r_number
            )
            sr.fit(X_train, y_train)
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)



            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:
                train.append(i)
            for i in y_pred_train:
                train_pred.append(i)
            for i in y_test:
                test.append(i)
            for i in y_pred_test:
                test_pred.append(i)
            mse0.append(mean_squared_error(y_train, y_pred_train))
            mae0.append(mean_absolute_error(y_train, y_pred_train))
            r20.append(r2_score(y_train, y_pred_train))
            mse1.append(mean_squared_error(y_test, y_pred_test))
            mae1.append(mean_absolute_error(y_test, y_pred_test))
            r21.append(r2_score(y_test, y_pred_test))

        mse0 = np.array(mse0)
        mae0 = np.array(mae0)
        r20 = np.array(r20)
        mse1 = np.array(mse1)
        mae1 = np.array(mae1)
        r21 = np.array(r21)
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle("bayesian optimization of %s" % filename[0:15])
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(X_train, X_train, color='black', linewidth=1.0)
        ax1.scatter(train, train_pred, color='b')
        ax1.text(0.06, 0.9, s='train_mae=%f' % mae0.mean())
        ax1.text(0.06, 0.85, s='train_mse=%f' % mse0.mean())
        ax1.text(0.06, 0.8, s='train_r2 score=%f' % r20.mean())
        ax1.set_xlabel("Actual value")
        ax1.set_ylabel("Predict value")
        ax2.plot(X_test, X_test, color='black', linewidth=1.0)
        ax2.scatter(test, test_pred, color='b')
        ax2.set_xlabel("Actual value")
        ax2.set_ylabel("Predict value")
        ax2.text(0.06, 0.9, s='test_mae=%f' % mae1.mean())
        ax2.text(0.06, 0.85, s='test_mse=%f' % mse1.mean())
        ax2.text(0.06, 0.8, s='test_r2 score=%f' % r21.mean())
        ax2.set_xlim(0.05, 0.95)
        ax2.set_ylim(0.05, 0.95)
        ax1.set_xlim(0.05, 0.95)
        ax1.set_ylim(0.05, 0.95)
        if not os.path.exists('picture'):
            os.mkdir('picture')
        fig.savefig('picture/%svalue.png' % filename[7:13])
        plt.show()


def plot_r2_ssvr(filename, data, targets):
    with open(filename, 'r', encoding='utf-8', errors='replace')as f:
        lines = f.readlines()
        line = eval(lines[-1])
        dic = line["params"]
        # loss_n = int(round(dic['loss_n']))
        ker = int(dic['ker'])
        epsilon = int(dic['epsilon'])
        C = int(dic['C'])
        loss_ = ['mae', 'mse']
        t_size = 0.2
        mse0 = []
        mae0 = []
        r20 = []
        mse1 = []
        mae1 = []
        r21 = []
        train = []
        train_pred = []
        test = []
        test_pred = []

        for r_number in range(10):
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size,
                                                                random_state=r_number)
            scaler_zscore = preprocessing.StandardScaler()

            sc_ale = scaler_zscore.fit(X_train)
            X_train = scaler_zscore.transform(X_train)
            X_test = scaler_zscore.transform(X_test)
            sr = SVR(
                # loss_n=loss_n,
                epsilon=epsilon,
                C=C,
                kernel=ker
                # ac=ac,
                # so=so,
                # criterion=loss_[loss_n],
                # random_state=r_number
            )
            sr.fit(X_train, y_train)
            y_pred_test = sr.predict(X_test)
            y_pred_train = sr.predict(X_train)



            y_train = np.array(y_train).reshape(-1, )
            y_test = np.array(y_test).reshape(-1, )

            for i in y_train:
                train.append(i)
            for i in y_pred_train:
                train_pred.append(i)
            for i in y_test:
                test.append(i)
            for i in y_pred_test:
                test_pred.append(i)
            mse0.append(mean_squared_error(y_train, y_pred_train))
            mae0.append(mean_absolute_error(y_train, y_pred_train))
            r20.append(r2_score(y_train, y_pred_train))
            mse1.append(mean_squared_error(y_test, y_pred_test))
            mae1.append(mean_absolute_error(y_test, y_pred_test))
            r21.append(r2_score(y_test, y_pred_test))

        mse0 = np.array(mse0)
        mae0 = np.array(mae0)
        r20 = np.array(r20)
        mse1 = np.array(mse1)
        mae1 = np.array(mae1)
        r21 = np.array(r21)
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle("bayesian optimization of %s" % filename[0:15])
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(X_train, X_train, color='black', linewidth=1.0)
        ax1.scatter(train, train_pred, color='b')
        ax1.text(0.06, 0.9, s='train_mae=%f' % mae0.mean())
        ax1.text(0.06, 0.85, s='train_mse=%f' % mse0.mean())
        ax1.text(0.06, 0.8, s='train_r2 score=%f' % r20.mean())
        ax1.set_xlabel("Actual value")
        ax1.set_ylabel("Predict value")
        ax2.plot(X_test, X_test, color='black', linewidth=1.0)
        ax2.scatter(test, test_pred, color='b')
        ax2.set_xlabel("Actual value")
        ax2.set_ylabel("Predict value")
        ax2.text(0.06, 0.9, s='test_mae=%f' % mae1.mean())
        ax2.text(0.06, 0.85, s='test_mse=%f' % mse1.mean())
        ax2.text(0.06, 0.8, s='test_r2 score=%f' % r21.mean())
        ax2.set_xlim(0.05, 0.95)
        ax2.set_ylim(0.05, 0.95)
        ax1.set_xlim(0.05, 0.95)
        ax1.set_ylim(0.05, 0.95)
        if not os.path.exists('picture'):
            os.mkdir('picture')
        fig.savefig('picture/%svalue.png' % filename[7:13])
        #plt.show()

if __name__ == '__main__':
    #optimize_rf(X, targets=Y)
    #optimize_mlp(X, targets=Y)
    #optimize_svr(X, targets=Y)
    #optimize_ssvr(X, targets=Y)
    #plot_r2_rf("randomlogs.json", X, Y)
    plot_r2_mlp("MLPreglogs.json", X, Y)
    #plot_r2_svr("linsvrlogs.json", X, Y)
    #plot_r2_ssvr("SVRmaclogs.json", X, Y)
    #plot_optimization("randomlogs.json")
    #plot_optimization("MLPreglogs.json")
    # plot_optimization("linsvrlogs.json")
    # plot_optimization("SVRmaclogs.json")
