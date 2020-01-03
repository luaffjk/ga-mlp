from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint
import random
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae

iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def inicializacao_populacao_mlp(size_mlp):
    activation = ['identity','logistic', 'tanh', 'relu']
    solver = ['lbfgs','sgd', 'adam']
    pop =  np.array([[random.choice(activation), random.choice(solver), randint(2,100),randint(2,50)]])
    for i in range(0, size_mlp-1):
        pop = np.append(pop, [[random.choice(activation), random.choice(solver), randint(2,50),randint(2,50)]], axis=0)
    return pop

def cruzamento_mlp(pai_1, pai_2):
    child = [pai_1[0], pai_2[1], pai_1[2], pai_2[3]]    
    return child

def mutacao_mlp(child, prob_mut):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() >= prob_mut:
            k = randint(2,3)
            child_[c,k] = int(child_[c,k]) + randint(1, 4)
    return child_


def function_fitness_mlp(pop, X_train, y_train, X_test, y_test, size_mlp): 
    fitness = {}
    j = 0
    for w in pop:
        clf = MLPClassifier(activation=w[0], solver=w[1], alpha=1e-5, hidden_layer_sizes=(int(w[2]), int(w[3])),  max_iter=10000, n_iter_no_change=100)
        try:
            clf.fit(X_train, y_train)
            fitness[accuracy_score(clf.predict(X_test), y_test)] = [clf, w]
        except:
            pass
    return fitness


def ag_mlp(X_train, y_train, X_test, y_test, num_epochs = 10, size_mlp=10, prob_mut=0.8):
    pop = inicializacao_populacao_mlp(size_mlp)
    fitness = function_fitness_mlp(pop,  X_train, y_train, X_test, y_test, size_mlp)
    pop_fitness_sort = dict(reversed(sorted(fitness.items())))

    for j in range(0, num_epochs):
        #seleciona os pais
        parent_1 = np.array(list(dict(list(pop_fitness_sort.items())[:len(pop_fitness_sort)//2]).values()))[:, 1]
        parent_2 = np.array(list(dict(list(pop_fitness_sort.items())[len(pop_fitness_sort)//2:]).values()))[:, 1]

        #cruzamento
        child_1 = [cruzamento_mlp(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [cruzamento_mlp(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutacao_mlp(child_2, prob_mut)
        
        #calcula o fitness dos filhos para escolher quem vai passar pra próxima geração
        fitness_child_1 = function_fitness_mlp(child_1,X_train, y_train, X_test, y_test, size_mlp)
        fitness_child_2 = function_fitness_mlp(child_2, X_train, y_train, X_test, y_test, size_mlp)
        pop_fitness_sort.update(fitness_child_1)
        pop_fitness_sort.update(fitness_child_2)
        sort = dict(reversed(sorted(pop_fitness_sort.items())))
        
        #seleciona individuos da proxima geração
        pop_fitness_sort = dict(list(sort.items())[:size_mlp])
        best = list(reversed(sorted(pop_fitness_sort.keys())))[0]
        best_individual = pop_fitness_sort[best][0]
        print (pop_fitness_sort[best][1], best)
        
    return best_individual

melhor_result = ag_mlp(X_train, y_train, X_test, y_test, num_epochs = 10, size_mlp=20, prob_mut=0.5)
print (accuracy_score(melhor_result.predict(X_test), y_test))
