import celery
import numpy as np
from copy import deepcopy
import json
import random
from scipy.stats import multivariate_normal
import os

# Make sure that the 'myguest' user exists with 'myguestpwd' on the RabbitMQ server and your load balancer has been set up correctly.
# My load balancer address is'RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com'.
# Below you will need to change it to your load balancer's address.

app = celery.Celery('gmm_workers',
                    broker='amqp://myguest:myguestpwd@10745-hw1-78ed39ec05bd3e11.elb.us-east-2.amazonaws.com',
                    backend='rpc://myguest:myguestpwd@10745-hw1-78ed39ec05bd3e11.elb.us-east-2.amazonaws.com')

iter = 0
mus=[]
pis=[]
sigmas=[]
rs=[]
R=[]

X = []
n_clusters = 0
n_features = 0


# worker_name='worker_'+str(random.randint(1, 10000))
# worker_name = os.getpid()

@app.task
def gmm_tasks(task, **kwargs):
    json_dump = kwargs['json_dump']
    json_load = json.loads(json_dump)
    global mus,sigmas, pis, R, X, n_clusters, n_features, rs, iter
    if task == 'E_step':
        sigmas = np.asarray(json_load["sigmas"])
        results = E_step()
        return results
    elif task=='M_step1':
        R = np.asarray(json.load["R"])
        results = M_step1()
        return results
    elif task=='M_step3':
        mus = np.asarray(json_load["mus"])
        pis = np.asarray(json_load["pis"])
        results = M_step3()
        return results
    elif task == 'data_to_workers':
        iter = 0
        mus = np.asarray(json_load["mus"])
        sigmas = np.asarray(json_load["sigmas"])
        pis = np.asarray(json_load["pis"])
        X = np.asarray(json_load["X"])
        n_clusters = json_load["n_clusters"]
        n_features = json_load["n_features"]
        print('n_clusters:', n_clusters)
        print('n_features:', n_features)
        return f'success: {len(X)} points uploaded'
    else:
        raise ValueError('undefined task')


# Euclidean Distance Caculator
def norm(mu,sigma,x):
    return multivariate_normal.pdf(x,mu,sigma)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def data_to_workers(C, X):
    print('*** we are sending data to the workers ***')
    # print('C',C)

    print("length of X:", len(X))
    for i in range(len(X)):
        # print(X[i])
        distances = dist(X[i]["points"], C)
        cluster = np.argmin(distances)
        X[i]["label"] = cluster
        # print(X[i].label)
    print(" *** Labels updated. E-step done ***")
    return json.dumps({'X': deepcopy(X)}, cls=NumpyEncoder)


def E_step():
    global iter
    print('*** we are in the E-Step ***')
    iter = iter + 1
    print("E step iter:", iter)
    rs=np.zeros((len(X),n_clusters))
    for j in range(len(X)):
        # print(X[i])
        deno = 0
        for i in range(n_clusters):
            deno += norm(mus[i],sigmas[i],X[j])*pis[i]
        for i in range(n_clusters):
            rs[j][i] = norm(mus[i],sigmas[i],X[j])*pis[i] / deno
        # print(X[i].label)
    print(" *** E-step done ***")
    return json.dumps({'s':deepcopy(np.sum(rs,axis=0))},cls=NumpyEncoder)

def M_step1():
    global ws
    print('*** we are in the M-Step1 ***')
    ws = rs / R
    muhats = np.zeros((n_clusters,2))
    for j in range(len(X)):
        for i in range(n_clusters):
            muhats[i] += ws[j][i]*X[j]
    print(" *** M-step1 done ***")
    return json.dumps({'muhats': deepcopy(muhats)},cls=NumpyEncoder)

def M_step3():
    print('*** we are in the M-Step3 ***')
    sigmahats = np.zeros((n_clusters,2,2))
    for j in range(len(X)):
        for i in range(n_clusters):
            sigmahats[i] += ws[j][i] * np.outer(X[j]-mus[i],X[j]-mus[i])
    print(" *** M-step3 done ***")
    return json.dumps({'sigmahats': deepcopy(sigmahats)},cls=NumpyEncoder)
