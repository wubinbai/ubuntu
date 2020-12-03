h = help
import numpy as np
import pandas as pd
from pandas import read_csv as pdrc
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
#import seaborn as sns
# for ipython to display all results in the jupyter notebook:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=100)
plt.ion()

def plot_whole(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(df)

# Better help function he():

def he(): 
    global ar
    ar = input('Enter the function name for help:')
    help(eval(ar))
# for . operation of dir
# use eval(repr(xxx.i))


def my_plot(data_array):
    plt.figure()
    plt.plot(data_array)
    plt.grid()

def my_plotas(data_array):
    plt.figure()
    plt.plot(data_array)
    plt.plot(data_array,'b*')
    plt.grid()

def save_model_keras(model,save_path):
    from keras.utils import plot_model
    plot_model(model,show_shapes=True,to_file=save_path)



def torchviz_pdf(model,input_tensor):
    from torchviz import make_dot
    vis_graph = make_dot(model(input_tensor), params=dict(model.named_parameters()))
    vis_graph.view()  # 会在当前目录下保存一个“Digraph.gv.pdf”文件，并在默认浏览器中打开

def torch_summary(model,input_size):
    from torchsummary import summary
    summary(model.cuda(), input_size=input_size)

