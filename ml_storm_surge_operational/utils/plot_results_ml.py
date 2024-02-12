import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def plot_learning_curves(history, metric, station, title='', path=None):
    station = "\n " + station
    if title:
        title = "\n " + title

    metric_str = metric.replace('_', ' ').title()    
    if metric =='root_mean_squared_error':
        plt.figure()
        plt.plot(np.sqrt(history.history['mean_squared_error']))
        plt.plot(np.sqrt(history.history["val_" + 'mean_squared_error']))
        plt.title("Model " + metric + station + title)
        plt.ylabel(metric + ' [m]', fontsize="large")
        plt.xlabel("Epoch", fontsize="large")
        plt.legend(["Train", "Validation"], loc="best")
        plt.grid()
        if path:
            plt.savefig(path)
        plt.show()
        plt.close()
    else:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("Model " + metric + station + title)
        plt.ylabel(metric + ' [m]', fontsize="large")
        plt.xlabel("Epoch", fontsize="large")
        plt.legend(["Train", "Validation"], loc="best")
        plt.grid()
        if path:
            plt.savefig(path)
        plt.show()
        plt.close()
    
def plot_rmse_vector(rmse_vector, horizon, station, path=None):
    station = "\n " + station
    
    print('horizon: ', horizon)
    print('rmse_vector.shape: ', rmse_vector.shape)
    print('rmse_vector: ', rmse_vector)
     
    plt.plot(np.arange(1, horizon + 1), rmse_vector)
    plt.grid()
    plt.ylabel('RMSE [m]')
    plt.xlabel('Hours')
    plt.title('RMSE' + station)
    if path:
            plt.savefig(path)
            
    plt.show()
    plt.close()