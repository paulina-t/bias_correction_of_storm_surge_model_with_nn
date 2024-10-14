import numpy as np
from ml_storm_surge_operational.utils.helpers import mae, rmse, rmse_vector

class EvaluateModels():
    
    def __init__(self, x_test_norm, y_test_norm, multi_dense_model):
        self.x_test_norm = x_test_norm
        self.y_test_norm = y_test_norm
        self.multi_dense_model = multi_dense_model
        
    def loss_accuracy(self):
        # TODO: Save in log
        # test_loss, test_acc_mae, test_acc_mse = self.multi_dense_model.evaluate(self.x_test_norm, self.y_test_norm)
        test_loss, test_acc_mae = self.multi_dense_model.evaluate(self.x_test_norm, self.y_test_norm)

        print("Test accuracy MAE", test_acc_mae)
        #print("Test accuracy MSE", test_acc_mse)
        print("Test loss", test_loss) 
        
        # TODO: return all vars, there might be more 
        return test_loss, test_acc_mae#, test_acc_mse
    
    
    def predict(self, x_y_dict, normalized=True, t=None):
        y_test_norm_pred = self.multi_dense_model.predict(self.x_test_norm)
        #y_test_norm_pred = y_test_norm_pred.transpose()
        print('y_test_norm_pred: ', y_test_norm_pred)
        print('y_test_norm_pred.shape: ', y_test_norm_pred.shape)
        y_train_norm_pred = self.multi_dense_model.predict(x_y_dict['x_train_norm'])
        #y_train_norm_pred = y_train_norm_pred.transpose()
        
        if type(t) == int:
            if normalized:
                bool(x_y_dict) == True
                y_test_pred = (y_test_norm_pred * x_y_dict['y_train_std'][t]) + x_y_dict['y_train_mean'][t]
                print('y_test_pred: ', y_test_pred)
                print('y_test_pred.shape: ', y_test_pred.shape)
                y_train_pred = (y_train_norm_pred * x_y_dict['y_train_std'][t]) + x_y_dict['y_train_mean'][t]
                y_train = x_y_dict['y_train'][:, t].reshape(-1, 1)
                y_test = x_y_dict['y_test'][:, t].reshape(-1, 1)
                print('y_test: ', y_test)
                print('y_test.shape: ', y_test.shape)
    
            else:
                y_test_pred = y_test_norm_pred
                y_train_pred = y_train_norm_pred
                y_train = x_y_dict['y_train_norm'][:, t].reshape(-1, 1)
                y_test = x_y_dict['y_test_norm'][:, t].reshape(-1, 1)
                
        else:
            if normalized:
                bool(x_y_dict) == True
                y_test_pred = (y_test_norm_pred * x_y_dict['y_train_std']) + x_y_dict['y_train_mean']
                y_train_pred = (y_train_norm_pred * x_y_dict['y_train_std']) + x_y_dict['y_train_mean']
                y_train = x_y_dict['y_train']
                y_test = x_y_dict['y_test']
    
            else:
                y_test_pred = y_test_norm_pred
                y_train_pred = y_train_norm_pred
                y_train = x_y_dict['y_train_norm']
                y_test = x_y_dict['y_test_norm']
                
        print('y_test.shape: ', y_test.shape)
        print('y_test_pred.shape: ', y_test_pred.shape)
        
        train_pred_dict = {
            'y_train' : y_train,
            'y_test' : y_test,
            'y_train_pred' : y_train_pred,
            'y_test_pred' : y_test_pred
            }
        
        return train_pred_dict
    
    def evaluate(self, metric:str, train_pred_dict):
        
        y_train = train_pred_dict['y_train']
        y_test = train_pred_dict['y_test']
        y_train_pred = train_pred_dict['y_train_pred']
        y_test_pred = train_pred_dict['y_test_pred']
        
        if metric == 'mae': 
            train_eval = mae(y_train, y_train_pred)
            test_eval = mae(y_test, y_test_pred)
            
        if metric == 'rmse': 
            train_eval = rmse(y_train, y_train_pred)
            test_eval = rmse(y_test, y_test_pred)
            
        if metric == 'rmse_vector': 
            train_eval = rmse_vector(y_train, y_train_pred)
            test_eval = rmse_vector(y_test, y_test_pred)
            
        return train_eval, test_eval

    
    
    
    