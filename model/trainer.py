import numpy as np
import sys
import glob
import pdb
from sklearn.model_selection import train_test_split
import pointer_capsnet as model
sys.path.append('/home/sanket/MS_Thesis/Pointer_CapsNet/Dataset')
from dataset_iter import dataset_iterator
import tensorflow as tf
import logging
import os
from config import cfg

from result_dir import *
# from store_result import *





def calculate_accuracy(prediction, labels):
    #pdb.set_trace()
    prediction = np.argmax(prediction, axis = -1 )
    correct = np.sum(prediction == labels)
    #pdb.set_trace()
    return (correct/ (len(prediction)*len(prediction[0])))

def calculate_class_accuracy(prediction, labels):
    prediction = np.argmax(prediction, axis = -1 )
    key_dict = {'Car':1, 'Van': 2, 'Truck':3,
             'Pedestrian':4, 'Person_sitting':5,
             'Cyclist': 6 , 'Tram' : 7 ,
             'Misc' : 0 , 'DontCare': 0}
    c2 = []
    for i in range(1,8):
        indx1 =  np.where(prediction == i)[1]
        indx2 =  np.where(labels  == i)[1]
        if len(indx2) ==  0:
            pass
        else:
            correct = np.sum(np.in1d(indx1,indx2))
            correct = (correct/ len(indx2))
            c2.append(correct)
    if len(c2) > 0 : #checking for empty array for no class frame
        return (np.mean(c2)/10.)
    else:
        return 0.


def calculate_car_accuracy(pred, label):
    pred = np.argmax(pred, axis = -1)
    c1 = np.where(pred == 1)[1]
    c2 = np.where(label == 1)[1]
    if len(c2) == 0:
        return 0.
    else:
        correct = np.sum(np.in1d(c1,c2))
        correct = (correct/ (len(c2)))
        return correct/10.


def get_one_hot_label(label):
    shape_l = label.shape
    one_hot = np.zeros((shape_l[0],shape_l[1],2))
    for j in range(shape_l[0]):
        for i in range(shape_l[1]):
            try:
                one_hot[j][i][int(label[j][i])] = 1
            except:
                pdb.set_trace()
    return one_hot

def make_new_class(label):
    shape_label = label.shape
    new_label = np.zeros((shape_label[0], shape_label[1]))
    for i in range(shape_label[0]):
        for j in range(shape_label[1]):
            if label[i][j] == 0:
                new_label[i][j] = 0
            else:
                new_label[i][j] = 1
    return new_label

def train(dataset_iterator, num_iteration, loss, pred):
    optimizer = tf.train.AdamOptimizer()
    train_op =  optimizer.minimize(loss)
    # pdb.set_trace()
    result_repo = make_result_def('/home/sanket/MS_Thesis/Pointer_CapsNet/results','Capsnet')
    logging.basicConfig(level=logging.DEBUG, filename=result_repo + "/log/log.txt", filemode="a+",
                        format="%(asctime)-15s %(message)s")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        data, label , iteration_num ,batch_num = dataset_iterator.get_batch()
        while(iteration_num< num_iteration):
            _, batch_loss,net_pred = sess.run([train_op,loss, pred],feed_dict = {pcl_placeholder : data,
                                            label_placeholder : label,
                                            is_training_pl:True})

            # accuracy = calculate_accuracy(seg_predict, label_seg)*100
            # class_accuracy = calculate_class_accuracy(seg_predict, label_seg)*100
            # car_acc = calculate_car_accuracy(seg_predict,label_seg)*100
            print ("Iter : ", iteration_num , "Batch : " , batch_no ,  "  Loss : ", batch_loss )
                    # " Accuracy : ",accuracy, " Class Accuracy : ", class_accuracy , " Car class accuracy " , car_acc )
            # log = "Iter : " + str(iteration_num) + " Batch : " + str(batch_no) ,  "  Loss : " + str(batch_loss) + " Accuracy : " + str(accuracy) +  " Class Accuracy : "+ str(class_accuracy)
            # logging.info(log)
            # loss_ar.append(batch_loss)
            # acc_all.append(accuracy)
            # class_acc.append(class_accuracy)

            #print('Instance passed')
            data, label , iteration_num ,batch_num = dataset_iterator.get_batch()

            # data, label_instance ,label_seg, iteration_num , batch_no= dataset_iterator.get_batch()
            # label_seg  = make_new_class(label_seg)
            # label_seg_ = get_one_hot_label(label_seg)
            # #pdb.set_trace()

            if(batch_no == 0):
                batch_accuracy = np.mean(acc_all)
                class_accuracy = np.mean(class_acc)
                batch_loss_mean= np.mean(loss_ar)
                log = "**** Iteration : " +  str(iteration_num) + " loss : " + str(batch_loss_mean) + " Accuracy: " + str(batch_accuracy) +" Class Accuracy : " + str(class_accuracy)
                logging.info(log)
                print (log)

            if ((iteration_num % 10  == 0)and (batch_no == 0)):
                path = result_repo + '/checkpoints/instance_pointer2__'
                save_path = saver.save(sess, path +str(iteration_num) +"_"+ str(batch_no) +".ckpt")
                print("Model saved in path: %s" % save_path)
                infer_model(test_iter,sess, pred, result_repo , iteration_num , num_samples = 14)
        return result_repo





if __name__=='__main__':
    dataset_iterator = sample  = dataset_iterator(cfg.dataset_folder + cfg.dataset_file,
        cfg.dataset_folder + cfg.labels_file,cfg.batch_size , n_points = 1000 )
    pcl_placeholder,label_placeholder = model.input_placeholder(cfg.batch_size,num_point = 1000)
    is_training_pl = tf.placeholder(tf.bool, shape=())
    net_out = model.get_model(pcl_placeholder, is_training = is_training_pl, n_outputs = 40)
    loss_model = model.get_loss(net_out,label_placeholder,batch_size = cfg.batch_size)
    result_repo = train(dataset_iterator,num_iteration = 200, loss= loss_model, pred = net_out)
    #path  = "/home/srgujar/Pointwise-segmentation/results/pointer_M2_2_1_11_57"
    #model_path = path +  "/checkpoints/pointer2__3_0.ckpt"
    #infer_model_trained(dataset_iterator_test, model_path, net_pred,path)
