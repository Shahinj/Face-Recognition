from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

###
#os.chdir("C:/Users/Shahin/Documents/School/Skule/Year 3 - Robo/second semester/csc411/project_1_face_recognition")
###

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

   

def get_data(act):

    
    testfile = urllib.URLopener()  
    
    
    #Note: you need to create the uncropped folder first in order 
    #for this to work
    
    
    for a in act:
        name = a.split()[1]
        i = 0
        files = ["facescrub_actors.txt", "facescrub_actresses.txt"]
        for gender in files:
            for line in open(gender):
                if a in line:
                    filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                    #A version without timeout (uncomment in case you need to 
                    #unsupress exceptions, which timeout() does)
                    #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                    #timeout is used to stop downloading images which take too long to download
                    try:
                        os.mkdir(os.getcwd() + '/uncropped/'+a)
                        os.mkdir(os.getcwd() + '/cropped/'+a)
                        os.mkdir(os.getcwd() + '/edited/'+a)
                    except OSError:
                        pass
                    timeout(testfile.retrieve, (line.split()[4], "uncropped/"+a+'/'+filename), {}, 3000)
                    if not os.path.isfile("uncropped/"+a+'/'+filename):
                        continue
                    print line
                    ###crop and convert
                    crop_data = line.split('\t')[4].split(',')      #0 is x0, 1 is y0, 2 is x1, 3 is y2
                    try:
                        image = imread(os.getcwd() + '/uncropped/'+a+'/'+filename)
                    except IOError:
                        continue
                    if (len(image.shape) < 3):
                        continue
                    img_cropped = image[int(crop_data[1]):int(crop_data[3]),int(crop_data[0]):int(crop_data[2])]
                    imsave(os.path.join(os.getcwd(), 'cropped/'+a+'/'+filename),img_cropped)
                    img_gray = rgb2gray(img_cropped)
                    img_scaled = imresize(img_gray,(32,32))
                    imsave(os.path.join(os.getcwd(), 'edited/'+a+'/'+filename),img_scaled)
                    
                    ###
                    
                    print filename
                    i += 1
    
    
    
def get_baskets(act,train_size, valid_size, test_size):
    '''
    input: actors list, size of each set per actor
    output: 3 baskets (list), training, validation and test
    '''
    training = []
    validation = []
    test = []
    cnt = 0
    label = 0
    for a in act:
        if a == 'Alec Baldwin':
            label = 1
        elif a== 'Steve Carell':
            label = -1
        cnt += 1
        i = 0
        while len(training) !=  cnt*train_size :
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            training.append((img_scaled,label))
        
        while len(validation) !=  cnt*valid_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            validation.append((img_scaled,label))
        
        while len(test) !=  cnt*test_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            test.append((img_scaled,label))
        
    return training,validation,test
    
    
def get_baskets_gender(act,train_size, valid_size, test_size):
    '''
    input: actors list, size of each set per actor
    output: 3 baskets (list), training, validation and test
    '''
    training = []
    validation = []
    test = []
    cnt = 0
    label = 0
    male = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Gerard Butler'  ,'Daniel Radcliffe' , 'Michael Vartan']
    female = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'America Ferrera',  'Fran Drescher' , 'Kristin Chenoweth']
    for a in act:
        if a in male:
            label = 1
        elif a in female:
            label = -1
        cnt += 1
        i = 0
        while len(training) !=  cnt*train_size :
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            training.append((img_scaled,label))

        
        while len(validation) !=  cnt*valid_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            validation.append((img_scaled,label))
        
        while len(test) !=  cnt*test_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            test.append((img_scaled,label))
        
    return training,validation,test
    
    
def train(training):
    '''
    input: training basket
    output: a matrix of thetas for linear regression
    '''
    
    ### how to obtain the best learning rate
    '''
    init_theta = np.zeros((1+32*32,1))
    t_min = init_theta
    l_rates = [1,0.1,0.01,0.001,0.0001,0.00001,0.00001,0.000001]
    #learning_rate = 0.00001
    x = np.zeros((1 + 32*32,len(training)))
    y = np.zeros((1,len(training)))
    x[0,:] = 1
    i = 0
    f_min = 0
    for learning_rate in l_rates:
        for i in range(len(training)):
            x[1:,i] = training[i][0].flatten() 
            y[0,i] = training[i][1] 
        theta = grad_descent(f,df, x, y,init_theta,learning_rate)
        fun_value = f(x,y,theta)
        if (i == 0 or fun_value < f_min):
            i += 1
            f_min = fun_value
            t_min = theta 
        print (fun_value)
    return t_min
    '''
    
    ###using the found learning rate
    init_theta = np.zeros((1+32*32,1))
    learning_rate = 0.0000001
    x = np.zeros((1 + 32*32,len(training)))
    y = np.zeros((1,len(training)))
    x[0,:] = 1
    for i in range(len(training)):
        x[1:,i] = training[i][0].flatten() 
        y[0,i] = training[i][1] 
    theta = grad_descent(f,df, x, y,init_theta,learning_rate)
    
    return theta
    
def train_4(training,setting):
    '''
    input: training basket
    output: a matrix of thetas for linear regression
    '''
    
    ###using the found learning rate
    if(setting == 'zeros'):
        init_theta = np.zeros((1+32*32,1))
    elif(setting == 'ones'):
        init_theta = np.ones((1+32*32,1))
    elif(setting == 'random'):
        init_theta = np.random.rand(1+32*32,1)

    learning_rate = 0.00001
    x = np.zeros((1 + 32*32,len(training)))
    y = np.zeros((1,len(training)))
    x[0,:] = 1
    for i in range(len(training)):
        x[1:,i] = training[i][0].flatten() 
        y[0,i] = training[i][1] 
    theta = grad_descent(f,df, x, y,init_theta,learning_rate)
    
    return theta

    
def f(x, y, theta):
    '''
    cost function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: cost function
    '''
    return sum( (y - dot(theta.T,x)) ** 2)      #J
    
def df(x,y,theta):
    '''
    gradient function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: derivative of cost function
    '''
    return -2*sum( (y - dot(theta.T,x)) * x, 1).T        #J, axis=1 indicates that row is constant, columns add
    
    
def f_mat(x,y,theta):
    '''
    cost function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: cost function
    '''
    #return sum(sum((np.matmul(theta.T, x)-y)**2,0))
    #return sum(sum((np.matmul(theta.T, x)-y)**2,1))
    return sum((np.matmul(theta.T, x)-y)**2,1)

def df_mat(x,y,theta):
    '''
    gradient function
    input: training set and their label(x,y = 0 or 1) and thetas
    output: derivative of cost function
    '''
    return 2*(np.matmul(x,(np.matmul(theta.T,x)-y).T))       #J, axis=1 indicates that row is constant, columns add    


def finite_dif(func,xs,ys,thetas,h):
    '''
    finite difference computation of the gradient
    input: function and h
    output: gradient
    '''
    gradient = np.zeros((thetas.shape))
    for i in range(thetas.shape[0]):
        #derivative = 0
        #for j in range(x.shape[1]):
        new_theta = thetas.copy()
        new_theta[i,:] += h
        # print new_x
        # print xs
        u1 = func(xs,ys,new_theta)
        u2 = func(xs,ys,thetas)
        # print u1
        # print u2
        derivative = (func(xs,ys,new_theta) - func(xs,ys,thetas)) / float(h)

        gradient[i,:] = derivative
        
    return gradient

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    sub = np.zeros(init_t.shape)
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        sub = alpha*df(x, y, t).reshape(sub.shape)
        t -= sub
        iter += 1
    return t        #t is the fitted thetas
    
def classify(test_set, theta):
    '''
    input: test set, thetas
    output: percentage correct classified
    '''
    
    correct = 0
    total = 0
    column = np.zeros((1 + 32*32,1))
    for point in test_set:
        column[0,0] = 1
        column[1:,0] = point[0].flatten() 
        label_act = point[1]
        if ( dot(theta.T , column) >= 0):
            label_comp = 1
        else:
            label_comp = -1

        if (label_act == label_comp):
            correct += 1
        total += 1.0
        
    return correct/total * 100
    

    
def disp_theta(theta):
    k = theta[1:].reshape((32,32))
    plt.figure()
    imshow(k, cmap = cm.coolwarm_r)
    plt.show()
    
def get_cost(set,theta):
    x = np.zeros((1 + 32*32,len(set)))
    y = np.zeros((1,len(set)))
    x[0,:] = 1
    for i in range(len(set)):
        x[1:,i] = set[i][0].flatten() 
        y[0,i] = set[i][1] 
    
    return f(x,y,theta)
    
def get_baskets_new_labeling(act,train_size, valid_size, test_size):
    '''
    input: actors list, size of each set per actor
    output: 3 baskets (list), training, validation and test
    '''
    training = []
    validation = []
    test = []
    cnt = 0
    label = 0
    act_labels = []
    for a_num in range(len(act)):
        a = act[a_num]
        label = np.zeros((len(act),1))
        label[a_num,0] = 1
        act_labels.append((a,label))
        cnt += 1
        i = 0
        while len(training) !=  cnt*train_size :
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            training.append((img_scaled,label))
        
        while len(validation) !=  cnt*valid_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            validation.append((img_scaled,label))
        
        while len(test) !=  cnt*test_size:
            name = a.split()[1]
            filename = name+str(i)
            try:
                image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.JPG')
                img_gray = rgb2gray(image)
                img_scaled = imresize(img_gray,(32,32)) / 255.0
            except IOError:
                try:
                    image = imread(os.getcwd() + '/cropped/'+ a +'/'+filename+ '.png')
                    img_gray = rgb2gray(image)
                    img_scaled = imresize(img_gray,(32,32)) / 255.0
                except:
                    i += 1
                    continue
            i += 1
            test.append((img_scaled,label))
        
    return training,validation,test,act_labels
    
def train_new_labeling(training):

    init_theta = np.zeros((1+32*32,len(training[0][1])))
    learning_rate = 0.0000001
    x = np.zeros((1 + 32*32,len(training)))
    y = np.zeros((len(act),len(training)))
    x[0,:] = 1
    for i in range(len(training)):
        x[1:,i] = training[i][0].flatten() 
        y[:,i] = training[i][1].flatten() 
    theta = grad_descent(f_mat,df_mat, x, y,init_theta,learning_rate)
    
    return theta
    
    
def classify_new_labeling(test_set, theta):
    '''
    input: test set, thetas
    output: percentage correct classified
    '''
    
    correct = 0
    total = 0
    column = np.zeros((1 + 32*32,1))
    for point in test_set:
        column[0,0] = 1
        column[1:,0] = point[0].flatten() 
        label_act = point[1]
        
        hyp = dot(theta.T , column)
        max = np.max(hyp)
        hyp /= max
        hyp = hyp.astype(int)
        
        if ( np.max(dot(hyp.T , label_act)) == 1):
            correct += 1
        total += 1.0
        
    return correct/total * 100
    
    
if (__name__ == "__main__"):

    
    part = '8'
    
    if( part == '2'):
        ##part 2
        act = ['Steve Carell', 'Alec Baldwin']
        training,validation,test = get_baskets(act,70,10,10)
    elif(part == 'data_extraction'):
        act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        get_data()
    elif ( part == '3'):
        ##part 3
        
        act = ['Steve Carell', 'Alec Baldwin']
        training,validation,test = get_baskets(act,70,10,10)
        theta = train(training)
        training_performance = classify(training,theta)
        validation_performance = classify(validation,theta)
        test_performance = classify(test,theta)
        cost_validation = get_cost(validation,theta) / float(len(validation))
        cost_training = get_cost(training,theta) / float(len(training))
        print('cost on training set per image: %g' %cost_training)
        print('cost on validation set per image: %g' %cost_validation)
        print('performance on training set: %g%%' %training_performance)
        print('performance on validation set: %g%%' %validation_performance)
        
    elif ( part == '4a'):
        ##part 4
        ###a)
        
        act = ['Steve Carell', 'Alec Baldwin']
        training,validation,test = get_baskets(act,2,0,0)
        theta = train(training)
        disp_theta(theta)
        
        training,validation,test = get_baskets(act,70,0,0)
        theta = train(training)
        disp_theta(theta)
        
    elif ( part == '4b'):
        ###b)
        #train 4 is the the same function as train, but was made for experimentation purposes of this part
        
        act = ['Steve Carell', 'Alec Baldwin']
        training,validation,test = get_baskets(act,2,1,1)
        theta = train_4(training,'zeros')
        disp_theta(theta)
        theta = train_4(training,'ones')
        disp_theta(theta)
        theta = train_4(training,'random')
        disp_theta(theta)
                        
        training,validation,test = get_baskets(act,70,1,1)
        theta = train_4(training,'zeros')
        disp_theta(theta)
        theta = train_4(training,'ones')
        disp_theta(theta)
        theta = train_4(training,'random')
        disp_theta(theta)
        
        
        
    elif ( part == '5'):
        ##part 5
        #first it did not converge, changed alpha from 0.0000001 to 
        
        act_2 = ['America Ferrera',  'Fran Drescher' , 'Kristin Chenoweth', 'Gerard Butler'  ,'Daniel Radcliffe' , 'Michael Vartan']
    
        act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        size = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        performance = np.zeros((len(size),4))
        i = 0
        training_2,validation_2,test_2 = get_baskets_gender(act_2,0,50,0)
        print ('done reading other actors')
        for s in size:
            training,validation,test = get_baskets_gender(act,s,10,0)
            theta = train(training)
            validation_performance = classify(validation,theta)
            training_performance = classify(training,theta)
            validation_performance_2 = classify(validation_2,theta)           
            performance[i,0] = len(training)
            performance[i,1] = validation_performance
            performance[i,2] = training_performance
            performance[i,3] = validation_performance_2
            i += 1
            print (s)
            
        #need to plot
        plt.figure()
        plt.plot(performance[:,0],performance[:,1])   #validation plot
        plt.plot(performance[:,0],performance[:,2])   #training plot
        plt.plot(performance[:,0],performance[:,3])   #validation plot, not in act
        plt.title('Gender Classifier')
        plt.xlabel('number of training images')
        plt.ylabel('Performance(percentage)')
        plt.legend(['Classifier performance on actors in act, Validation set','Classifier performance on actors in act, Training set','Classifier performance on actors not in act, Validation set'])
        plt.show()
        
    elif ( part == '6'):
        ##part 6
        
        act = ['Steve Carell', 'Alec Baldwin']
        training,validation,test,labels = get_baskets_new_labeling(act,5,1,1)
        x = np.zeros((1 + 32*32,len(training)))
        y = np.zeros((len(act),len(training)))
        x[0,:] = 1
        t = np.ones((1+32*32,len(act)))
        for i in range(len(training)):
            x[1:,i] = training[i][0].flatten() 
            y[:,i] = training[i][1].flatten()
        grd = df_mat(x,y,t)
        print 'vectorized gradient is:'
        print grd
        h = 0.00001
        fd = finite_dif(f_mat,x,y,t,h)
        print 'gradient using finite difference is:'
        print fd
        
        dif = abs(grd - fd) / grd * 100
        print ('maximum percentage difference between vectorized gradient and finite difference approximation with h of %g is %g%%' %(h,np.max(dif))) 
        
    elif ( part == '7'):
    
        ##part 7
        
        act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        training,validation,test,labels = get_baskets_new_labeling(act,70,30,0)
        theta = train_new_labeling(training)
        validation_performance = classify_new_labeling(validation,theta)
        training_performance = classify_new_labeling(training,theta)
        print ('performance on validation set is %g%%' %validation_performance)
        print ('performance on training set is %g%%' %training_performance)
         
    elif ( part == '8'):
        ##part 8
        
        act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        training,validation,test,labels = get_baskets_new_labeling(act,80,30,0)
        theta = train_new_labeling(training)
        validation_performance = classify_new_labeling(validation,theta)
        training_performance = classify_new_labeling(training,theta)
        print ('performance on validation set is %g%%' %validation_performance)
        print ('performance on training set is %g%%' %training_performance)
        i = 0
        for artist in labels:
            name = artist[0]
            k = theta[1:,i].reshape((32,32))
            plt.figure()
            imshow(k, cmap = cm.coolwarm_r)
            plt.title(name)
            plt.show()
            i += 1
        
    else:
        print ('please select the part you wish to obtain the results for in the main function')

    



    

    
  

    

    