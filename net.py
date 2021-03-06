import numpy as np
import tensorflow as tf 
import os,sys, json
from tensorflow.contrib import rnn
import random
import collections
import datetime
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true')
parser.add_argument('--test',action='store_true')
parser.add_argument('--ckptdir',type=str)
parser.add_argument('--ckptnum',type=str)


args = parser.parse_args()
ckptpath=None

print(args)




if args.train is False and args.test is  False:
    print("no train and no test. nothing to do. abort")
    sys.exit(0)
elif args.train is True and args.test is True:
    print("--train and --test cannot both exist at the same time. abort")
    sys.exit(1)


def check_ckpt_files_exist(ckpt_abs_path):
    meta_file = "{}.meta".format(ckpt_abs_path)
    index_file = "{}.index".format(ckpt_abs_path)
    data_file = "{}.data-00000-of-00001".format(ckpt_abs_path)

    if os.path.exists(meta_file) and os.path.exists(index_file) and os.path.exists(data_file):
        return True
    else:
        return False


if args.test:
    if args.ckptdir is None or args.ckptnum is None:
        print("ckptdir or ckptnum value doesn't exsit")
        sys.exit(1)
    # first see if the file exists assuming that the given parameter is absolute path
    # the value of args.ckpt should be path/ckpt_number.ckpt 
    # first check if the directory exists

    # check if cpktdir exists
    
    if os.path.exists(args.ckptdir):
        print("{} exists".format(args.ckptdir))
    else:
        # perhaps the input was a relative path. check existence after converting it
        # to absolute path
        ckptdir_abspath = os.path.join(os.getcwd(),args.ckptdir)
        if(os.path.exists(ckptdir_abspath)):
            print("cpkt exists in {}".format(ckptdir_abspath))
            args.ckptdir = ckptdir_abspath
        else:
            print("cannot locate the ckpt in {}. abort".format(ckptdir_abspath))
            sys.exit(1)
    
    # check if the ckptnum file exists
    extensions_to_check=['.meta','.data-00000-of-00001','.index']

    for check in extensions_to_check:

        basename = "{}.ckpt{}".format(args.ckptnum,check)
        filepath = os.path.join(args.ckptdir,basename)

        if not os.path.exists(filepath):
            print("{} doesn't exist. abort".format(filepath))
            sys.exit(1)
    
    # if reach here it means that all ckpt checks has been passed
    # global ckptpath

    # we need the absolute path
    ckptpath_temp = os.path.join(args.ckptdir,"{}.ckpt".format(args.ckptnum))
    ckptpath = os.path.join(os.getcwd(),ckptpath_temp)


# # testing up to args.parsing
# print("end of parsing test")
# sys.exit(0)


# Parameters
learning_rate = 0.001
training_iters = 3
display_step = 1000
n_input = 4
input_size = 6 # x,y,w,h,e,m

output_size = 3 # it will be one hot among three possible values(left, right, nothing)

# number of units in RNN cell
n_hidden = 3
DATETIME_TIMESTAMP = datetime.datetime.now().strftime("%y%m%d_%H%M")

# print(DATETIME_TIMESTAMP)



CURRENT_DIR = os.getcwd()
SAVE_DIR = os.path.join(CURRENT_DIR,"ckpt",DATETIME_TIMESTAMP)


# if we are testing, then we don't need to create a save directory since we are not doing training
if args.train is True:
    os.makedirs(SAVE_DIR)





def convert_to_onehot_vector(outputval):
    if outputval > 2 or outputval < 0:
        raise Exception("invalid outputval")
        return
    
    if outputval == 0:
        return np.array([1,0,0],dtype=np.float)
    elif outputval == 1:
        return np.array([0,1,0],dtype=np.float)
    elif outputval ==2:
        return np.array([0,0,1],dtype=np.float)
    else:
        raise Exception("invalid integer")
        return


## reading data

DATA_DIR = 'testdata'

if not os.path.exists(DATA_DIR):
    print("ERROR: DATA_DIR doesn't exist")
    sys.exit(1)

# gather all json files in DATA_DIR
os.chdir(DATA_DIR)

allfiles = os.listdir()

# filter only json files
jsonfiles = []
for onefile in allfiles:
    _, ext = os.path.splitext(onefile)
    if ext == '.json':
        jsonfiles.append(onefile)

# sort by name
jsonfiles.sort()

print("jsonfiles = {}".format(jsonfiles))

jsonlist = []
# read all json files and create data streak
for f in jsonfiles:
    js = json.load(open(f))
    jsonlist.append(js)


# print("jsons: {}".format(jsonlist))

## rearrange the values to fit for NN input
initial_m_val=0

rearrangedlist=[]
outputlist=[]
m_temp=initial_m_val
for js in jsonlist[:-1]:
    
    newjs={}
    newjs['x']=js['x']
    newjs['y']=js['y']
    newjs['w'] = js['w']
    newjs['h'] = js['h']
    newjs['m'] = m_temp
    
    m_temp = js['m']
    outputlist.append(m_temp)
    newjs['e'] = js['e']

    rearrangedlist.append(newjs)


# print("rearranged list = {}".format(rearrangedlist))

print("rearranged list len={}, jsonlist len={}, outputlist len = {}".format(len(rearrangedlist),len(jsonlist),len(outputlist)))



# for i in range(0,len(rearrangedlist)):
#     rearr = rearrangedlist[i]
#     jss = jsonlist[i]

#     print("{}th m value, rearr: {}, jss: {}".format(i,rearr['m'],jss['m']))

nparrays = []

for item in rearrangedlist:
    temp = np.array([item['x'],item['y'],item['w'],item['h'],item['e'],item['m']])
    nparrays.append(temp)

alldata = np.array(nparrays)

print("nparray rearranged={}".format(alldata))
print("shape={}".format(alldata.shape))


# sys.exit(0)


# Target log path
logs_path = 'tmp/rnn_test'
logs_path = os.path.join(CURRENT_DIR,logs_path)
print("logs_path={}".format(logs_path))
writer = tf.summary.FileWriter(logs_path)


###
# CREATING GRAPH
#
#
# 
# 
# 
### 




# tf Graph input
x = tf.placeholder("float", [None, n_input, 6])
y = tf.placeholder("float", [None, output_size])


weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, output_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([output_size]))
}

batch_size = 5


# def RNN(x, weights, biases):

# pred = RNN(x, weights, biases)



# reshape to [1, n_input]
x1 = tf.reshape(x, [batch_size, n_input,input_size],name="x_input_reshape")

# Generate a n_input-element sequence of inputs
# (eg. [had] [a] [general] -> [20] [6] [33])
x2 = tf.split(x1,n_input,1,name="x_input_split")

x3 = tf.squeeze(x2)
# x3 = None

for index,tensor in enumerate(x2):
    x2[index] = tf.squeeze(tensor)

# initializer = tf.zeros([batch_size,input_size])
# x4 = tf.scan(lambda a, x_i: tf.squeeze(x_i), x2, initializer=tf.constant(0,shape=[batch_size,input_size]) )
x4=tf.Variable([0])

# 2-layer LSTM, each layer has n_hidden units.
# Average Accuracy= 95.20% at 50k iter
rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden)])

# 1-layer LSTM with n_hidden units but with lower accuracy.
# Average Accuracy= 90.60% 50k iter
# Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
# rnn_cell = rnn.BasicLSTMCell(n_hidden)

# generate prediction
outputs, states = rnn.static_rnn(rnn_cell, x2,dtype=tf.float32)

# there are n_input outputs but
# we only want the last output
multiply_weight = tf.matmul(outputs[-1], weights['out'],name="multiply_weight")

pred2 =  tf.add(multiply_weight,biases['out'],name="final_output")

pred = tf.nn.softmax(pred2)



# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
pred_max = tf.argmax(pred,1)
max_ped = tf.reduce_max(pred,reduction_indices=[1])
label_max = tf.argmax(y,1)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


saver = tf.train.Saver()

### 
# 
# 
# setting the net input and output
#
# 
###



pythonlist_inputbatch=[]
start_sequence_offset = 0
for i in range(batch_size):
    pythonlist_inputbatch.append(alldata[start_sequence_offset+i:start_sequence_offset+i+n_input])

firstbatch=np.array(pythonlist_inputbatch)
print("firsbatch shape={}".format(firstbatch.shape))

batch_input = np.reshape(firstbatch,[-1,n_input,input_size])
print("batch_input shape ={}".format(batch_input.shape))

##
# preparing the output
# 
#
##

onehot_converted_outputs=[]
for i in range(batch_size):
    out = outputlist[start_sequence_offset+n_input]
    try:
        ret = convert_to_onehot_vector(out)
        onehot_converted_outputs.append(ret)
    except Exception:
        sys.exit(1)



# coonvert single integer value of output to a one-hot vector
# raw_output_list = outputlist[0:3]
# onehot_vector_list = []
# for item in raw_output_list:
    
#     try:
#         ret = convert_to_onehot_vector(item)
#         onehot_vector_list.append(ret)
#     except Exception:
#         sys.exit(1)
# print("onehot_vector_list = {}".format(onehot_vector_list))

batch_output = np.reshape(onehot_converted_outputs,[-1,3])

print("batch_output={}".format(batch_output))

# firstbatch_output = np.reshape(onehot_vector_list,[-1,1,3])

# print("firstbatch_output = {}".format(firstbatch_output))

# print("firstbatch output last = {}".format(firstbatch_output[-1]))





# Launch the graph
with tf.Session() as session:
    
    

    writer.add_graph(session.graph)
    print("graph added to writer")

    writer.flush()
    writer.close()
    print("graph writer closed. written in {}".format(writer.get_logdir()))

    # _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
    #                                             feed_dict={x: firstbatch_input, y: firstbatch_output[-1]})

    

    if args.test:
        # global ckptpath
        print("ckptpath={}".format(ckptpath))
        saver.restore(session,ckptpath)
        print("model restored")

        o_pred, o_outputs = session.run([pred,outputs],feed_dict={x: batch_input, y: batch_output})

        print("pred={}".format(o_pred))
        print("outputs={}".format(o_outputs))
    
    if args.train:
        
        # if we are training from scratch, we need to initialize the variables.
        # TODO: of course, this is not acceptable when we want to train based on a pretrained ckpt.
        session.run(init)

        for step in range(training_iters):

            print("== step: {}".format(step))

            o_x1,o_x2,o_pred, o_outputs, o_states, o_multiply_weight,\
            _,o_accuracy,o_cost, o_pred_max, o_label_max, o_maxpred = session.run([x1,x2,pred,outputs,states,multiply_weight,optimizer,accuracy,cost,pred_max, label_max, max_ped], feed_dict={x: batch_input, y: batch_output})

            # print("x={}".format(batch_input))
            # print("x1={}".format(o_x1))
            # print("x2={}".format(o_x2))
            # print("pred={}".format(o_pred))
            
            # print("o_outputs={}",format(o_outputs[-1]))
            # print("last o_output shape={}".format(o_outputs[-1].shape))
            # print("outputs shape={}".format(np.array(outputs).shape))

            print("cost={}".format(o_cost))
            print("accuracy={}".format(o_accuracy))
            print("max pred = {}".format(o_maxpred))
            print("o_outputs={}".format(o_outputs))
            print("o_states={}".format(o_states))
            # print("pred_max ={}".format(o_pred_max))
            # print("label max = {}".format(o_label_max))


            save_path = saver.save(session,os.path.join(SAVE_DIR,"{:05d}.ckpt".format(step)))
            print("ckpt saved in {}".format(save_path))


    # o_x1, o_x2, o_x3, o_x4  = session.run([x1,x2,x3,x4],feed_dict={x:batch_input,y:batch_output})

    # print("x1={}".format(o_x1))
    # print("x2={}".format(o_x2))
    # # print("x2 shape={}".format(o_x2.shape))
    # print("x3={}".format(o_x3))
    # print("x3 shape = {}".format(o_x3.shape))
    # print("x4={}".format(o_x4))







print("end of code")




    # while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.

      
        


        # if offset > (len(training_data)-end_offset):
        #     offset = random.randint(0, n_input+1)

        # symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        # symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        # symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        # symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
        #                                         feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        # loss_total += loss
        # acc_total += acc
        # if (step+1) % display_step == 0:
        #     print("Iter= " + str(step+1) + ", Average Loss= " + \
        #           "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
        #           "{:.2f}%".format(100*acc_total/display_step))
        #     acc_total = 0
        #     loss_total = 0
        #     symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
        #     symbols_out = training_data[offset + n_input]
        #     symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
        #     print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        # step += 1
        # offset += (n_input+1)



    # print("Optimization Finished!")
    # print("Elapsed time: ", elapsed(time.time() - start_time))
    # print("Run on command line.")
    # print("\ttensorboard --logdir=%s" % (logs_path))
    # print("Point your web browser to: http://localhost:6006/")
    # while True:
    #     prompt = "%s words: " % n_input
    #     sentence = input(prompt)
    #     sentence = sentence.strip()
    #     words = sentence.split(' ')
    #     if len(words) != n_input:
    #         continue
    #     try:
    #         symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
    #         for i in range(32):
    #             keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
    #             onehot_pred = session.run(pred, feed_dict={x: keys})
    #             onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
    #             sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
    #             symbols_in_keys = symbols_in_keys[1:]
    #             symbols_in_keys.append(onehot_pred_index)
    #         print(sentence)
    #     except:


    #         print("Word not in dictionary")
