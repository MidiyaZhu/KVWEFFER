dataset = ["sst-2"]
# dataset = ["isear"]
tuning_param  = ["lambda_loss","main_learning_rate","batch_size","nepoch","temperature","label_list","SEED","dataset"] ## list of possible paramters to be tuned can be increased
lambda_loss = [0.5]
temperature = [0.3]
batch_size = [12]
decay = 1e-02
main_learning_rate = [5e-05]  #2e-05c  #5e-5 for aman 4e-5 for isear


hidden_size = 768
nepoch = [15]
criterion = "emo_acc"
run_name = "" ## for subset use "subset" else leave empty
loss_type = "lcl"
# model_type = "electra"
model_type = "bert"
label_list = [None]

SEED = [0]


param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"batch_size":batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"criterion":criterion,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"label_list":label_list,"decay":decay,"SEED":SEED,"model_type":model_type}


