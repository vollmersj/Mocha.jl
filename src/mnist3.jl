using Mocha
using HDF5
cd("../examples/mnist")
srand(12345678)
h5_file = h5open("data/restricttrain.hdf5", "r")
images=h5_file["data"][:,:,:,:]
dlabel=h5_file["label"][:,:]
# 0 or 1 digits

#data_layer  = AsyncHDF5DataLayer(name="train-data", source="data/train.txt", batch_size=64, shuffle=true)
#data_layer=MemoryDataLayer(name="train-data", data=Array[images,dlabel],batch_size=64)
data_layer=MemoryDataLayerInds(name="train-data", data=Array[images,dlabel],batch_size=4)
conv_layer  = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
# data_layer=MemoryDataLayer2(name="train-data", data=Array[images,dlabel],batch_size=64)


pool_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer   = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])


backend = use_gpu ? GPUBackend() : CPUBackend()
init(backend)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
net = Net("MNIST-train", backend, [data_layer, common_layers..., loss_layer])
1
net.states[1].blobs

net.backend

exp_dir = "snapshots$(use_gpu ? "-gpu" : "-cpu")"

params = SolverParameters(max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75))
solver = SGD(params)


solver_state = SolverState()
solver_state = load_snapshot(net, solver.params.load_from, solver_state)
solver_state.learning_rate = get_learning_rate(solver.params.lr_policy, solver_state)
solver_state.momentum = get_momentum(solver.params.mom_policy, solver_state)

  # we init network AFTER loading. If the parameters are loaded from file, the
  # initializers will be automatically set to NullInitializer
init(net)
# Initial forward iteration
@show solver_state.obj_val = forward(net, solver.params.regu_coef)
@show solver_state.obj_val = forward(net, solver.params.regu_coef)
@show solver_state.obj_val = forward(net, solver.params.regu_coef)
backward(net, solver.params.regu_coef)
exit()

println("achieved backward")
net.states[4].parameters[1].gradient.data

@debug("Initializing coffee breaks")
setup(solver.coffee_lounge, solver_state, net)

# coffee break for iteration 0, before everything starts
check_coffee_break(solver.coffee_lounge, solver_state, net)
i_state = setup(solver, net, solver_state)



# param_states  = map(i -> net.states[i],filter(i -> has_param(net.layers[i]) && !is_frozen(net.states[i]), 1:length(net.layers)))


setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000,max_iter=50)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=5)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_layer_test = HDF5DataLayer(name="test-data", source="data/restricttest.txt", batch_size=5)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:ip2, :label])
test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
