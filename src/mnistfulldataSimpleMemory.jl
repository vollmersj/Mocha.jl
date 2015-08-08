cd("/Users/sjv/git/Mocha/examples/mnist")
use_gpu = false

if use_gpu
  ENV["MOCHA_USE_CUDA"] = "true"
else
  ENV["MOCHA_USE_NATIVE_EXT"] = "true"
  ENV["OMP_NUM_THREADS"] = 1
  blas_set_num_threads(1)
end

using Mocha
using HDF5
srand(12345678)
h5_file = h5open("../exmaples/mnist/data/train.hdf5", "r")
images=h5_file["data"][:,:,:,:]
dlabel=h5_file["label"][:,:]
#Simpler model
#data_layer  = AsyncHDF5DataLayer(name="train-data", source="data/train.txt", batch_size=64, shuffle=true)
data_layer=MemoryDataLayer(name="train-data", data=Array[images,dlabel],batch_size=64)
fc1_layer   = InnerProductLayer(name="ip1", output_dim=100, neuron=Neurons.ReLU(), bottoms=[:data], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

backend = use_gpu ? GPUBackend() : CPUBackend()
init(backend)

common_layers = [fc1_layer, fc2_layer]
net = Net("MNIST-train", backend, [data_layer, common_layers..., loss_layer])

exp_dir = "snapshots$(use_gpu ? "-gpu" : "-cpu")"

params = SolverParameters(max_iter=10000, regu_coef=0.0005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75))
solver = SGD(params)

setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
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
