@defstruct MemoryDataLayerInds Layer (
  name :: String = "memory-data",
  (tops :: Vector{Symbol} = Symbol[:data,:label], length(tops) > 0),
  (batch_size :: Int = 0, batch_size > 0),
  (data :: Vector{Array} = Array[], length(data) == length(tops)),
  transformers :: Vector = [],
)
@characterize_layer(MemoryDataLayerInds,
  is_source => true
)

type MemoryDataLayerIndsState <: LayerState
  layer :: MemoryDataLayerInds
  blobs :: Vector{Blob}
  epoch :: Int
  trans :: Vector{Vector{DataTransformerState}}
  inds :: Array{Int64}

  curr_idx :: Int

  MemoryDataLayerIndsState(backend::Backend, layer::MemoryDataLayerInds) = begin
    blobs = Array(Blob, length(layer.tops))
    trans = Array(Vector{DataTransformerState}, length(layer.tops))
    transformers = convert(Vector{@compat(Tuple{Symbol, DataTransformerType})}, layer.transformers)
    for i = 1:length(blobs)
      dims = tuple(size(layer.data[i])[1:end-1]..., layer.batch_size)
      idxs = map(x -> 1:x, dims)

      blobs[i] = make_blob(backend, eltype(layer.data[i]), dims...)
      trans[i] = [setup(backend, convert(DataTransformerType, t), blobs[i])
          for (k,t) in filter(kt -> kt[1] == layer.tops[i], transformers)]
    end
    inds=[1:layer.batch_size];
    new(layer, blobs, 0, trans, 1)
  end
end

function setup(backend::Backend, layer::MemoryDataLayerInds, inputs::Vector{Blob}, diffs::Vector{Blob})
  @assert length(inputs) == 0
  for i = 2:length(layer.data)
    @assert eltype(layer.data[i]) == eltype(layer.data[1])
  end

  state = MemoryDataLayerIndsState(backend, layer)
  return state
end
function shutdown(backend::Backend, state::MemoryDataLayerIndsState)
  map(destroy, state.blobs)
  map(ts -> map(t -> shutdown(backend, t), ts), state.trans)
end

function forward(backend::Backend, state::MemoryDataLayerIndsState, inputs::Vector{Blob})
  inds=state.inds
  for j=1:length(inds)
    for i = 1:length(state.blobs)
      dset = state.layer.data[i]
      idx = map(x -> 1:x, size(state.blobs[i])[1:end-1])
      the_data = dset[idx..., ind[j]]
      set_blob_data(the_data, state.blobs[i],j)
    end
  end

  for i = 1:length(state.blobs)
    for j = 1:length(state.trans[i])
      forward(backend, state.trans[i][j], state.blobs[i])
    end
  end
end

function backward(backend::Backend, state::MemoryDataLayerIndsState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

