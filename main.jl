using Flux
using CUDA
using Random
using Revise
using Base.Iterators: take
using Zygote: Params, gradient
using ProgressMeter: Progress, next!

include("utils.jl")
include("data.jl")
include("models.jl")
include("train.jl")

# Packages 
using FRAP

 
function main(batch_size, n_batches)
    @info "Cuda functional: $(CUDA.functional())"

    @info "Initializing..."
    rng = MersenneTwister(1234)
    
    @info "Loading model..."
    model = fc()
    
    # Define training parameters
    loss(x, y)  = Flux.Losses.mse(model(x), y)
    θ           = Flux.params(model)
    optimizer   = Flux.Momentum(1e-5, 0.99)

    # Load parameters from file
    @info "Loading parameters..."
    experiment, bath = FRAP.from_config("range.yml")
    
    # Create a dataset
    @info "Indexing data..."
    data = DataGenerator(n_batches*batch_size, experiment, bath, rng; batch_size = batch_size, mode = :rc)

    # Train the model parameters
    @info "Starting training..."

    train!(loss, θ, data, optimizer)
    @info "Training complete!"



end


if length(ARGS) > 0
    main(convert(Int64, ARGS[0]), convert(Int64, ARGS[1]))
else

    main(256, 200)
    
end
