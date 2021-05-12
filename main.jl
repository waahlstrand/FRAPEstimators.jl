using Flux
using CUDA
using Random
using Revise
using Base.Iterators: take
using Zygote: Params, gradient
using ProgressMeter: Progress, next!

include(joinpath("..", "FRAP.jl", "src", "FRAP.jl"))

include("utils.jl")
include("data.jl")


# Packages 
import .FRAP

function train!(loss, θ, data::DataGenerator, optimizer)
    # Trains a model with parameters θ with respect to a 
    # loss function and optimizer for a number of iterations in a dataset
        
    p = Progress(length(data); showspeed=true)

    # Declare loss to make it loggable
    local L

    # Flux gradient wants a Params struct
    θ = θ |> Params

    # Yield new generated data in a loop
    for d in data

        # Calculate the gradients from the loss
        ∇θ = gradient(θ) do

            L = loss(d...)

            return L
        end

        # Update progressmeter
        next!(p; showvalues = [(:L,L)])

        # Update the weights and optimizer
        Flux.update!(optimizer, θ, ∇θ)

    end
end
 
function main(batch_size, n_batches)

    @info "Initializing..."
    rng = MersenneTwister(1234);

    @info "Loading model..."
    init = Flux.glorot_uniform(rng)
    n_channels = 110
    model = Chain(
        Conv((4,4), n_channels => n_channels, relu, stride = 1; init=init),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu, stride = 1; init=init),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu; init=init),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu; init=init),
        MaxPool((2,2)),
        Conv((3,3), n_channels => n_channels, relu; init=init),
        MaxPool((3,3)),
        Flux.flatten,
        Dense(n_channels, 1024, relu; initW=init, initb=init),
        Dense(1024, 512, relu; initW=init, initb=init),
        Dense(512, 3; initW=init, initb=init)
    ) |> gpu
    
    # Define training parameters
    loss(x, y)  = Flux.Losses.mse(model(x), y)
    θ           = Flux.params(model)
    optimizer   = Flux.Descent(1e-6)

    # Load parameters from file
    @info "Loading parameters..."
    experiment, bath = FRAP.from_config("../FRAP.jl/configs/range.yml")
    
    # Create a dataset
    @info "Indexing data..."
    data = DataGenerator(n_batches*batch_size, experiment, bath, rng; batch_size = batch_size)

    # Train the model parameters
    @info "Starting training..."
    train!(loss, θ, data, optimizer)
    @info "Training complete!"


end


main(4, 10)
