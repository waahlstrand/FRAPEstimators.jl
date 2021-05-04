using Flux
using CUDA
using Random
using Revise

include(joinpath("..", "FRAP.jl", "src", "FRAP.jl"))

include("utils.jl")
include("data.jl")


# Packages 
import .FRAP

function train!(loss, θ, data::DataGenerator, optimizer)
    # Trains a model with parameters θ with respect to a 
    # loss function and optimizer for a number of iterations in a dataset
        
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

        # Update the weights and optimizer
        update!(optimizer, θ, ∇θ)

    end
end
 
function main(batch_size)

    n_channels = 114
    model = Chain(
        Conv((4,4), n_channels => n_channels, relu, stride = 1),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu, stride = 1),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu),
        MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, relu),
        MaxPool((2,2)),
        Conv((3,3), n_channels => n_channels, relu),
        MaxPool((3,3)),
        flatten,
        Dense(n_channels, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, 3)
    )
    
    # loss(x, y) = Flux.Losses.mse(model(x), y)
    # θ = Flux.params(model)
    # optimizer = ADAM()

    # X = Float32.(randn((512, 512, 110, 8)))

    # y = model(X)
    # print(size(y))

    rng = MersenneTwister(1234);

    # n_pixels     = 256
    # n_pad_pixels = 128
    # pixel_size   = 7.5e-7

    # n_prebleach_frames    = 10
    # n_bleach_frames       = 4
    # n_postbleach_frames   = 100
    # n_frames              = n_prebleach_frames + n_postbleach_frames
    # n_elements            = n_pixels + 2*n_pad_pixels

    # x = 128
    # y = 128
    # r = 15e-6


    # c₀ = [0.5, 1.0]
    # ϕₘ = 1.0
    # D_SI = [1e-12, 1e-9]; # m^2/s
    # D = D_SI ./ pixel_size^2

    # δt = 0.1

    # α = [0.5, 0.95]
    # β = 1.0
    # γ = 0.0
    # a = [0.01, 0.1]
    # b = 0.0

    # #############################################
    # # Run the experiment
    

    # experiment  = FRAP.ExperimentParams(c₀, ϕₘ, D, δt, α, β, γ, a, b)
    # bath        = FRAP.BathParams(n_pixels, n_pad_pixels, pixel_size, 
    #                             n_prebleach_frames, n_bleach_frames, n_postbleach_frames, 
    #                             x, y, r)

    # Load parameters from file
    experiment, bath = FRAP.from_config("../FRAP.jl/configs/range.yml")

    # Create a dataset
    data = DataGenerator(3*batch_size, experiment, bath, rng; batch_size = batch_size)

    # Move to gpu 
    model = model |> gpu

    for (x, y) in data
        # println(typeof(x))
        # println(size(x), size(y))
        println(model(x))
    end

end




main(8)