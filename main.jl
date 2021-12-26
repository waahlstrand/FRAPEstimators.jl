using Flux
using CUDA
using Random
using Revise
using Base.Iterators: take
using Zygote: Params, gradient
using ProgressMeter: Progress, next!
using ArgParse

include("utils.jl")
include("data.jl")
include("models.jl")
include("train.jl")

# Packages 
using FRAP

 
function main()

    Random.seed!(1234)

    args = parse()

    batch_size = args["batch-size"]
    n_batches  = args["n-batches"]
    which     = args["device"]

    # Select device to train on
    device = which == "gpu" ? gpu : cpu

    @info "Parsing input: Batch size = $(batch_size), Number of batches = $(n_batches)"

    @info "Cuda functional: $(CUDA.functional())"
    @info "Using: $(which)"

    @info "Initializing..."
    
    @info "Loading model..."
    model = fc() |> device
    
    # Define training parameters
    loss(x, y)  = Flux.Losses.mse(model(x), y) 
    θ           = Flux.params(model) .|> device
    optimizer   = Flux.Momentum(1e-5, 0.99)

    # Load parameters from file
    @info "Loading parameters..."
    experiment, bath = FRAP.from_config("range.yml")
    
    # Create a dataset
    @info "Indexing data..."
    data = DataGenerator(n_batches*batch_size, experiment, bath; batch_size = batch_size, mode = :rc)

    # Train the model parameters
    @info "Starting training..."

    train!(loss, θ, data, optimizer)
    @info "Training complete!"



end

function parse()

    s = ArgParseSettings()
    @add_arg_table! s begin
        "--batch-size", "-b"
            help = "Number of samples to average for loss, must be larger than 1."
            arg_type = Int
            required = true
        "--n-batches", "-n"
            help = "Number of batches train, must be larger than 1."
            arg_type = Int
            default = 1
            required = true
        "--device", "-d"
            help = "Device to train on, either 'cpu' or 'gpu'."
            required = false
            default = "cpu"
    end    

    return parse_args(s)
end

# Run file and train
main()