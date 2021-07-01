# Adapted from Knet's src/data.jl (author: Deniz Yuret)
import .FRAP
import Flux

struct DataGenerator{D}
    n_obs::D
    batch_size::Int
    partial::Bool
    imax::Int
    experiment::FRAP.ExperimentParams
    bath::FRAP.BathParams
    rng::Random.AbstractRNG
end

function DataGenerator(n_obs, experiment, bath, rng; batch_size=1, partial=true)
    batch_size > 0 || throw(ArgumentError("Need positive batchsize"))

    if n_obs < batch_size
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batch_size = n_obs
    end
    imax = partial ? n_obs : n_obs - batch_size + 1
    DataGenerator(n_obs, batch_size, partial, imax, experiment, bath, rng)
end

@Base.propagate_inbounds function Base.iterate(d::DataGenerator, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing

    nexti = min(i + d.batch_size, d.n_obs)
    batch = generate(d)

    return (batch, nexti)
end

function Base.length(d::DataGenerator)
    n = d.n_obs / d.batch_size
    d.partial ? ceil(Int,n) : floor(Int,n)
end

Base.eltype(::DataGenerator{D}) where D = D


function generate(d::DataGenerator)

    # Dimensions
    width   = d.bath.n_pixels + 2*d.bath.n_pad_pixels
    height  = width
    channels = d.bath.n_prebleach_frames + d.bath.n_postbleach_frames

    # Pre-allocate batch
    X = zeros((width, height, channels, d.batch_size)) |> gpu
    y = zeros((3, d.batch_size)) 

    # Could possibly be parallelized
    for b in 1:d.batch_size

        # Generate experiment parameters and targets
        D   = sample(d.experiment, :D) # D
        c₀  = sample(d.experiment, :c₀) # c₀
        α   = sample(d.experiment, :α) # α
        a   = sample(d.experiment, :a) # α

        # Create a new set of experiment parameters for each run
        # but keep the bath parameters
        experiment = FRAP.ExperimentParams(c₀, 
                                      d.experiment.ϕₘ, 
                                      D, 
                                      d.experiment.δt, 
                                      α, 
                                      d.experiment.β, 
                                      d.experiment.γ,
                                      a, 
                                      d.experiment.b
                                      )
        y[:,b] = [D, c₀, α]
        X[:,:,:,b] = FRAP.run(experiment, d.bath, d.rng)

    end

    y = y |> gpu

    return (X, y)

end

function sample(experiment::FRAP.ExperimentParams, key::Symbol)

    # Get parameter bounds
    bounds = getfield(experiment, key)

    if length(bounds) > 1

        if key == :D 

            sample = logunirand(bounds[1], bounds[2])

        elseif key == :c₀

            sample = unirand(bounds[1], bounds[2])

        elseif key == :α

            sample = unirand(bounds[1], bounds[2])

        elseif key == :a

            sample = logunirand(bounds[1], bounds[2])

        end

    else
        
        sample = bounds

    end

    # Return a random sample
    return sample
end

