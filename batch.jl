include(joinpath("..", "FRAP.jl", "src", "FRAP.jl"))

# Packages 
import .FRAP

function generate(experiment, bath, batch_size)
    
    # Dimensions
    width   = experiment.n_pixels + 2 experiment.n_pad_pixels
    height  = width
    channels = experiment.n_prebleach + experiment.n_postbleach

    # Pre-allocate batch
    X = zeros((width, height, channels, batch_size))
    y = zeros((3, batch_size))

    # Could possibly be parallelized
    for b in batch_size

        # Generate experiment parameters and targets
        y[:,b] .= 0 #...
        X[:,:,:,b] .= FRAP.run(experiment, bath, rng)

    end

    return (X, y)

end