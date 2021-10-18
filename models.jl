
function fcnn()
    
    n_channels = 110
    model = Chain(
        Conv((4,4), n_channels => n_channels, stride = 1),
            BatchNorm(n_channels, relu),
            MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels, stride = 1),
            BatchNorm(n_channels, relu),
            MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels),
            BatchNorm(n_channels, relu),
            MaxPool((3,3)),
        Conv((3,3), n_channels => n_channels),
            BatchNorm(n_channels, relu),
            MaxPool((2,2)),
        Conv((3,3), n_channels => n_channels),
            BatchNorm(n_channels, relu),
            MaxPool((3,3)),
        Flux.flatten,
        Dense(n_channels, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, 3)
    ) |> Flux.gpu

    return model

end


function fc()
    n_channels = 110
    model = Chain(
        Dense(n_channels, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, 256, relu),
        Dense(256, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 3, relu)
    ) |> Flux.gpu

    return model
    
end