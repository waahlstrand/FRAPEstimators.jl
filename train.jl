using FRAP

function train!(loss, θ, data::DataGenerator, optimizer)
    # Trains a model with parameters θ with respect to a 
    # loss function and optimizer for a number of iterations in a dataset
        
    # p = Progress(length(data); showspeed=true)

    # Declare loss to make it loggable
    local L

    # Flux gradient wants a Params struct
    θ = θ |> Params

    # Yield new generated data in a loop
    for (i, (x, y)) in enumerate(data)

        # Calculate the gradients from the loss
        ∇θ = gradient(θ) do

            L = loss(x, y)

            return L
        end

        # Update progressmeter
        # next!(p; showvalues = [(:L,L)])

        @info "$i\t$L"

        # Update the weights and optimizer
        Flux.update!(optimizer, θ, ∇θ)

    end
end