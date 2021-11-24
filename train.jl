using FRAP

function train!(loss, θ, data::DataGenerator, optimizer)
    # Trains a model with parameters θ with respect to a 
    # loss function and optimizer for a number of iterations in a dataset

    # p = Progress(length(data); showspeed=true)
    f = open("losses.csv", "w")
    write(f, string("batch;loss;time\n"))
    close(f)
    f = open("losses.csv", "a")

    # Declare loss to make it loggable
    local L

    # Flux gradient wants a Params struct
    θ = θ |> Params

    # Yield new generated data in a loop
    t_old = time_ns()
    for (i, (x, y)) in enumerate(data)

        # Calculate the gradients from the loss
        ∇θ = gradient(θ) do

            L = loss(x, y)

            return L
        end

        # Update progressmeter
        # next!(p; showvalues = [(:L,L)])

        #@info "$i\t$L"

        # Update the weights and optimizer
        Flux.update!(optimizer, θ, ∇θ)

        # Save results to file.
        t_new = time_ns()
        t_exec = (t_new - t_old) / 1.0e9
        t_old = t_new
        print(string(i, ";", L, ";", t_exec, "\n"))
        write(f, string(i, ";", L, ";", t_exec, "\n"))
        flush(f)
    end
    close(f)
end