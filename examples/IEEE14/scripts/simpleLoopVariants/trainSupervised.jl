

function trainModelSupervised(model, optimizer, train_dataloader, test_in, test_out; epochs=100)
    ps = Flux.params(model)
    opt_state = Flux.setup(optimizer, model)
    test_loss_history = []
    training_time = 0
    
    for epoch in 1:epochs
        for (x, y) in train_dataloader
            lv, grads = Flux.withgradient(model) do m  
                prediction = m(x)
                Flux.mse(prediction, y)
            end
            Flux.update!(opt_state, model, grads[1])
        end
        push!(test_loss_history, Flux.mse(model(test_in), test_out))
    end
    return model, test_loss_history, training_time
end