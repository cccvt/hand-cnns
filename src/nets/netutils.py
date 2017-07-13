def print_net(model):
    num_params = [param.numel() for param in model.parameters()]
    print('---- Network ----')
    print('Total number of parameters: {0}'.format(sum(num_params)))
    print(model)

