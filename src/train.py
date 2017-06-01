import torch


def train_net(dataloader, model, optimizer, criterion, epochs=10):
    for i, (image, target) in enumerate(dataloader):
        target = target.cuda()
        input_var = torch.autorgrad.Variable(image)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do descent step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
