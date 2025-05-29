from .train_config import *

def train_one_epoch(model, train_loader, loss_fn, optimizer, 
                    metric, scheduler= None, epoch= None, clip= 1):
    model.train()
    loss_train= MeanMetric()
    metric.reset()

    with tqdm.tqdm(train_loader, unit= 'batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch: tepoch.set_description(f'Epoch {epoch}')

            inputs, targets= inputs.to(device), targets.to(device)
            outputs= model(inputs, targets[:, :-1])

            loss= loss_fn(outputs.permute(1, 2, 0), targets[:, 1:])
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm= clip)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler: scheduler.step()
            loss_train.update(loss.item(), weight= len(targets))

            outputs, targets= postprocess(outputs, targets)
            metric.update(outputs, targets)

            tepoch.set_postfix(loss= loss_train.compute().item(),
                                metric= metric.compute().item())

    return model, loss_train.compute().item(), metric.compute().item()


def evaluate(model, test_loader, loss_fn, metric):
    model.eval()
    loss_eval= MeanMetric()
    metric.reset()

    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs, targets= inputs.to(device), targets.to(device)
            outputs= model(inputs, targets[:, :-1])

            loss= loss_fn(outputs.permute(0, 2, 1), targets[:, 1:])
            loss_eval.update(loss.item(), weight= len(targets))

            outputs, targets= postprocess(outputs, targets)
            metric.update(outputs, targets)
    return loss_eval.compute().item(), metric.compute().item()