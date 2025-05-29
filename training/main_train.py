from .one_epoch_fn import *

num_epochs= 30

loss_train_hist, loss_valid_hist= [], []
metric_train_hist, metric_valid_hist= [], []
best_loss_valid, epoch_counter= torch.inf, 0

for epoch in range(num_epochs):
    model, loss_train, metric_train= train_one_epoch(model, train_loader, loss_fn,
                                                     optimizer, metric, None, epoch+1)

    loss_valid, metric_valid= evaluate(model, valid_loader, loss_fn, metric)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)
    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)

    if loss_valid < best_loss_valid:
        torch.save(model, f'./output/model.pt')
        best_loss_valid = loss_valid
        print('Model Saved!')

    print(f'Valid: Loss= {loss_valid:.4}, Metric= {metric_valid:.4}')
    print()

    epoch_counter +=1