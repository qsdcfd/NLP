with torch.no_grad():
    import random

    index = int(random.random() * test_x.size(0))

    recon = model(test_x[index].view(1,-1)).squeeze()

    show_image(test_x[index])
    show_image(recon)